use either::Either;
use memmap::Mmap;
use num_traits::ToPrimitive;
use parking_lot::Mutex;
use radix_trie::Trie;
use radix_trie::TrieCommon;
use rayon::prelude::*;
use std::collections::{HashSet, VecDeque, HashMap};
use std::fmt::Write;
use std::fs;
use std::fs::File;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use t1ha::T1haHashMap;
use rand::Rng;
use itertools::{Itertools, MinMaxResult};

pub const DEFAULT_VLEN: usize = 100_000;
pub const MAX_RETRY_ON_COLLISION: usize = 500;

pub type SparseVector = sprs::CsVec<f32>;
pub type BasisVector = Vec<u8>;

pub fn euclidean_len(vec: &SparseVector) -> f32 {
    vec.l2_norm()
}

#[inline]
pub fn vec_2_sparse_vec<T>(vec: Vec<T>, len: usize) -> SparseVector
where
    T: ToPrimitive + PartialEq,
{
    let mut sparse_vec = SparseVector::empty(len);
    for (i, v) in vec.iter().enumerate() {
        if (*v).to_isize().unwrap() == 0 {
            continue;
        }
        sparse_vec.append(i, (*v).to_f32().unwrap());
    }
    sparse_vec
}

pub struct DataSource {
    source: Either<PathBuf, Vec<u8>>,
    len: usize,
    open: bool,
    mmap: Option<Mmap>,
}

impl DataSource {
    fn is_file(&self) -> bool {
        return self.source.is_left();
    }

    fn get(&self, range: Range<usize>) -> Vec<u8> {
        if !self.open {
            panic!("Can not read from data source before 'open'!");
        }

        if self.is_file() {
            self.mmap.as_ref().unwrap()[range].to_owned()
        } else {
            self.source.as_ref().expect_right("vec source")[range].to_owned()
        }
    }

    fn open(&mut self) {
        if self.is_file() {
            let the_file =
                File::open(&self.source.as_ref().expect_left("file source")).expect("open");
            self.mmap = Some(unsafe { Mmap::map(&the_file).expect("mmap") });
        }
        self.open = true;
    }

    fn close(&mut self) {
        if self.is_file() {
            self.mmap = None;
        }
        self.open = false;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn read_all(&self) -> Vec<u8> {
        if self.is_file() {
            fs::read(self.source.as_ref().expect_left("file source")).expect("read")
        } else {
            self.source.as_ref().expect_right("vec source").clone()
        }
    }
    
    fn description(&self) -> String {
        if self.is_file() {
            self.source.as_ref().expect_left("file source").to_str().unwrap().to_string()
        } else {
            format!("Vec DataSource len = {}", self.source.as_ref().expect_right("vec source").len())
        }
    }
}

impl Clone for DataSource {
    fn clone(&self) -> Self {
        DataSource {
            source: self.source.clone(),
            len: self.len.clone(),
            open: false,
            mmap: None,
        }
    }
}

impl From<&PathBuf> for DataSource {
    fn from(p: &PathBuf) -> Self {
        let len = p.metadata().unwrap().len() as usize;
        DataSource {
            source: Either::Left(p.clone()),
            len,
            open: false,
            mmap: None,
        }
    }
}

impl From<Vec<u8>> for DataSource {
    fn from(v: Vec<u8>) -> Self {
        let len = v.len();
        DataSource {
            source: Either::Right(v),
            len,
            open: false,
            mmap: None,
        }
    }
}

#[derive(Default)]
pub struct StatisticsCollector {
    pub sequence_frequency: Vec<(Vec<u8>, usize)>
}

#[derive(Debug, PartialEq, Clone)]
pub struct DataVec {
    pub vec: SparseVector,
    pub euclidean_len: f32,
}

#[derive(Clone)]
pub struct App {
    n: usize,
    vlen: usize,
    data_vec: Vec<DataSource>,
    data_count: usize,
    data_sizes: Vec<usize>,
    data_sizes_sums: Vec<usize>,
    debug: bool,
}

impl App {
    pub fn new(n: usize, vlen: usize, data_vec: Vec<DataSource>, debug: bool) -> Self {
        let data_sizes = data_vec.iter().map(|d| d.len()).collect::<Vec<_>>();
        let mut data_sizes_sums = Vec::with_capacity(data_vec.len());
        let mut sum = 0;
        for data in &data_vec {
            sum += data.len();
            data_sizes_sums.push(sum);
        }
        let data_count = data_vec.len();

        if debug {
            println!("n = {}", n);
            println!("m = {}", vlen);
        }

        App {
            n,
            vlen,
            data_vec,
            data_count,
            data_sizes,
            data_sizes_sums,
            debug,
        }
    }
    
    pub fn mapping_string(&self) -> String {
        let mut map = String::new();
        for (index, data) in self.data_vec.iter().enumerate() {
            writeln!(&mut map, "{} {}", index, data.description());
        }
        map
    }

    /// Selects m (== `vlen`) random byte sequences with length n from
    /// all the provided files, and returns the resulting sequences in a trie.
    pub fn generate_basis(&mut self) -> Trie<BasisVector, ()> {
        // Generate vlen random indices
        let mut indices = (0..self.vlen)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut data_index = 0;
                let mut byte_index = 0;
                loop { // FIXME this may loop infinitely if no file > n exists
                    let random = rng.gen_range(0, *self.data_sizes_sums.last().unwrap() as u64) as usize;
                    data_index = 0;
                    loop {
                        if self.data_sizes_sums[data_index] > random {
                            break;
                        }
                        data_index += 1;
                    }
                    let data_size = self.data_sizes[data_index];
                    if data_size < self.n {
                        continue;
                    } else if data_size == self.n {
                        byte_index = 0;
                    } else {
                        byte_index = rng.gen_range(0, (data_size - self.n) as u64);
                    }
                    
                    break;
                }
                
                (data_index, byte_index as usize)
            })
            .collect::<Vec<(usize, usize)>>();

        indices.sort_by_key(|i| i.0);

        let mut basis = Trie::new();
        let mut current_data_index = indices[0].0;
        let mut current_data = &mut self.data_vec[current_data_index];
        current_data.open();

        for (data_index, byte_index) in indices {
            if current_data_index != data_index {
                current_data_index = data_index;
                current_data.close();
                current_data = &mut self.data_vec[data_index];
                current_data.open();
            }

            // Try to add this sequence or one of the following ones until out of retries
            let mut index = byte_index;
            let mut bytes = current_data.get(index..(index + self.n));
            while basis.insert(bytes, ()) != None
                && index + 1 + self.n <= self.data_sizes[data_index] as usize
                && index + 1 - byte_index < MAX_RETRY_ON_COLLISION
            {
                index += 1;
                bytes = current_data.get(index..(index + self.n));
            }
        }

        current_data.close();

        if basis.len() < self.vlen && self.debug {
            println!(
                "Warning: Could not find enough distinct byte sequences (found: {}; requested: {}; {:.2}%)",
                basis.len(),
                self.vlen,
                basis.len() as f32 / self.vlen as f32 * 100.0
            );
            println!("Warning: Remaining will be padded with zero!");
        }

        return basis;
    }

    /// Given a set of sequences to count (the "basis"), this methods
    /// goes through all files and counts the number of occurences
    /// of each sequence. This results in one sparse vector (`DataVec`) for
    /// each file.
    pub fn build_file_vectors<'a>(&self, basis: Trie<BasisVector, ()>, statistics_collector: impl Into<Option<&'a mut StatisticsCollector>>) -> Vec<DataVec> {
        let optional_collector: Option<&mut StatisticsCollector> = Into::into(statistics_collector);
        let collect_statistics = optional_collector.is_some();
        
        let mut hashed_seqs: Vec<u64> = basis
            .iter()
            .map(|(sequence, _)| RollingHash::new(self.n).feed_slice(sequence))
            .collect();
        let minimum;
        let maximum;
        if let MinMaxResult::MinMax(&min, &max) = hashed_seqs.iter().minmax() {
            minimum = min;
            maximum = max;
        } else {
            panic!("minmax");
        }
        
        let mut indexmap = T1haHashMap::with_capacity_and_hasher(self.vlen, Default::default());
        hashed_seqs.iter().enumerate().for_each(|(index, sequence)| {
            indexmap.insert(sequence, index);
        });

        let file_vecs: Vec<DataVec> = self
            .data_vec
            .par_iter()
            .map(|file: &DataSource| {
                let mut vec = vec![0; self.vlen];
                let bytes = file.read_all();
                let mut roll = RollingHash::new(self.n);
                if bytes.len() >= self.n {
                    for byte in bytes {
                        let hash = roll.feed(byte);
                        if !roll.valid()
                            || hash < minimum
                            || hash > maximum
                        {
                            continue;
                        } else if let Some(index) = indexmap.get(&hash) {
                            vec[*index] += 1; //thread '<unnamed>' panicked at 'index out of bounds: the len is 10000000 but the index is 32294021', src/lib.rs:344:29
                        }
                    }
                }

                let sparse_vec = vec_2_sparse_vec(vec, self.vlen);
                let len = euclidean_len(&sparse_vec);
                DataVec {
                    vec: sparse_vec,
                    euclidean_len: len,
                }
            })
            .collect();
        
        if collect_statistics {
            let sparse_vec = file_vecs.iter().fold(vec_2_sparse_vec(vec![0.0; self.vlen], self.vlen), |mut acc, x| { acc + &x.vec });
            let mut sequence_frequency: Vec<_> = basis.iter().enumerate().map(|(index, (sequence, _))| (sequence.clone(), *sparse_vec.get(index).unwrap_or(&0.0) as usize)).collect();
            sequence_frequency.sort_by_key(|s| s.1);
            sequence_frequency.reverse();
            optional_collector.unwrap().sequence_frequency = sequence_frequency;
        }

        return file_vecs;
    }
}

pub fn cosine_similarity(fv0: &DataVec, fv1: &DataVec) -> f32 {
    fv0.vec.dot(&fv1.vec) / (fv0.euclidean_len * fv1.euclidean_len)
}

#[allow(clippy::needless_range_loop)]
pub fn generate_similarity_matrix_string(vecs: Vec<DataVec>) -> String {
    let n = vecs.len();
    let matrix = Mutex::new(vec![vec![0.0; n]; n]);
    let vecs = Arc::new(vecs);
    let mut indices: Vec<(usize, usize)> = Vec::with_capacity(n * n);

    for j in 0..n {
        for i in j..n {
            indices.push((i, j));
        }
    }

    indices.par_iter().for_each(|(i, j)| {
        let i = *i;
        let j = *j;
        if vecs[i].euclidean_len != 0.0 && vecs[j].euclidean_len != 0.0 {
            let sim = if i == j {
                1.0
            } else {
                cosine_similarity(&vecs[i], &vecs[j])
            };
            {
                let mut m = matrix.lock();
                m[i][j] = sim;
                m[j][i] = sim;
            }
        }
    });

    let matrix = matrix.lock();
    let mut mat = String::new();
    for i in 0..n {
        for j in 0..n {
            write!(&mut mat, "{:4.4}", matrix[i][j]).expect("write");
            if j < n - 1 {
                write!(&mut mat, ",").expect("write");
            }
        }
        writeln!(&mut mat).expect("writeln");
    }
    mat
}

struct RollingHash {
    buffer: VecDeque<u8>,
    len: usize,
    read: usize,
    a: usize,
    hash: u64,
}

impl RollingHash {
    pub fn new(len: usize) -> Self {
        RollingHash {
            buffer: VecDeque::with_capacity(len),
            len,
            a: 31,
            read: 0,
            hash: 0,
        }
    }

    pub fn feed_slice(&mut self, bytes: &[u8]) -> u64 {
        for byte in bytes {
            self.feed(*byte);
        }
        self.hash
    }

    pub fn feed(&mut self, byte: u8) -> u64 {
        self.buffer.push_back(byte);
        self.hash *= self.a as u64;
        self.hash += byte as u64;
        if self.read >= self.len {
            self.hash -=
                self.a.pow(self.len as u32) as u64 * self.buffer.pop_front().unwrap() as u64;
        }
        self.read += 1;
        self.hash
    }

    pub fn valid(&self) -> bool {
        self.read >= self.len
    }
}

#[cfg(test)]
mod tests {
    use crate::{App, DataSource, BasisVector, DataVec, vec_2_sparse_vec, generate_similarity_matrix_string, StatisticsCollector};
    use radix_trie::Trie;
    use radix_trie::TrieCommon;
    use num_traits::Pow;
    use num_traits::real::Real;
    
    fn contains(basis: &Trie<BasisVector, ()>, elem: &BasisVector) -> bool {
        basis.get(elem) == Some(&())
    }
    
    #[test]
    fn test_basis_1(){
        let mut app = App::new(4, 2, vec![
            DataSource::from(vec![0x1, 0x2, 0x3, 0x4]),
            DataSource::from(vec![0x5, 0x6, 0x7, 0x8])
        ], true);
        
        let basis = app.generate_basis();
        assert!(contains(&basis, &vec![0x1, 0x2, 0x3, 0x4]) || contains(&basis, &vec![0x5, 0x6, 0x7, 0x8]));
    }
    
    #[test]
    fn test_basis_2(){
        let mut app = App::new(2, 2, vec![
            DataSource::from(vec![0x1, 0x2, 0x3, 0x4]),
            DataSource::from(vec![0x5, 0x6, 0x7, 0x8])
        ], true);
        
        let basis = app.generate_basis();
        assert!(basis.len() == 2);
    }
    
    #[test]
    fn test_basis_3(){
        let mut vec = vec![0x1; 1000];
        vec[499] = 0x2;
        vec[999] = 0x2;
        let mut app = App::new(2, 2, vec![
            DataSource::from(vec)
        ], true);
        
        let basis = app.generate_basis();
        assert!(basis.len() == 2);
    }
    
    #[test]
    fn test_file_vector_1(){
        let mut app = App::new(2, 2, vec![
            DataSource::from(vec![0x1, 0x2, 0x3, 0x4]),
            DataSource::from(vec![0x5, 0x3, 0x4])
        ], true);
        
        let mut basis = Trie::new();
        basis.insert(vec![0x1, 0x2], ());
        basis.insert(vec![0x3, 0x4], ());
        
        let vecs = app.build_file_vectors(basis, None);
        
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], DataVec { vec: vec_2_sparse_vec(vec![1, 1], 2), euclidean_len: 2.0f32.sqrt() } );
        assert_eq!(vecs[1], DataVec { vec: vec_2_sparse_vec(vec![0, 1], 2), euclidean_len: 1.0 } );
    }
    
    #[test]
    fn test_file_vector_2(){
        let mut vec = vec![0x1; 1000];
        vec[599] = 0x2;
        let mut app = App::new(2, 2, vec![
            DataSource::from(vec)
        ], true);
        
        let mut basis = Trie::new();
        basis.insert(vec![0x1, 0x1], ());
        basis.insert(vec![0x1, 0x2], ());
        
        let vecs = app.build_file_vectors(basis, None);
    
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0], DataVec { vec: vec_2_sparse_vec(vec![997, 1], 2), euclidean_len: ((997.0f32.pow(2) + 1.0f32) as f32).sqrt() } );
    }
    
    #[test]
    fn test_matrix_1(){
        let mut app = App::new(2, 2, vec![
            DataSource::from(vec![0x1, 0x2, 0x3, 0x4]),
            DataSource::from(vec![0x5, 0x3, 0x4])
        ], true);
        
        let mut basis = Trie::new();
        basis.insert(vec![0x1, 0x2], ());
        basis.insert(vec![0x3, 0x4], ());
        let vecs = app.build_file_vectors(basis, None);
        
        let mut str = generate_similarity_matrix_string(vecs);
        assert_eq!(str, format!("{:4.4},{:4.4}\n{:4.4},{:4.4}\n", 1.0, 1.0 / 2.0f32.sqrt(),  1.0 / 2.0f32.sqrt(), 1.0));
    }
    
    #[test]
    fn test_matrix_2(){
        let mut vec = vec![0x1; 1000];
        vec[599] = 0x2;
        let mut app = App::new(2, 2, vec![
            DataSource::from(vec)
        ], true);
        
        let mut basis = Trie::new();
        basis.insert(vec![0x1, 0x1], ());
        basis.insert(vec![0x1, 0x2], ());
        let vecs = app.build_file_vectors(basis, None);
    
        let mut str = generate_similarity_matrix_string(vecs);
        assert_eq!(str, format!("{:4.4}\n", 1.0));
    }
    
    #[test]
    fn test_matrix_3(){
        let mut app = App::new(2, 2, vec![
            DataSource::from(vec![0x1, 0x2, 0x3, 0x4]),           // 1, 1, len = sqrt(2)
            DataSource::from(vec![0x5, 0x3, 0x4]),                // 0, 1, len = 1
            DataSource::from(vec![0x5, 0x1, 0x2]),                // 1, 0, len = 1
            DataSource::from(vec![0x1, 0x2, 0x3, 0x4, 0x3, 0x4]), // 1, 2, len = sqrt(5)
        ], true);
    
        let mut basis = Trie::new();
        basis.insert(vec![0x1, 0x2], ());
        basis.insert(vec![0x3, 0x4], ());
        let vecs = app.build_file_vectors(basis, None);
    
        let sqrt2 = 2.0f32.sqrt();
        let sqrt5 = 5.0f32.sqrt();
        let sqrt10 = 10.0f32.sqrt();
        let mut str = generate_similarity_matrix_string(vecs);
        assert_eq!(str, format!("\
            {:4.4},{:4.4},{:4.4},{:4.4}\n\
            {:4.4},{:4.4},{:4.4},{:4.4}\n\
            {:4.4},{:4.4},{:4.4},{:4.4}\n\
            {:4.4},{:4.4},{:4.4},{:4.4}\n",

            1.0,          1.0 / sqrt2, 1.0 / sqrt2, 3.0 / sqrt10,
            1.0 / sqrt2,  1.0,         0.0,         2.0 / sqrt5,
            1.0 / sqrt2,  0.0,         1.0,         1.0 / sqrt5,
            3.0 / sqrt10, 2.0 / sqrt5, 1.0 / sqrt5, 1.0
        ));
    }
    
    #[test]
    fn test_statistics(){
        let mut app = App::new(2, 2, vec![
            DataSource::from(vec![0x1, 0x2, 0x3, 0x4]),           // 1, 1, len = sqrt(2)
            DataSource::from(vec![0x5, 0x3, 0x4]),                // 0, 1, len = 1
            DataSource::from(vec![0x5, 0x1, 0x2]),                // 1, 0, len = 1
            DataSource::from(vec![0x1, 0x2, 0x3, 0x4, 0x3, 0x4]), // 1, 2, len = sqrt(5)
        ], true);
        
        let mut basis = Trie::new();
        basis.insert(vec![0x1, 0x2], ());
        basis.insert(vec![0x3, 0x4], ());
        let mut stats = StatisticsCollector::default();
        app.build_file_vectors(basis, &mut stats);
        assert!(stats.sequence_frequency.contains(&(vec![0x1, 0x2], 3)));
        assert!(stats.sequence_frequency.contains(&(vec![0x3, 0x4], 4)));
        assert_eq!(stats.sequence_frequency.len(), 2);
    }
}
