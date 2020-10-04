use either::Either;
use memmap::Mmap;
use num_traits::ToPrimitive;
use parking_lot::Mutex;
use radix_trie::Trie;
use radix_trie::TrieCommon;
use oorandom::*;
use rayon::prelude::*;
use std::collections::{HashSet, VecDeque};
use std::fmt::Write;
use std::fs;
use std::fs::File;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

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

#[derive(Clone)]
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

    pub fn generate_basis(&mut self) -> Trie<BasisVector, ()> {
        // Generate vlen random indices
        let mut rng = Rand64::new(1334u128);
        let mut indices = (0..self.vlen)
            .into_iter()
            .map(|_| {
                let mut data_index = 0;
                let mut byte_index = 0;
                loop { // FIXME this may loop infinitely if no file > n exists
                    let random = rng.rand_range(0 .. *self.data_sizes_sums.last().unwrap() as u64) as usize;
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
                        byte_index = rng.rand_range(0 .. (data_size - self.n) as u64);
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
            if self.data_sizes[current_data_index] < self.n {
                if self.debug {
                    println!("Warning: Tried to sample file that was smaller that n = {}! (actual file size: {})", self.n, self.data_sizes[current_data_index]);
                }
                continue;
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

    pub fn build_file_vectors(&self, basis: Trie<BasisVector, ()>) -> Vec<DataVec> {
        let mut hashed_seqs = basis
            .iter()
            .map(|(sequence, _)| RollingHash::new(self.n).feed_slice(sequence))
            .collect::<Vec<u64>>();
        hashed_seqs.sort();

        let minimum = hashed_seqs[0];
        let maximum = *hashed_seqs.last().unwrap();
        let mut hashset = HashSet::with_capacity(self.vlen);
        hashed_seqs.iter().for_each(|sequence| {
            hashset.insert(sequence);
        });

        let file_vecs = self
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
                            || !hashset.contains(&hash)
                        {
                            continue;
                        } else if let Ok(index) = hashed_seqs.binary_search(&hash) {
                            vec[index] += 1;
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

    for i in 0..n {
        for j in i..n {
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
