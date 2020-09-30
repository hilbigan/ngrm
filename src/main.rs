use memmap::Mmap;
use std::fs::File;
use std::path::PathBuf;
use structopt::*;
use radix_trie::Trie;
use radix_trie::TrieCommon;
use rand::Rng;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fs;
use parking_lot::Mutex;
use std::sync::Arc;
use std::fmt::Write;
use std::time::SystemTime;

const DEFAULT_VLEN: usize = 5;
const MAX_RETRY_ON_COLLISION: usize = 500;

#[derive(StructOpt, Debug)]
#[structopt(name = "bngram")]
struct CLIArgs {
    #[structopt(short)]
    n: usize,

    #[structopt(short, long = "vector-length")]
    m: Option<usize>,

    #[structopt(name = "FILES", parse(from_os_str))]
    files: Vec<PathBuf>,
}

struct FileVec {
    vec: Vec<usize>,
    euclidean_len: f32
}

struct App {
    n: usize,
    vlen: usize,
    files: Vec<PathBuf>,
    files_len: usize,
    file_sizes: Vec<u64>,
    file_size_sums: Vec<u64>,
}

impl App {
    pub fn new(n: usize, vlen: usize, files: Vec<PathBuf>) -> Self {
        let file_sizes = files
            .iter()
            .map(|f| f.metadata().expect("metadata").len())
            .collect::<Vec<_>>();

        let mut file_size_sums = Vec::with_capacity(files.len());
        let mut sum = 0;
        for file in &files {
            sum += file.metadata().expect("metadata").len();
            file_size_sums.push(sum);
        }

        println!("n = {}", n);
        println!("m = {}", vlen);
        //println!("Files: {:?}", file_sizes);
        //println!("FileSums: {:?}", file_size_sums);

        let files_len = files.len();
        App {
            n,
            vlen,
            files,
            files_len,
            file_sizes,
            file_size_sums,
        }
    }

    pub fn generate_basis(&self) -> Vec<Vec<u8>> {
        // Generate vlen random indices
        let mut indices = (0..self.vlen)
            .into_iter()
            .map(|_| {
                let mut thread_rng = rand::thread_rng();
                let random = thread_rng.gen_range(0, self.file_size_sums.last().unwrap());
                let mut file_index = 0;
                loop {
                    if self.file_size_sums[file_index] > random {
                        break;
                    }
                    file_index += 1;
                }
                let file_size = self.file_sizes[file_index];
                let byte_index = thread_rng.gen_range(0, file_size - self.n as u64);

                (file_index, byte_index as usize)
            })
            .collect::<Vec<(usize, usize)>>();

        indices.sort_by_key(|i| i.0);

        //println!("Indices: {:?}", indices);

        let mut basis = Trie::new();
        let mut current_file = indices[0].0;
        let the_file = File::open(&self.files[current_file]).expect("open");
        let mut mmap = unsafe { Mmap::map(&the_file).expect("mmap") };
        for (file_index, byte_index) in indices {
            if current_file != file_index {
                current_file = file_index;
                let the_file = File::open(&self.files[file_index]).expect("open");
                mmap = unsafe { Mmap::map(&the_file).expect("mmap") };
            }
            if self.file_sizes[current_file] < self.n as u64 {
                println!("Warning: Tried to sample file that was smaller that n = {}! (actual file size: {})", self.n, self.file_sizes[current_file]);
                continue;
            }

            // Try to add this sequence or one of the following ones until out of retries
            let mut index = byte_index;
            let mut bytes = mmap[index..(index + self.n)].to_owned();
            while basis.insert(bytes, ()) != None
                && index + 1 + self.n <= self.file_sizes[file_index] as usize
                && index + 1 - byte_index < MAX_RETRY_ON_COLLISION
            {
                index += 1;
                bytes = mmap[index..(index + self.n)].to_owned();
            }
        }

        let mut basis_vec = Vec::with_capacity(self.vlen);
        basis.iter().for_each(|(k, _)| {
            basis_vec.push(k.clone());
        });

        //println!("Basis: {:?}", basis_vec);

        return basis_vec;
    }

    pub fn build_file_vectors(&self, basis: Vec<Vec<u8>>) -> Vec<FileVec> {
        let mut hashed_seqs = basis
            .iter()
            .map(|sequence| RollingHash::new(self.n).feed_slice(sequence))
            .collect::<Vec<u64>>();
        hashed_seqs.sort();

        //println!("Hashes: {:?}", hashed_seqs);
        let minimum = hashed_seqs[0];
        let maximum = *hashed_seqs.last().unwrap();

        let mut file_vecs = self.files.par_iter()
            .map(|file| {
                let mut vec = vec![0usize; self.vlen];
                let bytes = fs::read(file).expect("read");
                let mut roll = RollingHash::new(self.n);
                if bytes.len() >= self.n {
                    for byte in bytes {
                        let hash = roll.feed(byte);
                        if !roll.valid() || hash < minimum || hash > maximum {
                            continue;
                        } else if let Ok(index) = hashed_seqs.binary_search(&hash) {
                            vec[index] += 1;
                        }
                    }
                }
    
                //println!("FVec: {:?}", vec);
                let len = vec.iter().map(|x| (x * x) as f32).sum::<f32>().sqrt();
                FileVec { vec, euclidean_len: len }
            })
            .collect();
        
        return file_vecs;
    }
}

#[allow(clippy::needless_range_loop)]
fn generate_similarity_matrix_string(vecs: Vec<FileVec>) -> String {
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
        let sim = if i == j { 1.0 } else { cosine_similarity(&vecs[i], &vecs[j]) };
        {
            let mut m = matrix.lock();
            m[i][j] = sim;
            m[j][i] = sim;
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

fn cosine_similarity(fv0: &FileVec, fv1: &FileVec) -> f32 {
    let elem_sum: f32 = (0..fv0.vec.len())
        .collect::<Vec<usize>>()
        .iter()
        .map(|i| (fv0.vec[*i] * fv1.vec[*i]) as f32)
        .sum();
    
    elem_sum / (fv0.euclidean_len * fv1.euclidean_len)
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

fn main() {
    let opt: CLIArgs = CLIArgs::from_args();
    let app = App::new(opt.n, opt.m.unwrap_or(DEFAULT_VLEN), opt.files);

    let mut now = SystemTime::now();
    let basis = app.generate_basis();
    if basis.len() < app.vlen {
        println!(
            "Warning: Could not find enough distinct byte sequences (found: {}; requested: {})",
            basis.len(),
            app.vlen
        );
        println!("Warning: Remaining will be padded with zero!");
    }
    println!("Time (generate_basis): {}", now.elapsed().unwrap().as_millis());
    now = SystemTime::now();
    
    let file_vecs = app.build_file_vectors(basis);
    println!("Time (build_file_vectors): {}", now.elapsed().unwrap().as_millis());
    now = SystemTime::now();
    let matrix = generate_similarity_matrix_string(file_vecs);
    println!("Time (generate_similarity_matrix): {}", now.elapsed().unwrap().as_millis());
    
    //println!("{}", matrix);
}
