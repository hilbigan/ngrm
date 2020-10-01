use abng::{generate_similarity_matrix_string, App};
use std::path::PathBuf;
use std::time::SystemTime;
use structopt::*;

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

fn main() {
    let opt: CLIArgs = CLIArgs::from_args();
    let mut app = App::new(opt.n, opt.m.unwrap_or(abng::DEFAULT_VLEN), opt.files.iter().map(|p| p.into()).collect(), true);

    let mut now = SystemTime::now();
    let basis = app.generate_basis();

    println!(
        "Time (generate_basis): {}",
        now.elapsed().unwrap().as_millis()
    );
    now = SystemTime::now();

    let file_vecs = app.build_file_vectors(basis);
    println!(
        "Time (build_file_vectors): {}",
        now.elapsed().unwrap().as_millis()
    );
    now = SystemTime::now();
    let matrix = generate_similarity_matrix_string(file_vecs);
    println!(
        "Time (generate_similarity_matrix): {}",
        now.elapsed().unwrap().as_millis()
    );

    //println!("{}", matrix);
}

/*
   let long_vec = vec![0usize; 100_000];
   let file_vecs = (0..300).into_iter().map(|i| { FileVec { vec: long_vec.clone(), euclidean_len: 1.0 } }).collect();
*/
