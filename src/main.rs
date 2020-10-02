use abng::{generate_similarity_matrix_string, App};
use std::path::PathBuf;
use std::time::SystemTime;
use structopt::*;

#[derive(StructOpt, Debug)]
#[structopt(name = "abng")]
struct CLIArgs {
    /// The n in n-gram.
    #[structopt(short)]
    n: usize,

    /// The desired vector size.
    /// Higher values give more precise results.
    /// The required working memory will be O(n * m).
    /// This value should be chosen as high as your RAM permits (default is 100,000).
    #[structopt(short, long = "vector-length")]
    m: Option<usize>,

    /// Debug Level.
    /// -v: Debug messages.
    /// -vv: Debug messages only.
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,

    /// Input files.
    #[structopt(name = "FILES", parse(from_os_str))]
    files: Vec<PathBuf>,
}

fn main() {
    let opt: CLIArgs = CLIArgs::from_args();
    let mut app = App::new(
        opt.n,
        opt.m.unwrap_or(abng::DEFAULT_VLEN),
        opt.files.iter().map(|p| p.into()).collect(),
        opt.verbose > 0,
    );

    let mut now = SystemTime::now();
    let basis = app.generate_basis();

    if opt.verbose > 0 {
        println!(
            "Time (generate_basis): {}",
            now.elapsed().unwrap().as_millis()
        );
    }
    now = SystemTime::now();

    let file_vecs = app.build_file_vectors(basis);

    if opt.verbose > 0 {
        println!(
            "Time (build_file_vectors): {}",
            now.elapsed().unwrap().as_millis()
        );
    }
    now = SystemTime::now();
    let matrix = generate_similarity_matrix_string(file_vecs);

    if opt.verbose > 0 {
        println!(
            "Time (generate_similarity_matrix): {}",
            now.elapsed().unwrap().as_millis()
        );
    }

    if opt.verbose < 2 {
        print!("{}", matrix);
    }
}
