use abng::{generate_similarity_matrix_string, App};
use std::path::PathBuf;
use std::time::SystemTime;
use structopt::*;
use std::{fs, io};
use std::io::BufRead;

#[derive(StructOpt, Debug)]
#[structopt(name = "Approximate Byte n-gram Analysis Tool", author = "Aaron Hilbig")]
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

    /// Silence output.
    #[structopt(short, long)]
    silent: bool,
    
    /// Read input file paths from stdin instead of the arguments
    #[structopt(long)]
    stdin: bool,
    
    /// Output file for similarity matrix
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,
    
    /// Optional output file with a mapping of file -> index in matrix
    #[structopt(short = "M", long, parse(from_os_str))]
    mapping: Option<PathBuf>,

    /// Input files.
    #[structopt(name = "FILES", parse(from_os_str))]
    files: Vec<PathBuf>,
}

fn main() {
    let opt: CLIArgs = CLIArgs::from_args();
    let verbose = !opt.silent;
    let files = if opt.stdin {
        io::stdin().lock().lines().map(|line| PathBuf::from(line.unwrap())).collect()
    } else {
        opt.files
    };
    if files.is_empty() {
        eprintln!("Nothing to do.");
        return;
    }
    
    let mut app = App::new(
        opt.n,
        opt.m.unwrap_or(abng::DEFAULT_VLEN),
        files.iter().map(|p| p.into()).collect(),
        verbose,
    );

    let total = SystemTime::now();
    let mut now = SystemTime::now();
    let basis = app.generate_basis();

    if verbose {
        println!(
            "Time (generate_basis): {}",
            now.elapsed().unwrap().as_millis()
        );
    }
    now = SystemTime::now();

    let file_vecs = app.build_file_vectors(basis);

    if verbose {
        println!(
            "Time (build_file_vectors): {}",
            now.elapsed().unwrap().as_millis()
        );
    }
    now = SystemTime::now();
    let matrix = generate_similarity_matrix_string(file_vecs);

    if verbose {
        println!(
            "Time (generate_similarity_matrix): {}",
            now.elapsed().unwrap().as_millis()
        );
    }

    fs::write(&opt.output, matrix);
    
    if opt.mapping.is_some() {
        fs::write(opt.mapping.as_ref().unwrap(), app.mapping_string());
        if verbose {
            println!(
                "Wrote mapping to file: {}",
                opt.mapping.unwrap().to_str().unwrap()
            );
        }
    }
    
    if verbose {
        println!(
            "Wrote output to file: {}",
            opt.output.to_str().unwrap()
        );
        println!(
            "Time (total): {}",
            total.elapsed().unwrap().as_millis()
        );
    }
}
