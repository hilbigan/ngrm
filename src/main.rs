mod lib;

use ngrm::{generate_similarity_matrix_string, App, StatisticsCollector};
use std::io::BufRead;
use std::path::PathBuf;
use std::time::SystemTime;
use std::{fs, io};
use structopt::*;
use std::process::exit;

#[derive(StructOpt, Debug)]
#[structopt(
    name = "Approximate Byte n-gram Analysis Tool",
    author = "Aaron Hilbig"
)]
struct CLIArgs {
    /// The n in n-gram (Byte sequence length).
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

    /// Read input file paths from stdin instead of the arguments.
    #[structopt(long)]
    stdin: bool,
    
    /// Print similarity matrix to stdout (Recommended only for small matrices).
    #[structopt(long)]
    stdout: bool,

    /// Output file for similarity matrix.
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,

    /// Optional output file with a mapping of (file |-> index in matrix).
    #[structopt(short = "M", long, parse(from_os_str))]
    mapping: Option<PathBuf>,

    /// Input files.
    #[structopt(name = "FILES", parse(from_os_str))]
    files: Vec<PathBuf>,

    /// Print top 25 most common sequences.
    #[structopt(long)]
    stats: bool,
}

fn main() {
    let opt: CLIArgs = CLIArgs::from_args();
    let verbose = !opt.silent;
    let files = if opt.stdin {
        io::stdin()
            .lock()
            .lines()
            .map(|line| PathBuf::from(line.unwrap()))
            .collect()
    } else {
        opt.files
    };
    if files.is_empty() {
        eprintln!("Nothing to do.");
        return;
    }
    let files_clone = if opt.stdout {
        Some(files.clone())
    } else {
        None
    };

    let mut app = App::new(
        opt.n,
        opt.m.unwrap_or(ngrm::DEFAULT_VLEN),
        files.iter().map(|p| p.into()).collect(),
        verbose,
    );
    
    if !app.check_input_ok() {
        eprintln!(
            "The provided files are too small! No file is larger than n!",
        );
        exit(1)
    }

    let total = SystemTime::now();
    let mut now = SystemTime::now();
    let basis = app.generate_basis();

    if verbose {
        println!(
            "Time (generate_basis): {} ms",
            now.elapsed().unwrap().as_millis()
        );
    }
    now = SystemTime::now();

    let mut statistics_collector = StatisticsCollector::default();
    let file_vecs = app.build_file_vectors(
        basis,
        if opt.stats {
            Some(&mut statistics_collector)
        } else {
            None
        },
    );

    if verbose {
        println!(
            "Time (build_file_vectors): {} ms",
            now.elapsed().unwrap().as_millis()
        );
    }
    now = SystemTime::now();
    let matrix = generate_similarity_matrix_string(file_vecs);

    if verbose {
        println!(
            "Time (generate_similarity_matrix): {} ms",
            now.elapsed().unwrap().as_millis()
        );
    }

    fs::write(&opt.output, &matrix);
    
    if opt.stdout {
        let files = files_clone.unwrap();
        let file_name = |file: &PathBuf| {
            file.file_name().unwrap().to_str().unwrap().to_string()
        };
        print!("\n       ");
        for f in &files {
            print!("{:6.6} ", file_name(f))
        }
        println!();
        matrix.split("\n")
            .into_iter()
            .enumerate()
            .for_each(|(i, str)| {
                if i < files.len() {
                    println!("{:6.6} {}", file_name(&files[i]), str);
                } else {
                    println!("{}", str);
                }
            });
    }

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
        println!("Wrote output to file: {}", opt.output.to_str().unwrap());
        println!("Time (total): {} ms", total.elapsed().unwrap().as_millis());
    }

    if opt.stats {
        println!("Top 25 most common sequences:");
        statistics_collector
            .sequence_frequency
            .iter()
            .take(25)
            .for_each(|(s, f)| {
                println!("{:02x?} x {}", s, f);
            });
    }
}
