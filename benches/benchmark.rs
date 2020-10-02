use abng::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::fs;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use rand::distributions::Standard;
use abng::euclidean_len;

fn random_array<T>(rng: &mut ChaCha20Rng, len: usize) -> Vec<T>
    where Standard: Distribution<T>
{
    (0..len).into_iter().map(|_| rng.gen()).collect::<Vec<T>>()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(1337);
    
    c.bench_function("generate_similarity_matrix", |b| {
        let file_vecs: Vec<_> = (0..100).into_iter().map(|i| {
            let long_vec = vec_2_sparse_vec(random_array::<f32>(&mut rng, 10_000), 10_000);
            let len = euclidean_len(&long_vec);
            DataVec {
                vec: long_vec,
                euclidean_len: len
            }
        }).collect();
        
        b.iter(|| generate_similarity_matrix_string(file_vecs.clone()));
    });
    
    let mut group = c.benchmark_group("generate_basis");
    for n in [4, 8].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let mut app = App::new(n,
                                   1_000,
                                   (0..100).into_iter().map(|i| { DataSource::from(random_array::<u8>(&mut rng, 10_000)) }).collect(),
                                   false);
            
            b.iter(|| app.generate_basis());
        });
    }
    group.finish();
    
    let mut group = c.benchmark_group("build_file_vectors");
    for n in [4, 8].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let mut app = App::new(n,
                                   1_000,
                                   (0..50).into_iter().map(|i| { DataSource::from(random_array::<u8>(&mut rng, 10_000)) }).collect(),
                                   false);
            let basis = app.generate_basis();
            
            b.iter(|| app.build_file_vectors(basis.clone()));
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
