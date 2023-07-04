use multivariate_optimization::distributions::*;

use criterion::{BatchSize, Criterion};
use rand::{distributions::Distribution, thread_rng};

fn main() {
    let mut c = Criterion::default().configure_from_args();
    c.bench_function("StdNormDist::sample", |b| {
        let rng = &mut thread_rng();
        b.iter(|| {
            let x: f64 = StdNormDist.sample(rng);
            x
        });
    });
    let mut c = Criterion::default().configure_from_args();
    c.bench_function("StdNormDistPair::sample", |b| {
        let rng = &mut thread_rng();
        b.iter(|| {
            let (x, y): (f64, f64) = StdNormDistPair.sample(rng);
            (x, y)
        });
    });
    c.bench_function("NormDist::sample", |b| {
        let rng = &mut thread_rng();
        b.iter(|| {
            NormDist::<f64> {
                average: 1.0,
                stddev: 2.0,
            }
            .sample(rng)
        });
    });
    c.bench_function("MultivarNormDist::new", |b| {
        let averages: Vec<f64> = vec![1.0, 2.0, 3.0];
        let covariances = Triangular::<f64>::new(3, |idx| match idx {
            (0, 0) => 1.0,
            (1, 1) => 0.5,
            (2, 2) => 2.0,
            (2, 0) => 0.25,
            (0, 2) => 0.25,
            _ => 0.0,
        });
        b.iter_batched(
            || (averages.clone(), covariances.clone()),
            |(averages, covariances)| MultivarNormDist::new(averages, covariances),
            BatchSize::SmallInput,
        );
    });
    c.bench_function("MultivarNormDist(3)::sample", |b| {
        let rng = &mut thread_rng();
        let averages: Vec<f64> = vec![1.0, 2.0, 3.0];
        let covariances = Triangular::<f64>::new(3, |idx| match idx {
            (0, 0) => 1.0,
            (1, 1) => 0.5,
            (2, 2) => 2.0,
            (2, 0) => 0.25,
            (0, 2) => 0.25,
            _ => 0.0,
        });
        let d = MultivarNormDist::new(averages, covariances);
        b.iter(|| d.sample(rng));
    });
    c.bench_function("MultivarNormDist(100)::sample", |b| {
        let rng = &mut thread_rng();
        let averages: Vec<f64> = vec![0.0; 100];
        let covariances =
            Triangular::<f64>::new(100, |(row, col)| if row == col { 1.0 } else { 0.0 });
        let d = MultivarNormDist::new(averages, covariances);
        b.iter(|| d.sample(rng));
    });
}
