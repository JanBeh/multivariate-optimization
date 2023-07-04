use multivariate_optimization::optimize::*;
use multivariate_optimization::testfuncs::rastrigin;

use criterion::{BatchSize, Criterion};

fn main() {
    let mut c = Criterion::default().configure_from_args();
    c.bench_function("Solver::initialize", |b| {
        const DIM: usize = 100;
        const POPULATION: usize = 1000;
        b.iter_batched(
            || {
                let search_space: Vec<SearchRange> = vec![(-5.12..=5.12).into(); DIM];
                Solver::new(search_space, |params| {
                    let cost = rastrigin(&params);
                    BasicSpecimen { params, cost }
                })
            },
            |mut solver| {
                solver.initialize(POPULATION);
            },
            BatchSize::SmallInput,
        );
    });
    c.bench_function("Solver::evolution", |b| {
        const DIM: usize = 100;
        const POPULATION: usize = 1000;
        b.iter_batched(
            || {
                let search_space: Vec<SearchRange> = vec![(-5.12..=5.12).into(); DIM];
                let mut solver = Solver::new(search_space, |params| {
                    let cost = rastrigin(&params);
                    BasicSpecimen { params, cost }
                });
                solver.initialize(POPULATION);
                solver
            },
            |mut solver| {
                solver.evolution(POPULATION);
            },
            BatchSize::SmallInput,
        );
    });
}
