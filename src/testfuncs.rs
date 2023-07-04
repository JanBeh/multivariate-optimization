//! Test problems for optimization algorithms.

/// Rastrigin function.
///
/// Typically used from `-5.12..=5.12`. Minimum is at `0.0`.
pub fn rastrigin(args: &[f64]) -> f64 {
    const A: f64 = 10.0;
    const PI: f64 = std::f64::consts::PI;
    let n = args.len() as f64;
    A * n
        + args
            .iter()
            .copied()
            .map(|x| x * x - A * (2.0 * PI * x).cos())
            .sum::<f64>()
}

/// Sum of Rosenbrock functions.
///
/// Multidimensional generalization of Rosenbrock function.
/// If `args.len() % 2 == 1`, then the last argument is ignored.
pub fn rosenbrock(mut args: &[f64]) -> f64 {
    const A: f64 = 1.0;
    const B: f64 = 100.0;
    let mut sum: f64 = 0.0;
    while args.len() >= 2 {
        let x = args[0];
        let y = args[1];
        sum += (A - x).powi(2) + B * (y - x.powi(2)).powi(2);
        args = &args[2..];
    }
    sum
}
