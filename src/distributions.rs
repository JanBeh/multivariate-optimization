//! Univariate and multivariate probability distributions.

pub use crate::triangular::Triangular;

use num::traits::{Float, FloatConst, NumAssignOps, NumCast};
use rand::{
    distributions::{uniform::SampleUniform, Distribution, OpenClosed01},
    Rng,
};

/// Numbers supported by generic items of this module.
pub trait Num
where
    Self: Float + FloatConst + NumCast + NumAssignOps,
    Self: SampleUniform,
{
    /// Generate number between zero (exclusive) and one (inclusive).
    ///
    /// This method is needed due to Rust issue [#20671]:
    /// it avoids having to add `OpenClosed01: Distribution<T>` bounds.
    ///
    /// [#20671]: https://github.com/rust-lang/rust/issues/20671
    fn sample_open_closed_01<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

impl<T> Num for T
where
    T: Float + FloatConst + NumCast + NumAssignOps,
    T: SampleUniform,
    OpenClosed01: Distribution<T>,
{
    fn sample_open_closed_01<R: Rng + ?Sized>(rng: &mut R) -> Self {
        OpenClosed01.sample(rng)
    }
}

/// Multivariate distribution.
pub trait MultivarDist<T>: Distribution<Vec<T>> {
    /// Dimensionality
    fn dim(&self) -> usize;
}

/// Univariate standard normal distribtion.
#[derive(Clone, Copy, Default, Debug)]
pub struct StdNormDist;

impl<T: Num> Distribution<T> for StdNormDist {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        let pi: T = T::PI();
        let r = (T::from(-2.0).unwrap() * T::ln(T::sample_open_closed_01(rng))).sqrt();
        let sin = rng.gen_range(-pi..pi).sin();
        r * sin
    }
}

/// Univariate standard normal distribtion with pairwise output.
#[derive(Clone, Copy, Default, Debug)]
pub struct StdNormDistPair;

impl<T: Num> Distribution<[T; 2]> for StdNormDistPair {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [T; 2] {
        let pi: T = T::PI();
        let r = (T::from(-2.0).unwrap() * T::ln(T::sample_open_closed_01(rng))).sqrt();
        let (sin, cos) = rng.gen_range(-pi..pi).sin_cos();
        [r * sin, r * cos]
    }
}

impl<T: Num> Distribution<(T, T)> for StdNormDistPair {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> (T, T) {
        let [x, y]: [T; 2] = self.sample(rng);
        (x, y)
    }
}

/// Univariate standard normal distribtion with variable length output.
#[derive(Clone, Copy, Default, Debug)]
pub struct StdNormDistVec(
    /// Number of elements in [`Vec`] sample.
    pub usize,
);

impl<T: Num> Distribution<Vec<T>> for StdNormDistVec {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<T> {
        let dim = self.0;
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim / 2 {
            let (x, y) = StdNormDistPair.sample(rng);
            vec.push(x);
            vec.push(y);
        }
        if dim % 2 == 1 {
            vec.push(StdNormDist.sample(rng));
        }
        vec
    }
}

/// Univariate (non-standard) normal distribtion with given average and
/// standard deviation.
#[derive(Clone, Debug)]
pub struct NormDist<T> {
    /// Average of created samples.
    pub average: T,
    /// Standard deviation of created samples.
    pub stddev: T,
}

impl<T> NormDist<T> {
    /// Create distribution with given average and standard deviation.
    pub const fn new(average: T, stddev: T) -> NormDist<T> {
        NormDist { average, stddev }
    }
}

impl<T: Num> Distribution<T> for NormDist<T> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        let x: T = StdNormDist.sample(rng);
        x * self.stddev + self.average
    }
}

/// Multivariate normal distribution for given
/// average vectors and covariance matrices.
///
/// # Example
///
/// ```
/// use multivariate_optimization::distributions::MultivarNormDist;
/// use multivariate_optimization::triangular::Triangular;
/// use rand::{thread_rng, Rng};
///
/// let averages: Vec<f64> = vec![1.51, 1.57, -0.29];
/// let covariances = Triangular::<f64>::new(3, |(i, j)| {
///     // needs to handle only cases for i >= j
///     // assert!(i >= j);
///     if i == j { 1.0 } else { 0.5 }
/// });
/// let dist = MultivarNormDist::new(averages, covariances);
///
/// let mut rng = thread_rng();
/// let value = rng.sample(dist);
/// println!("Sample vector: {:?}", value);
/// ```
#[derive(Debug)]
pub struct MultivarNormDist<T> {
    averages: Vec<T>,
    factors: Triangular<T>,
}

impl<T: Num> MultivarNormDist<T> {
    /// Create multivariate normal distribution for given averages and
    /// covariances.
    pub fn new(averages: Vec<T>, covariances: Triangular<T>) -> MultivarNormDist<T> {
        let dim = covariances.dim();
        assert_eq!(
            averages.len(),
            dim,
            "dimensions of averages and covariances do not match"
        );
        let mut factors = covariances;
        for i in 0..dim {
            for j in 0..=i {
                // SAFETY:
                //  *  `i` is smaller than `factors.len()`
                //  *  `j` is equal to or smaller than `i`
                //  *  `k` is equal to or smaller than `j`
                unsafe {
                    let mut sum = *factors.get_unchecked((i, j));
                    for k in 0..j {
                        sum -= *factors.get_unchecked((i, k)) * *factors.get_unchecked((j, k));
                    }
                    let v = if i == j {
                        sum.sqrt()
                    } else {
                        sum / *factors.get_unchecked((j, j))
                    };
                    *factors.get_unchecked_mut((i, j)) = if v.is_finite() { v } else { T::zero() }
                }
            }
        }
        MultivarNormDist { averages, factors }
    }
}

impl<T: Num> MultivarDist<T> for MultivarNormDist<T> {
    fn dim(&self) -> usize {
        self.factors.dim()
    }
}

impl<T: Num> Distribution<Vec<T>> for MultivarNormDist<T> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<T> {
        let dim = self.dim();
        let mut result: Vec<T> = StdNormDistVec(dim).sample(rng);
        // SAFETY:
        //  *  `i` is smaller than `self.dim()` and thus smaller
        //     than `result.len()` and `self.factors.dim()`
        //  *  `j` is equal to or smaller than `i`
        unsafe {
            for i in (0..dim).rev() {
                let mut sum = T::zero();
                for j in 0..=i {
                    sum += *result.get_unchecked(j) * *self.factors.get_unchecked((i, j));
                }
                *result.get_unchecked_mut(i) = sum;
            }
            for i in 0..dim {
                *result.get_unchecked_mut(i) += *self.averages.get_unchecked(i);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{MultivarNormDist, StdNormDist};
    use crate::triangular::Triangular;
    use rand::{distributions::Distribution, thread_rng};
    const STATCNT: usize = 10000;
    #[test]
    fn test_std_norm_dist() {
        let values: Vec<f64> = StdNormDist
            .sample_iter(thread_rng())
            .take(STATCNT)
            .collect();
        let (mut avg, mut var): (f64, f64) = (0.0, 0.0);
        for value in values.iter() {
            avg += value;
            var += value * value;
        }
        avg /= STATCNT as f64;
        var /= STATCNT as f64;
        assert!(avg.abs() < 0.1);
        assert!((var - 1.0).abs() < 0.1);
    }
    #[test]
    fn test_multivar_norm_dist() {
        let avg_goal: Vec<f64> = vec![0.0, 0.0, 0.0];
        let cov_goal = Triangular::<f64>::new(3, |idx| match idx {
            (0, 0) => 1.0,
            (1, 1) => 0.5,
            (2, 2) => 2.0,
            (2, 0) => 0.25,
            (0, 2) => 0.25,
            _ => 0.0,
        });
        let cov_goal = cov_goal;
        let dist = MultivarNormDist::new(avg_goal.clone(), cov_goal.clone());
        let values: Vec<Vec<f64>> = dist.sample_iter(thread_rng()).take(STATCNT).collect();
        let mut avg: Vec<f64> = vec![0.0; 3];
        for value in values.iter() {
            for i in 0..3 {
                avg[i] += value[i];
            }
        }
        for i in 0..3 {
            avg[i] /= STATCNT as f64;
        }
        let mut cov = Triangular::<f64>::new(3, |(i, j)| {
            let mut sum = 0.0;
            for value in values.iter() {
                sum += (value[i] - avg[i]) * (value[j] - avg[j]);
            }
            sum
        });
        for i in 0..3 {
            for j in 0..=i {
                cov[(i, j)] /= STATCNT as f64;
            }
        }
        for i in 0..3 {
            assert!((avg[i] - avg_goal[i]).abs() < 0.1);
            for j in 0..=i {
                assert!((cov[(i, j)] - cov_goal[(i, j)]).abs() < 0.1);
            }
        }
    }
}
