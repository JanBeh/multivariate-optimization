//! Multivariate optimization (through estimation of distribution).
//!
//! # Example
//!
//! ```
//! use multivariate_optimization::optimize::*;
//! use multivariate_optimization::testfuncs::rastrigin;
//! use rand::{Rng, thread_rng};
//! use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! let mut rng = thread_rng();
//!
//! const DIM: usize = 3;
//! let search_space: Vec<SearchRange> = (0..DIM).map(|_| {
//!     SearchRange::Finite {
//!         low: rng.gen_range(-7.68..=-2.56),
//!         high: rng.gen_range(2.56..=7.68),
//!     }
//! }).collect();
//!
//! const POPULATION: usize = 1000;
//! const MAX_ITERATIONS: usize = 1000;
//! let constructor = |params: Vec<f64>| {
//!     let cost = rastrigin(&params);
//!     BasicSpecimen { params, cost }
//! };
//! let into_specimens =
//!     |x: Vec<Vec<f64>>| x.into_par_iter().map(constructor).collect::<Vec<_>>();
//!
//! // `into_specimens` may be replaced with another mechanism, e.g. an `async` one
//! // or, if no parallel computation is desired, one can simply use:
//! // let into_specimens = |x: Vec<Vec<f64>>| x.into_iter().map(constructor);
//!
//! let mut solver = Solver::new(search_space);
//! solver.set_speed_factor(0.5);
//!
//! let initial_params = solver.random_params(POPULATION);
//! solver.extend_specimens(into_specimens(initial_params));
//! for iter in 0..MAX_ITERATIONS {
//!     let specimens = solver.specimens();
//!     println!(
//!         "{} {}",
//!         specimens[0].cost,
//!         specimens[specimens.len()-1].cost,
//!     );
//!     if solver.converged() {
//!         break;
//!     }
//!     let new_params = solver.recombined_params(POPULATION, 0.0);
//!     solver.replace_worst_specimens(into_specimens(new_params));
//! }
//! let specimen = solver.into_specimen();
//! assert_eq!(specimen.cost, 0.0);
//! ```
//!
//! See also [`Solver`].

use crate::distributions::{MultivarNormDist, NormDist};
use crate::splitter::Splitter;
use crate::triangular::Triangular;

use futures::stream::{FuturesOrdered, StreamExt};
use rand::distributions::{Distribution, Standard, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

use std::cmp::Ordering;
use std::future::Future;
use std::ops::RangeInclusive;

/// Search range for a single dimension.
///
/// `SearchRange` describes a search range for a single dimension and
/// `Vec<SearchRange>` describes a multidimensional search space (which can be
/// passed to [`Solver::new`]).
#[derive(Copy, Clone, Debug)]
pub enum SearchRange {
    /// Finite search range.
    Finite {
        /// Lower bound.
        low: f64,
        /// Upper bound.
        high: f64,
    },
    /// Infinite search range.
    Infinite {
        /// Initial average value for starting search.
        average: f64,
        /// Initial standard deviation for starting search.
        stddev: f64,
    },
}

impl From<RangeInclusive<f64>> for SearchRange {
    fn from(range: RangeInclusive<f64>) -> Self {
        SearchRange::Finite {
            low: *range.start(),
            high: *range.end(),
        }
    }
}

/// Enum with one-dimensional distribution used for initial search.
#[derive(Clone, Debug)]
enum SearchDist {
    Finite(Uniform<f64>),
    Infinite(NormDist<f64>),
}

impl From<SearchRange> for SearchDist {
    /// Create one-dimensional distribution for given SearchRange.
    fn from(search_range: SearchRange) -> SearchDist {
        match search_range {
            SearchRange::Finite { low, high } => {
                SearchDist::Finite(Uniform::new_inclusive(low, high))
            }
            SearchRange::Infinite { average, stddev } => {
                SearchDist::Infinite(NormDist::new(average, stddev))
            }
        }
    }
}

impl Distribution<f64> for SearchDist {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            SearchDist::Finite(dist) => dist.sample(rng),
            SearchDist::Infinite(dist) => dist.sample(rng),
        }
    }
}

/// Specimen in evolutionary process.
///
/// Two methods are required to be implemented:
/// * The [`params`](Specimen::params) method needs to return the parameters
///   originally used to create the specimen (see `constructor` argument to
///   function [`Solver::new`]).
/// * The [`cmp_cost`](Specimen::cmp_cost) method compares specimen by their
///   fitness (lower cost values are better).
///
/// For most cases, the module provides a [`BasicSpecimen`] which is an
/// implementation only storing the `params` and `cost` values.
pub trait Specimen {
    /// Parameters used for creation.
    fn params(&self) -> &[f64];
    /// Compare specimen's fitness ([`Less`] means better).
    ///
    /// [`Less`]: Ordering::Less
    fn cmp_cost(&self, other: &Self) -> Ordering;
    /// Euclidean distance between two specimens' parameters
    fn params_dist(&self, other: &Self) -> f64 {
        self.params()
            .iter()
            .copied()
            .zip(other.params().iter().copied())
            .map(|(a, b)| (a - b).powf(2.0))
            .sum()
    }
}

/// Most simple implementation of a [`Specimen`].
#[derive(Clone, Debug)]
pub struct BasicSpecimen {
    /// Coefficients used to create specimen
    pub params: Vec<f64>,
    /// Fitness (smaller values are better)
    pub cost: f64,
}

impl Specimen for BasicSpecimen {
    fn params(&self) -> &[f64] {
        &self.params
    }
    fn cmp_cost(&self, other: &Self) -> Ordering {
        let a = self.cost;
        let b = other.cost;
        a.partial_cmp(&b)
            .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()).then(Ordering::Equal))
    }
}

/// Parallel solver for multidimensional problems.
///
/// Usual workflow:
///
/// * [`Solver::new`]
/// * [`Solver::set_speed_factor`]
/// * Convert result of [`Solver::random_params`] into [specimens] and pass
///   them to [`Solver::extend_specimens`]
/// * In a loop:
///     * Inspect first element (or more elements) of [`Solver::specimens`],
///       e.g. for a break condition
///     * Convert result of [`Solver::recombined_params`] into [specimens] and
///       pass them to [`Solver::replace_worst_specimens`]
/// * Extract best specimen, e.g. using [`Solver::into_specimen`]
///
/// See [module level documentation] for a code example.
///
/// [specimens]: Specimen
/// [module level documentation]: self
#[derive(Clone, Debug)]
pub struct Solver<S> {
    search_space: Vec<SearchRange>,
    search_dists: Vec<SearchDist>,
    division_count: usize,
    min_population: usize,
    is_sorted: bool,
    specimens: Vec<S>,
}

impl<S> Solver<S> {
    /// Create `Solver` for search space.
    ///
    /// See [module level documentation] for a code example.
    ///
    /// [module level documentation]: self
    pub fn new(search_space: Vec<SearchRange>) -> Self {
        let search_dists: Vec<SearchDist> = search_space
            .iter()
            .copied()
            .map(|search_range| SearchDist::from(search_range))
            .collect();
        let mut solver = Solver {
            search_space,
            search_dists,
            division_count: Default::default(),
            min_population: Default::default(),
            is_sorted: true,
            specimens: vec![],
        };
        solver.set_division_count(1);
        solver
    }
    /// Dimensionality of search space.
    pub fn dim(&self) -> usize {
        self.search_space.len()
    }
    /// Simplify calculation by dividing dimensions into a given number of
    /// groups.
    ///
    /// Divides dimensions into (almost) equally sized groups when calculating
    /// covariances to reduce computation time. The number of groups is given
    /// as an integer.
    ///
    /// See [`Solver::set_speed_factor`] for a high-level interface.
    pub fn set_division_count(&mut self, mut division_count: usize) {
        let dim = self.dim();
        if dim == 0 {
            self.division_count = 1;
            self.min_population = 0;
        } else {
            if division_count < 1 {
                division_count = 1;
            } else if division_count > dim {
                division_count = dim;
            }
            self.division_count = division_count;
            self.min_population = (dim - 1) / division_count + 1;
        }
    }
    /// Simplify calculation by dividing dimensions according to speed factor.
    ///
    /// This method is a high-level interface for
    /// [`Solver::set_division_count`].
    ///
    /// Uses a `speed_factor` to divide dimensions into (almost) equally sized
    /// groups when calculating covariances to reduce computation time. Higher
    /// speed factors generally result in faster (but possibly inaccurate)
    /// calculation.
    ///
    /// Speed factors range from `0.0` to `1.0`, however, a factor greater than
    /// `0.75` is not recommended due to the introduced overhead.
    pub fn set_speed_factor(&mut self, speed_factor: f64) {
        assert!(
            speed_factor >= 0.0 && speed_factor <= 1.0,
            "speed_factor must be between 0.0 and 1.0"
        );
        self.set_division_count((self.dim() as f64).powf(speed_factor).round() as usize);
    }
    /// Simply calculation by dividing dimensions into groups of specified
    /// maximum size.
    ///
    /// This method is an alternative interface for
    /// [`Solver::set_division_count`] where the maximum size of each division
    /// is specified.
    /// See [`Solver::set_speed_factor`] for a high-level interface.
    pub fn set_max_division_size(&mut self, max_division_size: usize) {
        self.set_division_count((self.dim() as f64 / max_division_size as f64).ceil() as usize);
    }
    /// Number of groups into which dimensions are split when calculating
    /// covariances.
    pub fn division_count(&self) -> usize {
        self.division_count
    }
    /// Minimum required population.
    ///
    /// The number depends on the number of dimensions and the number of groups
    /// into which the dimensions are split when calculating covariances.
    pub fn min_population(&self) -> usize {
        self.min_population
    }
}

impl<S> Solver<S>
where
    S: Specimen + Send,
{
    /// Ensures that speciments are sorted based on cost (best first).
    fn sort(&mut self) {
        if !self.is_sorted {
            self.specimens.par_sort_by(S::cmp_cost);
            self.is_sorted = true;
        }
    }
    /// Add specimens to population.
    pub fn extend_specimens<I: IntoIterator<Item = S>>(&mut self, iter: I) {
        self.is_sorted = false;
        self.specimens.extend(iter);
    }
    /// Replace worst specimens in population.
    pub fn replace_worst_specimens<I: IntoIterator<Item = S>>(&mut self, iter: I) {
        let count = self.specimens.len();
        self.extend_specimens(iter);
        self.truncate_specimens(count);
    }
    /// Add specimens to population asynchronously.
    pub async fn extend_specimens_async<F, I>(&mut self, iter: I)
    where
        F: Future<Output = S> + Send,
        I: IntoIterator<Item = F>,
    {
        let new_specimens = FuturesOrdered::from_iter(iter);
        self.specimens.reserve(new_specimens.len());
        self.is_sorted = false;
        new_specimens
            .for_each(|specimen| {
                self.specimens.push(specimen);
                async { () }
            })
            .await;
    }
    /// Replace worst specimens in population asynchronously.
    pub async fn replace_worst_specimens_async<F, I>(&mut self, iter: I)
    where
        F: Future<Output = S> + Send,
        I: IntoIterator<Item = F>,
    {
        let count = self.specimens.len();
        self.extend_specimens_async(iter).await;
        self.truncate_specimens(count);
    }
    /// Truncate population of specimens to given `count`
    /// (drops worst fitting specimens).
    pub fn truncate_specimens(&mut self, count: usize) {
        self.sort();
        self.specimens.truncate(count);
    }
    /// Return true if specimens have converged.
    ///
    /// Note that this method only returns true if the [cost] of all specimens
    /// is exactly equal. Additional, more sensitive termination conditions may
    /// be required in practice in order to avoid endless optimization.
    ///
    /// [cost]: Specimen::cmp_cost
    pub fn converged(&mut self) -> bool {
        let len = self.specimens.len();
        if len == 0 {
            true
        } else {
            self.sort();
            self.specimens[0].cmp_cost(&self.specimens[len - 1]) == Ordering::Equal
        }
    }
    /// Sorted population of [`Specimen`]s as shared slice (best first).
    pub fn specimens(&mut self) -> &[S] {
        self.sort();
        &self.specimens
    }
    /// Sorted population of [`Specimen`]s as mutable [`Vec`] (best first).
    ///
    /// The `Vec` may be modified and doesn't need to be (re-)sorted by the
    /// caller after modifying.
    pub fn specimens_mut(&mut self) -> &mut Vec<S> {
        self.sort();
        self.is_sorted = false;
        &mut self.specimens
    }
    /// Consume [`Solver`] and return all [`Specimen`]s, ordered by fitness
    /// (best first).
    pub fn into_specimens(mut self) -> Vec<S> {
        self.sort();
        self.specimens
    }
    /// Consume [`Solver`] and return best [`Specimen`].
    pub fn into_specimen(self) -> S {
        self.into_specimens()
            .into_iter()
            .next()
            .expect("solver contains no specimen")
    }
}

impl<S> Solver<S>
where
    S: Specimen + Send + Sync,
{
    /// Create parameters for a single specimen.
    fn random_single_params<R>(&self, rng: &mut R) -> Vec<f64>
    where
        R: Rng + ?Sized,
    {
        self.search_dists
            .iter()
            .map(|dist| dist.sample(rng))
            .collect::<Vec<f64>>()
    }
    /// Create parameters for random specimens.
    pub fn random_params(&self, count: usize) -> Vec<Vec<f64>> {
        (0..count)
            .into_par_iter()
            .map_init(
                || rand::thread_rng(),
                |rng, _| self.random_single_params(rng),
            )
            .collect()
    }
    /// Create parameters for recombined specimens.
    ///
    /// Setting the `local_factor` to a value greater than `0.0` (but smaller
    /// than `1.0`) selects a particular specimen with a correspondingly
    /// proportional chance to be modified. This allows performing more
    /// localized searches. A reasonable value seems to be
    /// `0.01 / self.dim() as f64`.
    pub fn recombined_params(&mut self, children_count: usize, local_factor: f64) -> Vec<Vec<f64>> {
        self.sort();
        let total_count = self.specimens.len();
        let total_weight = total_count as f64;
        let splitter = Splitter::new(&mut rand::thread_rng(), self.dim(), self.division_count());
        let sub_averages = splitter
            .groups()
            .par_iter()
            .map(|group| {
                (0..group.len())
                    .into_par_iter()
                    .map(|i| {
                        let i_orig = group[i];
                        self.specimens
                            .par_iter()
                            .map(|specimen| specimen.params()[i_orig])
                            .sum::<f64>()
                            / total_weight
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(); // TODO: use boxed slice when supported by rayon
        let sub_dists = splitter
            .groups()
            .par_iter()
            .zip(sub_averages.into_par_iter())
            .map(|(group, averages)| {
                let covariances = Triangular::<f64>::par_new(group.len(), |(i, j)| {
                    let i_orig = group[i];
                    let j_orig = group[j];
                    self.specimens
                        .iter()
                        .map(|specimen| {
                            let a = specimen.params()[i_orig] - averages[i];
                            let b = specimen.params()[j_orig] - averages[j];
                            a * b
                        })
                        .sum::<f64>()
                        / total_weight
                });
                MultivarNormDist::new(averages, covariances)
            })
            .collect::<Vec<_>>(); // TODO: use boxed slice when supported by rayon
        let local_exp = if local_factor > 0.0 {
            1.0 / local_factor
        } else {
            f64::INFINITY
        };
        (0..children_count)
            .into_par_iter()
            .map_init(
                || rand::thread_rng(),
                |rng, _| {
                    let param_groups_iters: Box<[_]> = sub_dists
                        .iter()
                        .map(|dist| dist.sample(rng).into_iter())
                        .collect();
                    let mut params: Vec<_> = splitter.merge(param_groups_iters).collect();
                    let specimen = self.specimens.choose(rng).unwrap();
                    let parent_params = specimen.params();
                    let factor1: f64 = Standard.sample(rng);
                    let factor1 = 2.0 * factor1.powf(local_exp);
                    let factor2: f64 = 1.0 - factor1;
                    for i in 0..params.len() {
                        params[i] = factor1 * parent_params[i] + factor2 * params[i];
                        if let SearchRange::Finite { low, high } = self.search_space[i] {
                            if !(low..=high).contains(&params[i]) {
                                return self.random_single_params(rng);
                            }
                        }
                    }
                    params
                },
            )
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{BasicSpecimen, SearchRange, Solver, Specimen as _};
    use rand::{thread_rng, Rng};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    #[test]
    fn test_solver() {
        let mut rng = thread_rng();
        const PARAMCNT: usize = 3;
        let ranges = vec![-1.0..=1.0; PARAMCNT];
        let search_space: Vec<SearchRange> = ranges.iter().cloned().map(Into::into).collect();
        let goals: Vec<f64> = ranges
            .iter()
            .cloned()
            .map(|range| rng.gen_range(range) * 0.75)
            .collect();
        let constructor = |params: Vec<f64>| {
            let mut cost: f64 = 0.0;
            for (param, goal) in params.iter().zip(goals.iter()) {
                cost += (param - goal) * (param - goal);
            }
            BasicSpecimen { params, cost }
        };
        let into_specimens =
            |x: Vec<Vec<f64>>| x.into_par_iter().map(constructor).collect::<Vec<_>>();
        let mut solver = Solver::new(search_space);
        let initial_params = solver.random_params(200);
        solver.extend_specimens(into_specimens(initial_params));
        for _ in 0..1000 {
            let new_params = solver.recombined_params(10, 0.0);
            solver.replace_worst_specimens(into_specimens(new_params));
        }
        for (param, goal) in solver
            .specimens
            .first()
            .unwrap()
            .params()
            .iter()
            .zip(goals.iter())
        {
            assert!((param - goal).abs() < 0.01);
        }
    }
}
