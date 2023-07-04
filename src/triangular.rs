//! Triangular numbers and matrices

use rayon::prelude::*;

/// Calculates a triangular number, where `trinum(x) == x*(x+1)/2`
pub fn trinum(x: usize) -> usize {
    x * (x + 1) / 2
}

/// Same as [`trinum`], but returns [`None`] if calculation would overflow
pub fn checked_trinum(x: usize) -> Option<usize> {
    x.checked_add(1)
        .and_then(|t| t.checked_mul(x))
        .map(|t| t / 2)
}

/// Triangular matrix
#[derive(Clone, Debug)]
pub struct Triangular<T> {
    /// Number of both rows and columns
    dim: usize,
    /// Elements as linear slice
    linear: Box<[T]>,
}

impl<T> Triangular<T> {
    /// Create triangular matrix with given dimension
    ///
    /// Fills initial elements by calling `contents` with `(i, j)` as argument,
    /// where `i < dim && j <= i`.
    pub fn new<F>(dim: usize, mut contents: F) -> Triangular<T>
    where
        F: FnMut((usize, usize)) -> T,
    {
        let mut linear = Vec::with_capacity(trinum(dim));
        for row in 0..dim {
            for col in 0..=row {
                linear.push(contents((row, col)));
            }
        }
        let linear = linear.into_boxed_slice();
        Triangular { dim, linear }
    }
    /// Same as [`Triangular::new`], but execute in parallel
    pub fn par_new<F>(dim: usize, contents: F) -> Triangular<T>
    where
        F: Sync + Fn((usize, usize)) -> T,
        T: Send,
    {
        let contents = &contents;
        let linear = (0..dim)
            .into_par_iter()
            .flat_map(|row| {
                (0..=row)
                    .into_par_iter()
                    .map(move |col| (contents)((row, col)))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(); // TODO: directly collect into boxed slice, if possible
        Triangular { dim, linear }
    }
    /// Number of both rows and columns
    pub fn dim(&self) -> usize {
        self.dim
    }
    fn linear_index(&self, (row, col): (usize, usize)) -> usize {
        trinum(row) + col
    }
    fn checked_linear_index(&self, (row, col): (usize, usize)) -> Result<usize, &'static str> {
        if !(row < self.dim) {
            return Err("first index out of bounds");
        }
        if !(col <= row) {
            return Err("second index larger than first index");
        }
        Ok(self.linear_index((row, col)))
    }
    /// Immutable unchecked indexing through `(i, j)`, where `j <= i < dim()`
    ///
    /// # Safety
    ///
    /// * `row` (first tuple field) must be smaller than `self.dim()`
    /// * `col` (second tuple field) must be equal to or smaller than `row`
    pub unsafe fn get_unchecked(&self, (row, col): (usize, usize)) -> &T {
        let idx = self.linear_index((row, col));
        // SAFETY: `row` and `col` are valid
        unsafe { self.linear.get_unchecked(idx) }
    }
    /// Mutable unchecked indexing through `(i, j)`, where `j <= i < dim()`
    ///
    /// # Safety
    ///
    /// * `row` (first tuple field) must be smaller than `self.dim()`
    /// * `col` (second tuple field) must be equal to or smaller than `row`
    pub unsafe fn get_unchecked_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        let idx = self.linear_index((row, col));
        // SAFETY: `row` and `col` are valid
        unsafe { self.linear.get_unchecked_mut(idx) }
    }
}

/// Immutable indexing through `(i, j)`, where `j <= i < dim()`
impl<T> std::ops::Index<(usize, usize)> for Triangular<T> {
    type Output = T;
    fn index(&self, (row, col): (usize, usize)) -> &T {
        let idx = match self.checked_linear_index((row, col)) {
            Ok(x) => x,
            Err(x) => panic!("invalid indices for triangular matrix: {x}"),
        };
        // SAFETY: `checked_linear_index` returns valid index on success
        unsafe { self.linear.get_unchecked(idx) }
    }
}

/// Mutable indexing through `(i, j)`, where `j <= i < dim()`
impl<T> std::ops::IndexMut<(usize, usize)> for Triangular<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        let idx = match self.checked_linear_index((row, col)) {
            Ok(x) => x,
            Err(x) => panic!("invalid indices when mutably indexing triangular matrix: {x}"),
        };
        // SAFETY: `checked_linear_index` returns valid index on success
        unsafe { self.linear.get_unchecked_mut(idx) }
    }
}

#[cfg(test)]
mod tests {
    use super::{checked_trinum, trinum, Triangular};
    #[test]
    fn test_trinum() {
        assert_eq!(checked_trinum(0), Some(0));
        assert_eq!(checked_trinum(1), Some(1));
        assert_eq!(checked_trinum(2), Some(3));
        assert_eq!(checked_trinum(3), Some(6));
        assert_eq!(checked_trinum(4), Some(10));
        assert_eq!(checked_trinum(5), Some(15));
        assert_eq!(checked_trinum(6), Some(21));
        assert_eq!(checked_trinum(7), Some(28));
        assert_eq!(checked_trinum(8), Some(36));
        assert_eq!(checked_trinum(100), Some(5050));
        assert_eq!(checked_trinum(usize::MAX / 16), None);
        for i in 0..100 {
            assert_eq!(Some(trinum(i)), checked_trinum(i));
        }
    }
    #[test]
    fn test_triangular() {
        let calc = |(i, j)| (10 * i + j) as i16;
        let mut m = Triangular::<i16>::new(5, calc);
        assert_eq!(m.dim(), 5);
        for i in 0..5 {
            for j in 0..=i {
                assert_eq!(m[(i, j)], calc((i, j)));
            }
        }
        m[(0, 0)] = -1;
        m[(3, 0)] = -2;
        m[(4, 3)] = -3;
        m[(4, 4)] = -4;
        assert_eq!(m[(0, 0)], -1);
        assert_eq!(m[(3, 0)], -2);
        assert_eq!(m[(4, 3)], -3);
        assert_eq!(m[(4, 4)], -4);
    }
    #[test]
    #[should_panic]
    fn test_triangular_index_too_large() {
        let m = Triangular::<()>::new(3, |_| ());
        m[(3, 0)];
    }
    #[test]
    #[should_panic]
    fn test_triangular_index_wrongly_ordered() {
        let m = Triangular::<()>::new(3, |_| ());
        m[(1, 2)];
    }
}
