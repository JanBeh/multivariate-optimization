//! Randomly assign indices to a fixed number of groups (to split work into
//! smaller parts).

use rand::{seq::SliceRandom, Rng};

use std::ops::Deref;

/// Memorizes random assignment of indices to groups.
#[derive(Debug)]
pub struct Splitter {
    assignments: Box<[usize]>,
    groups: Box<[Vec<usize>]>,
}

impl Splitter {
    /// Randomly assign indices to fixed number of groups.
    ///
    /// Create a `Splitter` struct, by randomly assigning indices (from
    /// `0..source_len`) to a fixed number of groups (`group_count`).
    /// The returned struct provides access to the created groups (containing
    /// their assigned indices, see [`groups`]) and allows merging of iterators
    /// (see [`merge`]).
    /// If `group_count` is smaller than `source_len`, the number of groups is
    /// set (i.e. limited) to `input_len`.
    ///
    /// [`groups`]: Self::groups
    /// [`merge`]: Self::merge
    pub fn new<R: Rng + ?Sized>(rng: &mut R, source_len: usize, group_count: usize) -> Self {
        assert_ne!(group_count, 0, "group_count must be positive");
        let group_count = group_count.min(source_len);
        let mut assignments: Box<[usize]> = (0..source_len)
            .into_iter()
            .map(|i| i % group_count)
            .collect();
        let mut groups: Box<[Vec<usize>]> = (0..group_count).map(|_| Vec::<usize>::new()).collect();
        if source_len > 0 {
            assignments.shuffle(rng);
            for group in groups.iter_mut() {
                group.reserve((source_len - 1) / group_count + 1);
            }
            for (source_idx, assignment) in assignments.iter().copied().enumerate() {
                groups[assignment].push(source_idx);
            }
        }
        Self {
            assignments,
            groups,
        }
    }
    /// Return slice containing index of assigned group for each original index.
    pub fn assignments(&self) -> &[usize] {
        &self.assignments
    }
    /// Return slice containing indices of created groups.
    pub fn groups(&self) -> &[impl Deref<Target = [usize]>] {
        &self.groups
    }
    /// Merge iterators returning results for each group to a single iterator.
    pub fn merge<'a, T, I, R>(&'a self, results: R) -> impl 'a + Iterator<Item = T>
    where
        I: Iterator<Item = T> + 'a,
        R: Iterator<Item = I>,
    {
        let mut results = results.collect::<Box<_>>();
        self.assignments.iter().copied().map(move |assignment| {
            results[assignment]
                .next()
                .unwrap_or_else(|| panic!("iterator for group #{} exhausted", assignment))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Splitter;
    use rand::thread_rng;
    #[test]
    fn test_internal() {
        let mut rng = thread_rng();
        let c = Splitter::new(&mut rng, 100, 3);
        let group_count = c.groups.len();
        assert_eq!(group_count, 3);
        for group in 0..group_count {
            let group_size = c.groups[group].len();
            assert!(group_size >= 33 && group_size <= 34);
            for element in 0..group_size {
                assert_eq!(c.assignments[c.groups[group][element]], group);
            }
        }
    }
    #[test]
    fn test_run() {
        let mut rng = thread_rng();
        let specimens: Vec<char> = vec!['A', 'B', 'C', 'D', 'E'];
        let c = Splitter::new(&mut rng, specimens.len(), 2);
        let parts = c
            .groups()
            .iter()
            .map(|group| group.iter().map(|idx| specimens[*idx]));
        let merged: Vec<char> = c.merge(parts).collect();
        assert_eq!(specimens, merged);
    }
}
