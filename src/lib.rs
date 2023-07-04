//! Multivariate optimization
//!
//! See module [`optimize`] for multivariate optimization.
//! The other modules are (public) helper modules.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod conquer;
pub mod distributions;
pub mod optimize;
pub mod triangular;

pub use optimize::*;
