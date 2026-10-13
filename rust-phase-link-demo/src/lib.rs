//! Minimal EVD phase linking, ported from
//! `docs/notebooks/theory-phase-linking.ipynb`.
//!
//! This crate is intentionally small. The pipeline is:
//!
//! 1. Read a stack of complex SLCs into an `(nslc, rows, cols)` array
//!    ([`crate::pipeline::run_in_memory`]).
//! 2. For each output pixel, take a sliding window of samples and form the
//!    sample coherence matrix ([`crate::coherence::compute_sample_coherence`]).
//! 3. Solve for the largest eigenvector of the coherence matrix via power
//!    iteration ([`crate::evd::largest_eigenpair`]) and reference the phases
//!    to the first SLC ([`crate::evd::link_phases_evd`]).
//!
//! The library has no I/O dependencies; the optional `gdal-io` feature adds
//! a CLI binary that reads/writes GeoTIFFs.

pub mod coherence;
pub mod evd;
pub mod pipeline;

use num_complex::Complex;

/// Single-precision complex SLC sample type.
pub type C32 = Complex<f32>;
