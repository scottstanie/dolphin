//! Per-pixel sliding-window EVD over an SLC stack.
//!
//! Mirrors the double-loop at the end of the notebook:
//!
//! ```python
//! for row in range(half_row, rows - half_row):
//!     for col in range(half_col, cols - half_col):
//!         win = stack[:, row-h:row+h+1, col-w:col+w+1]
//!         coh = compute_sample_coherence(win.reshape(nslc, -1))
//!         output[:, row, col] = link_phases_evd(coh)
//! ```
//!
//! …but parallel over pixels via `rayon`.

use ndarray::{s, Array2, Array3, ArrayView3};
use rayon::prelude::*;

use crate::coherence::compute_sample_coherence;
use crate::evd::link_phases_evd;
use crate::C32;

/// Run EVD phase linking on an in-memory SLC stack.
///
/// * `stack` — shape `(nslc, rows, cols)`, complex SLC samples.
/// * `half_row`, `half_col` — half window sizes (full window = `2·half + 1`).
///
/// Returns a `(nslc, rows, cols)` float32 phase array, referenced to the
/// first SLC at each pixel. Border pixels (where the window does not fit)
/// remain zero.
pub fn run_in_memory(stack: ArrayView3<C32>, half_row: usize, half_col: usize) -> Array3<f32> {
    let (nslc, rows, cols) = stack.dim();
    assert!(rows > 2 * half_row && cols > 2 * half_col, "stack too small for window");

    let win_h = 2 * half_row + 1;
    let win_w = 2 * half_col + 1;
    let nsamp = win_h * win_w;

    let row_lo = half_row;
    let row_hi = rows - half_row;
    let col_lo = half_col;
    let col_hi = cols - half_col;

    // Build (row, col) work items, then map in parallel. Each item produces
    // a `(nslc,)` phase vector. We collect into a flat Vec keyed by index
    // and scatter into the output below — this keeps the parallel section
    // free of shared mutable state.
    let coords: Vec<(usize, usize)> = (row_lo..row_hi)
        .flat_map(|r| (col_lo..col_hi).map(move |c| (r, c)))
        .collect();

    let phases: Vec<Vec<f32>> = coords
        .par_iter()
        .map(|&(r, c)| {
            // Extract the (nslc, win_h, win_w) window and flatten samples.
            let win = stack.slice(s![..,
                r - half_row..r + half_row + 1,
                c - half_col..c + half_col + 1]);
            let mut samples = Array2::<C32>::zeros((nslc, nsamp));
            for k in 0..nslc {
                let mut idx = 0;
                for i in 0..win_h {
                    for j in 0..win_w {
                        samples[[k, idx]] = win[[k, i, j]];
                        idx += 1;
                    }
                }
            }
            let coh = compute_sample_coherence(samples.view());
            link_phases_evd(coh.view()).to_vec()
        })
        .collect();

    let mut out = Array3::<f32>::zeros((nslc, rows, cols));
    for ((r, c), phase) in coords.into_iter().zip(phases) {
        for k in 0..nslc {
            out[[k, r, c]] = phase[k];
        }
    }
    out
}
