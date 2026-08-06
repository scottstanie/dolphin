//! Integration test: phase ramp recovery on a noiseless rank-1 stack.

use ndarray::Array3;
use num_complex::Complex;

use phase_link_demo::{pipeline::run_in_memory, C32};

fn wrap_pi(x: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut y = x.rem_euclid(two_pi);
    if y > std::f32::consts::PI {
        y -= two_pi;
    }
    y
}

#[test]
fn noiseless_rank_one_recovers_truth() {
    let nslc = 10usize;
    let rows = 30usize;
    let cols = 30usize;
    let half = 3usize;

    let true_phase: Vec<f32> = (0..nslc).map(|k| 0.25 * k as f32).collect();

    let mut stack = Array3::<C32>::zeros((nslc, rows, cols));
    for k in 0..nslc {
        let s = Complex::from_polar(1.0, true_phase[k]);
        for r in 0..rows {
            for c in 0..cols {
                stack[[k, r, c]] = s;
            }
        }
    }

    let phase = run_in_memory(stack.view(), half, half);

    for k in 0..nslc {
        let expected = true_phase[0] - true_phase[k];
        for r in half..rows - half {
            for c in half..cols - half {
                let err = wrap_pi(phase[[k, r, c]] - expected).abs();
                assert!(err < 1e-3, "k={k} r={r} c={c}: err={err}");
            }
        }
    }
}
