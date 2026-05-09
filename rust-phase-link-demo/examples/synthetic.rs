//! Sanity-check the EVD pipeline on a small in-memory synthetic stack.
//!
//! Generates a stack where every pixel has the *same* true phase history
//! (a smooth ramp over time), corrupts each SLC sample with circular complex
//! Gaussian noise, runs the pipeline, and reports the mean error vs. truth.
//!
//! Run with: `cargo run --example synthetic --release`

use ndarray::{Array2, Array3, Axis};
use num_complex::Complex;

use phase_link_demo::{pipeline::run_in_memory, C32};

fn main() {
    let nslc = 20usize;
    let rows = 60usize;
    let cols = 60usize;
    let half = 5usize;

    // True phase history (one value per SLC, shared across all pixels).
    let true_phase: Vec<f32> = (0..nslc).map(|k| 0.4 * k as f32).collect();

    // Simple PCG-style PRNG so we don't take a `rand` dep.
    // Take the upper 32 bits so we get a full-range u32 (not biased).
    let mut rng_state = 0xdeadbeefcafebabeu64;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (rng_state >> 32) as u32;
        (bits as f32) / (u32::MAX as f32) - 0.5
    };

    // Stack of complex SLCs: signal · exp(j φ_k) + noise.
    let amp = 1.0f32;
    let noise_sigma = 0.5f32;
    let mut stack = Array3::<C32>::zeros((nslc, rows, cols));
    for k in 0..nslc {
        let signal = C32::from_polar(amp, true_phase[k]);
        for r in 0..rows {
            for c in 0..cols {
                let n = Complex::new(next_f32() * noise_sigma, next_f32() * noise_sigma);
                stack[[k, r, c]] = signal + n;
            }
        }
    }

    println!("Running phase linking on ({nslc}, {rows}, {cols}) stack with half-window {half}…");
    let phase = run_in_memory(stack.view(), half, half);

    // Compare interior (where window fits) against ground truth.
    let mut total_err = 0.0f32;
    let mut count = 0usize;
    for k in 0..nslc {
        let view: Array2<f32> = phase.index_axis(Axis(0), k).to_owned();
        // Reference is index 0, so expected = -(true_phase[k] - true_phase[0])
        // = true_phase[0] - true_phase[k] (the link returns angle(v[0]·conj(v))).
        let expected = true_phase[0] - true_phase[k];
        for r in half..rows - half {
            for c in half..cols - half {
                let diff = wrap_pi(view[[r, c]] - expected);
                total_err += diff.abs();
                count += 1;
            }
        }
    }
    let mean_err = total_err / count as f32;
    println!("mean |phase error| over interior pixels = {mean_err:.4} rad");
    assert!(mean_err < 0.1, "phase error too large: {mean_err}");
    println!("OK");
}

fn wrap_pi(x: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut y = x.rem_euclid(two_pi);
    if y > std::f32::consts::PI {
        y -= two_pi;
    }
    y
}
