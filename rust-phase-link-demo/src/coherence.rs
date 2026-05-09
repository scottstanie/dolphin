//! Sample coherence estimation.
//!
//! Mirrors `compute_sample_coherence` from the notebook:
//!
//! ```python
//! ifgs = slc_samples @ slc_samples.conj().T          # (nslc, nslc)
//! amp  = (|slc_samples|**2).sum(axis=1)              # (nslc,)
//! coh  = ifgs / sqrt(outer(amp, amp))                # (nslc, nslc)
//! ```

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::C32;

/// Compute the sample coherence matrix from a `(nslc, nsamples)` block.
///
/// The `(i, j)` element of the result is the complex correlation
/// `ρ_{ij} · exp(j φ_{ij})` between SLCs `i` and `j` over the sample window.
pub fn compute_sample_coherence(samples: ArrayView2<C32>) -> Array2<C32> {
    let nslc = samples.nrows();

    // Numerator: ifgs[i, j] = Σ_k samples[i, k] · conj(samples[j, k])
    // Equivalent to samples · samples^H.
    let conj_t: Array2<C32> = samples.t().mapv(|c| c.conj());
    let ifgs: Array2<C32> = samples.dot(&conj_t);

    // amp[i] = Σ_k |samples[i, k]|^2
    let amp: Array1<f32> = samples.map_axis(Axis(1), |row| {
        row.iter().map(|c| c.norm_sqr()).sum::<f32>()
    });

    // coherence[i, j] = ifgs[i, j] / sqrt(amp[i] · amp[j])
    let mut coh = Array2::<C32>::zeros((nslc, nslc));
    for i in 0..nslc {
        for j in 0..nslc {
            let denom = (amp[i] * amp[j]).sqrt();
            if denom > 1e-12 {
                coh[[i, j]] = ifgs[[i, j]] / C32::new(denom, 0.0);
            }
        }
    }
    coh
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn diagonal_is_unit() {
        // For any non-zero pixel, coherence(i, i) should be 1.
        let samples = array![
            [C32::new(1.0, 2.0), C32::new(0.5, -0.3), C32::new(-1.1, 0.8)],
            [C32::new(0.4, 0.9), C32::new(1.2, 0.1), C32::new(0.7, -0.5)],
        ];
        let coh = compute_sample_coherence(samples.view());
        for i in 0..2 {
            assert!((coh[[i, i]].re - 1.0).abs() < 1e-5);
            assert!(coh[[i, i]].im.abs() < 1e-5);
        }
    }

    #[test]
    fn perfectly_correlated_phases() {
        // If SLC j is SLC i rotated by phi, the coherence should be exp(-i phi)
        // (under the `samples · samples^H` convention used here).
        let nsamp = 50usize;
        let phi = 0.7f32;
        let s0: Array1<C32> = (0..nsamp)
            .map(|k| C32::new(k as f32 + 1.0, 0.0))
            .collect();
        let s1: Array1<C32> = s0.iter().map(|c| c * C32::from_polar(1.0, phi)).collect();
        let mut samples = Array2::<C32>::zeros((2, nsamp));
        samples.row_mut(0).assign(&s0);
        samples.row_mut(1).assign(&s1);

        let coh = compute_sample_coherence(samples.view());
        // |coh| = 1 off-diagonal.
        assert!((coh[[0, 1]].norm() - 1.0).abs() < 1e-5);
        // arg(coh[0, 1]) = -phi (because of the conj on s1).
        assert!((coh[[0, 1]].arg() + phi).abs() < 1e-5);
    }
}
