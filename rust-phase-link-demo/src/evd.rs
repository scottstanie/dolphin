//! Largest-eigenpair extraction via power iteration.
//!
//! The notebook calls `scipy.linalg.eigh(T, subset_by_index=[n-1, n-1])`.
//! For the small Hermitian matrices that come out of phase linking
//! (`nslc × nslc`, where `nslc` is typically 10–60), power iteration
//! converges in a handful of iterations and avoids a LAPACK dependency.
//!
//! This is also what dolphin's batched implementation in
//! `src/dolphin/phase_link/_eigenvalues.py` does under the hood for stacks
//! of matrices: power iteration on `T` with periodic Rayleigh-quotient
//! convergence checks.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::C32;

/// Power iteration for the largest eigenvalue of a Hermitian matrix `T`.
///
/// Returns `(λ_max, v_max)` with `v_max` unit-norm and `λ_max` real.
pub fn largest_eigenpair(t: ArrayView2<C32>, max_iters: usize, tol: f32) -> (f32, Array1<C32>) {
    let n = t.nrows();
    debug_assert_eq!(n, t.ncols(), "T must be square");

    // Start from a uniform real vector. Any vector with a non-zero projection
    // onto the dominant eigenvector works; uniform is robust for our inputs.
    let mut v = Array1::from_elem(n, C32::new(1.0 / (n as f32).sqrt(), 0.0));

    let mut lambda_prev = f32::INFINITY;
    for _ in 0..max_iters {
        let w = matvec(t, v.view());
        let norm: f32 = w.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt();
        if norm < 1e-20 {
            return (0.0, v);
        }
        v = w.mapv(|c| c / norm);

        // Rayleigh quotient: λ = vᴴ T v (real for Hermitian T).
        let tv = matvec(t, v.view());
        let lambda: C32 = v.iter().zip(tv.iter()).map(|(a, b)| a.conj() * b).sum();
        let lambda = lambda.re;

        if (lambda - lambda_prev).abs() <= tol * lambda.abs().max(1e-12) {
            return (lambda, v);
        }
        lambda_prev = lambda;
    }
    (lambda_prev, v)
}

fn matvec(t: ArrayView2<C32>, v: ArrayView1<C32>) -> Array1<C32> {
    let n = t.nrows();
    let mut out = Array1::<C32>::zeros(n);
    for i in 0..n {
        let mut acc = C32::new(0.0, 0.0);
        for j in 0..n {
            acc += t[[i, j]] * v[j];
        }
        out[i] = acc;
    }
    out
}

/// EVD ("CAESAR") phase linking solution for a coherence matrix `T`.
///
/// Returns the wrapped phase vector of length `nslc`, referenced to index 0.
/// Mirrors `link_phases_evd` in the notebook:
///
/// ```python
/// _, v = largest_eigenpair(T)
/// phase = np.angle(v[0] * v.conj())
/// ```
pub fn link_phases_evd(t: ArrayView2<C32>) -> Array1<f32> {
    let (_, v) = largest_eigenpair(t, 50, 1e-6);
    let v0 = v[0];
    v.mapv(|c| (v0 * c.conj()).arg())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Build a rank-1 Hermitian matrix T = v vᴴ with known dominant eigenvector.
    fn rank_one(v: &Array1<C32>) -> Array2<C32> {
        let n = v.len();
        let mut t = Array2::<C32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                t[[i, j]] = v[i] * v[j].conj();
            }
        }
        t
    }

    #[test]
    fn rank_one_recovers_phase_ramp() {
        // True phase: linear ramp.
        let n = 8;
        let phi: Vec<f32> = (0..n).map(|k| 0.3 * k as f32).collect();
        let v: Array1<C32> = phi.iter().map(|&p| C32::from_polar(1.0, p)).collect();
        let t = rank_one(&v);

        let recovered = link_phases_evd(t.view());
        // recovered[k] = angle(v[0] * conj(v[k])) = -phi[k]  (since phi[0]=0).
        for k in 0..n {
            let expected = -phi[k];
            // Wrap into (-pi, pi].
            let diff = ((recovered[k] - expected + std::f32::consts::PI)
                .rem_euclid(2.0 * std::f32::consts::PI))
                - std::f32::consts::PI;
            assert!(diff.abs() < 1e-4, "k={k}: got {} expected {}", recovered[k], expected);
        }
    }

    #[test]
    fn largest_eigenvalue_is_n_for_rank_one() {
        // For T = v vᴴ with ||v|| = sqrt(n), the only nonzero eigenvalue is n.
        let n = 6;
        let v: Array1<C32> = (0..n).map(|_| C32::new(1.0, 0.0)).collect();
        let t = rank_one(&v);
        let (lambda, _) = largest_eigenpair(t.view(), 100, 1e-7);
        assert!((lambda - n as f32).abs() < 1e-3, "lambda = {lambda}");
    }
}
