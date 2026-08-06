// Package evd extracts the largest eigenpair via power iteration.
//
// The notebook calls scipy.linalg.eigh(T, subset_by_index=[n-1, n-1]).
// For the small Hermitian matrices phase linking produces (nslc x nslc,
// nslc typically 10-60), power iteration converges in a handful of steps
// and avoids a LAPACK/eigensolver dependency. gonum's high-level mat
// package has no complex matmul or complex Hermitian eigensolver as of
// v0.17, so a hand-rolled kernel is both simpler and dependency-free here.
package evd

import (
	"math"

	"github.com/scottstanie/dolphin/go-phase-link-demo/internal/c64"
)

// matvec computes t (n x n, row-major) times v (length n).
func matvec(t, v []complex64, n int) []complex64 {
	out := make([]complex64, n)
	for i := 0; i < n; i++ {
		row := t[i*n : i*n+n]
		var acc complex64
		for j := 0; j < n; j++ {
			acc += row[j] * v[j]
		}
		out[i] = acc
	}
	return out
}

// LargestEigenpair runs power iteration for the dominant eigenpair of the
// Hermitian matrix t (n x n, row-major). Returns the (real) eigenvalue and
// a unit-norm eigenvector.
func LargestEigenpair(t []complex64, n, maxIters int, tol float32) (float32, []complex64) {
	v := make([]complex64, n)
	start := complex(float32(1.0/math.Sqrt(float64(n))), 0)
	for i := range v {
		v[i] = start
	}

	lambdaPrev := float32(math.Inf(1))
	for iter := 0; iter < maxIters; iter++ {
		w := matvec(t, v, n)
		var norm2 float32
		for _, z := range w {
			norm2 += c64.Norm2(z)
		}
		if norm2 < 1e-40 {
			return 0, v
		}
		norm := complex(float32(math.Sqrt(float64(norm2))), 0)
		for i := range w {
			v[i] = w[i] / norm
		}

		// Rayleigh quotient lambda = v^H t v (real for Hermitian t).
		tv := matvec(t, v, n)
		var lam complex64
		for i := 0; i < n; i++ {
			lam += c64.Conj(v[i]) * tv[i]
		}
		lambda := real(lam)

		scale := float32(math.Abs(float64(lambda)))
		if scale < 1e-12 {
			scale = 1e-12
		}
		if float32(math.Abs(float64(lambda-lambdaPrev))) <= tol*scale {
			return lambda, v
		}
		lambdaPrev = lambda
	}
	return lambdaPrev, v
}

// LinkPhasesEVD is the EVD ("CAESAR") phase linking solution for coherence
// matrix t (n x n, row-major). Returns the wrapped phase vector of length
// n, referenced to index 0. Mirrors link_phases_evd in the notebook:
//
//	_, v = largest_eigenpair(T)
//	phase = np.angle(v[0] * v.conj())
func LinkPhasesEVD(t []complex64, n int) []float32 {
	_, v := LargestEigenpair(t, n, 50, 1e-6)
	v0 := v[0]
	out := make([]float32, n)
	for k := 0; k < n; k++ {
		out[k] = c64.Arg(v0 * c64.Conj(v[k]))
	}
	return out
}
