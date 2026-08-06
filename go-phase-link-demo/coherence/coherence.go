// Package coherence estimates the sample coherence matrix.
//
// Mirrors compute_sample_coherence from
// docs/notebooks/theory-phase-linking.ipynb:
//
//	ifgs = slc_samples @ slc_samples.conj().T   # (nslc, nslc)
//	amp  = (|slc_samples|**2).sum(axis=1)        # (nslc,)
//	coh  = ifgs / sqrt(outer(amp, amp))          # (nslc, nslc)
package coherence

import (
	"math"

	"github.com/scottstanie/dolphin/go-phase-link-demo/internal/c64"
)

// Compute returns the (nslc x nslc) sample coherence matrix, row-major.
//
// samples is a (nslc x nsamp) block stored row-major, i.e. the sample k of
// SLC i is samples[i*nsamp+k]. The (i, j) output element is the complex
// correlation rho_{ij} * exp(j*phi_{ij}) between SLCs i and j.
func Compute(samples []complex64, nslc, nsamp int) []complex64 {
	// amp[i] = sum_k |samples[i, k]|^2
	amp := make([]float32, nslc)
	for i := 0; i < nslc; i++ {
		base := i * nsamp
		var a float32
		for k := 0; k < nsamp; k++ {
			a += c64.Norm2(samples[base+k])
		}
		amp[i] = a
	}

	coh := make([]complex64, nslc*nslc)
	for i := 0; i < nslc; i++ {
		ib := i * nsamp
		for j := 0; j < nslc; j++ {
			jb := j * nsamp
			// numerator: sum_k samples[i,k] * conj(samples[j,k])
			var acc complex64
			for k := 0; k < nsamp; k++ {
				acc += samples[ib+k] * c64.Conj(samples[jb+k])
			}
			denom := float32(math.Sqrt(float64(amp[i] * amp[j])))
			if denom > 1e-12 {
				coh[i*nslc+j] = acc / complex(denom, 0)
			}
		}
	}
	return coh
}
