package coherence

import (
	"math"
	"testing"

	"github.com/scottstanie/dolphin/go-phase-link-demo/internal/c64"
)

func TestDiagonalIsUnit(t *testing.T) {
	// For any non-zero pixel, coherence(i, i) should be 1.
	nslc, nsamp := 2, 3
	samples := []complex64{
		complex(1, 2), complex(0.5, -0.3), complex(-1.1, 0.8),
		complex(0.4, 0.9), complex(1.2, 0.1), complex(0.7, -0.5),
	}
	coh := Compute(samples, nslc, nsamp)
	for i := 0; i < nslc; i++ {
		d := coh[i*nslc+i]
		if math.Abs(float64(real(d)-1)) > 1e-5 || math.Abs(float64(imag(d))) > 1e-5 {
			t.Fatalf("coh[%d,%d] = %v, want 1+0i", i, i, d)
		}
	}
}

func TestPerfectlyCorrelatedPhases(t *testing.T) {
	// SLC 1 = SLC 0 rotated by phi. Coherence magnitude should be 1 and
	// the phase should be -phi (samples * samples^H convention).
	const nsamp = 50
	const phi = 0.7
	samples := make([]complex64, 2*nsamp)
	for k := 0; k < nsamp; k++ {
		s := complex(float32(k+1), 0)
		samples[k] = s
		samples[nsamp+k] = s * complex(float32(math.Cos(phi)), float32(math.Sin(phi)))
	}
	coh := Compute(samples, 2, nsamp)
	off := coh[1] // (0,1)
	mag := math.Sqrt(float64(c64.Norm2(off)))
	if math.Abs(mag-1) > 1e-4 {
		t.Fatalf("|coh[0,1]| = %f, want 1", mag)
	}
	if math.Abs(float64(c64.Arg(off))+phi) > 1e-4 {
		t.Fatalf("arg(coh[0,1]) = %f, want %f", c64.Arg(off), -phi)
	}
}
