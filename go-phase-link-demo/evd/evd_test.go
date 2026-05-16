package evd

import (
	"math"
	"testing"

	"github.com/scottstanie/dolphin/go-phase-link-demo/internal/c64"
)

// rankOne builds the Hermitian matrix T = v vᴴ (row-major).
func rankOne(v []complex64) []complex64 {
	n := len(v)
	t := make([]complex64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			t[i*n+j] = v[i] * c64.Conj(v[j])
		}
	}
	return t
}

func wrapPi(x float32) float32 {
	const twoPi = 2 * math.Pi
	y := float32(math.Mod(float64(x), twoPi))
	if y > math.Pi {
		y -= twoPi
	} else if y < -math.Pi {
		y += twoPi
	}
	return y
}

func TestRankOneRecoversPhaseRamp(t *testing.T) {
	n := 8
	phi := make([]float32, n)
	v := make([]complex64, n)
	for k := 0; k < n; k++ {
		phi[k] = 0.3 * float32(k)
		v[k] = complex(float32(math.Cos(float64(phi[k]))), float32(math.Sin(float64(phi[k]))))
	}
	tm := rankOne(v)

	got := LinkPhasesEVD(tm, n)
	for k := 0; k < n; k++ {
		// recovered[k] = angle(v[0]*conj(v[k])) = -phi[k] (phi[0] = 0).
		if d := wrapPi(got[k] - (-phi[k])); math.Abs(float64(d)) > 1e-4 {
			t.Fatalf("k=%d: got %f want %f", k, got[k], -phi[k])
		}
	}
}

func TestLargestEigenvalueIsNForRankOne(t *testing.T) {
	n := 6
	v := make([]complex64, n)
	for i := range v {
		v[i] = complex(1, 0)
	}
	tm := rankOne(v)
	lam, _ := LargestEigenpair(tm, n, 100, 1e-7)
	if math.Abs(float64(lam)-float64(n)) > 1e-3 {
		t.Fatalf("lambda = %f, want %d", lam, n)
	}
}
