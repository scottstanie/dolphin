package pipeline

import (
	"math"
	"testing"
)

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

// Noiseless rank-1 stack: every pixel must recover the true phase ramp.
func TestNoiselessRankOneRecoversTruth(t *testing.T) {
	nslc, rows, cols, half := 10, 30, 30, 3
	truePhase := make([]float64, nslc)
	st := &Stack{
		Data: make([]complex64, nslc*rows*cols),
		Nslc: nslc,
		Rows: rows,
		Cols: cols,
	}
	for k := 0; k < nslc; k++ {
		truePhase[k] = 0.25 * float64(k)
		s := complex(float32(math.Cos(truePhase[k])), float32(math.Sin(truePhase[k])))
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				st.Data[((k*rows)+r)*cols+c] = s
			}
		}
	}

	out := RunInMemory(st, half, half, 0)

	for k := 0; k < nslc; k++ {
		expected := float32(truePhase[0] - truePhase[k])
		for r := half; r < rows-half; r++ {
			for c := half; c < cols-half; c++ {
				e := math.Abs(float64(wrapPi(out[((k*rows)+r)*cols+c] - expected)))
				if e > 1e-3 {
					t.Fatalf("k=%d r=%d c=%d: err=%g", k, r, c, e)
				}
			}
		}
	}
}
