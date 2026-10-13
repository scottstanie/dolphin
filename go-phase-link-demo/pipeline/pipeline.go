// Package pipeline runs a per-pixel sliding-window EVD over an SLC stack.
//
// Mirrors the double loop at the end of the notebook, but parallel over
// output rows using a goroutine worker pool. Phase linking is
// embarrassingly parallel per output pixel: each goroutine writes a
// disjoint set of output pixels, so no locking is needed.
package pipeline

import (
	"runtime"
	"sync"

	"github.com/scottstanie/dolphin/go-phase-link-demo/coherence"
	"github.com/scottstanie/dolphin/go-phase-link-demo/evd"
)

// Stack is a complex SLC stack of shape (nslc, rows, cols), stored
// row-major: element (k, r, c) lives at ((k*Rows)+r)*Cols + c.
type Stack struct {
	Data       []complex64
	Nslc       int
	Rows, Cols int
}

func (s *Stack) at(k, r, c int) complex64 {
	return s.Data[((k*s.Rows)+r)*s.Cols+c]
}

// RunInMemory runs EVD phase linking with a (2*halfRow+1, 2*halfCol+1)
// window. Returns a (nslc, rows, cols) float32 phase array (same layout as
// the input) referenced to the first SLC at each pixel. Border pixels
// where the window does not fit are left at zero.
//
// workers <= 0 means runtime.GOMAXPROCS(0).
func RunInMemory(s *Stack, halfRow, halfCol, workers int) []float32 {
	if s.Rows <= 2*halfRow || s.Cols <= 2*halfCol {
		panic("stack too small for window")
	}
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}

	winH := 2*halfRow + 1
	winW := 2*halfCol + 1
	nsamp := winH * winW
	nslc := s.Nslc

	out := make([]float32, nslc*s.Rows*s.Cols)

	rowLo, rowHi := halfRow, s.Rows-halfRow
	colLo, colHi := halfCol, s.Cols-halfCol

	var wg sync.WaitGroup
	// Static row-band partition: contiguous output rows per worker means
	// each worker touches a disjoint slice of `out`.
	band := (rowHi - rowLo + workers - 1) / workers
	for w := 0; w < workers; w++ {
		r0 := rowLo + w*band
		r1 := r0 + band
		if r1 > rowHi {
			r1 = rowHi
		}
		if r0 >= r1 {
			break
		}
		wg.Add(1)
		go func(r0, r1 int) {
			defer wg.Done()
			// Reuse one scratch sample buffer per goroutine.
			samples := make([]complex64, nslc*nsamp)
			for r := r0; r < r1; r++ {
				for c := colLo; c < colHi; c++ {
					for k := 0; k < nslc; k++ {
						kb := k * nsamp
						idx := 0
						for i := r - halfRow; i <= r+halfRow; i++ {
							for j := c - halfCol; j <= c+halfCol; j++ {
								samples[kb+idx] = s.at(k, i, j)
								idx++
							}
						}
					}
					coh := coherence.Compute(samples, nslc, nsamp)
					phase := evd.LinkPhasesEVD(coh, nslc)
					for k := 0; k < nslc; k++ {
						out[((k*s.Rows)+r)*s.Cols+c] = phase[k]
					}
				}
			}
		}(r0, r1)
	}
	wg.Wait()
	return out
}
