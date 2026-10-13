// Command synthetic sanity-checks the EVD pipeline on an in-memory stack
// and prints timing, so you can compare against the Rust demo's
// `cargo run --release --example synthetic`.
//
// Every pixel shares the same true phase history (a smooth ramp over
// time); each SLC sample is corrupted with circular complex Gaussian
// noise. We run the pipeline and report the mean phase error vs. truth
// (summarised with gonum/stat) and the wall-clock time.
//
//	go run ./cmd/synthetic
//	go run ./cmd/synthetic -nslc 20 -rows 200 -cols 200 -half 5
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"time"

	"gonum.org/v1/gonum/stat"

	"github.com/scottstanie/dolphin/go-phase-link-demo/pipeline"
)

func main() {
	nslc := flag.Int("nslc", 20, "number of SLCs")
	rows := flag.Int("rows", 60, "stack rows")
	cols := flag.Int("cols", 60, "stack cols")
	half := flag.Int("half", 5, "half window (both axes)")
	workers := flag.Int("workers", 0, "worker goroutines (0 = GOMAXPROCS)")
	noise := flag.Float64("noise", 0.5, "per-component Gaussian noise sigma")
	flag.Parse()

	truePhase := make([]float64, *nslc)
	for k := range truePhase {
		truePhase[k] = 0.4 * float64(k)
	}

	rng := rand.New(rand.NewSource(0xC0FFEE))
	gauss := func() float32 { return float32(rng.NormFloat64() * *noise) }

	st := &pipeline.Stack{
		Data: make([]complex64, *nslc**rows**cols),
		Nslc: *nslc,
		Rows: *rows,
		Cols: *cols,
	}
	const amp = 1.0
	for k := 0; k < *nslc; k++ {
		sig := complex(float32(amp*math.Cos(truePhase[k])), float32(amp*math.Sin(truePhase[k])))
		for r := 0; r < *rows; r++ {
			for c := 0; c < *cols; c++ {
				n := complex(gauss(), gauss())
				st.Data[((k**rows)+r)**cols+c] = sig + n
			}
		}
	}

	w := *workers
	if w <= 0 {
		w = runtime.GOMAXPROCS(0)
	}
	fmt.Printf("Running phase linking on (%d, %d, %d) stack, half-window %d, %d workers...\n",
		*nslc, *rows, *cols, *half, w)

	start := time.Now()
	out := pipeline.RunInMemory(st, *half, *half, *workers)
	elapsed := time.Since(start)

	var errs []float64
	for k := 0; k < *nslc; k++ {
		expected := float32(truePhase[0] - truePhase[k])
		for r := *half; r < *rows-*half; r++ {
			for c := *half; c < *cols-*half; c++ {
				diff := wrapPi(out[((k**rows)+r)**cols+c] - expected)
				errs = append(errs, math.Abs(float64(diff)))
			}
		}
	}
	meanErr := stat.Mean(errs, nil)
	fmt.Printf("mean |phase error| over interior pixels = %.4f rad\n", meanErr)
	fmt.Printf("elapsed: %s  (%d output pixels)\n", elapsed,
		(*rows-2**half)*(*cols-2**half))

	if meanErr > 0.1 {
		fmt.Println("FAIL: phase error too large")
		os.Exit(1)
	}
	fmt.Println("OK")
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
