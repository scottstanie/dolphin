// Command phase-link reads a stack of complex SLC GeoTIFFs, runs EVD phase
// linking with a sliding window, and writes one float32 phase GeoTIFF per
// date. It mirrors src/bin/phase_link.rs from the Rust demo.
//
// This is a separate nested module so godal's large cgo dependency tree
// never touches the core algorithm packages. Building it requires libgdal
// + cgo:
//
//	cd cmd/phase-link && go build .
//	./phase-link \
//	    -input-glob 'synthetic-demo/slcs/2*.tif' \
//	    -output-dir synthetic-demo/phase-linked \
//	    -half-row 5 -half-col 5
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/airbusgeo/godal"

	"github.com/scottstanie/dolphin/go-phase-link-demo/pipeline"
)

func main() {
	inputGlob := flag.String("input-glob", "", "glob for input SLC GeoTIFFs")
	outputDir := flag.String("output-dir", "", "directory for per-date phase rasters")
	halfRow := flag.Int("half-row", 5, "half window in rows")
	halfCol := flag.Int("half-col", 5, "half window in cols")
	workers := flag.Int("workers", 0, "worker goroutines (0 = GOMAXPROCS)")
	flag.Parse()

	if *inputGlob == "" || *outputDir == "" {
		fmt.Fprintln(os.Stderr, "both -input-glob and -output-dir are required")
		os.Exit(2)
	}

	godal.RegisterAll()

	paths, err := filepath.Glob(*inputGlob)
	if err != nil {
		fatal("invalid glob: %v", err)
	}
	if len(paths) == 0 {
		fatal("no files matched %s", *inputGlob)
	}
	sort.Strings(paths)
	fmt.Printf("Found %d SLCs\n", len(paths))

	first, gt, proj, sx, sy, err := readComplexSLC(paths[0])
	if err != nil {
		fatal("%v", err)
	}
	nslc := len(paths)
	fmt.Printf("Stack shape: (%d, %d, %d)\n", nslc, sy, sx)

	st := &pipeline.Stack{
		Data: make([]complex64, nslc*sy*sx),
		Nslc: nslc,
		Rows: sy,
		Cols: sx,
	}
	copy(st.Data[0:sy*sx], first)
	for i := 1; i < nslc; i++ {
		buf, _, _, bx, by, err := readComplexSLC(paths[i])
		if err != nil {
			fatal("%v", err)
		}
		if bx != sx || by != sy {
			fatal("shape mismatch at %s: (%d,%d) != (%d,%d)", paths[i], by, bx, sy, sx)
		}
		copy(st.Data[i*sy*sx:(i+1)*sy*sx], buf)
	}

	fmt.Printf("Running EVD phase linking with half-window (%d, %d)...\n", *halfRow, *halfCol)
	start := time.Now()
	out := pipeline.RunInMemory(st, *halfRow, *halfCol, *workers)
	fmt.Printf("phase linking done in %s\n", time.Since(start))

	if err := os.MkdirAll(*outputDir, 0o755); err != nil {
		fatal("%v", err)
	}
	band := make([]float32, sy*sx)
	for i, p := range paths {
		stem := strings.TrimSuffix(filepath.Base(p), filepath.Ext(p))
		outPath := filepath.Join(*outputDir, stem+".phase.tif")
		copy(band, out[i*sy*sx:(i+1)*sy*sx])
		if err := writeFloatGeoTIFF(outPath, band, sx, sy, gt, proj); err != nil {
			fatal("%v", err)
		}
	}
	fmt.Printf("Wrote %d phase rasters to %s\n", nslc, *outputDir)
}

func readComplexSLC(path string) (data []complex64, gt [6]float64, proj string, sx, sy int, err error) {
	ds, err := godal.Open(path)
	if err != nil {
		return nil, gt, "", 0, 0, fmt.Errorf("opening %s: %w", path, err)
	}
	defer ds.Close()

	band := ds.Bands()[0]
	bs := band.Structure()
	if bs.DataType != godal.CFloat32 {
		return nil, gt, "", 0, 0, fmt.Errorf("%s: expected CFloat32 SLC, got %v", path, bs.DataType)
	}
	sx, sy = bs.SizeX, bs.SizeY
	data = make([]complex64, sx*sy)
	if err := band.Read(0, 0, data, sx, sy); err != nil {
		return nil, gt, "", 0, 0, fmt.Errorf("reading %s: %w", path, err)
	}
	if g, err := ds.GeoTransform(); err == nil {
		gt = g
	} else {
		gt = [6]float64{0, 1, 0, 0, 0, 1}
	}
	proj = ds.Projection()
	return data, gt, proj, sx, sy, nil
}

func writeFloatGeoTIFF(path string, data []float32, sx, sy int, gt [6]float64, proj string) error {
	ds, err := godal.Create(godal.GTiff, path, 1, godal.Float32, sx, sy)
	if err != nil {
		return fmt.Errorf("creating %s: %w", path, err)
	}
	defer ds.Close()

	if err := ds.Bands()[0].Write(0, 0, data, sx, sy); err != nil {
		return fmt.Errorf("writing %s: %w", path, err)
	}
	_ = ds.SetGeoTransform(gt)
	if proj != "" {
		_ = ds.SetProjection(proj)
	}
	return nil
}

func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
