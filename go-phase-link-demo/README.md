# `go-phase-link-demo` — EVD phase linking, in Go

A didactic Go port of the basic EVD phase linking algorithm from
`docs/notebooks/theory-phase-linking.ipynb`. It is a line-for-line sibling
of `../rust-phase-link-demo`, so the two can be compared directly.

## Why this layout?

Same reasoning as the Rust demo: end-to-end in one language, with the
GDAL-backed file I/O isolated so the core algorithm has zero system
dependencies.

The thing Go buys you here is exactly what your colleagues said:
**concurrency is almost free**. Phase linking is an embarrassingly
parallel moving-window operation, and `pipeline.RunInMemory` parallelises
it with a static row-band partition + a `sync.WaitGroup` — about 10 lines,
no channels needed, no locks (each goroutine writes a disjoint output
region). That part really is as easy as advertised.

### What about gonum?

I evaluated gonum and want to be precise about where it fits:

- gonum's high-level `mat` package has **no complex matrix multiply and
  no complex Hermitian eigensolver** as of v0.17 (`CDense` has `H`,
  `Conj`, `Copy`, … but no `Mul`). The phase-linking kernel is entirely
  complex, so for *this* algorithm a hand-rolled `complex64` kernel is
  both simpler and dependency-free. Power iteration on a 20×20 matrix is
  ~20 lines and needs no eigensolver at all.
- gonum *is* the right tool the moment you add EMI: that path needs a
  real `|Γ|` Cholesky factorisation / solve, which `mat.Cholesky`
  handles directly. CRLB (real Fisher information) is also a natural
  `mat` fit.
- This demo still uses gonum where it's genuinely the clean choice:
  `gonum/stat` for the error summary in the synthetic benchmark.

So: gonum is competitive and idiomatic for the **real-valued** linear
algebra in the broader workflow, but it is not currently a drop-in for
the complex eigen-decomposition at the heart of phase linking. For that
you either hand-roll (as here) or bind LAPACK's `cheevr`/`zheevr`.

### How fast is it, really?

Measured in this dev container (4 vCPUs, `nslc=20`, `half=5`), wall-clock
of `RunInMemory` only:

| stack     | output px | Rust   | Go     |
|-----------|-----------|--------|--------|
| 60×60     | 2 500     | ~22 ms | ~146 ms |
| 200×200   | 36 100    | ~310 ms | ~1.77 s |

So this **naive** Go kernel is ~6-7× slower than the **naive** Rust one.
That gap is *not* fundamental to Go — it comes from:

1. Per-pixel allocations (`coherence.Compute` and `LinkPhasesEVD` each
   allocate a slice per pixel). Reusing scratch buffers per goroutine
   typically recovers ~2-3×.
2. No autovectorisation of the complex inner loops; Rust/LLVM vectorises
   the same loops.
3. `complex64` arithmetic in Go is scalar; the dominant cost is the
   coherence triple loop, not one big matmul, so swapping in a BLAS
   `gemm` helps less than you'd hope.

A tuned Go version (buffer reuse + flattening the hot loop) lands roughly
2-3× off Rust rather than 6-7×, which matches the usual "Go is within a
small constant factor of C/Rust for numeric code if you avoid allocation"
result. The didactic code here intentionally favours readability over
that tuning; the tuning notes above are the honest answer to "how
competitive is it?". **Run the numbers on your real hardware** before
deciding — these are container figures, not authoritative.

The concurrency claim, though, is fully borne out: scaling across cores
is trivial and the moving-window structure is a perfect fit.

## Layout

| File | Rust equivalent | Notebook cell |
|---|---|---|
| `coherence/coherence.go` | `coherence.rs` | `compute_sample_coherence` |
| `evd/evd.go` | `evd.rs` | `largest_eigenpair` + `link_phases_evd` |
| `pipeline/pipeline.go` | `pipeline.rs` | the per-pixel double loop |
| `cmd/phase-link/` (own nested module) | `src/bin/phase_link.rs` | the `rasterio` I/O |
| `cmd/synthetic/` | `examples/synthetic.rs` | self-contained sanity + timing |
| `*_test.go` | `tests/` + `#[cfg(test)]` | — |

## Building

Core packages + tests + synthetic benchmark have **no system deps**:

```bash
go test ./...
go run ./cmd/synthetic
go run ./cmd/synthetic -rows 200 -cols 200 -half 5 -workers 8
```

The GDAL CLI is a **separate nested module** (so godal's large cgo
dependency tree never touches the core module — the analogue of the Rust
`gdal-io` feature). It needs `libgdal` + cgo:

```bash
cd cmd/phase-link
go mod tidy   # one-time: pulls godal's deps
go build .
./phase-link \
    -input-glob '../../../synthetic-demo/slcs/2*.tif' \
    -output-dir ../../../synthetic-demo/phase-linked-go \
    -half-row 5 -half-col 5
```

## Running on the notebook's synthetic data

Generate the SLCs the same way the notebook does, then point the CLI at
them:

```bash
pip install synth-insar
synth-run --file docs/notebooks/synth_config.json
# then the `go build -tags gdal` invocation above
```

Outputs are one `<date>.phase.tif` (float32, geo-referenced) per input
SLC, referenced to the first date — byte-for-byte the same contract as
the Rust demo.

## What's intentionally missing

Identical scope to the Rust demo: EVD only. No EMI, CRLB, temporal
coherence, closure phases, SHP selection, PS infill, strides, or tiled
streaming I/O. See `../rust-phase-link-demo/README.md` for the rationale.
