# `phase-link-demo` — EVD phase linking, in Rust

A didactic Rust port of the basic EVD phase linking algorithm from
`docs/notebooks/theory-phase-linking.ipynb`. The goal is to mirror that
notebook step-for-step, in idiomatic Rust, so the algorithm reads top-to-bottom
without any framework machinery in the way.

## Why this layout?

The question I started from was: *should everything happen in Rust
(load → process → write), or should Python orchestrate and Rust just process
in-memory chunks via PyO3?*

For a **demo**, end-to-end Rust is the cleanest answer:

- The pipeline is small enough (≈ 250 lines of algorithm code) that there's
  no value in straddling two languages.
- Phase linking is embarrassingly parallel per output pixel — `rayon` makes
  the per-pixel loop one line, and the result is trivially faster than the
  naive Python loop in the notebook.
- The 20×20 Hermitian matrices that come out of the coherence step are
  small enough that **plain power iteration** beats LAPACK for both
  simplicity and (with the JIT/cache effects) often speed. No
  `ndarray-linalg` / `lapack-src` dependency needed.
- GDAL via `gdal-rs` is the direct analogue of `rasterio` in Python and
  handles complex GeoTIFFs natively, so the I/O layer stays familiar.

For **production** (calling this from a larger Python workflow), the right
move is to wrap `pipeline::run_in_memory` with `pyo3` so Python passes in a
`numpy` array and gets one back. That keeps the I/O path in Python (where
all the existing dolphin file-format handling lives) and reuses the Rust
kernel without reading/writing GeoTIFFs from Rust at all. That's
deliberately *not* in this demo — it's a follow-on.

A short note on chunking: this demo loads the whole stack into memory,
matching the notebook. For real scenes you'd want a tiled outer loop —
read a `(tile_h + 2·half, tile_w + 2·half)` block of every SLC, process
the interior, write the interior, advance. `pipeline::run_in_memory`
is structured so that's a one-screen change later.

## Layout

| File | Notebook cell |
|---|---|
| `src/coherence.rs` | `compute_sample_coherence` |
| `src/evd.rs` | `largest_eigenpair` + `link_phases_evd` |
| `src/pipeline.rs` | the per-pixel double-loop at the end |
| `src/bin/phase_link.rs` | the file I/O wrapping (replaces `rasterio`) |
| `examples/synthetic.rs` | self-contained sanity check (no GDAL) |
| `tests/integration.rs` | round-trip on a known phase ramp |

## Building

The library + example + tests have **no system dependencies**:

```bash
cargo test --release
cargo run --release --example synthetic
# the synthetic bench prints timing; size is configurable via env vars:
ROWS=200 COLS=200 HALF=5 NSLC=20 cargo run --release --example synthetic
```

> There is a sibling **Go** port in `../go-phase-link-demo` with the same
> structure and CLI contract, plus a Rust-vs-Go timing comparison and an
> honest assessment of where gonum does/doesn't fit. See its README.

The CLI binary requires `libgdal` (`apt install libgdal-dev` on Debian/Ubuntu,
`brew install gdal` on macOS), then:

```bash
cargo build --release --features gdal-io --bin phase-link
```

## Running on the notebook's synthetic data

First generate the synthetic SLCs the same way the notebook does:

```bash
pip install synth-insar
synth-run --file docs/notebooks/synth_config.json
```

Then run phase linking:

```bash
cargo run --release --features gdal-io --bin phase-link -- \
    --input-glob 'synthetic-demo/slcs/2*.tif' \
    --output-dir synthetic-demo/phase-linked \
    --half-row 5 --half-col 5
```

Outputs are one `<date>.phase.tif` (float32, geo-referenced) per input SLC,
referenced to the first date.

## Mapping to the notebook

| Notebook | Rust |
|---|---|
| `T = slc_samples @ slc_samples.conj().T / sqrt(outer(amp, amp))` | `coherence::compute_sample_coherence` |
| `eigh(T, subset_by_index=[n-1, n-1])` | `evd::largest_eigenpair` (power iteration) |
| `np.angle(eig_vecs[0] * eig_vecs.conj())` | `evd::link_phases_evd` |
| Double `for row, col` loop | `pipeline::run_in_memory` (parallel via `rayon`) |
| `rio.open(...).read(1)` and writing | `src/bin/phase_link.rs` (GDAL) |

## What's intentionally missing

This is a *demo* of the basics, not a port of the production
`dolphin.phase_link.run_phase_linking`. Things deliberately left out:

- **EMI** — only EVD is implemented. EMI needs a complex Hermitian inverse
  (Cholesky), which is a bigger lift; a follow-on could add it via the
  faer crate or a hand-rolled Cholesky on the small matrices.
- **CRLB** estimate, temporal coherence, closure phases — production metrics
  from `_core.py:65` on. Each is its own ~50-line addition.
- **SHP** (statistically homogeneous pixel) selection — uses a fixed
  rectangular window, like the notebook.
- **PS pixel infill**, nodata masking, strides, multi-baseline lag,
  compressed SLCs.
- **Tiled streaming I/O** — see the chunking note above.
