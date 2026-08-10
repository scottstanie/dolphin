#!/usr/bin/env python
"""Extract the top-N EVD scatterers (CAESAR "other" scatterers) from an SLC stack.

This is a small, self-contained research helper. It is intentionally *not* part
of the main ``dolphin`` workflow (``dolphin.workflows.single``): it skips PS
infilling, SHP neighborhoods, compressed-SLC creation, CRLB, etc., and just runs
``dolphin.phase_link.run_evd_cpl`` block-by-block so you can inspect the second
(and further) eigenvector solutions on real data before deciding whether the
approach is worth wiring into the production pipeline.

For each of the ``n_eigenvectors`` leading eigenvectors it writes, into
``<output_dir>/scatterer_<k>/``:

* ``phase.tif`` -- the phase-linked complex (unit-magnitude) SLCs, one band per
  input date (band ``i`` <-> input SLC ``i``).
* ``temporal_coherence.tif`` -- the goodness-of-fit of that scatterer's solution.
* ``eigenvalue.tif`` -- the corresponding eigenvalue of the coherence matrix.

Examples
--------
    python scripts/run-evd-multi-scatterer.py \
        --output-dir ./evd_scatterers \
        --n-eigenvectors 2 \
        --half-window 11 11 \
        --strides 2 2 \
        slcs/*.slc.tif

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from dolphin import io
from dolphin._types import HalfWindow, Strides
from dolphin.io import StridedBlockManager, VRTStack
from dolphin.phase_link import run_evd_cpl


def run(
    slc_files: list[Path],
    output_dir: Path,
    n_eigenvectors: int = 2,
    half_window: tuple[int, int] = (11, 11),
    strides: tuple[int, int] = (1, 1),
    block_shape: tuple[int, int] = (512, 512),
    reference_idx: int = 0,
    weight_by_coherence: bool = False,
) -> None:
    """Run the multi-scatterer EVD estimator over a stack and write the outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vrt = VRTStack(sorted(map(str, slc_files)), outfile=output_dir / "slc_stack.vrt")
    nslc, nrows, ncols = vrt.shape
    hw = HalfWindow(y=half_window[0], x=half_window[1])
    st = Strides(y=strides[0], x=strides[1])
    stride_dict = {"x": st.x, "y": st.y}
    print(f"Stack: {nslc} SLCs, {nrows} x {ncols} pixels")

    # Pre-create the output rasters for each scatterer (empty; filled by block).
    phase_files, tcoh_files, eig_files = [], [], []
    for k in range(n_eigenvectors):
        sdir = output_dir / f"scatterer_{k}"
        sdir.mkdir(exist_ok=True)
        phase_f = sdir / "phase.tif"
        tcoh_f = sdir / "temporal_coherence.tif"
        eig_f = sdir / "eigenvalue.tif"
        io.write_arr(
            arr=None,
            output_name=phase_f,
            like_filename=vrt.outfile,
            nbands=nslc,
            dtype=np.complex64,
            strides=stride_dict,
        )
        for name, dtype in ((tcoh_f, np.float32), (eig_f, np.float32)):
            io.write_arr(
                arr=None,
                output_name=name,
                like_filename=vrt.outfile,
                nbands=1,
                dtype=dtype,
                strides=stride_dict,
            )
        phase_files.append(phase_f)
        tcoh_files.append(tcoh_f)
        eig_files.append(eig_f)

    block_manager = StridedBlockManager(
        arr_shape=(nrows, ncols),
        block_shape=block_shape,
        strides=st,
        half_window=hw,
    )
    blocks = list(block_manager.iter_blocks())
    for out_block, out_trim, in_block, _in_no_pad, _in_trim in tqdm(blocks):
        out_rows, out_cols = out_block
        out_trim_rows, out_trim_cols = out_trim
        in_rows, in_cols = in_block

        cur = np.asarray(vrt[:, in_rows, in_cols]).astype(np.complex64)
        # Skip blocks that are entirely nodata.
        if np.all(cur == 0) or np.all(np.isnan(cur)):
            continue

        out = run_evd_cpl(
            cur,
            half_window=hw,
            strides=st,
            n_eigenvectors=n_eigenvectors,
            reference_idx=reference_idx,
            weight_by_coherence=weight_by_coherence,
        )

        for k in range(n_eigenvectors):
            # cpx_phase[k]: (nslc, out_block_rows, out_block_cols)
            phase = np.asarray(out.cpx_phase[k])[:, out_trim_rows, out_trim_cols]
            tcoh = np.asarray(out.temp_coh[k])[out_trim_rows, out_trim_cols]
            eig = np.asarray(out.eigenvalues[k])[out_trim_rows, out_trim_cols]
            io.write_block(phase, phase_files[k], out_rows.start, out_cols.start)
            io.write_block(tcoh, tcoh_files[k], out_rows.start, out_cols.start)
            io.write_block(eig, eig_files[k], out_rows.start, out_cols.start)

    print(f"Wrote {n_eigenvectors} scatterers to {output_dir}")


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "slc_files",
        nargs="+",
        type=Path,
        help="Co-registered single-look complex files (GDAL-readable).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("evd_scatterers"),
        help="Directory to write the per-scatterer outputs into.",
    )
    parser.add_argument(
        "-n",
        "--n-eigenvectors",
        type=int,
        default=2,
        help="Number of leading eigenvectors (scatterers) to extract.",
    )
    parser.add_argument(
        "--half-window",
        type=int,
        nargs=2,
        default=(11, 11),
        metavar=("Y", "X"),
        help="Half-window (y, x); full window is 2*half + 1.",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs=2,
        default=(1, 1),
        metavar=("Y", "X"),
        help="Output strides (y, x).",
    )
    parser.add_argument(
        "--block-shape",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("ROWS", "COLS"),
        help="Size of the (full-res) processing block.",
    )
    parser.add_argument(
        "--reference-idx",
        type=int,
        default=0,
        help="Acquisition index each scatterer's phase is referenced to.",
    )
    parser.add_argument(
        "--weight-by-coherence",
        action="store_true",
        help=(
            "Decompose the `C * |C|` operator used by dolphin's EVD fallback "
            "instead of the plain (PSD) coherence matrix. Makes the dominant "
            "scatterer match `use_evd=True` exactly, but the secondary "
            "eigenbasis is less physically interpretable."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _get_parser().parse_args()
    run(
        slc_files=args.slc_files,
        output_dir=args.output_dir,
        n_eigenvectors=args.n_eigenvectors,
        half_window=tuple(args.half_window),
        strides=tuple(args.strides),
        block_shape=tuple(args.block_shape),
        reference_idx=args.reference_idx,
        weight_by_coherence=args.weight_by_coherence,
    )
