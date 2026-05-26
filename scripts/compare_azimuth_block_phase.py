#!/usr/bin/env python
r"""Compare stitched phase outputs across azimuth_blocks runs.

Validates that ``azimuth_blocks=N`` (PR #717) produces phase that matches the
``azimuth_blocks=1`` baseline. The halo-overlap bug in the original PR
manifested as edge-degraded data winning along block-boundary rows after
gdal_merge's last-pixel-wins stitch; this script makes that visible.

What it does
------------
For each work directory pair, finds matching stitched interferograms (by date
pair parsed from filename), cross-multiplies as

    diff = baseline * conj(test)      # complex
    phase_diff_rad = angle(diff)
    mag_ratio = |test| / |baseline|

and reports

1. Per-date-pair: median |phase diff|, 95th percentile |phase diff|, RMS of
   row-mean |phase diff| (a striping detector).
2. Aggregate row profile of mean |phase diff| over all date pairs.
3. If matplotlib is available and ``--plot`` is passed, saves PNGs of the
   row-profile + the highest-RMS date pair as a 2D phase-diff image, with
   horizontal lines drawn at the expected block boundaries (read from
   ``BASELINE_DIR / block_*`` if those directories exist).

Usage
-----
::

    # Two-way compare (baseline vs test):
    python compare_azimuth_block_phase.py BASELINE_DIR TEST_DIR

    # Three-way (Marin's original buggy vs my fix, both against baseline):
    python compare_azimuth_block_phase.py BASELINE_DIR BUGGY_DIR \\
        --extra-label buggy
    python compare_azimuth_block_phase.py BASELINE_DIR FIXED_DIR \\
        --extra-label fixed

Expected result on the halo-overlap bug
---------------------------------------
- ``baseline vs buggy`` should show sharp peaks in the row-profile at
  ``y_pixel = i * (ny // N)`` for i in 1..N-1.
- ``baseline vs fixed`` should be a smooth low-noise floor (small differences
  from per-block SHP/EVD decisions, but NO boundary discontinuities).

Inputs
------
Each work directory is the ``work_directory`` from a dolphin config — the
script looks for stitched interferograms at
``<work_dir>/<interferogram_network._directory_name>/`` (default
``interferograms/``). Pass ``--ifg-subdir`` if your network directory has a
different name (e.g. ``ifgs/``, ``single_reference_ifgs/``).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


DATE_PAIR_RE = re.compile(r"(\d{8})[_T-](\d{8})")


def find_ifgs(work_dir: Path, ifg_subdir: str) -> dict[tuple[str, str], Path]:
    """Map (date1, date2) -> path for stitched .int.tif files in work_dir."""
    ifg_dir = work_dir / ifg_subdir
    if not ifg_dir.is_dir():
        msg = f"No interferogram directory at {ifg_dir}"
        raise FileNotFoundError(msg)
    out: dict[tuple[str, str], Path] = {}
    for p in sorted(ifg_dir.glob("*.int.tif")):
        m = DATE_PAIR_RE.search(p.name)
        if m is None:
            logger.debug("Skipping %s: no date pair in filename", p.name)
            continue
        out[(m.group(1), m.group(2))] = p
    if not out:
        msg = f"No *.int.tif files matched in {ifg_dir}"
        raise FileNotFoundError(msg)
    return out


def block_row_boundaries(work_dir: Path) -> list[int] | None:
    """Detect whether the work dir has ``block_NN`` subdirs.

    Looks for ``block_NN/`` subdirectories under ``work_dir`` (the convention
    ``displacement.run`` uses when ``azimuth_blocks > 1``). Returns ``None``
    if none are found; ``[]`` (an empty list) if they are present but the
    caller needs to supply the actual pixel rows via ``--boundary-rows``.
    """
    block_dirs = sorted(d for d in work_dir.glob("block_*") if d.is_dir())
    if not block_dirs:
        return None
    # We can't easily reconstruct absolute pixel indices from disk alone — the
    # caller can pass them via --boundary-rows. Return [] so plotting code
    # knows "yes there are blocks, but I don't know where" and can prompt.
    return []


def open_complex(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    """Open a complex GeoTIFF as a 2D complex64 ndarray + geotransform."""
    # Local import so the script doesn't require the GDAL stack at module load
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(str(path))
    arr = ds.ReadAsArray()
    if arr.dtype != np.complex64 and arr.dtype != np.complex128:
        msg = f"{path} is {arr.dtype}, not complex"
        raise TypeError(msg)
    gt = ds.GetGeoTransform()
    return arr.astype(np.complex64), gt


def align_to_baseline(
    baseline: np.ndarray,
    baseline_gt: tuple[float, ...],
    test: np.ndarray,
    test_gt: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return baseline/test arrays trimmed to their common extent in pixel coords.

    Both rasters should already be in the same CRS and pixel size — this
    function only corrects for an integer-pixel offset/extent mismatch, which
    can happen when the stitched mosaic for the blocked run has a slightly
    different bounding box than the full-frame run.
    """
    bx0, dx, _, by0, _, dy = baseline_gt
    tx0, _, _, ty0, _, _ = test_gt
    px_x, px_y = dx, abs(dy)
    # Offset of test's UL corner in baseline pixel coords
    col_off = round((tx0 - bx0) / px_x)
    row_off = round((by0 - ty0) / px_y)  # dy is negative; both UL corners
    # Slice both to overlap
    base_ny, base_nx = baseline.shape
    test_ny, test_nx = test.shape
    # Region in baseline coords that overlaps with test
    r0_b = max(0, row_off)
    c0_b = max(0, col_off)
    r1_b = min(base_ny, row_off + test_ny)
    c1_b = min(base_nx, col_off + test_nx)
    if r0_b >= r1_b or c0_b >= c1_b:
        msg = "Baseline and test rasters don't overlap in pixel space"
        raise ValueError(msg)
    # Same region in test coords
    r0_t = r0_b - row_off
    c0_t = c0_b - col_off
    r1_t = r1_b - row_off
    c1_t = c1_b - col_off
    return (
        baseline[r0_b:r1_b, c0_b:c1_b],
        test[r0_t:r1_t, c0_t:c1_t],
    )


def compare_pair(baseline_path: Path, test_path: Path) -> dict[str, float | np.ndarray]:
    """Compute phase-diff statistics + the row profile for one date pair."""
    base, base_gt = open_complex(baseline_path)
    test, test_gt = open_complex(test_path)
    base, test = align_to_baseline(base, base_gt, test, test_gt)

    # Mask: ignore pixels that are zero/NaN/nodata in either
    valid = np.isfinite(base) & np.isfinite(test) & (base != 0) & (test != 0)
    if not valid.any():
        return {
            "median_abs_phase": float("nan"),
            "p95_abs_phase": float("nan"),
            "row_profile": np.full(base.shape[0], np.nan, dtype=np.float32),
            "row_valid_count": np.zeros(base.shape[0], dtype=np.int64),
            "n_valid": 0,
        }

    diff = base * np.conj(test)
    phase = np.angle(diff)  # radians, in [-pi, pi]
    abs_phase = np.abs(phase)
    abs_phase_masked = np.where(valid, abs_phase, np.nan)

    # Row-mean of |phase diff| ignoring nans -> the striping detector
    with np.errstate(invalid="ignore"):
        row_profile = np.nanmean(abs_phase_masked, axis=1)
    row_valid_count = valid.sum(axis=1)

    median_abs_phase = float(np.median(abs_phase[valid]))
    p95_abs_phase = float(np.percentile(abs_phase[valid], 95))

    return {
        "median_abs_phase": median_abs_phase,
        "p95_abs_phase": p95_abs_phase,
        "row_profile": row_profile.astype(np.float32),
        "row_valid_count": row_valid_count,
        "n_valid": int(valid.sum()),
        "phase_image": phase.astype(np.float32),
        "valid_image": valid,
    }


def aggregate_row_profile(
    per_pair_profiles: list[np.ndarray],
    per_pair_counts: list[np.ndarray],
) -> np.ndarray:
    """Pixel-count-weighted mean of |phase diff| per row across all date pairs."""
    if not per_pair_profiles:
        return np.array([], dtype=np.float32)
    # Each profile is the row mean; weight by count to get a true overall mean
    profiles = np.stack(per_pair_profiles, axis=0)  # (npairs, ny)
    counts = np.stack(per_pair_counts, axis=0).astype(np.float64)
    # Replace NaNs in profiles with 0, since their weight is also 0
    numer = np.nansum(profiles * counts, axis=0)
    denom = counts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, numer / denom, np.nan).astype(np.float32)


def find_worst_pair(
    results: dict[tuple[str, str], dict[str, float | np.ndarray]],
) -> tuple[str, str] | None:
    """Date pair with the largest RMS-of-row-profile (most striping-like)."""
    if not results:
        return None
    worst = None
    worst_rms = -1.0
    for key, r in results.items():
        rp = r["row_profile"]
        if not isinstance(rp, np.ndarray) or rp.size == 0:
            continue
        rms = float(np.sqrt(np.nanmean(rp**2)))
        if np.isfinite(rms) and rms > worst_rms:
            worst = key
            worst_rms = rms
    return worst


def write_plots(
    out_prefix: Path,
    *,
    agg_profile: np.ndarray,
    worst_key: tuple[str, str] | None,
    worst_result: dict | None,
    boundary_rows: list[int] | None,
    label: str,
) -> None:
    """Save row-profile + worst-pair image PNGs. No-op if matplotlib missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots")
        return

    # Row profile
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(agg_profile, lw=0.5)
    ax.set_xlabel("row (pixel)")
    ax.set_ylabel("mean |phase diff| (rad)")
    ax.set_title(f"{label}: row-mean |angle(baseline · conj(test))|")
    if boundary_rows:
        for r in boundary_rows:
            ax.axvline(r, color="red", ls="--", alpha=0.5, lw=0.8)
    fig.tight_layout()
    p1 = out_prefix.with_name(f"{out_prefix.name}_row_profile.png")
    fig.savefig(p1, dpi=120)
    logger.info("wrote %s", p1)
    plt.close(fig)

    # Worst-pair image
    if worst_key is None or worst_result is None:
        return
    img = worst_result["phase_image"]
    valid = worst_result["valid_image"]
    img_masked = np.where(valid, img, np.nan)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        img_masked,
        cmap="RdBu",
        vmin=-np.pi / 4,
        vmax=np.pi / 4,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(
        f"{label}: phase diff for date pair {worst_key[0]}_{worst_key[1]} (rad, ±π/4)"
    )
    if boundary_rows:
        for r in boundary_rows:
            ax.axhline(r, color="lime", ls="--", alpha=0.6, lw=0.8)
    fig.colorbar(im, ax=ax, label="phase (rad)")
    fig.tight_layout()
    p2 = out_prefix.with_name(f"{out_prefix.name}_worst_pair.png")
    fig.savefig(p2, dpi=120)
    logger.info("wrote %s", p2)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description="Compare stitched phase outputs between two dolphin runs."
    )
    p.add_argument(
        "baseline_dir",
        type=Path,
        help="Work directory of the azimuth_blocks=1 run (reference).",
    )
    p.add_argument(
        "test_dir",
        type=Path,
        help="Work directory of the azimuth_blocks=N run to validate.",
    )
    p.add_argument(
        "--ifg-subdir",
        default="interferograms",
        help=(
            "Subdirectory under each work dir holding stitched "
            ".int.tif files (default: interferograms)."
        ),
    )
    p.add_argument(
        "--boundary-rows",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Pixel row indices (in baseline coords) of expected "
            "block boundaries, for the plot overlay. If omitted, "
            "the script tries to infer them from baseline_dir/"
            "block_NN subdirs; if those aren't present, the lines "
            "are skipped."
        ),
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Write PNG plots of the row profile and worst-pair phase-diff image.",
    )
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("phase_compare"),
        help="Prefix for output files (default: ./phase_compare).",
    )
    p.add_argument(
        "--label",
        default="baseline vs test",
        help="Label for plot titles (e.g. 'baseline vs buggy').",
    )
    args = p.parse_args(argv)

    base_ifgs = find_ifgs(args.baseline_dir, args.ifg_subdir)
    test_ifgs = find_ifgs(args.test_dir, args.ifg_subdir)
    common = sorted(set(base_ifgs).intersection(test_ifgs))
    only_base = sorted(set(base_ifgs) - set(test_ifgs))
    only_test = sorted(set(test_ifgs) - set(base_ifgs))
    logger.info(
        "baseline ifgs: %d, test ifgs: %d, common: %d",
        len(base_ifgs),
        len(test_ifgs),
        len(common),
    )
    if only_base:
        logger.warning("baseline-only date pairs: %s", only_base[:5])
    if only_test:
        logger.warning("test-only date pairs: %s", only_test[:5])
    if not common:
        logger.error("No common date pairs to compare.")
        return 2

    results: dict[tuple[str, str], dict] = {}
    for k in common:
        logger.info("comparing %s_%s", k[0], k[1])
        results[k] = compare_pair(base_ifgs[k], test_ifgs[k])

    # Per-pair stats
    print()
    print(
        f"{'date pair':<22} {'median |phase|':>16} {'p95 |phase|':>14} "
        f"{'row-profile RMS':>18} {'n_valid':>12}"
    )
    print("-" * 86)
    for k in common:
        r = results[k]
        rp = r["row_profile"]
        rms = float(np.sqrt(np.nanmean(rp**2))) if rp.size else float("nan")
        print(
            f"{k[0]}_{k[1]:<13} {r['median_abs_phase']:>16.4f} "
            f"{r['p95_abs_phase']:>14.4f} {rms:>18.4f} {r['n_valid']:>12d}"
        )

    agg = aggregate_row_profile(
        [results[k]["row_profile"] for k in common],
        [results[k]["row_valid_count"] for k in common],
    )

    # Look at the top/bottom 5 rows by phase-diff to spot stripes quickly
    if agg.size:
        idx_sorted = np.argsort(-np.nan_to_num(agg, nan=-1.0))
        print("\nTop 10 rows by aggregate row-mean |phase diff|:")
        for i in idx_sorted[:10]:
            print(f"  row {i:>6d}: {agg[i]:.4f} rad")

    # Plot overlays — need boundary rows
    boundary_rows = args.boundary_rows
    if boundary_rows is None and block_row_boundaries(args.baseline_dir) == []:
        logger.warning(
            "Found block_NN subdirs in baseline_dir but couldn't infer pixel "
            "boundaries. Re-run with --boundary-rows R1 R2 ... to highlight "
            "them in the plot. The plot still renders without the overlay."
        )
        boundary_rows = None

    if args.plot:
        worst_key = find_worst_pair(results)
        worst_result = results[worst_key] if worst_key else None
        write_plots(
            args.out_prefix,
            agg_profile=agg,
            worst_key=worst_key,
            worst_result=worst_result,
            boundary_rows=boundary_rows,
            label=args.label,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
