#!/usr/bin/env python
"""End-to-end smoke test for the azimuth-block fix using 2 real NISAR granules.

Crops each granule's `frequencyA/HH` to a small ROI as a GeoTIFF, builds two
dolphin configs (azimuth_blocks=1 baseline and azimuth_blocks=2 blocked), runs
both with `run_unwrap=false`, and invokes
`compare_azimuth_block_phase.py` to look for block-boundary stripes in the
phase-diff.

The whole thing should take well under a minute on a laptop.

What this can and cannot test
-----------------------------
This test will catch:

- Pipeline crashes on real NISAR data (file open, VRT, stitching, etc.)
- Catastrophic shape / extent regressions
- Block-boundary artifacts in neighborhood-statistic outputs (``similarity``,
  ``shp_counts``) — those DO differ between buggy and fixed and the diffs
  cluster at the central-region boundary row.

This test cannot distinguish the halo-overlap bug in the per-block ``.int.tif``
itself when the input stack has only 2 SLCs: a 2-SLC pairwise interferogram is
mathematically invariant to the trivial 2x2 phase-linking EVD that each block
runs, so the IFGs (and temp_coh / interferometric correlation, all of which
flow from the pairwise product) come out byte-identical in both branches.
With >= 3 SLCs the per-SLC phase rotations no longer cancel and the IFG
itself acquires a boundary artifact in the buggy branch.

Usage::

    python scripts/smoke_test_azimuth_block_nisar.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# 2 real NISAR granules sitting on the user's external drive
NISAR_DIR = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar")
SUBDATASET = "/science/LSAR/GSLC/grids/frequencyA/HH"

# A small ROI inside the granule footprint with real (non-zero) data.
# Picked once via gdal inspection at the granule center; UTM zone 10N (EPSG 32610).
ROI_XMIN, ROI_YMIN, ROI_XMAX, ROI_YMAX = 794920.0, 3828680.0, 804920.0, 3848680.0
# Resulting size at the native 5 m pixel: 4000 rows x 2000 cols.


def crop_granule_to_tif(src_h5: Path, dst_tif: Path) -> None:
    """gdal.Translate the NETCDF: subdataset out of one NISAR granule into a TIF.

    Writes a complex64 GeoTIFF with the full granule CRS preserved.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    uri = f'NETCDF:"{src_h5}":"//{SUBDATASET.lstrip("/")}"'
    # projWin: (ulx, uly, lrx, lry)
    gdal.Translate(
        str(dst_tif),
        uri,
        projWin=(ROI_XMIN, ROI_YMAX, ROI_XMAX, ROI_YMIN),
        format="GTiff",
        outputType=gdal.GDT_CFloat32,
        creationOptions=[
            "COMPRESS=LZW",
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "BIGTIFF=YES",
        ],
    )


def extract_date_from_filename(name: str) -> str:
    """Pull the YYYYMMDD start-of-acquisition tag out of a NISAR filename."""
    # NISAR_L2_..._A_<YYYYMMDD>T<HHMMSS>_<YYYYMMDD>T<HHMMSS>_...
    import re

    m = re.search(r"_A_(\d{8})T\d{6}_", name)
    if m is None:
        msg = f"Couldn't find YYYYMMDD tag in {name}"
        raise ValueError(msg)
    return m.group(1)


def write_config(
    work_dir: Path,
    cslc_tifs: list[Path],
    *,
    azimuth_blocks: int,
    n_parallel_bursts: int,
) -> Path:
    """Write a minimal dolphin YAML for the smoke test."""
    cslc_list = "\n".join(f"  - {p}" for p in cslc_tifs)
    cfg_text = f"""# Smoke-test config: 2 GSLC tifs, 1 ministack, run_unwrap=false
work_directory: {work_dir}
worker_settings:
  gpu_enabled: false
  threads_per_worker: 1
  n_parallel_bursts: {n_parallel_bursts}
  block_shape: [512, 512]
cslc_file_list:
{cslc_list}
input_options:
  cslc_date_fmt: '%Y%m%d'
  azimuth_blocks: {azimuth_blocks}
output_options:
  strides:
    y: 5
    x: 5
phase_linking:
  ministack_size: 2
  half_window:
    y: 3
    x: 3
  write_crlb: false
  write_closure_phase: false
interferogram_network:
  reference_idx: 0
unwrap_options:
  run_unwrap: false
timeseries_options:
  run_inversion: false
  run_velocity: false
"""
    cfg_path = work_dir / "dolphin_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(cfg_text)
    return cfg_path


def run_dolphin(cfg_path: Path) -> None:
    """Invoke dolphin via the in-tree CLI, importing this branch's source."""
    repo = Path(__file__).resolve().parent.parent
    env_overlay = {"PYTHONPATH": str(repo / "src")}
    logger.info("running dolphin on %s", cfg_path)
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dolphin.cli import main; main()",
            "run",
            str(cfg_path),
        ],
        env={**__import__("os").environ, **env_overlay},
        check=False,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        logger.error("dolphin failed (rc=%d). stderr tail:", r.returncode)
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(r.returncode)
    logger.info("dolphin finished cleanly")


def list_ifgs(work_dir: Path) -> list[Path]:
    """Find the stitched .int.tif outputs (network dir is named `interferograms`)."""
    ifg_dir = work_dir / "interferograms"
    return sorted(ifg_dir.glob("*.int.tif"))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory to keep test outputs. Default: a fresh tempdir "
            "(deleted at exit unless --keep is passed)."
        ),
    )
    p.add_argument(
        "--keep",
        action="store_true",
        help="Don't delete --out-dir at exit.",
    )
    p.add_argument(
        "--skip-crop",
        action="store_true",
        help="Reuse existing tif crops in --out-dir/cslcs if present.",
    )
    args = p.parse_args(argv)

    granules = sorted(NISAR_DIR.glob("NISAR_L2_*.h5"))
    if len(granules) < 2:
        logger.error("Need 2 NISAR granules in %s; found %d", NISAR_DIR, len(granules))
        return 2

    cleanup = False
    if args.out_dir is None:
        args.out_dir = Path(tempfile.mkdtemp(prefix="azblock_smoke_"))
        cleanup = not args.keep
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("smoke-test workspace: %s", args.out_dir)

    try:
        # 1. Crop each granule to a small TIF
        cslcs_dir = args.out_dir / "cslcs"
        cslcs_dir.mkdir(exist_ok=True)
        cslc_tifs: list[Path] = []
        for g in granules:
            date = extract_date_from_filename(g.name)
            dst = cslcs_dir / f"{date}.slc.tif"
            if dst.exists() and args.skip_crop:
                logger.info("reusing %s", dst.name)
            else:
                logger.info("cropping %s → %s", g.name, dst.name)
                crop_granule_to_tif(g, dst)
            cslc_tifs.append(dst)

        # Sanity: do they have the right shape?
        from osgeo import gdal

        gdal.UseExceptions()
        ds = gdal.Open(str(cslc_tifs[0]))
        logger.info(
            "cropped tif size: %d x %d  gt=%s",
            ds.RasterYSize,
            ds.RasterXSize,
            ds.GetGeoTransform(),
        )

        # 2. Run baseline (azimuth_blocks=1)
        baseline_dir = args.out_dir / "baseline_az1"
        baseline_cfg = write_config(
            baseline_dir,
            cslc_tifs,
            azimuth_blocks=1,
            n_parallel_bursts=1,
        )
        run_dolphin(baseline_cfg)
        baseline_ifgs = list_ifgs(baseline_dir)
        logger.info("baseline ifgs: %s", [p.name for p in baseline_ifgs])

        # 3. Run blocked (azimuth_blocks=2)
        blocked_dir = args.out_dir / "blocked_az2"
        blocked_cfg = write_config(
            blocked_dir,
            cslc_tifs,
            azimuth_blocks=2,
            n_parallel_bursts=2,
        )
        run_dolphin(blocked_cfg)
        blocked_ifgs = list_ifgs(blocked_dir)
        logger.info("blocked ifgs: %s", [p.name for p in blocked_ifgs])

        if not baseline_ifgs or not blocked_ifgs:
            logger.error("Missing stitched ifgs; see dolphin logs in run dirs.")
            return 3

        # 4. Compare. The block boundary in baseline-pixel-coords is at the
        # midpoint of the cropped frame.
        ds = gdal.Open(str(baseline_ifgs[0]))
        boundary_row = ds.RasterYSize // 2
        logger.info("expected block-boundary row in baseline coords: %d", boundary_row)

        compare = Path(__file__).resolve().parent / "compare_azimuth_block_phase.py"
        cmd = [
            sys.executable,
            str(compare),
            str(baseline_dir),
            str(blocked_dir),
            "--plot",
            "--label",
            "az1 vs az2 (fixed)",
            "--out-prefix",
            str(args.out_dir / "compare"),
            "--boundary-rows",
            str(boundary_row),
        ]
        logger.info("compare: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

        logger.info("\n%s", "=" * 60)
        logger.info("DONE. Outputs in %s", args.out_dir)
        logger.info("  baseline:  %s", baseline_dir)
        logger.info("  blocked:   %s", blocked_dir)
        logger.info("  plots:     %s_*.png", args.out_dir / "compare")
        logger.info("Expectation: row-mean |phase diff| should be small")
        logger.info(
            "everywhere (no peak at row %d). The plot tells the story.", boundary_row
        )
        return 0
    finally:
        if cleanup:
            logger.info("cleaning up %s (pass --keep to retain)", args.out_dir)
            shutil.rmtree(args.out_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
