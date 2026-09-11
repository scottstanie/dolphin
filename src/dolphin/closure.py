"""Raster-level closure phase products built from the per-triplet closure rasters.

The per-pixel closure phases themselves are computed in
`dolphin.phase_link._closure_phase`; this module works on the rasters that the
phase-linking workflow writes for each triplet of consecutive acquisitions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
from opera_utils import get_dates

from dolphin import io
from dolphin._types import Filename

logger = logging.getLogger("dolphin")

__all__ = ["write_cumulative_closure_phase"]


def write_cumulative_closure_phase(
    closure_phase_files: Sequence[Filename],
    output_dir: Filename,
    wavelength: float | None = None,
    nearest_only: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    """Sum nearest-triplet closure phases through time, one output raster per date.

    Summing the closure phases of consecutive triplets ``(i, i+1, i+2)`` up to
    date ``m`` gives, exactly and before any wrapping, twice the disagreement
    between the bandwidth-1 (nearest-neighbor chain) and bandwidth-2 phase time
    series at that date. The cumulative raster is therefore a map of how far the
    shortest-baseline estimate has drifted from the next-shortest one: the
    systematic, estimator-dependent part of a closure-producing bias. It says
    nothing about a nuisance phase that is consistent across pairs.

    Parameters
    ----------
    closure_phase_files : Sequence[Filename]
        ``closure_phase_<d0>_<d1>_<d2>.tif`` rasters (radians), one per triplet.
    output_dir : Filename
        Directory for ``cumulative_closure_phase_<d1>.tif``. Each output is
        labeled with the middle date of the last triplet summed.
    wavelength : float, optional
        Radar wavelength in meters. If given, outputs are displacement in meters
        with positive toward the sensor, ``-wavelength / (4 pi)`` times the summed
        phase, the same convention as the ``timeseries/`` rasters. If None,
        outputs stay in radians.
    nearest_only : bool
        Keep only triplets whose three dates are consecutive in the sorted set of
        all dates seen. Sequential runs also produce triplets that start at a
        compressed SLC's base date, which span a long pair and are not part of
        the bandwidth-1 versus bandwidth-2 identity. Default True.
    overwrite : bool
        Recompute even if every output already exists. Default False.

    Returns
    -------
    list[Path]
        The written (or already existing) cumulative rasters, in date order.

    """
    output_dir = Path(output_dir)
    triplets: list[tuple[tuple[datetime, ...], Path]] = []
    for f in closure_phase_files:
        dates = tuple(get_dates(Path(f)))
        if len(dates) != 3:
            logger.debug(f"Skipping {f}: expected three dates in the filename")
            continue
        triplets.append((dates, Path(f)))
    if not triplets:
        logger.info("No closure phase triplet rasters found; nothing to accumulate.")
        return []

    if nearest_only:
        all_dates = sorted({d for dates, _ in triplets for d in dates})
        index = {d: i for i, d in enumerate(all_dates)}
        kept = [
            (dates, f)
            for dates, f in triplets
            if [index[d] for d in dates]
            == list(range(index[dates[0]], index[dates[0]] + 3))
        ]
        n_dropped = len(triplets) - len(kept)
        if n_dropped:
            logger.info(f"Dropping {n_dropped} non-consecutive closure triplets")
        triplets = kept

    # Order by the middle date, then by the last: a triplet's closure is a
    # property of its central acquisition's neighborhood in time.
    triplets.sort(key=lambda t: (t[0][1], t[0][2], t[0][0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [
        output_dir / f"cumulative_closure_phase_{dates[1]:%Y%m%d}.tif"
        for dates, _ in triplets
    ]
    if not overwrite and all(p.exists() for p in out_paths):
        logger.info(f"Cumulative closure phase rasters exist in {output_dir}; skipping")
        return out_paths

    if wavelength is not None:
        scale, units = -float(wavelength) / (4 * np.pi), "meters"
    else:
        scale, units = 1.0, "radians"

    running = None
    for (_, fin), fout in zip(triplets, out_paths, strict=True):
        block = io.load_gdal(fin, masked=True)
        block = np.ma.filled(block, 0).astype("float64")
        running = block if running is None else running + block
        io.write_arr(
            arr=(running * scale).astype("float32"),
            output_name=fout,
            like_filename=fin,
            options=io.EXTRA_COMPRESSED_TIFF_OPTIONS,
        )
        io.set_raster_units(fout, units)
    logger.info(
        f"Wrote {len(out_paths)} cumulative closure phase rasters to {output_dir}"
    )
    return out_paths
