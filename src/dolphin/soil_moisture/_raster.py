"""File-based workflow for soil moisture index computation.

This module provides functions to compute the InSAR Soil Moisture Index from
closure phase raster files and write the results to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from dolphin import io
from dolphin._overviews import ImageType, create_overviews

from ._core import (
    compute_cumulative_closure_phase,
    compute_soil_moisture_index,
    detrend_cumulative_closure_phase,
)

logger = logging.getLogger("dolphin")

__all__ = [
    "create_soil_moisture_index",
]


def create_soil_moisture_index(
    closure_phase_files: Sequence[Path | str],
    output_dir: Path | str,
    *,
    temporal_coherence_file: Path | str | None = None,
    coherence_threshold: float = 0.5,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
    add_overviews: bool = True,
) -> tuple[list[Path], Path, Path]:
    """Compute InSAR Soil Moisture Index from closure phase raster files.

    This function reads a stack of closure phase files, computes the cumulative
    closure phase and detrended soil moisture index, and writes the results
    to disk.

    Parameters
    ----------
    closure_phase_files : Sequence[Path | str]
        List of paths to closure phase raster files.
        Files should be in chronological order.
        Each file contains a single band of closure phase values (in radians).
    output_dir : Path | str
        Directory to write output files.
    temporal_coherence_file : Path | str, optional
        Path to temporal coherence file for quality masking.
        Pixels below `coherence_threshold` will be masked (set to NaN).
    coherence_threshold : float, optional
        Threshold for temporal coherence masking. Default is 0.5.
    block_shape : tuple[int, int], optional
        Block shape for processing. Default is (256, 256).
    num_threads : int, optional
        Number of parallel threads. Default is 4.
    add_overviews : bool, optional
        If True, add overviews to output rasters. Default is True.

    Returns
    -------
    ismi_files : list[Path]
        List of paths to output ISMI (soil moisture index) raster files.
        One file per input closure phase file.
    cumulative_file : Path
        Path to multi-band cumulative closure phase file.
    trend_file : Path
        Path to linear trend raster file.

    Notes
    -----
    The output ISMI is a **relative** soil moisture product. To obtain absolute
    volumetric soil moisture, calibration against external data (e.g., SMAP,
    in-situ measurements) is required.

    Output file naming:
    - ISMI files: `ismi_YYYYMMDD.tif` (one per triplet date)
    - Cumulative closure phase: `cumulative_closure_phase.tif` (multi-band)
    - Trend: `closure_phase_trend.tif` (single band)

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    closure_phase_files = [Path(f) for f in closure_phase_files]
    n_files = len(closure_phase_files)

    if n_files == 0:
        raise ValueError("No closure phase files provided")

    logger.info(f"Computing soil moisture index from {n_files} closure phase files")

    # Create output file paths
    # Extract dates from input filenames for output naming
    ismi_files = [
        output_dir / f"ismi_{f.stem.replace('closure_phase_', '')}.tif"
        for f in closure_phase_files
    ]
    cumulative_file = output_dir / "cumulative_closure_phase.tif"
    trend_file = output_dir / "closure_phase_trend.tif"

    # Check if outputs already exist
    if all(f.exists() for f in ismi_files) and cumulative_file.exists():
        logger.info("All output files exist, skipping computation")
        return ismi_files, cumulative_file, trend_file

    # Create VRT for reading closure phase stack
    vrt_file = output_dir / "closure_phase_stack.vrt"
    reader = io.VRTStack(
        file_list=closure_phase_files,
        outfile=vrt_file,
        skip_size_check=True,
    )

    # Read temporal coherence mask if provided
    if temporal_coherence_file is not None:
        temp_coh = io.load_gdal(temporal_coherence_file, masked=True)
        bad_pixel_mask = temp_coh.filled(0) < coherence_threshold
    else:
        bad_pixel_mask = None

    def process_block(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[np.ndarray, slice, slice]:
        """Process one block of closure phase data."""
        closure_stack = readers[0][:, rows, cols]

        # Apply mask if provided
        if bad_pixel_mask is not None:
            block_mask = bad_pixel_mask[rows, cols]
            closure_stack = closure_stack.astype(np.float64)
            closure_stack[:, block_mask] = np.nan

        # Compute cumulative closure phase
        cumulative = compute_cumulative_closure_phase(closure_stack, axis=0)

        # Detrend to get soil moisture index
        ismi, trend = detrend_cumulative_closure_phase(
            cumulative, axis=0, return_trend=True
        )

        # Stack outputs: ISMI layers, cumulative layers, then trend
        # Shape: (2*n_files + 1, block_rows, block_cols)
        output = np.vstack([ismi, cumulative, trend[np.newaxis]])

        return output, rows, cols

    # Create output writers
    all_output_files = [*ismi_files, cumulative_file, trend_file]

    # For the trend file, we only write one band
    # BackgroundStackWriter expects one file per output band
    # So we need to handle this differently

    # Alternative approach: write ISMI and cumulative as stacks, trend separately
    writer = io.BackgroundStackWriter(
        [*ismi_files, *[cumulative_file] * n_files, trend_file],
        like_filename=closure_phase_files[0],
        nodata=np.nan,
    )

    io.process_blocks(
        readers=[reader],
        writer=writer,
        func=process_block,
        block_shape=block_shape,
        num_threads=num_threads,
    )

    writer.notify_finished()

    # Set units metadata
    for f in ismi_files:
        io.set_raster_units(f, units="radians (relative)")
    io.set_raster_units(cumulative_file, units="radians")
    io.set_raster_units(trend_file, units="radians/acquisition")

    if add_overviews:
        logger.info("Creating overviews for soil moisture index files")
        create_overviews(ismi_files, image_type=ImageType.DEFAULT, max_workers=2)

    logger.info("Completed soil moisture index computation")
    return ismi_files, cumulative_file, trend_file


def create_soil_moisture_index_from_arrays(
    closure_phases: NDArray[np.floating],
    output_dir: Path | str,
    *,
    dates: Sequence[str] | None = None,
    like_filename: Path | str | None = None,
    temporal_coherence: NDArray[np.floating] | None = None,
    coherence_threshold: float = 0.5,
    add_overviews: bool = True,
) -> tuple[list[Path], Path, Path]:
    """Compute and save soil moisture index from in-memory closure phase arrays.

    This is a convenience function for when closure phases are already loaded
    in memory (e.g., directly after phase linking).

    Parameters
    ----------
    closure_phases : NDArray
        Array of closure phases with shape (n_triplets, rows, cols).
        Units should be radians.
    output_dir : Path | str
        Directory to write output files.
    dates : Sequence[str], optional
        Date strings for output file naming.
        If not provided, uses sequential integers.
    like_filename : Path | str, optional
        Reference raster file for geospatial metadata.
        If not provided, outputs will not have geospatial information.
    temporal_coherence : NDArray, optional
        Temporal coherence array for quality masking.
        Shape: (rows, cols).
    coherence_threshold : float, optional
        Threshold for temporal coherence masking. Default is 0.5.
    add_overviews : bool, optional
        If True, add overviews to output rasters. Default is True.

    Returns
    -------
    ismi_files : list[Path]
        List of paths to output ISMI files.
    cumulative_file : Path
        Path to cumulative closure phase file.
    trend_file : Path
        Path to trend file.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    n_triplets = closure_phases.shape[0]

    # Generate output filenames
    if dates is None:
        dates = [f"{i:04d}" for i in range(n_triplets)]

    ismi_files = [output_dir / f"ismi_{d}.tif" for d in dates]
    cumulative_file = output_dir / "cumulative_closure_phase.tif"
    trend_file = output_dir / "closure_phase_trend.tif"

    # Compute soil moisture index
    result = compute_soil_moisture_index(
        closure_phases,
        temporal_coherence=temporal_coherence,
        coherence_threshold=coherence_threshold,
    )

    # Write outputs
    for i, (ismi_file, ismi_band) in enumerate(zip(ismi_files, result.ismi)):
        io.write_arr(
            arr=ismi_band,
            output_name=ismi_file,
            like_filename=like_filename,
            nodata=np.nan,
        )
        io.set_raster_units(ismi_file, units="radians (relative)")

    # Write cumulative closure phase as multi-band
    io.write_arr(
        arr=result.cumulative_closure_phase,
        output_name=cumulative_file,
        like_filename=like_filename,
        nodata=np.nan,
    )
    io.set_raster_units(cumulative_file, units="radians")

    # Write trend
    io.write_arr(
        arr=result.trend,
        output_name=trend_file,
        like_filename=like_filename,
        nodata=np.nan,
    )
    io.set_raster_units(trend_file, units="radians/acquisition")

    if add_overviews:
        logger.info("Creating overviews for soil moisture index files")
        create_overviews(ismi_files, image_type=ImageType.DEFAULT, max_workers=2)

    return ismi_files, cumulative_file, trend_file
