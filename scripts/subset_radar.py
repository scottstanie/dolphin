#!/usr/bin/env python
"""Subset a radar-coordinate file using a lon/lat bounding box.

This script takes a geographic bounding box (in lon/lat) and converts it to
radar pixel coordinates using longitude/latitude lookup files. It can either
print the pixel bounding box or create a VRT file for subsetting.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import rasterio
import tyro
from osgeo import gdal


def lonlat_to_bbox(
    lon_file: Path,
    lat_file: Path,
    bounds: tuple[float, float, float, float],
    buffer: int = 0,
) -> tuple[int, int, int, int]:
    """Convert a lon/lat bounding box to radar pixel coordinates.

    Parameters
    ----------
    lon_file : Path
        Path to the longitude raster file (in radar coordinates).
    lat_file : Path
        Path to the latitude raster file (in radar coordinates).
    bounds : tuple[float, float, float, float]
        Bounding box as (left, bottom, right, top) in lon/lat.
    buffer : int, optional
        Number of pixels to buffer the bounding box, by default 0.

    Returns
    -------
    tuple[int, int, int, int]
        Pixel bounding box as (x_min, y_min, x_max, y_max) in radar coordinates.

    """
    left, bottom, right, top = bounds

    with rasterio.open(lon_file) as lon_ds, rasterio.open(lat_file) as lat_ds:
        lon_data = lon_ds.read(1)
        lat_data = lat_ds.read(1)

    # Find pixels within the lon/lat bounds
    mask = (
        (lon_data >= left)
        & (lon_data <= right)
        & (lat_data >= bottom)
        & (lat_data <= top)
    )

    if not mask.any():
        raise ValueError("No pixels found within the specified lon/lat bounds")

    # Get the row/col indices where the mask is True
    rows, cols = np.where(mask)

    # Get bounding box with buffer
    row_min = rows.min()
    row_max = rows.max()
    col_min = cols.min()
    col_max = cols.max()

    row_min_buffered = row_min - buffer
    row_max_buffered = row_max + buffer
    col_min_buffered = col_min - buffer
    col_max_buffered = col_max + buffer

    y_min = max(0, row_min_buffered)
    y_max = row_max_buffered
    x_min = max(0, col_min_buffered)
    x_max = col_max_buffered

    n_rows, n_cols = lon_data.shape
    if (
        row_min_buffered < 0
        or col_min_buffered < 0
        or row_max_buffered >= n_rows
        or col_max_buffered >= n_cols
    ):
        warnings.warn(
            "Buffered radar bounding box exceeds lon/lat raster extent; "
            f"requested lon/lat bounds={bounds}; "
            "requested pixel extents="
            f"(x_min={col_min_buffered}, y_min={row_min_buffered}, "
            f"x_max={col_max_buffered}, y_max={row_max_buffered})",
            RuntimeWarning,
            stacklevel=2,
        )

    return (x_min, y_min, x_max, y_max)


def create_subset_vrt(
    input_file: Path,
    output_vrt: Path,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> None:
    """Create a VRT file subsetting the input file to the specified pixel bounds.

    Parameters
    ----------
    input_file : Path
        Input raster file in radar coordinates.
    output_vrt : Path
        Output VRT file path.
    x_min : int
        Minimum x (column) pixel coordinate.
    y_min : int
        Minimum y (row) pixel coordinate.
    x_max : int
        Maximum x (column) pixel coordinate.
    y_max : int
        Maximum y (row) pixel coordinate.

    """
    width = x_max - x_min
    height = y_max - y_min

    gdal.Translate(
        str(output_vrt),
        str(input_file),
        format="VRT",
        srcWin=[x_min, y_min, width, height],
    )


def main(
    lon_file: Path,
    lat_file: Path,
    *,
    bounds: tuple[float, float, float, float],
    buffer: int = 0,
    input_file: Path | None = None,
    output_vrt: Path | None = None,
) -> None:
    """Subset radar-coordinate files using a lon/lat bounding box.

    Parameters
    ----------
    lon_file : Path
        Longitude raster file in radar coordinates.
    lat_file : Path
        Latitude raster file in radar coordinates.
    bounds : tuple[float, float, float, float]
        Bounding box as (left, bottom, right, top) in lon/lat.
    buffer : int, optional
        Extra buffer in pixels added to the subset, by default 0.
    input_file : Path, optional
        Radar raster to subset and write as VRT, by default None.
    output_vrt : Path, optional
        Output VRT path; defaults to ``input_file`` with ``.vrt`` suffix.

    """
    bbox = lonlat_to_bbox(
        lon_file,
        lat_file,
        bounds,
        buffer,
    )

    left, bottom, right, top = bounds
    print(f"Lon/Lat bounds: ({left}, {bottom}, {right}, {top})")
    print(f"Radar pixel bbox (x_min, y_min, x_max, y_max): {bbox}")

    if input_file:
        output_vrt = output_vrt or input_file.with_suffix(".vrt")
        create_subset_vrt(input_file, output_vrt, *bbox)
        print(f"Created VRT: {output_vrt}")


if __name__ == "__main__":
    tyro.cli(main)
