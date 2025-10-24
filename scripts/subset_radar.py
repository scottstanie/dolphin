#!/usr/bin/env python
"""Subset a radar-coordinate file using a lon/lat bounding box.

Converts a geographic bounding box (lon/lat) to a pixel window in radar geometry
using longitude/latitude lookup rasters. Can print the window or create a VRT
subset.

Conventions
-----------
- Bounds are (left, bottom, right, top) in degrees.
- The pixel window returned is (col_off, row_off, width, height),
    i.e., suitable for GDAL srcWin
- Handles antimeridian crossing (e.g., left=170, right=-170).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import rasterio
import tyro
from osgeo import gdal


def _normalize_lons_to(lons: np.ndarray, v: float) -> float:
    """Normalize longitude value v to the same wrapping as array lons."""
    finite = np.isfinite(lons)
    if not finite.any():
        return v
    lon_min = np.nanmin(lons[finite])
    lon_max = np.nanmax(lons[finite])
    # If data looks like [0, 360], map v to that; else to [-180, 180]
    if lon_min >= -1 and lon_max > 180:
        # target [0,360)
        vv = v % 360.0
        if vv < 0:
            vv += 360.0
        return vv
    else:
        # target [-180,180)
        vv = ((v + 180.0) % 360.0) - 180.0
        return vv


def _lon_mask(
    lon_data: np.ndarray, left: float, right: float, eps: float
) -> np.ndarray:
    """Wrap-aware longitude selection mask."""
    # Normalize requested bounds to LUT wrapping
    left_n = _normalize_lons_to(lon_data, left)
    right_n = _normalize_lons_to(lon_data, right)

    if math.isclose(left_n, right_n, abs_tol=eps):
        # Degenerate case: treat as full wrap
        return np.isfinite(lon_data)

    if left_n <= right_n:
        return (lon_data >= left_n - eps) & (lon_data <= right_n + eps)
    else:
        # Dateline crossing: select values >= left OR <= right
        return (lon_data >= left_n - eps) | (lon_data <= right_n + eps)


def lonlat_to_window(
    lon_file: Path,
    lat_file: Path,
    bounds: tuple[float, float, float, float],
    buffer: int = 0,
) -> tuple[int, int, int, int]:
    """Convert a lon/lat bbox to a GDAL/Rasterio-style pixel window.

    Parameters
    ----------
    lon_file : Path
        Path to longitude raster (radar geometry).
    lat_file : Path
        Path to latitude raster (radar geometry).
    bounds : (left, bottom, right, top)
        Geographic bbox in degrees.
    buffer : int
        Extra pixels to include on all sides (non-negative).

    Returns
    -------
    (col_off, row_off, width, height)
        Pixel window suitable for GDAL srcWin / rasterio window.

    """
    left, bottom, right, top = bounds
    if buffer < 0:
        raise ValueError("buffer must be >= 0")

    with rasterio.open(lon_file) as lon_ds, rasterio.open(lat_file) as lat_ds:
        if lon_ds.width != lat_ds.width or lon_ds.height != lat_ds.height:
            raise ValueError("lon/lat rasters must have identical shape")
        lon_nodata = lon_ds.nodata
        lat_nodata = lat_ds.nodata
        lon_data = lon_ds.read(1, masked=False)
        lat_data = lat_ds.read(1, masked=False)

    # Build validity mask (exclude NaN and explicit nodata)
    valid = np.isfinite(lon_data) & np.isfinite(lat_data)
    if lon_nodata is not None:
        valid &= lon_data != lon_nodata
    if lat_nodata is not None:
        valid &= lat_data != lat_nodata

    eps = 1e-9
    m_lon = _lon_mask(lon_data, left, right, eps)
    m_lat = (lat_data >= bottom - eps) & (lat_data <= top + eps)

    mask = valid & m_lon & m_lat
    if not mask.any():
        raise ValueError("No pixels found within the specified lon/lat bounds")

    rows, cols = np.where(mask)

    # Inclusive min/max indices
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())

    # Apply buffer (still inclusive indices)
    rmin_b = rmin - buffer
    rmax_b = rmax + buffer
    cmin_b = cmin - buffer
    cmax_b = cmax + buffer

    n_rows, n_cols = lon_data.shape

    # Clamp to image bounds
    rmin_b = max(0, rmin_b)
    cmin_b = max(0, cmin_b)
    rmax_b = min(n_rows - 1, rmax_b)
    cmax_b = min(n_cols - 1, cmax_b)

    # Convert inclusive max to width/height counts
    width = cmax_b - cmin_b + 1
    height = rmax_b - rmin_b + 1

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Computed non-positive window: (col_off={cmin_b}, row_off={rmin_b}, "
            f"width={width}, height={height})"
        )

    # Warn if requested buffer exceeded raster extent
    if (
        (rmin_b != rmin - buffer)
        or (cmin_b != cmin - buffer)
        or (rmax_b != rmax + buffer)
        or (cmax_b != cmax + buffer)
    ):
        warnings.warn(
            "Buffered window clipped to raster extent.",
            RuntimeWarning,
            stacklevel=2,
        )

    return (cmin_b, rmin_b, width, height)


def create_subset_vrt(
    input_file: Path,
    output_vrt: Path,
    col_off: int,
    row_off: int,
    width: int,
    height: int,
) -> None:
    """Create a VRT subset from a pixel window.

    Parameters
    ----------
    input_file : Path
        Input raster file in radar coordinates.
    output_vrt : Path
        Output VRT file path.
    col_off : int
        Column offset (x).
    row_off : int
        Row offset (y).
    width : int
        Width in pixels.
    height : int
        Height in pixels.

    """
    gdal.Translate(
        str(output_vrt),
        str(input_file),
        format="VRT",
        srcWin=[col_off, row_off, width, height],
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
    r"""Subset radar-coordinate files using a lon/lat bounding box.

    Parameters
    ----------
    lon_file : Path
        Longitude raster file (in radar coordinates).
    lat_file : Path
        Latitude raster file (in radar coordinates).
    bounds : tuple[float, float, float, float]
        Geographic bounding box as (left, bottom, right, top) in degrees.
        Example: (-123.5, 37.6, -123.0, 38.1)
        Handles antimeridian crossing, e.g., (170, 60, -170, 62)
    buffer : int
        Extra buffer in pixels added to all sides (default: 0).
    input_file : Path | None
        Radar raster to subset. If provided, creates a VRT file.
    output_vrt : Path | None
        Output VRT path. Defaults to input_file with .vrt suffix.

    Examples
    --------
    Print the radar pixel window only:
        python subset_radar.py lon.tif lat.tif --bounds -80.124 25.894 -80.122 25.896

    Create a VRT subsetting a radar file:
        python subset_radar.py lon.tif lat.tif \\
            --bounds -80.124 25.894 -80.122 25.896 \\
            --input-file velocity.tif --output-vrt velocity_subset.vrt

    With a buffer across the antimeridian:
        python subset_radar.py lon.tif lat.tif \\
            --bounds 170 60 -170 62 --buffer 10 --input-file data.tif

    """
    window = lonlat_to_window(lon_file, lat_file, bounds, buffer)

    left, bottom, right, top = bounds
    print(f"Lon/Lat bounds: ({left}, {bottom}, {right}, {top})")
    print(f"Radar pixel window (col_off, row_off, width, height): {window}")

    if input_file:
        output_vrt = output_vrt or input_file.with_suffix(".vrt")
        create_subset_vrt(input_file, output_vrt, *window)
        print(f"Created VRT: {output_vrt}")


if __name__ == "__main__":
    tyro.cli(main)
