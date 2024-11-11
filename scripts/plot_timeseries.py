#!/usr/bin/env python
from pathlib import Path
from typing import Optional, Tuple, Union

import cartopy.crs as ccrs
import click
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray
import xarray as xr
from cartopy.io import img_tiles
from matplotlib.gridspec import GridSpec
from opera_utils import get_dates

from dolphin._types import Bbox


def plot_insar_timeseries(
    velocity_da: xr.DataArray,
    timeseries_da: xr.DataArray,
    poi: Union[Tuple[float, float], None] = None,
    poi_is_latlon: bool = True,
    reference_point: Union[Tuple[int, int], None] = None,
    mask_zeros: bool = True,
    tile_zoom_level: int = 9,
    figsize: Tuple[float, float] = (12, 8),
    cmap: str = "RdYlBu",
    cbar_label: Optional[str] = None,
):
    """Create a publication-ready InSAR time series plot using xarray data.

    Parameters
    ----------
    velocity_da : xr.DataArray
        Velocity data in EPSG:4326 (lat/lon coordinates)
    timeseries_da : xr.DataArray
        Time series data in EPSG:4326
    poi : tuple, optional
        Point of interest as (lat, lon) if poi_is_latlon=True,
        or (row, col) if poi_is_latlon=False
    poi_is_latlon : bool, default=True
        Whether poi coordinates are in lat/lon or row/col
    reference_point : tuple, optional
        Reference point as (row, col)
    mask_zeros : bool, default=True
        Whether to mask zero values in velocity data
    tile_zoom_level : int, default=9
        Zoom level for background satellite imagery
    figsize : tuple, default=(12, 8)
        Figure size in inches
    cmap : str, default='RdYlBu'
        Colormap for velocity plot
    cbar_label : str, optional
        Label for colorbar. If None, tries to construct from metadata

    """
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[2, 1])

    # Top subplot: Map
    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

    # Add satellite imagery
    tiler = img_tiles.GoogleTiles(style="satellite")
    ax_map.add_image(tiler, tile_zoom_level, interpolation="bicubic")

    # Mask zeros if requested
    velocity_plot = velocity_da.where(velocity_da != 0) if mask_zeros else velocity_da

    # Get extent from velocity data
    extent = _padded_extent(velocity_da.rio.bounds(), 0.1)

    if reference_point:
        row, col = reference_point
        ref_lat = timeseries_da.y[row].item()
        ref_lon = timeseries_da.x[col].item()
        # ref_timeseries = timeseries_da.isel[row, col]
        ax_map.plot(
            ref_lon,
            ref_lat,
            "k^",
            markersize=10,
            transform=ccrs.PlateCarree(),
            label="Reference",
        )
        ax_map.legend()

    im = ax_map.imshow(
        velocity_plot,
        origin="upper",
        extent=extent,
        transform=ccrs.PlateCarree(),
        zorder=2,
        cmap=cmap,
    )

    if cbar_label is None:
        try:
            units = velocity_da.attrs.get("units", "mm/yr")
            cbar_label = f"Velocity ({units})"
        except AttributeError:
            cbar_label = "Velocity"

    cbar = fig.colorbar(im, ax=ax_map)
    cbar.set_label(cbar_label)

    ax_map.set_extent(extent, crs=ccrs.PlateCarree())

    # Add coastlines and gridlines
    ax_map.coastlines(resolution="10m")
    gl = ax_map.gridlines(draw_labels=True, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # Bottom subplot: Time series
    ax_ts = fig.add_subplot(gs[1])

    # Extract time series at point of interest
    if poi is not None:
        if poi_is_latlon:
            lon, lat = poi
            ts_data = timeseries_da.sel(y=lat, x=lon, method="nearest")
        else:
            row, col = poi
            lat = timeseries_da.y[row].item()
            lon = timeseries_da.x[col].item()
            ts_data = timeseries_da.isel(y=row, x=col)

        # Plot time series
        dates = pd.to_datetime(timeseries_da.time.values)
        ax_ts.plot(dates, ts_data, "o-", markersize=4)

        # Mark point on map
        ax_map.plot(
            lon,
            lat,
            "r*",
            markersize=10,
            transform=ccrs.PlateCarree(),
            label="Point of Interest",
        )

    # Format time series plot
    ax_ts.set_xlabel("Date")
    try:
        units = timeseries_da.attrs.get("units", "mm")
        ax_ts.set_ylabel(f"Displacement ({units})")
    except AttributeError:
        ax_ts.set_ylabel("Displacement")
    ax_ts.grid(True)

    # Rotate x-axis labels
    plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    return fig, (ax_map, ax_ts)


def _prep(ds):
    """Preprocess individual dataset when loading with open_mfdataset."""
    fname = ds.encoding["source"]
    date = get_dates(fname)[1] if len(get_dates(fname)) > 1 else get_dates(fname)[0]
    if len(ds.band) == 1:
        ds = ds.sel(band=ds.band[0])
    return ds.expand_dims(time=[pd.to_datetime(date)])


@click.command()
@click.argument(
    "timeseries_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "-v",
    "--velocity-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to velocity TIF file (default: timeseries_dir/velocity.tif)",
)
@click.option(
    "-p",
    "--poi",
    type=float,
    nargs=2,
    help="Point of interest as 'lat lon' or 'lat,lon'",
)
@click.option(
    "--poi-rowcol",
    is_flag=True,
    help="Interpret POI coordinates as row/col instead of lat/lon",
)
@click.option(
    "-r",
    "--reference-point",
    type=float,
    nargs=2,
    help="Reference point as 'row col' or 'row,col'",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file path (PNG/PDF/SVG)",
)
@click.option(
    "--figsize",
    type=float,
    nargs=2,
    default=(8, 6),
    help="Figure size in inches as 'width height' or 'width,height'",
)
@click.option(
    "--zoom-level",
    type=int,
    default=9,
    help="Zoom level for satellite imagery (default: 9)",
)
@click.option(
    "--cmap",
    type=str,
    default="RdYlBu",
    help="Colormap for velocity plot (default: RdYlBu)",
)
@click.option(
    "--mask-zeros/--no-mask-zeros",
    default=True,
    help="Mask zero values in velocity data",
)
@click.option("--show/--no-show", default=True, help="Show the plot window")
def main(
    timeseries_dir: Path,
    velocity_file: Path,
    poi: Tuple[float, float],
    poi_rowcol: bool,
    reference_point: Tuple[float, float],
    output_file: Path,
    figsize: Tuple[float, float],
    zoom_level: int,
    cmap: str,
    mask_zeros: bool,
    show: bool,
):
    r"""Create publication-ready InSAR time series plots using xarray.

    This tool generates a figure combining velocity maps and time series data
    from InSAR observations, overlaid on satellite imagery.

    Example usage:

    \b
    # Basic usage with lat/lon point of interest
    insar_plot timeseries/ -p -156 19.4

    \b
    # With row/col point of interest
    insar_plot timeseries/ --poi 100 200 --poi-rowcol

    \b
    # Full example
    insar_plot timeseries/ \\
        -v timeseries/velocity.tif \\
        -p -156 19.4 \\
        -r 385 1180 \\
        -o figure.png \\
        --figsize 12 8 \\
        --zoom-level 10 \\
        --cmap RdYlBu
    """
    # Set default velocity file if not provided
    if velocity_file is None:
        velocity_file = timeseries_dir / "velocity.tif"
        if not velocity_file.exists():
            raise click.UsageError(
                f"No velocity file provided and {velocity_file} does not exist"
            )

    # Load velocity data
    click.echo("Loading velocity data...")
    velocity_ds = rioxarray.open_rasterio(velocity_file).sel(band=1)
    velocity_latlon = velocity_ds.rio.reproject("EPSG:4326")

    # Load time series data
    click.echo("Loading time series data...")
    ts_pattern = str(timeseries_dir / "2*tif")
    ds = xr.open_mfdataset(
        ts_pattern,
        preprocess=_prep,
        engine="rasterio",
        concat_dim="time",
        combine="nested",
    ).rename({"band_data": "displacement"})

    # Reproject to lat/lon
    ts_latlon = ds.rio.reproject("EPSG:4326")

    # Handle reference point from file if needed
    from dolphin.timeseries import _read_reference_point

    if not reference_point:
        try:
            reference_point = _read_reference_point(
                timeseries_dir / "reference_point.txt"
            )
        except (ValueError, OSError) as e:
            raise click.UsageError(f"Error reading reference point file: {e}")

    click.echo("Creating plot...")

    fig, (ax_map, ax_ts) = plot_insar_timeseries(
        velocity_da=velocity_latlon,
        timeseries_da=ts_latlon.displacement,
        poi=poi,
        poi_is_latlon=not poi_rowcol,
        reference_point=reference_point,
        mask_zeros=mask_zeros,
        tile_zoom_level=zoom_level,
        figsize=figsize,
        cmap=cmap,
    )

    if output_file:
        click.echo(f"Saving figure to {output_file}...")
        fig.savefig(output_file, bbox_inches="tight", dpi=300)

    if show:
        plt.show()


def _padded_extent(bbox: Bbox, pad_pct: float) -> Bbox:
    """Return a padded extent, given a bbox and a percentage of padding."""
    left, bot, right, top = bbox
    padx = pad_pct * (right - left) / 2
    pady = pad_pct * (top - bot) / 2
    return (left - padx, right + padx, bot - pady, top + pady)


if __name__ == "__main__":
    main()
