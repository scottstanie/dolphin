"""Geocode a swath file from ISCE2 using latitude and longitude files."""

from __future__ import annotations

import tempfile
from pathlib import Path

import tyro
from osgeo import gdal, osr

from ._types import Bbox, PathOrStr


def geocode_using_gdal_warp(
    swath_file: PathOrStr,
    lat_file: PathOrStr,
    lon_file: PathOrStr,
    output_file: PathOrStr,
    insrs: int = 4326,
    outsrs: str | None = None,
    spacing: float | None = None,
    fmt: str = "GTiff",
    bounds: Bbox | None = None,
    method: str = "near",
) -> Path:
    """Geocode a swath file using the corresponding latitude and longitude files.

    This function creates a temporary VRT file with geolocation metadata and then uses
    GDAL Warp (with geoloc=True) to produce a geocoded output.

    Parameters
    ----------
    swath_file : PathOrStr
        Path to the input swath file.
    lat_file : PathOrStr
        Path to the file holding the latitude data.
    lon_file : Union[str, Path]
        Path to the file holding the longitude data.
    output_file : PathOrStr
        Path where the geocoded output will be saved.
    insrs : int
        EPSG code for the input spatial reference (default is 4326).
    outsrs : str | None
        Output spatial reference (e.g. "EPSG:4326" or a full WKT string). If None,
        no reprojection is applied.
    spacing : float | None
        Pixel spacing (resolution) to request in the output.
        If None, the native resolution is used.
    fmt : str
        Output file format. Default is 'GTiff'.
    bounds : Bbox | None
        Output bounds as (left, bottom, right, top).
        If provided, the output image will be clipped.
    method : str
        Resampling algorithm to use (e.g. "near", "bilinear"). Default is "near".

    Returns
    -------
    Path
        The path to the output geocoded file.

    """
    # Ensure we are working with Path objects.
    swath_file = Path(swath_file)
    lat_file = Path(lat_file)
    lon_file = Path(lon_file)
    output_file = Path(output_file)

    # Create a temporary VRT file.
    temp_vrt: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as tmp:
            temp_vrt = tmp.name

        driver = gdal.GetDriverByName("VRT")
        in_ds = gdal.Open(str(swath_file), gdal.GA_ReadOnly)
        if in_ds is None:
            raise ValueError(f"Unable to open input swath file: {swath_file}")

        vrt_ds = driver.Create(temp_vrt, in_ds.RasterXSize, in_ds.RasterYSize, 0)
        if vrt_ds is None:
            raise RuntimeError("Failed to create VRT dataset.")

        # Build a simple XML snippet to point each band back to the source file.
        sourcexmltmpl = (
            "    <SimpleSource>\n"
            "      <SourceFilename>{0}</SourceFilename>\n"
            "      <SourceBand>{1}</SourceBand>\n"
            "    </SimpleSource>"
        )

        # Loop over each band of the input and add it to the VRT.
        for i in range(in_ds.RasterCount):
            band = in_ds.GetRasterBand(i + 1)
            vrt_ds.AddBand(band.DataType)
            vrt_ds.GetRasterBand(i + 1).SetMetadata(
                {"source_0": sourcexmltmpl.format(str(swath_file), i + 1)},
                "vrt_sources",
            )

        # Set up the geolocation metadata.
        sref = osr.SpatialReference()
        sref.ImportFromEPSG(insrs)
        srswkt = sref.ExportToWkt()
        vrt_ds.SetMetadata(
            {
                "SRS": srswkt,
                "X_DATASET": str(lon_file),
                "X_BAND": "1",
                "Y_DATASET": str(lat_file),
                "Y_BAND": "1",
                "PIXEL_OFFSET": "0",
                "LINE_OFFSET": "0",
                "PIXEL_STEP": "1",
                "LINE_STEP": "1",
            },
            "GEOLOCATION",
        )

        # Close datasets so that the VRT is flushed.
        vrt_ds = None
        in_ds = None

        # Set up the GDAL warp options.
        warp_opts = gdal.WarpOptions(
            format=fmt,
            xRes=spacing,
            yRes=spacing,
            dstSRS=outsrs,
            outputBounds=bounds,
            resampleAlg=method,
            geoloc=True,
        )

        # Run the warp to perform geocoding.
        gdal.Warp(str(output_file), temp_vrt, options=warp_opts)
        return output_file
    finally:
        if temp_vrt is not None and Path(temp_vrt).exists():
            Path(temp_vrt).unlink()


def main() -> None:
    """Geocode a file using isce2 latitude and longitude files."""
    args = tyro.cli(geocode_using_gdal_warp)
    geocode_using_gdal_warp(**vars(args))


if __name__ == "__main__":
    main()
