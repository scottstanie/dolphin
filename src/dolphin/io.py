from os import fspath

import numpy as np
from osgeo import gdal

from dolphin.log import get_log
from dolphin.utils import Pathlike, numpy_to_gdal_type

gdal.UseExceptions()
logger = get_log()


def copy_projection(src_file: Pathlike, dst_file: Pathlike) -> None:
    """Copy projection/geotransform from `src_file` to `dst_file`."""
    ds_src = gdal.Open(fspath(src_file))
    projection = ds_src.GetProjection()
    geotransform = ds_src.GetGeoTransform()
    nodata = ds_src.GetRasterBand(1).GetNoDataValue()

    if projection is None and geotransform is None:
        logger.info("No projection or geotransform found on file %s", input)
        return
    ds_dst = gdal.Open(fspath(dst_file), gdal.GA_Update)

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        ds_dst.SetGeoTransform(geotransform)

    if projection is not None and projection != "":
        ds_dst.SetProjection(projection)

    if nodata is not None:
        ds_dst.GetRasterBand(1).SetNoDataValue(nodata)

    ds_src = ds_dst = None


def save_arr_like(*, arr, like_filename, output_name, driver="GTiff"):
    """Save an array to a file, copying projection/nodata from `like_filename`."""
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    ysize, xsize = arr.shape[-2:]
    nbands = arr.shape[0]
    gdal.UseExceptions()
    ds = gdal.Open(fspath(like_filename))
    if driver is None:
        driver = ds.GetDriver().ShortName
    drv = gdal.GetDriverByName(driver)
    out_ds = drv.Create(
        fspath(output_name),
        xsize,
        ysize,
        nbands,
        numpy_to_gdal_type(arr.dtype),
    )
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    for i in range(nbands):
        out_ds.GetRasterBand(i + 1).WriteArray(arr[i])
    # TODO: copy other metadata
    ds = out_ds = None


def load_gdal(filename, band=None):
    """Load a gdal file into a numpy array."""
    ds = gdal.Open(fspath(filename))
    return ds.ReadAsArray() if band is None else ds.GetRasterBand(band).ReadAsArray()