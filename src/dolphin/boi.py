from os import fspath

from osgeo import gdal

from dolphin import io, stitching


def compute_boi(ifg_file1, ifg_file2):
    assert io.get_raster_crs(ifg_file1) == io.get_raster_crs(ifg_file2)
    (left, bottom, right, top), nodata = stitching.get_combined_bounds_nodata(
        ifg_file1, ifg_file2
    )

    def grow_bounds(file):
        return gdal.Translate(
            "",
            fspath(file),
            format="VRT",  # Just creates a file that will warp on the fly
            resampleAlg="nearest",  # nearest neighbor for resampling
            projWin=(left, top, right, bottom),
        ).ReadAsArray()

    out1 = grow_bounds(ifg_file1)
    out2 = grow_bounds(ifg_file2)
    return out1 * out2.conj()
