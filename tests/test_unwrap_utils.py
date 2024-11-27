from pathlib import Path

import numpy as np
import pytest

from dolphin import io
from dolphin.unwrap import _utils, _post_process

# Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
pytestmark = pytest.mark.filterwarnings(
    "ignore::rasterio.errors.NotGeoreferencedWarning",
    "ignore:.*io.FileIO.*:pytest.PytestUnraisableExceptionWarning",
)


@pytest.fixture()
def corr_raster(raster_100_by_200):
    # Make a correlation raster of all 1s in the same directory as the raster
    d = Path(raster_100_by_200).parent
    arr = np.ones((100, 200), dtype="float32")
    # The first 20 rows have nodata
    arr[:20, :] = np.nan
    filename = d / "corr_raster.cor.tif"
    io.write_arr(
        arr=arr,
        output_name=filename,
        like_filename=raster_100_by_200,
        nodata=np.nan,
        driver="GTiff",
    )
    return filename


@pytest.fixture()
def mask_raster(raster_100_by_200):
    # Make a correlation raster of all 1s in the same directory as the raster
    d = Path(raster_100_by_200).parent
    filename = d / "mask.tif"
    arr = np.ones((100, 200), dtype=np.uint8)
    # Mask the first 20 columns
    arr[:, :20] = 0
    io.write_arr(
        arr=arr,
        output_name=filename,
        like_filename=raster_100_by_200,
        driver="GTiff",
    )
    return filename


def test_create_combined_mask(corr_raster, mask_raster):
    out_raster = _utils.create_combined_mask(
        mask_filename=mask_raster, image_filename=corr_raster
    )
    assert Path(out_raster).name == "combined_mask.tif"
    mask = io.load_gdal(out_raster)
    assert (mask[:20] == 0).all()
    assert (mask[:, :20] == 0).all()
    assert (mask[20:, 20:] == 1).all()


def test_set_nodata_values(corr_raster):
    import rasterio

    corr_copy = corr_raster.parent / "corr_copy.tif"
    # The first 20 cols have nodata
    arr = np.ones((100, 200), dtype="float32")
    arr[:, :20] = -1
    io.write_arr(
        arr=arr,
        output_name=corr_copy,
        like_filename=corr_raster,
        nodata=-1,
        driver="GTiff",
    )
    with rasterio.open(corr_copy) as src:
        assert src.nodata == -1
        a = src.read(1)
        assert np.all(a[:, :20] == -1)
        assert np.all(a[:, 20:] == 1)

    _utils.set_nodata_values(filename=corr_copy, like_filename=corr_raster)
    # Without specifying `output_nodata`, should keep existing nodata of `filename`
    with rasterio.open(corr_copy) as src:
        assert src.nodata == -1
        a = src.read(1)
        assert np.all(a[:20] == -1)
        assert np.all(a[:, :20] == -1)

    _utils.set_nodata_values(
        filename=corr_copy, output_nodata=-2, like_filename=corr_raster
    )
    with rasterio.open(corr_copy) as src:
        assert src.nodata == -2
        a = src.read(1)
        assert np.all(a[:20] == -2)
        assert np.all(a[:, :20] == -2)


class PostProcess:
    def test_interpolate_masked_gaps():
        pass

    unw = np.ones((100, 1)) * np.linspace(0, 20, 100).reshape(1, 100)
    ifg = np.exp(1j * unw)
    # ifg = np.exp(1j * 10 * np.random.randn(64, 64))
    mask = np.random.rand(100, 100) < 0.2
    unw_masked = unw.copy()
    unw_masked[mask] = np.nan

    _post_process.interpolate_masked_gaps(unw_masked, ifg)
    np.testing.assert_array_equal(unw[2:-2, 2:-2], unw_masked[2:-2, 2:-2])
