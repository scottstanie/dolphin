import numpy as np
import pytest
from pyproj import CRS

from dolphin import io
from dolphin.closure import write_cumulative_closure_phase


def _write_closure(path, value, shape=(4, 5)):
    arr = np.full(shape, value, dtype="float32")
    io.write_arr(
        arr=arr,
        output_name=path,
        shape=shape,
        geotransform=[0.0, 10.0, 0.0, 0.0, 0.0, -10.0],
        projection=CRS.from_epsg(32611).to_wkt(),
        nodata=0,
    )
    return path


@pytest.fixture
def closure_dir(tmp_path):
    d = tmp_path / "interferograms"
    d.mkdir()
    # Nearest triplets over dates 0101, 0113, 0125, 0206, 0218
    _write_closure(d / "closure_phase_20220101_20220113_20220125.tif", 0.1)
    _write_closure(d / "closure_phase_20220113_20220125_20220206.tif", 0.2)
    _write_closure(d / "closure_phase_20220125_20220206_20220218.tif", 0.3)
    # A long-pair triplet starting at the compressed SLC's base date, as the
    # sequential workflow produces at ministack boundaries
    _write_closure(d / "closure_phase_20220101_20220206_20220218.tif", 5.0)
    return d


def test_cumulative_sum_scaling_and_names(closure_dir, tmp_path):
    wavelength = 4 * np.pi  # makes the scale factor exactly -1
    out_dir = tmp_path / "out"
    written = write_cumulative_closure_phase(
        sorted(closure_dir.glob("closure_phase_*.tif")),
        output_dir=out_dir,
        wavelength=wavelength,
    )
    assert [p.name for p in written] == [
        "cumulative_closure_phase_20220113.tif",
        "cumulative_closure_phase_20220125.tif",
        "cumulative_closure_phase_20220206.tif",
    ]
    values = [float(io.load_gdal(p)[0, 0]) for p in written]
    # Outputs use the extra-compressed (quantized) GeoTIFF options
    np.testing.assert_allclose(values, [-0.1, -0.3, -0.6], rtol=1e-2)
    assert io.get_raster_units(written[0]) == "meters"


def test_radians_when_no_wavelength_and_long_pair_included(closure_dir, tmp_path):
    written = write_cumulative_closure_phase(
        sorted(closure_dir.glob("closure_phase_*.tif")),
        output_dir=tmp_path / "out",
        wavelength=None,
        nearest_only=False,
    )
    # The (0101, 0206, 0218) triplet sorts by its middle date, after the
    # (0113, 0125, 0206) triplet and before (0125, 0206, 0218); its label
    # collides with the nearest triplet centered on 0206, so it is the later of
    # the two 0206 outputs in the running sum
    names = [p.name for p in written]
    assert names.count("cumulative_closure_phase_20220206.tif") == 2
    values = [float(io.load_gdal(p)[0, 0]) for p in written]
    assert values[-1] == pytest.approx(0.1 + 0.2 + 5.0 + 0.3, rel=1e-2)
    assert io.get_raster_units(written[0]) == "radians"


def test_skips_existing_outputs(closure_dir, tmp_path):
    files = sorted(closure_dir.glob("closure_phase_*.tif"))
    out_dir = tmp_path / "out"
    first = write_cumulative_closure_phase(files, out_dir, wavelength=0.056)
    mtimes = [p.stat().st_mtime_ns for p in first]
    second = write_cumulative_closure_phase(files, out_dir, wavelength=0.056)
    assert second == first
    assert [p.stat().st_mtime_ns for p in second] == mtimes


def test_no_triplets_returns_empty(tmp_path):
    assert write_cumulative_closure_phase([], tmp_path / "out") == []
