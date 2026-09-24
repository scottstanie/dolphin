from pathlib import Path

import numpy as np
import pytest

from dolphin import filtering, io


def test_filter_long_wavelength():
    # Check filtering with ramp phase
    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    unw_ifg = np.pi * (x + y)

    corr = np.ones(unw_ifg.shape, dtype=np.float32)
    bad_pixel_mask = corr < 0.5

    # Filtering
    filtered_ifg = filtering.filter_long_wavelength(
        unw_ifg,
        bad_pixel_mask=bad_pixel_mask,
        pixel_spacing=100,
        wavelength_cutoff=1_000,
    )
    np.testing.assert_allclose(
        filtered_ifg[10:-10, 10:-10],
        np.zeros(filtered_ifg[10:-10, 10:-10].shape),
        atol=1.0,
    )


def test_filter_long_wavelength_too_large_cutoff():
    # Check filtering with ramp phase
    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    unw_ifg = np.pi * (x + y)
    bad_pixel_mask = np.zeros(unw_ifg.shape, dtype=bool)

    with pytest.raises(ValueError):
        filtering.filter_long_wavelength(
            unw_ifg,
            bad_pixel_mask=bad_pixel_mask,
            pixel_spacing=1,
            wavelength_cutoff=50_000,
        )


@pytest.fixture()
def unw_files(tmp_path):
    """Make series of files offset in lat/lon."""
    shape = (3, 9, 9)

    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    file_list = []
    for i in range(shape[0]):
        unw_arr = (i + 1) * np.pi * (x + y)
        fname = tmp_path / f"unw_{i}.tif"
        io.write_arr(arr=unw_arr, output_name=fname)
        file_list.append(Path(fname))

    return file_list


def test_filter(tmp_path, unw_files):
    output_dir = Path(tmp_path) / "filtered"
    filtering.filter_rasters(
        unw_filenames=unw_files,
        output_dir=output_dir,
        max_workers=1,
        wavelength_cutoff=50,
    )


def test_gaussian_filter_nan_uses_mode_with_nans():
    from scipy.ndimage import gaussian_filter

    ramp = np.tile(np.arange(64, dtype=float), (64, 1))
    with_nan = ramp.copy()
    with_nan[32, 32] = np.nan
    out = filtering.gaussian_filter_nan(with_nan, 2, mode="nearest")
    # Far from the NaN, the edges must follow the requested boundary mode.
    expected = gaussian_filter(ramp, 2, mode="nearest")
    np.testing.assert_allclose(out[:10, :10], expected[:10, :10])


@pytest.mark.parametrize("has_nan", [False, True])
def test_gaussian_filter_nan_preserves_constant_at_edges(has_nan):
    image = np.full((20, 20), 3.0)
    if has_nan:
        image[10, 10] = np.nan
    out = filtering.gaussian_filter_nan(image, 2)
    np.testing.assert_allclose(out, 3.0)
