import numpy as np
import pytest
from numpy.testing import assert_allclose
from osgeo import gdal

import dolphin.ps
from dolphin import io
from dolphin.ps import DispersionMetric


def test_ps_block(slc_stack):
    # Run the PS selector on entire stack
    amp_mean, amp_disp, ps_pixels = dolphin.ps.calc_ps_block(
        np.abs(slc_stack),
        amp_dispersion_threshold=0.25,  # should be too low for random data
    )
    assert amp_mean.shape == amp_disp.shape == ps_pixels.shape
    assert amp_mean.dtype == amp_disp.dtype == np.float32
    assert ps_pixels.dtype == bool

    assert ps_pixels.sum() == 0

    assert amp_mean.min() > 0
    assert amp_disp.min() >= 0


def test_ps_nodata(slc_stack):
    s_nan = slc_stack.copy()
    s_nan[:, 0, 0] = np.nan
    # Run the PS selector on entire stack
    amp_mean, amp_disp, ps_pixels = dolphin.ps.calc_ps_block(
        np.abs(s_nan),
        amp_dispersion_threshold=0.95,  # high thresh shouldn't matter for nodata
    )
    assert amp_mean[0, 0] == 0
    assert amp_disp[0, 0] == 0
    assert not ps_pixels[0, 0]


def test_ps_threshold(slc_stack):
    _, _, ps_pixels = dolphin.ps.calc_ps_block(
        np.abs(slc_stack),
        amp_dispersion_threshold=100000,
    )
    assert ps_pixels.sum() == ps_pixels.size
    _, _, ps_pixels = dolphin.ps.calc_ps_block(
        np.abs(slc_stack),
        amp_dispersion_threshold=0,
    )
    assert ps_pixels.sum() == 0


@pytest.fixture()
def vrt_stack(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    return io.VRTStack(slc_file_list, outfile=vrt_file)


def test_create_ps(tmp_path, vrt_stack):
    ps_mask_file = tmp_path / "ps_pixels.tif"

    amp_dispersion_file = tmp_path / "amp_disp.tif"
    amp_mean_file = tmp_path / "amp_mean.tif"
    dolphin.ps.create_ps(
        reader=vrt_stack,
        like_filename=vrt_stack.outfile,
        output_amp_dispersion_file=amp_dispersion_file,
        output_amp_mean_file=amp_mean_file,
        output_file=ps_mask_file,
    )
    assert io.get_raster_dtype(ps_mask_file) == np.uint8
    assert io.get_raster_dtype(amp_mean_file) == np.float32
    assert io.get_raster_dtype(amp_dispersion_file) == np.float32


@pytest.fixture()
def vrt_stack_with_nans(tmp_path, raster_with_nan_block):
    vrt_file = tmp_path / "test_with_nans.vrt"
    return io.VRTStack([raster_with_nan_block, raster_with_nan_block], outfile=vrt_file)


def _write_zeros(file, shape):
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(str(file), shape[1], shape[0], 1, gdal.GDT_Float32)
    bnd = ds.GetRasterBand(1)
    bnd.WriteArray(np.zeros(shape))
    ds.SetMetadataItem("N", "0", "ENVI")
    ds = bnd = None


def test_multilook_ps_file(tmp_path, vrt_stack):
    ps_mask_file = tmp_path / "ps_pixels.tif"

    amp_dispersion_file = tmp_path / "amp_disp.tif"
    amp_mean_file = tmp_path / "amp_mean.tif"
    dolphin.ps.create_ps(
        reader=vrt_stack,
        like_filename=vrt_stack.outfile,
        output_amp_dispersion_file=amp_dispersion_file,
        output_amp_mean_file=amp_mean_file,
        output_file=ps_mask_file,
    )
    output_ps_file, output_amp_disp_file = dolphin.ps.multilook_ps_files(
        strides={"x": 5, "y": 3},
        ps_mask_file=ps_mask_file,
        amp_dispersion_file=amp_dispersion_file,
    )
    assert io.get_raster_dtype(output_ps_file) == np.uint8
    assert io.get_raster_dtype(output_amp_disp_file) == np.float32


def test_compute_combined_amplitude_means():
    # Test basic functionality
    amplitudes = np.array([[[1.0, 1.0], [1.0, 1.0]], [[6.0, 6.0], [11.0, 21.0]]])
    N = np.array([9, 1])
    expected = np.array([[1.5, 1.5], [2.0, 3.0]])
    result = dolphin.ps.combine_means(amplitudes, N)
    assert_allclose(result, expected, rtol=1e-5)

    #  Test with multiple groups
    amplitudes = np.random.randn(10, 2, 2) ** 2
    amp_mean_1 = np.mean(amplitudes[:5], axis=0)
    amp_mean_2 = np.mean(amplitudes[5:9], axis=0)
    amp_3 = amplitudes[9]
    result = dolphin.ps.combine_means(
        np.stack([amp_mean_1, amp_mean_2, amp_3]), [5, 4, 1]
    )
    assert_allclose(result, np.mean(amplitudes, axis=0), rtol=1e-5)

    # Test with all equal weights
    expected_equal = np.mean(amplitudes, axis=0)
    result_equal = dolphin.ps.combine_means(amplitudes, np.ones(len(amplitudes)))
    assert_allclose(result_equal, expected_equal, rtol=1e-5)


def test_compute_combined_amplitude_dispersions():
    # Test basic functionality

    amplitudes = np.random.randn(10, 2, 2) ** 2

    amp_mean, amp_disp, _ = dolphin.ps.calc_ps_block(amplitudes)

    N = [5, 4, 1]

    amp_mean_1, amp_disp_1, _ = dolphin.ps.calc_ps_block(amplitudes[:5])
    amp_mean_2, amp_disp_2, _ = dolphin.ps.calc_ps_block(amplitudes[5:9])

    mean_inputs = np.stack([amp_mean_1, amp_mean_2, amplitudes[9]])
    # Note: a dispersion of N=1 isn't really defined. we dont use that
    disp_inputs = np.stack([amp_disp_1, amp_disp_2, np.zeros_like(amplitudes[9])])

    combined_disp, combined_mean = dolphin.ps.combine_amplitude_dispersions(
        dispersions=disp_inputs, means=mean_inputs, N=N
    )
    assert_allclose(combined_disp, amp_disp, rtol=1e-5)


def test_single_group():
    """Test with a group where all N=1 (meaning we passed in just the amplitudes)."""
    amplitudes = np.random.randn(10, 2, 2) ** 2
    amp_mean, amp_disp, _ = dolphin.ps.calc_ps_block(amplitudes)
    N = [1] * len(amplitudes)
    result = dolphin.ps.combine_means(amplitudes, N)
    assert_allclose(result, amp_mean, rtol=1e-5)

    result_disp, result_mean = dolphin.ps.combine_amplitude_dispersions(
        np.zeros_like(amplitudes), amplitudes, N
    )
    assert_allclose(result_disp, amp_disp, rtol=1e-5)
    assert_allclose(result_mean, amp_mean, rtol=1e-5)


def test_dispersion_metric_enum():
    """Test DispersionMetric enum functionality."""
    # Test from_any method
    assert DispersionMetric.from_any("nad") == DispersionMetric.NAD
    assert DispersionMetric.from_any("NAD") == DispersionMetric.NAD
    assert DispersionMetric.from_any("amp_dispersion") == DispersionMetric.NAD
    assert DispersionMetric.from_any("nmad") == DispersionMetric.NMAD
    assert DispersionMetric.from_any("NMAD") == DispersionMetric.NMAD
    assert DispersionMetric.from_any(DispersionMetric.NAD) == DispersionMetric.NAD
    assert DispersionMetric.from_any(DispersionMetric.NMAD) == DispersionMetric.NMAD

    # Test invalid input
    with pytest.raises(ValueError, match="Unknown dispersion metric"):
        DispersionMetric.from_any("invalid")


def test_ps_block_nmad(slc_stack):
    """Test PS block calculation with NMAD metric."""
    # Run the PS selector on entire stack with NMAD
    amp_median, nmad_disp, ps_pixels = dolphin.ps.calc_ps_block(
        np.abs(slc_stack),
        dispersion_threshold=0.25,
        dispersion_metric=DispersionMetric.NMAD,
    )
    assert amp_median.shape == nmad_disp.shape == ps_pixels.shape
    assert amp_median.dtype == nmad_disp.dtype == np.float32
    assert ps_pixels.dtype == bool

    # For random data, NMAD might find some PS pixels (depending on the distribution)

    assert amp_median.min() > 0
    assert nmad_disp.min() >= 0

    # Check that sigma_psi attribute is attached for NMAD
    assert hasattr(nmad_disp, "sigma_psi")
    sigma_psi = nmad_disp.sigma_psi
    assert sigma_psi.shape == nmad_disp.shape
    assert np.all(sigma_psi >= 0)


def test_ps_block_nad_vs_nmad(slc_stack):
    """Compare NAD and NMAD results on the same data."""
    stack_mag = np.abs(slc_stack)

    # NAD calculation
    mean_nad, disp_nad, ps_nad = dolphin.ps.calc_ps_block(
        stack_mag,
        dispersion_threshold=0.25,
        dispersion_metric=DispersionMetric.NAD,
    )

    # NMAD calculation
    median_nmad, disp_nmad, ps_nmad = dolphin.ps.calc_ps_block(
        stack_mag,
        dispersion_threshold=0.25,
        dispersion_metric=DispersionMetric.NMAD,
    )

    # They should have same shape but different values
    assert mean_nad.shape == median_nmad.shape
    assert disp_nad.shape == disp_nmad.shape
    assert ps_nad.shape == ps_nmad.shape

    # For random data, NAD and NMAD should generally be different
    assert not np.array_equal(mean_nad, median_nmad)
    assert not np.array_equal(disp_nad, disp_nmad)

    # NMAD should not have sigma_psi attribute
    assert not hasattr(disp_nad, "sigma_psi")
    # NMAD should have sigma_psi attribute
    assert hasattr(disp_nmad, "sigma_psi")


def test_create_ps_nmad(tmp_path, vrt_stack):
    """Test create_ps function with NMAD metric."""
    ps_mask_file = tmp_path / "ps_pixels_nmad.tif"
    dispersion_file = tmp_path / "nmad_disp.tif"
    amp_mean_file = tmp_path / "amp_median.tif"

    dolphin.ps.create_ps(
        reader=vrt_stack,
        like_filename=vrt_stack.outfile,
        output_dispersion_file=dispersion_file,
        output_amp_mean_file=amp_mean_file,
        output_file=ps_mask_file,
        dispersion_metric=DispersionMetric.NMAD,
        dispersion_threshold=0.25,
    )

    assert io.get_raster_dtype(ps_mask_file) == np.uint8
    assert io.get_raster_dtype(amp_mean_file) == np.float32
    assert io.get_raster_dtype(dispersion_file) == np.float32


def test_create_ps_backwards_compatibility(tmp_path, vrt_stack):
    """Test that old parameter names still work with deprecation warnings."""
    ps_mask_file = tmp_path / "ps_pixels_compat.tif"
    amp_dispersion_file = tmp_path / "amp_disp_compat.tif"
    amp_mean_file = tmp_path / "amp_mean_compat.tif"

    # Test using old parameter names
    with pytest.warns(DeprecationWarning):
        dolphin.ps.create_ps(
            reader=vrt_stack,
            like_filename=vrt_stack.outfile,
            output_amp_dispersion_file=amp_dispersion_file,
            output_amp_mean_file=amp_mean_file,
            output_file=ps_mask_file,
            amp_dispersion_threshold=0.25,
        )

    assert io.get_raster_dtype(ps_mask_file) == np.uint8
    assert io.get_raster_dtype(amp_mean_file) == np.float32
    assert io.get_raster_dtype(amp_dispersion_file) == np.float32


def test_multilook_ps_files_backwards_compatibility(tmp_path, vrt_stack):
    """Test multilook_ps_files backwards compatibility."""
    ps_mask_file = tmp_path / "ps_pixels.tif"
    dispersion_file = tmp_path / "disp.tif"
    amp_mean_file = tmp_path / "amp_mean.tif"

    dolphin.ps.create_ps(
        reader=vrt_stack,
        like_filename=vrt_stack.outfile,
        output_dispersion_file=dispersion_file,
        output_amp_mean_file=amp_mean_file,
        output_file=ps_mask_file,
    )

    # Test with old parameter name
    with pytest.warns(DeprecationWarning):
        output_ps_file, output_disp_file = dolphin.ps.multilook_ps_files(
            strides={"x": 5, "y": 3},
            ps_mask_file=ps_mask_file,
            dispersion_file=None,
            amp_dispersion_file=dispersion_file,
        )

    assert io.get_raster_dtype(output_ps_file) == np.uint8
    assert io.get_raster_dtype(output_disp_file) == np.float32
