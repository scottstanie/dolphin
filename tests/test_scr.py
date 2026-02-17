"""Tests for the Signal-to-Clutter Ratio (SCR) PS selection method."""

import jax.numpy as jnp
import numpy as np
import pytest

import dolphin.ps
from dolphin import io
from dolphin.ps._scr import (
    _compute_phase_residues,
    _estimate_scr,
    _phase_pdf_gaussian,
    _phase_pdf_single_look,
)


@pytest.fixture(scope="module")
def deterministic_slc_stack():
    """Create an SLC stack with a mix of PS-like and distributed scatterers.

    PS pixels have a strong deterministic signal (high SCR),
    while distributed scatterers have random phase (low SCR).
    """
    np.random.seed(42)
    n_slc, nrow, ncol = 20, 10, 10
    sigma = 0.5

    # Random clutter (distributed scatterers)
    noise = sigma * (
        np.random.randn(n_slc, nrow, ncol) + 1j * np.random.randn(n_slc, nrow, ncol)
    )

    # Add a strong deterministic signal to some pixels (PS candidates)
    signal = np.zeros((n_slc, nrow, ncol), dtype=np.complex64)
    # Make the first row strong PS (high amplitude, stable phase)
    signal[:, 0, :] = 5.0 + 0j

    slc = (signal + noise).astype(np.complex64)
    return slc


class TestPhaseResidues:
    def test_shape(self, slc_stack):
        slc_jax = jnp.asarray(slc_stack, dtype=jnp.complex64)
        n_slc = slc_stack.shape[0]
        ref_idx = n_slc // 2
        residues = _compute_phase_residues(slc_jax, ref_idx, window_size=5)
        assert residues.shape == (n_slc - 1, *slc_stack.shape[1:])

    def test_dtype(self, slc_stack):
        slc_jax = jnp.asarray(slc_stack, dtype=jnp.complex64)
        ref_idx = slc_stack.shape[0] // 2
        residues = _compute_phase_residues(slc_jax, ref_idx, window_size=5)
        assert residues.dtype == jnp.float32

    def test_range(self, slc_stack):
        """Phase residues should be in [-pi, pi]."""
        slc_jax = jnp.asarray(slc_stack, dtype=jnp.complex64)
        ref_idx = slc_stack.shape[0] // 2
        residues = _compute_phase_residues(slc_jax, ref_idx, window_size=5)
        assert np.all(np.asarray(residues) >= -np.pi)
        assert np.all(np.asarray(residues) <= np.pi)

    def test_ps_pixels_have_small_residues(self, deterministic_slc_stack):
        """PS pixels (with strong deterministic signal) should have small residues."""
        slc_jax = jnp.asarray(deterministic_slc_stack, dtype=jnp.complex64)
        ref_idx = deterministic_slc_stack.shape[0] // 2
        residues = np.asarray(_compute_phase_residues(slc_jax, ref_idx, window_size=5))
        # First row is PS (strong signal)
        ps_residues = np.abs(residues[:, 0, :])
        # Other rows are distributed
        ds_residues = np.abs(residues[:, 5:, :])

        # PS residues should be smaller on average
        assert np.mean(ps_residues) < np.mean(ds_residues)


class TestPhasePDFs:
    def test_gaussian_pdf_integrates_to_one(self):
        """PDF should integrate to approximately 1 over [-pi, pi]."""
        phi = jnp.linspace(-jnp.pi, jnp.pi, 1000)
        for gamma in [0.1, 1.0, 5.0, 10.0]:
            pdf = np.asarray(_phase_pdf_gaussian(gamma, phi))
            integral = np.trapezoid(pdf, np.asarray(phi))
            np.testing.assert_allclose(integral, 1.0, atol=0.02)

    def test_gaussian_pdf_positive(self):
        phi = jnp.linspace(-jnp.pi, jnp.pi, 100)
        for gamma in [0.0, 0.5, 2.0, 10.0]:
            pdf = np.asarray(_phase_pdf_gaussian(gamma, phi))
            assert np.all(pdf >= 0)

    def test_single_look_pdf_integrates_to_one(self):
        """Single-look PDF should integrate to approximately 1 over [-pi, pi]."""
        theta = jnp.linspace(-jnp.pi, jnp.pi, 1000)
        for gamma in [0.1, 1.0, 5.0, 10.0]:
            pdf = np.asarray(_phase_pdf_single_look(gamma, theta))
            integral = np.trapezoid(pdf, np.asarray(theta))
            np.testing.assert_allclose(integral, 1.0, atol=0.02)

    def test_single_look_pdf_positive(self):
        theta = jnp.linspace(-jnp.pi, jnp.pi, 100)
        for gamma in [0.0, 0.5, 2.0, 10.0]:
            pdf = np.asarray(_phase_pdf_single_look(gamma, theta))
            assert np.all(pdf >= 0)

    def test_high_scr_peaks_at_zero(self):
        """For high SCR, the single-look PDF should peak near zero phase."""
        theta = jnp.linspace(-jnp.pi, jnp.pi, 1000)
        pdf = np.asarray(_phase_pdf_single_look(20.0, theta))
        peak_idx = np.argmax(pdf)
        assert abs(float(theta[peak_idx])) < 0.1


class TestSCREstimation:
    def test_basic_shape(self, slc_stack):
        scr = dolphin.ps.calc_scr_block(slc_stack)
        assert scr.shape == slc_stack.shape[1:]
        assert scr.dtype == np.float32

    def test_ps_has_higher_scr(self, deterministic_slc_stack):
        """PS pixels should have higher SCR than distributed scatterers."""
        scr = dolphin.ps.calc_scr_block(deterministic_slc_stack, window_size=5)
        # First row is PS
        ps_scr = scr[0, :]
        # Middle rows are distributed
        ds_scr = scr[5:, :]
        assert np.mean(ps_scr) > np.mean(ds_scr)

    def test_gaussian_model(self, slc_stack):
        scr = dolphin.ps.calc_scr_block(slc_stack, model="gaussian")
        assert scr.shape == slc_stack.shape[1:]
        assert scr.dtype == np.float32

    def test_coherence_model(self, slc_stack):
        scr = dolphin.ps.calc_scr_block(slc_stack, model="coherence")
        assert scr.shape == slc_stack.shape[1:]
        assert scr.dtype == np.float32
        assert np.all(scr >= 0)

    def test_coherence_ps_vs_ds(self, deterministic_slc_stack):
        """Coherence model should also distinguish PS from DS."""
        scr = dolphin.ps.calc_scr_block(
            deterministic_slc_stack, model="coherence", window_size=5
        )
        assert np.mean(scr[0, :]) > np.mean(scr[5:, :])

    def test_requires_3d(self):
        with pytest.raises(AssertionError):
            dolphin.ps.calc_scr_block(np.ones((5, 5)))

    def test_requires_at_least_2_slcs(self):
        with pytest.raises(AssertionError):
            dolphin.ps.calc_scr_block(np.ones((1, 5, 5)))

    def test_non_negative(self, slc_stack):
        scr = dolphin.ps.calc_scr_block(slc_stack)
        assert np.all(scr >= 0)

    def test_reference_idx(self, slc_stack):
        """Passing different reference_idx should work without error."""
        scr0 = dolphin.ps.calc_scr_block(slc_stack, reference_idx=0)
        scr_mid = dolphin.ps.calc_scr_block(slc_stack)  # default N//2
        assert scr0.shape == scr_mid.shape


class TestMLEEstimation:
    def test_shape_3d(self):
        """Test with 3D phase residues (n_ifg, rows, cols)."""
        np.random.seed(0)
        phi = jnp.array(np.random.uniform(-np.pi, np.pi, size=(10, 4, 5)))
        scr = np.asarray(_estimate_scr(phi, model="constant"))
        assert scr.shape == (4, 5)

    def test_concentrated_phases_give_high_scr(self):
        """Phase residues near zero should yield high SCR."""
        np.random.seed(0)
        # Very small phase residues -> high SCR (3D: n_ifg, rows, cols)
        phi_ps = jnp.array(np.random.normal(0, 0.1, size=(20, 1, 5)))
        scr_ps = np.asarray(_estimate_scr(phi_ps, model="constant"))

        # Uniform phase residues -> low SCR
        phi_ds = jnp.array(np.random.uniform(-np.pi, np.pi, size=(20, 1, 5)))
        scr_ds = np.asarray(_estimate_scr(phi_ds, model="constant"))

        assert np.mean(scr_ps) > np.mean(scr_ds)


@pytest.fixture()
def vrt_stack(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    return io.VRTStack(slc_file_list, outfile=vrt_file)


def test_create_scr(tmp_path, vrt_stack):
    scr_file = tmp_path / "scr.tif"
    dolphin.ps.create_scr(
        reader=vrt_stack,
        output_file=scr_file,
        like_filename=vrt_stack.outfile,
    )
    assert scr_file.exists()
    assert io.get_raster_dtype(scr_file) == np.float32

    scr_data = io.load_gdal(scr_file)
    assert scr_data.shape == (vrt_stack.shape[1], vrt_stack.shape[2])
    assert np.all(scr_data >= 0)
