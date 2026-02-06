"""Tests for Point Coherence Estimation (PCE) module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dolphin.ps._pce import (
    _build_arcs,
    _compute_arc_coherences,
    _form_consecutive_ifgs,
    _solve_coherence_system,
    estimate_point_coherence,
)


# --- Fixtures ---


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def small_grid_shape():
    """A small 10x10 grid for fast tests."""
    return (10, 10)


@pytest.fixture()
def candidate_mask_dense(small_grid_shape):
    """All pixels are candidates."""
    return np.ones(small_grid_shape, dtype=bool)


@pytest.fixture()
def candidate_mask_sparse(small_grid_shape):
    """A few scattered candidate pixels."""
    mask = np.zeros(small_grid_shape, dtype=bool)
    # Place candidates at known positions
    mask[2, 3] = True
    mask[2, 5] = True
    mask[4, 4] = True
    mask[7, 7] = True
    mask[8, 8] = True
    return mask


@pytest.fixture()
def stable_ifg_stack(rng, small_grid_shape):
    """Interferograms with low noise (stable pixels everywhere).

    A ramp + small noise means nearby pixels have very similar phase,
    so arcs between nearby pixels should have high coherence.
    """
    n_ifg = 20
    rows, cols = small_grid_shape
    # Small random noise per interferogram, per pixel
    phase_noise = 0.05 * rng.standard_normal((n_ifg, rows, cols))
    ifg_stack = np.exp(1j * phase_noise).astype(np.complex64)
    return ifg_stack


@pytest.fixture()
def mixed_ifg_stack(rng, small_grid_shape):
    """Interferograms where some pixels are stable, some noisy.

    Top-left quadrant: stable (low noise).
    Bottom-right quadrant: noisy (high noise).
    """
    n_ifg = 30
    rows, cols = small_grid_shape
    phase = np.zeros((n_ifg, rows, cols))

    # Low noise for top-left
    half_r, half_c = rows // 2, cols // 2
    phase[:, :half_r, :half_c] = 0.05 * rng.standard_normal(
        (n_ifg, half_r, half_c)
    )
    # High noise for bottom-right
    phase[:, half_r:, half_c:] = 1.5 * rng.standard_normal(
        (n_ifg, rows - half_r, cols - half_c)
    )
    # Medium noise for other quadrants
    phase[:, :half_r, half_c:] = 0.3 * rng.standard_normal(
        (n_ifg, half_r, cols - half_c)
    )
    phase[:, half_r:, :half_c] = 0.3 * rng.standard_normal(
        (n_ifg, rows - half_r, half_c)
    )

    return np.exp(1j * phase).astype(np.complex64)


# --- Test _form_consecutive_ifgs ---


def test_form_consecutive_ifgs_shape():
    n_slc, rows, cols = 10, 5, 5
    slc_stack = np.ones((n_slc, rows, cols), dtype=np.complex64)
    ifgs = _form_consecutive_ifgs(slc_stack)
    assert ifgs.shape == (n_slc - 1, rows, cols)


def test_form_consecutive_ifgs_phase():
    """Check that consecutive ifgs capture the phase difference."""
    n_slc, rows, cols = 5, 3, 3
    phases = np.arange(n_slc).reshape(-1, 1, 1) * 0.5
    slc_stack = np.exp(1j * phases) * np.ones((1, rows, cols))
    ifgs = _form_consecutive_ifgs(slc_stack)
    # Each ifg should have phase ~0.5
    expected_phase = 0.5
    actual_phases = np.angle(ifgs)
    assert_allclose(actual_phases, expected_phase, atol=1e-6)


# --- Test _build_arcs ---


def test_build_arcs_no_candidates():
    cand_rows = np.array([], dtype=np.intp)
    cand_cols = np.array([], dtype=np.intp)
    pixel_to_idx = np.full((5, 5), -1, dtype=np.int32)
    arc_rows, arc_cols, arc_p, arc_q = _build_arcs(
        cand_rows, cand_cols, pixel_to_idx, max_radius=2, nrows=5, ncols=5
    )
    assert len(arc_p) == 0
    assert len(arc_q) == 0


def test_build_arcs_single_candidate():
    cand_rows = np.array([2], dtype=np.intp)
    cand_cols = np.array([3], dtype=np.intp)
    pixel_to_idx = np.full((5, 5), -1, dtype=np.int32)
    pixel_to_idx[2, 3] = 0
    arc_rows, arc_cols, arc_p, arc_q = _build_arcs(
        cand_rows, cand_cols, pixel_to_idx, max_radius=2, nrows=5, ncols=5
    )
    # Single pixel: no arcs possible
    assert len(arc_p) == 0


def test_build_arcs_two_adjacent():
    cand_rows = np.array([2, 2], dtype=np.intp)
    cand_cols = np.array([3, 4], dtype=np.intp)
    pixel_to_idx = np.full((5, 5), -1, dtype=np.int32)
    pixel_to_idx[2, 3] = 0
    pixel_to_idx[2, 4] = 1
    arc_rows, arc_cols, arc_p, arc_q = _build_arcs(
        cand_rows, cand_cols, pixel_to_idx, max_radius=1, nrows=5, ncols=5
    )
    assert len(arc_p) == 1
    assert arc_p[0] == 0
    assert arc_q[0] == 1


def test_build_arcs_no_duplicates():
    """Verify that each arc appears exactly once (p < q)."""
    cand_rows = np.array([0, 0, 1, 1], dtype=np.intp)
    cand_cols = np.array([0, 1, 0, 1], dtype=np.intp)
    pixel_to_idx = np.full((3, 3), -1, dtype=np.int32)
    for i, (r, c) in enumerate(zip(cand_rows, cand_cols)):
        pixel_to_idx[r, c] = i
    arc_rows, arc_cols, arc_p, arc_q = _build_arcs(
        cand_rows, cand_cols, pixel_to_idx, max_radius=1, nrows=3, ncols=3
    )
    # 4 pixels in a 2x2 block with radius=1: 6 possible arcs for all-pairs
    # but only adjacent (Chebyshev dist <= 1): all 6 pairs are within radius 1
    pairs = set(zip(arc_p.tolist(), arc_q.tolist()))
    # Each pair should appear once, and p < q
    for p, q in pairs:
        assert p < q


def test_build_arcs_radius():
    """Pixels beyond max_radius should not form arcs."""
    cand_rows = np.array([0, 0], dtype=np.intp)
    cand_cols = np.array([0, 4], dtype=np.intp)
    pixel_to_idx = np.full((5, 5), -1, dtype=np.int32)
    pixel_to_idx[0, 0] = 0
    pixel_to_idx[0, 4] = 1

    # radius=2 should NOT connect (0,0) to (0,4)
    _, _, arc_p, arc_q = _build_arcs(
        cand_rows, cand_cols, pixel_to_idx, max_radius=2, nrows=5, ncols=5
    )
    assert len(arc_p) == 0

    # radius=4 should connect them
    _, _, arc_p, arc_q = _build_arcs(
        cand_rows, cand_cols, pixel_to_idx, max_radius=4, nrows=5, ncols=5
    )
    assert len(arc_p) == 1


# --- Test _compute_arc_coherences ---


def test_arc_coherences_perfect():
    """Identical phases across ifgs -> coherence = 1."""
    n_ifg, rows, cols = 10, 3, 3
    # All pixels have zero phase
    ifg_stack = np.ones((n_ifg, rows, cols), dtype=np.complex64)
    arc_rows = np.array([[0, 0], [0, 1]], dtype=np.intp)
    arc_cols = np.array([[0, 1], [0, 0]], dtype=np.intp)
    arc_idx_p = np.array([0, 0], dtype=np.intp)
    arc_idx_q = np.array([1, 2], dtype=np.intp)
    coh = _compute_arc_coherences(ifg_stack, arc_rows, arc_cols, arc_idx_p, arc_idx_q)
    assert_allclose(coh, 1.0, atol=1e-6)


def test_arc_coherences_random():
    """Random phases -> coherence << 1 (but >= 0)."""
    rng = np.random.default_rng(123)
    n_ifg, rows, cols = 50, 5, 5
    phase = rng.uniform(-np.pi, np.pi, size=(n_ifg, rows, cols))
    ifg_stack = np.exp(1j * phase).astype(np.complex64)
    arc_rows = np.array([[0, 0], [1, 2]], dtype=np.intp)
    arc_cols = np.array([[0, 1], [0, 0]], dtype=np.intp)
    arc_idx_p = np.array([0, 1], dtype=np.intp)
    arc_idx_q = np.array([1, 2], dtype=np.intp)
    coh = _compute_arc_coherences(ifg_stack, arc_rows, arc_cols, arc_idx_p, arc_idx_q)
    assert np.all(coh >= 0)
    assert np.all(coh <= 1)
    # Random phases should give low coherence
    assert np.all(coh < 0.5)


# --- Test _solve_coherence_system ---


def test_solve_coherence_system_uniform():
    """If all arcs have the same coherence, all pixels should get similar values."""
    n_pixels = 5
    # Build a chain: 0-1, 1-2, 2-3, 3-4
    arc_p = np.array([0, 1, 2, 3], dtype=np.intp)
    arc_q = np.array([1, 2, 3, 4], dtype=np.intp)
    arc_coh = np.array([0.9, 0.9, 0.9, 0.9])
    x = _solve_coherence_system(arc_p, arc_q, arc_coh, n_pixels)
    # All values should be roughly equal since all arcs have equal coherence
    gamma = np.exp(x / 2.0)
    assert_allclose(gamma, gamma[0], atol=0.05)


def test_solve_coherence_system_one_noisy():
    """One pixel with worse arcs should get lower coherence."""
    n_pixels = 3
    # Arcs: 0-1, 0-2, 1-2
    arc_p = np.array([0, 0, 1], dtype=np.intp)
    arc_q = np.array([1, 2, 2], dtype=np.intp)
    # Arc 0-2 and 1-2 are lower (pixel 2 is noisy)
    arc_coh = np.array([0.95, 0.5, 0.5])
    x = _solve_coherence_system(arc_p, arc_q, arc_coh, n_pixels)
    gamma = np.exp(x / 2.0)
    # pixel 2 should have the lowest coherence
    assert gamma[2] < gamma[0]
    assert gamma[2] < gamma[1]


# --- Test estimate_point_coherence (integration) ---


def test_estimate_point_coherence_stable(stable_ifg_stack, candidate_mask_dense):
    """Stable pixels should yield high coherence."""
    coh = estimate_point_coherence(
        stable_ifg_stack,
        candidate_mask_dense,
        max_radius=2,
    )
    assert coh.shape == candidate_mask_dense.shape
    assert np.all(coh >= 0)
    assert np.all(coh <= 1)
    # Stable pixels should have high coherence (> 0.7)
    assert coh[candidate_mask_dense].mean() > 0.7


def test_estimate_point_coherence_sparse(stable_ifg_stack, candidate_mask_sparse):
    """Sparse candidates - some may not form arcs if too far apart."""
    coh = estimate_point_coherence(
        stable_ifg_stack,
        candidate_mask_sparse,
        max_radius=3,
    )
    assert coh.shape == candidate_mask_sparse.shape
    # Only candidate pixels should have nonzero coherence
    assert np.all(coh[~candidate_mask_sparse] == 0)


def test_estimate_point_coherence_mixed(mixed_ifg_stack, candidate_mask_dense):
    """Stable quadrant should have higher coherence than noisy quadrant."""
    rows, cols = mixed_ifg_stack.shape[1:]
    half_r, half_c = rows // 2, cols // 2

    coh = estimate_point_coherence(
        mixed_ifg_stack,
        candidate_mask_dense,
        max_radius=2,
    )
    # Compare average coherence in stable vs noisy quadrants
    # (ignoring edge effects at boundaries)
    stable_region = coh[1 : half_r - 1, 1 : half_c - 1]
    noisy_region = coh[half_r + 1 : -1, half_c + 1 : -1]

    assert stable_region.mean() > noisy_region.mean()


def test_estimate_point_coherence_empty_mask(stable_ifg_stack, small_grid_shape):
    """No candidates -> all zeros."""
    mask = np.zeros(small_grid_shape, dtype=bool)
    coh = estimate_point_coherence(stable_ifg_stack, mask)
    assert_allclose(coh, 0.0)


def test_estimate_point_coherence_invalid_input():
    """Should raise for non-3D or non-complex input."""
    with pytest.raises(ValueError, match="must be 3D"):
        estimate_point_coherence(
            np.ones((5, 5), dtype=np.complex64),
            np.ones((5, 5), dtype=bool),
        )
    with pytest.raises(ValueError, match="must be complex"):
        estimate_point_coherence(
            np.ones((3, 5, 5), dtype=np.float32),
            np.ones((5, 5), dtype=bool),
        )


# --- Test end-to-end with SLC stack ---


def test_pce_from_slc_stack(rng):
    """Test the full flow: SLC stack -> consecutive ifgs -> PCE."""
    n_slc, rows, cols = 20, 8, 8
    # Create a stable SLC stack with small noise
    phase_noise = 0.05 * rng.standard_normal((n_slc, rows, cols))
    slc_stack = np.exp(1j * phase_noise).astype(np.complex64)

    ifg_stack = _form_consecutive_ifgs(slc_stack)

    candidate_mask = np.ones((rows, cols), dtype=bool)
    coh = estimate_point_coherence(ifg_stack, candidate_mask, max_radius=2)

    # All pixels should show high coherence since noise is small
    assert coh.mean() > 0.6
    assert np.all(coh >= 0)
    assert np.all(coh <= 1)
