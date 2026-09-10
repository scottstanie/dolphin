import numpy as np

from dolphin._types import HalfWindow
from dolphin.phase_link._looks import (
    effective_looks_fraction,
    estimate_effective_looks_fraction,
    intensity_correlation,
)


def _speckle(rng, n, rows, cols):
    z = rng.normal(size=(n, rows, cols)) + 1j * rng.normal(size=(n, rows, cols))
    return z / np.sqrt(2)


def test_fraction_is_one_for_uncorrelated_pixels():
    r2 = np.array([1.0, 0.0, 0.0])
    assert effective_looks_fraction(r2, r2, 5, 7) == 1.0


def test_fraction_matches_brute_force_double_sum():
    # |r|^2 by lag along each axis, separable model
    r2_y = 0.6 ** np.arange(9)
    r2_x = 0.4 ** np.arange(5)
    ny, nx = 9, 5
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    iy, ix = iy.ravel(), ix.ravel()
    lag_y = np.abs(iy[:, None] - iy[None, :])
    lag_x = np.abs(ix[:, None] - ix[None, :])
    r2 = r2_y[lag_y] * r2_x[lag_x]
    inv_l_eff = r2.sum() / (ny * nx) ** 2
    expected = 1.0 / inv_l_eff / (ny * nx)
    assert np.isclose(effective_looks_fraction(r2_y, r2_x, ny, nx), expected)


def test_white_speckle_gives_fraction_near_one():
    rng = np.random.default_rng(0)
    z = _speckle(rng, 4, 400, 400)
    frac = estimate_effective_looks_fraction(z, HalfWindow(y=5, x=3))
    assert 0.9 < frac <= 1.0


def test_smoothed_speckle_is_detected():
    rng = np.random.default_rng(1)
    z = _speckle(rng, 4, 400, 400)
    # Two-tap running mean along columns: complex |r| = 0.5 at lag 1, so |r|^2 = 0.25
    zs = (z[:, :, 1:] + z[:, :, :-1]) / np.sqrt(2)
    r2_y, r2_x = intensity_correlation(zs, max_lag_y=3, max_lag_x=3)
    assert abs(r2_x[1] - 0.25) < 0.05
    assert r2_x[2] < 0.05
    assert r2_y[1] < 0.05
    frac = estimate_effective_looks_fraction(zs, HalfWindow(y=2, x=2))
    expected = effective_looks_fraction([1.0, 0, 0, 0, 0], [1.0, 0.25, 0, 0, 0], 5, 5)
    assert abs(frac - expected) < 0.05
    assert frac < 0.9


def test_nodata_and_small_blocks_are_safe():
    rng = np.random.default_rng(2)
    z = _speckle(rng, 2, 60, 60)
    z[:, :10, :] = np.nan
    frac = estimate_effective_looks_fraction(z, HalfWindow(y=2, x=2))
    assert 0.0 < frac <= 1.0
    # Too few valid pixels: falls back to the uncorrelated answer
    tiny = _speckle(rng, 2, 8, 8)
    assert estimate_effective_looks_fraction(tiny, HalfWindow(y=3, x=3)) == 1.0
