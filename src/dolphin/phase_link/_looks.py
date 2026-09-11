"""Effective number of independent looks in a multilook window.

The Cramér-Rao bound scales as ``1 / L`` with the number of *independent* looks.
Geocoded SLCs are oversampled and spectrally weighted, so adjacent pixels are
correlated and the pixel count of a window overstates the information in it.
For circular Gaussian speckle the correlation of intensity equals the squared
magnitude of the complex correlation, ``corr(I_a, I_b) = |r_ab|^2``, which is the
quantity that sets the effective look count of a linear estimator:

    1 / L_eff = sum_{a,b} w_a w_b |r_ab|^2

for normalized window weights ``w``. This module measures ``|r|^2`` by lag from
the intensity of the stack itself and evaluates the sum for a uniform window
under a separable (row times column) correlation model.

Texture that survives the local detrending (bright targets, field boundaries)
also correlates intensity, so the estimate is conservative: it errs toward fewer
looks and a larger reported uncertainty.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter

from dolphin._types import HalfWindow

__all__ = [
    "CrlbLooksMethod",
    "effective_looks_fraction",
    "estimate_effective_looks_fraction",
    "estimate_stack_effective_looks_fraction",
    "intensity_correlation",
]


class CrlbLooksMethod(str, Enum):
    """How to count the looks that scale the CRLB standard deviation."""

    EFFECTIVE = "effective"
    """Neighbor count times the effective-looks fraction measured from the data."""

    SHP_COUNT = "shp_count"
    """Neighbor count (or full window size), treating every pixel as independent."""

    SQRT_HALF_WINDOW = "sqrt_half_window"
    """Legacy: ``sqrt(half_window_y * half_window_x)`` looks for every pixel."""


def intensity_correlation(
    slc_stack: ArrayLike,
    max_lag_y: int,
    max_lag_x: int,
    max_dates: int = 5,
    trend_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the intensity autocorrelation by lag along rows and along columns.

    Each selected date's intensity is divided by a moving mean (the local
    backscatter trend) and the correlation of the resulting fluctuations is
    computed at lags ``1 .. max_lag`` along each axis. The median over dates is
    returned, with lag 0 set to 1 and values clipped to ``[0, 1]``.

    Parameters
    ----------
    slc_stack : ArrayLike
        Complex SLC stack, shape ``(n_slc, rows, cols)``. NaNs mark nodata.
    max_lag_y, max_lag_x : int
        Largest lag along rows and columns.
    max_dates : int
        Number of dates (evenly spaced through the stack) to average over.
    trend_size : tuple[int, int], optional
        Size of the moving-mean window used to remove the backscatter trend.
        Default is about twice the largest lag in each direction.

    Returns
    -------
    r2_y, r2_x : np.ndarray
        ``|r|^2`` at lags ``0 .. max_lag`` along rows and along columns.

    """
    stack = np.asarray(slc_stack)
    n_slc, rows, cols = stack.shape
    max_lag_y = int(min(max_lag_y, max(rows - 2, 0)))
    max_lag_x = int(min(max_lag_x, max(cols - 2, 0)))
    if trend_size is None:
        trend_size = (2 * max_lag_y + 1, 2 * max_lag_x + 1)
    trend_size = (max(3, min(trend_size[0], rows)), max(3, min(trend_size[1], cols)))

    idxs = np.unique(
        np.linspace(0, n_slc - 1, min(max_dates, n_slc)).round().astype(int)
    )
    all_y, all_x = [], []
    for i in idxs:
        intensity = np.abs(stack[i]) ** 2
        valid = np.isfinite(intensity) & (intensity > 0)
        if valid.sum() < 4 * (trend_size[0] * trend_size[1]):
            continue
        filled = np.where(valid, intensity, intensity[valid].mean())
        trend = uniform_filter(filled, size=trend_size, mode="reflect")
        fluct = filled / np.maximum(trend, 1e-30) - 1.0
        fluct[~valid] = 0.0
        all_y.append(_lag_correlations(fluct, max_lag_y, axis=0))
        all_x.append(_lag_correlations(fluct, max_lag_x, axis=1))

    if not all_y:
        return np.r_[1.0, np.zeros(max_lag_y)], np.r_[1.0, np.zeros(max_lag_x)]
    r2_y = np.clip(np.median(all_y, axis=0), 0.0, 1.0)
    r2_x = np.clip(np.median(all_x, axis=0), 0.0, 1.0)
    r2_y[0] = r2_x[0] = 1.0
    return r2_y, r2_x


def _lag_correlations(fluct: np.ndarray, max_lag: int, axis: int) -> np.ndarray:
    """Correlate `fluct` with itself shifted by ``1..max_lag`` along `axis`."""
    out = np.ones(max_lag + 1)
    f = np.moveaxis(fluct, axis, 0)
    for k in range(1, max_lag + 1):
        a = f[k:] - f[k:].mean()
        b = f[:-k] - f[:-k].mean()
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
        out[k] = np.sum(a * b) / denom if denom > 0 else 0.0
    return out


def effective_looks_fraction(
    r2_y: ArrayLike, r2_x: ArrayLike, window_y: int, window_x: int
) -> float:
    """Ratio ``L_eff / L_nominal`` for a uniform window under separable correlation.

    Parameters
    ----------
    r2_y, r2_x : ArrayLike
        ``|r|^2`` by lag along rows and columns, starting with lag 0.
        Lags beyond the provided values are taken as uncorrelated.
    window_y, window_x : int
        Full window size in rows and columns.

    Returns
    -------
    float
        Fraction in ``(0, 1]``. Equals 1 for uncorrelated pixels.

    """

    def one_dim(r2: ArrayLike, n: int) -> float:
        r2 = np.asarray(r2, dtype=float)
        lags = np.arange(1, n)
        vals = np.zeros(n - 1)
        m = min(len(r2) - 1, n - 1)
        vals[:m] = np.clip(r2[1 : m + 1], 0.0, 1.0)
        # 1 / L_eff for a uniform 1-D window of length n
        return (n + 2.0 * np.sum((n - lags) * vals)) / n**2

    inv_l_eff = one_dim(r2_y, int(window_y)) * one_dim(r2_x, int(window_x))
    return float(min(1.0, 1.0 / (inv_l_eff * window_y * window_x)))


def estimate_effective_looks_fraction(
    slc_stack: ArrayLike, half_window: HalfWindow, max_dates: int = 5
) -> float:
    """Measure ``L_eff / L_nominal`` for the phase-linking window from the stack itself.

    Parameters
    ----------
    slc_stack : ArrayLike
        Complex SLC stack, shape ``(n_slc, rows, cols)``.
    half_window : HalfWindow
        Half window of the phase-linking estimator; the full window is
        ``2 * half + 1`` in each direction.
    max_dates : int
        Number of dates to use for the intensity correlation.

    Returns
    -------
    float
        Fraction in ``(0, 1]`` to multiply the pixel (or SHP) count by.

    """
    window_y, window_x = 2 * half_window[0] + 1, 2 * half_window[1] + 1
    r2_y, r2_x = intensity_correlation(
        slc_stack, max_lag_y=window_y - 1, max_lag_x=window_x - 1, max_dates=max_dates
    )
    return effective_looks_fraction(r2_y, r2_x, window_y, window_x)


def estimate_stack_effective_looks_fraction(
    reader,
    half_window: HalfWindow,
    block_shape: tuple[int, int] = (512, 512),
    max_blocks: int = 16,
    max_dates: int = 3,
    percentile: float = 50.0,
) -> float:
    """Estimate one ``L_eff / L_nominal`` for a whole stack from sampled blocks.

    The correlation between neighboring pixels comes from the product's
    oversampling and spectral weighting, which are the same everywhere in a
    frame, so the fraction should be a single number per stack. Estimating it
    block by block instead lets scene texture modulate it and prints the block
    grid into the CRLB rasters. Texture larger than a pixel (fields, roads) adds
    intensity correlation and lowers a block's estimate; isolated bright targets
    dilute it and raise the estimate. The median over blocks is robust to both.

    Parameters
    ----------
    reader : array-like
        Anything indexable as ``reader[date, rows, cols]`` with a ``shape`` of
        ``(n_slc, rows, cols)``, such as a `VRTStack` or a NumPy array.
    half_window : HalfWindow
        Half window of the phase-linking estimator.
    block_shape : tuple[int, int]
        Size of the blocks to sample.
    max_blocks : int
        Number of blocks, spread over the raster, to sample.
    max_dates : int
        Number of dates, spread over the stack, to read per block.
    percentile : float
        Percentile of the per-block fractions to return. Default 50 (median).

    Returns
    -------
    float
        Fraction in ``(0, 1]``; 1.0 if no block had enough valid data.

    """
    n_slc, rows, cols = reader.shape
    br, bc = min(block_shape[0], rows), min(block_shape[1], cols)
    n_side = max(1, int(np.ceil(np.sqrt(max_blocks))))
    row_starts = np.unique(np.linspace(0, rows - br, n_side).round().astype(int))
    col_starts = np.unique(np.linspace(0, cols - bc, n_side).round().astype(int))
    date_idxs = np.unique(
        np.linspace(0, n_slc - 1, min(max_dates, n_slc)).round().astype(int)
    )
    fractions = []
    for r0 in row_starts:
        for c0 in col_starts:
            block = np.stack(
                [
                    np.asarray(reader[int(i), r0 : r0 + br, c0 : c0 + bc])
                    for i in date_idxs
                ]
            )
            valid = np.isfinite(block).all(axis=0) & (np.abs(block) > 0).any(axis=0)
            if valid.mean() < 0.5:
                continue
            fractions.append(
                estimate_effective_looks_fraction(
                    block, half_window, max_dates=max_dates
                )
            )
    if not fractions:
        return 1.0
    return float(np.percentile(fractions, percentile))
