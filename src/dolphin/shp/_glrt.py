from __future__ import annotations

from functools import partial
from math import log

import jax.numpy as jnp
import numba
import numpy as np
from jax import Array, jit, lax, vmap
from numpy.typing import ArrayLike

from dolphin._types import HalfWindow, Strides
from dolphin.utils import compute_out_shape

from ._common import _make_loop_function, _read_cutoff_csv

NO_STRIDES = Strides(1, 1)


@numba.njit(nogil=True)
def _compute_glrt_test_stat(scale_1, scale_2):
    """Compute the GLRT test statistic."""
    scale_pooled = (scale_1 + scale_2) / 2
    return 2 * log(scale_pooled) - log(scale_1) - log(scale_2)


_loop_over_pixels = _make_loop_function(_compute_glrt_test_stat)


def estimate_neighbors(
    mean: ArrayLike,
    var: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    nslc: int,
    strides: Strides = NO_STRIDES,
    alpha: float = 0.05,
    prune_disconnected: bool = False,
):
    """Estimate the number of neighbors based on the GLRT.

    Based on the method described in [@Parizzi2011AdaptiveInSARStack].
    Assumes Rayleigh distributed amplitudes ([@Siddiqui1962ProblemsConnectedRayleigh])

    Parameters
    ----------
    mean : ArrayLike, 2D
        Mean amplitude of each pixel.
    var: ArrayLike, 2D
        Variance of each pixel's amplitude.
    halfwin_rowcol : tuple[int, int]
        Half the size of the block in (row, col) dimensions
    nslc : int
        Number of images in the stack used to compute `mean` and `var`.
        Used to compute the degrees of freedom for the t- and F-tests to
        determine the critical values.
    strides: dict, optional
        The (x, y) strides (in pixels) to use for the sliding window.
        By default {"x": 1, "y": 1}
    alpha : float, default=0.05
        Significance level at which to reject the null hypothesis.
        Rejecting means declaring a neighbor is not a SHP.
    prune_disconnected : bool, default=False
        If True, keeps only SHPs that are 8-connected to the current pixel.
        Otherwise, any pixel within the window may be considered an SHP, even
        if it is not directly connected.


    Notes
    -----
    When `strides` is not (1, 1), the output first two dimensions
    are smaller than `mean` and `var` by a factor of `strides`. This
    will match the downstream shape of the strided phase linking results.

    Returns
    -------
    is_shp : np.ndarray, 4D
        Boolean array marking which neighbors are SHPs for each pixel in the block.
        Shape is (out_rows, out_cols, window_rows, window_cols), where
            `out_rows` and `out_cols` are computed by
            `[dolphin.io.compute_out_shape][]`
            `window_rows = 2 * halfwin_rowcol[0] + 1`
            `window_cols = 2 * halfwin_rowcol[1] + 1`
    """
    half_row, half_col = halfwin_rowcol
    rows, cols = mean.shape

    threshold = get_cutoff(alpha=alpha, N=nslc)

    strides_rowcol = strides
    out_rows, out_cols = compute_out_shape((rows, cols), Strides(*strides_rowcol))
    is_shp = np.zeros(
        (out_rows, out_cols, 2 * half_row + 1, 2 * half_col + 1), dtype=np.bool_
    )
    return _loop_over_pixels(
        mean,
        var,
        halfwin_rowcol,
        strides_rowcol,
        threshold,
        prune_disconnected,
        is_shp,
    )


compute_out_shape_jax = jit(compute_out_shape, static_argnames=["shape", "strides"])


@partial(jit, static_argnames=["half_window", "strides", "nslc", "alpha"])
def estimate_neighbors_jax(
    mean: ArrayLike,
    var: ArrayLike,
    half_window: HalfWindow,
    nslc: int,
    strides: Strides = NO_STRIDES,
    alpha: float = 0.05,
):
    """Estimate the number of neighbors based on the GLRT."""
    # Convert mean/var to the Rayleigh scale parameter
    rows, cols = mean.shape
    row_strides, col_strides = strides
    half_row, half_col = half_window

    in_r_start = row_strides // 2
    in_c_start = col_strides // 2
    out_rows, out_cols = compute_out_shape((rows, cols), strides)

    scale_squared = (var + mean**2) / 2
    threshold = get_cutoff_jax(alpha=alpha, N=nslc)

    def _get_window(arr, r: int, c: int, half_row: int, half_col: int) -> Array:
        r0 = r - half_row
        c0 = c - half_col
        start_indices = (r0, c0)

        rsize = 2 * half_row + 1
        csize = 2 * half_col + 1
        slice_sizes = (rsize, csize)

        return lax.dynamic_slice(arr, start_indices, slice_sizes)

    def _process_row_col(out_r, out_c):
        in_r = in_r_start + out_r * row_strides
        in_c = in_c_start + out_c * col_strides

        scale_1 = scale_squared[in_r, in_c]  # One pixel
        # and one window for scale 2, will broadcast
        scale_2 = _get_window(scale_squared, in_r, in_c, half_row, half_col)

        # Compute the GLRT test statistic.
        scale_pooled = (scale_1 + scale_2) / 2
        test_stat = 2 * jnp.log(scale_pooled) - jnp.log(scale_1) - jnp.log(scale_2)

        return threshold > test_stat

    # Now make a 2D grid of indices to access all output pixels
    out_r_indices, out_c_indices = jnp.meshgrid(
        jnp.arange(out_rows), jnp.arange(out_cols), indexing="ij"
    )

    # Create the vectorized function in 2d
    _process_2d = vmap(_process_row_col)
    # Then in 3d
    _process_3d = vmap(_process_2d)
    return _process_3d(out_r_indices, out_c_indices)


def get_cutoff(alpha: float, N: int) -> float:
    r"""Compute the upper cutoff for the GLRT test statistic.

    Statistic is

    \[
    2\log(\sigma_{pooled}) - \log(\sigma_{p}) -\log(\sigma_{q})
    \]

    Parameters
    ----------
    alpha: float
        Significance level (0 < alpha < 1).
    N: int
        Number of samples.

    Returns
    -------
    float
        Cutoff value for the GLRT test statistic.
    """
    n_alpha_to_cutoff = _read_cutoff_csv("glrt")
    try:
        return n_alpha_to_cutoff[(N, alpha)]
    except KeyError as e:
        msg = f"Not implemented for {N = }, {alpha = }"
        raise NotImplementedError(msg) from e


get_cutoff_jax = jit(get_cutoff, static_argnames=["alpha", "N"])
