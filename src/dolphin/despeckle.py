"""Module for despeckling a stack of SLC amplitudes."""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, vmap
from jax.typing import ArrayLike

from dolphin import shp
from dolphin._types import HalfWindow, Strides
from dolphin.utils import compute_out_shape

DEFAULT_STRIDES = Strides(1, 1)

__all__ = ["despeckle"]


def despeckle(
    amp_stack: ArrayLike,
    half_window: HalfWindow,
    strides: Strides = DEFAULT_STRIDES,
    shp_alpha: float = 0.01,
) -> Array:
    """Estimate the linked phase at all pixels of `slc_stack`.

    Parameters
    ----------
    amp_stack : ArrayLike
        The SLC amplitude stack, with shape (n_slc, n_rows, n_cols).
    half_window : tuple[int, int]
        A (named) tuple of (y, x) sizes for the half window.
        The full window size is 2 * half_window + 1 for x, y.
    strides : tuple[int, int], optional
        The (y, x) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    shp_alpha : float, optional
        The significance level for the SHP test, by default 0.01.


    Returns
    -------
    despeckled_stack : np.ndarray
        The despeckled amplitude stack, with shape (n_slc, n_rows, n_cols).
    shp_counts : np.ndarray
        The number of SHPs at each pixel, with shape (n_rows, n_cols).

    """
    yhalf, xhalf = half_window

    amp_mean = np.mean(amp_stack, axis=0)
    amp_variance = np.var(amp_stack, axis=0)
    # Compute the neighbor_arrays for this block
    neighbor_arrays = shp.estimate_neighbors(
        halfwin_rowcol=(yhalf, xhalf),
        alpha=shp_alpha,
        strides=strides,
        mean=amp_mean,
        var=amp_variance,
        nslc=amp_stack.shape[0],
        method=shp.ShpMethod.GLRT,
    )
    despeckled_stack = compute_block(amp_stack, half_window, strides, neighbor_arrays)
    shp_counts = np.sum(neighbor_arrays, axis=(2, 3))
    return despeckled_stack, shp_counts


@partial(jit, static_argnames=["half_window", "strides"])
def compute_block(
    amp_stack: ArrayLike,
    half_window: HalfWindow,
    strides: Strides = DEFAULT_STRIDES,
    neighbor_arrays: Optional[np.ndarray] = None,
) -> Array:
    """Estimate the linked phase at all pixels of `amp_stack`."""
    nslc, rows, cols = amp_stack.shape

    row_strides = strides.y
    col_strides = strides.x
    half_row = half_window.y
    half_col = half_window.x

    out_rows, out_cols = compute_out_shape((rows, cols), strides)

    in_r_start = row_strides // 2
    in_c_start = col_strides // 2

    if neighbor_arrays is None:
        neighbor_arrays = jnp.ones(
            (out_rows, out_cols, 2 * half_window[0] + 1, 2 * half_window[1] + 1),
            dtype=bool,
        )

    def _process_row_col(out_r, out_c):
        """Get slices for, and process, one pixel's window."""
        in_r = in_r_start + out_r * row_strides
        in_c = in_c_start + out_c * col_strides
        # Get a 3D slice, size (row_window, col_window, nslc)
        slc_window = _get_stack_window(amp_stack, in_r, in_c, half_row, half_col)
        # Reshape to be (num_samples, nslc)
        slc_samples = slc_window.reshape(nslc, -1)
        cur_neighbors = neighbor_arrays[out_r, out_c, :, :]
        neighbor_mask = cur_neighbors.ravel()

        return average_single(slc_samples, neighbor_mask=neighbor_mask)

    # Now make a 2D grid of indices to access all output pixels
    out_r_indices, out_c_indices = jnp.meshgrid(
        jnp.arange(out_rows), jnp.arange(out_cols), indexing="ij"
    )

    # Create the vectorized function in 2d
    _process_2d = vmap(_process_row_col, out_axes=1)
    # Then in 3d
    _process_3d = vmap(_process_2d, out_axes=1)
    return _process_3d(out_r_indices, out_c_indices)


# Note: this is the transposed version from the one in phase_link/covariance.py,
# since it's not as speed critical/simpler to compute.
# Perhaps we should refactor the two to share a common function?
def _get_stack_window(
    stack: ArrayLike, r: int, c: int, half_row: int, half_col: int
) -> Array:
    """Dynamically slice the stack at (r, c) with size (2*half_row+1, 2*half_col+1).

    Expected shape of `stack` is (nslc, rows, cols).
    """
    # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
    # Note: out of bounds indexing for JAX clamps to the nearest value
    # This is fine as we want to trim the borders anyway, so we don't need the
    # extra checks in utils._get_slices

    # Center the slice on (r, c), so we need the starts to move up/left
    # The upper bound on `clamp` isn't used meaningless here
    r0 = lax.clamp(0, r - half_row, r)
    c0 = lax.clamp(0, c - half_col, c)
    start_indices = (0, r0, c0)

    # Note: we can't clamp the size using a max size,
    # TypeError: Shapes must be 1D sequences of concrete values of integer type,
    # got  Traced<ShapedArray(int32[],  ...
    dsize = stack.shape[0]
    rsize = 2 * half_row + 1
    csize = 2 * half_col + 1
    slice_sizes = (dsize, rsize, csize)
    return lax.dynamic_slice(stack, start_indices, slice_sizes)


@jit
def average_single(
    slc_samples: ArrayLike, neighbor_mask: Optional[ArrayLike] = None
) -> Array:
    """Given (n_samps, nslc) SLC samples, get the average for neighboring pixels."""
    nslc, nsamps = slc_samples.shape

    if neighbor_mask is None:
        neighbor_mask = jnp.ones(nsamps, dtype=jnp.bool_)
    valid_samples_mask = ~jnp.isnan(slc_samples)
    combined_mask = valid_samples_mask & neighbor_mask[None, :]

    # Mask the slc samples
    # note that it's not possible to change the size based on the mask
    # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#dynamic-shapes
    masked_slc = jnp.where(combined_mask, slc_samples, 0)
    # Average the selected pixels:
    return jnp.mean(masked_slc, axis=1)
