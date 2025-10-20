"""Module for estimating covariance matrices for stacks or single pixels.

Contains for CPU and GPU versions (which will not be available if no GPU).
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, vmap
from jax.typing import ArrayLike

from dolphin._types import HalfWindow, Strides
from dolphin.utils import compute_out_shape

DEFAULT_STRIDES = Strides(1, 1)

__all__ = ["coh_mat_single", "estimate_stack_covariance"]


@partial(jit, static_argnames=["half_window", "strides"])
def estimate_stack_covariance(
    slc_stack: ArrayLike,
    half_window: HalfWindow,
    strides: Strides = DEFAULT_STRIDES,
    neighbor_arrays: Optional[np.ndarray] = None,
) -> Array:
    """Estimate the linked phase at all pixels of `slc_stack`.

    Parameters
    ----------
    slc_stack : ArrayLike
        The SLC stack, with shape (n_slc, n_rows, n_cols).
    half_window : tuple[int, int]
        A (named) tuple of (y, x) sizes for the half window.
        The full window size is 2 * half_window + 1 for x, y.
    strides : tuple[int, int], optional
        The (y, x) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    neighbor_arrays : np.ndarray, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.

    Returns
    -------
    C_arrays : np.ndarray
        The covariance matrix at each pixel, with shape
        (n_rows, n_cols, n_slc, n_slc).

    Raises
    ------
    ValueError
        If `slc_stack` is not complex data.

    """
    if not np.iscomplexobj(slc_stack):
        msg = "The SLC stack must be complex."
        raise ValueError(msg)
    if neighbor_arrays is None:
        rows, cols = slc_stack.shape[1:]
        full_window = (2 * half_window.y + 1, 2 * half_window.x + 1)
        neighbor_arrays = jnp.ones((rows, cols, *full_window), dtype=bool)

    nslc, rows, cols = slc_stack.shape

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
        slc_window = _get_stack_window(slc_stack, in_r, in_c, half_row, half_col)
        # Reshape to be (nslc, num_samples)
        slc_samples = slc_window.reshape(nslc, -1)
        cur_neighbors = neighbor_arrays[out_r, out_c, :, :]
        neighbor_mask = cur_neighbors.ravel()

        return coh_mat_single(slc_samples, neighbor_mask=neighbor_mask)

    # Now make a 2D grid of indices to access all output pixels
    out_r_indices, out_c_indices = jnp.meshgrid(
        jnp.arange(out_rows), jnp.arange(out_cols), indexing="ij"
    )

    # Create the vectorized function in 2d
    _process_2d = vmap(_process_row_col)
    # Then in 3d
    _process_3d = vmap(_process_2d)
    return _process_3d(out_r_indices, out_c_indices)


@jit
def coh_mat_single(
    slc_samples: ArrayLike, neighbor_mask: Optional[ArrayLike] = None
) -> Array:
    """Given (n_slc, n_samps) SLC samples, get the (nslc, nslc) coherence matrix.

    Note this requires `slc_samples` to be transposed from `coh_mat_single`.
    """
    _nslc, nsamps = slc_samples.shape

    if neighbor_mask is None:
        neighbor_mask = jnp.ones(nsamps, dtype=jnp.bool_)
    valid_samples_mask = ~jnp.isnan(slc_samples)
    combined_mask = valid_samples_mask & neighbor_mask[None, :]

    # Mask the slc samples
    # note that it's not possible to change the size based on the mask
    # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#dynamic-shapes
    masked_slc = jnp.where(combined_mask, slc_samples, 0)

    # Compute cross-correlation
    numer = jnp.dot(masked_slc, jnp.conj(masked_slc.T))

    # Compute amplitudes so we normalize the covariance to a coherence matrix
    # a1 is shape (nslc,)
    amp_vec = jnp.sum(jnp.abs(masked_slc) ** 2, axis=1)
    # Form outer product of amplitudes for each slc
    power_mat = amp_vec[:, None] * amp_vec[None, :]
    amp_mat = jnp.sqrt(power_mat)

    # Compute coherence matrix
    # Make the output 0 where the amplitudes are 0 (instead of nan/divide by 0)
    return jnp.where(amp_mat > 1e-6, numer / amp_mat, 0 + 0j)


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


@partial(jit, static_argnames=["half_window", "strides", "oversample"])
def estimate_stack_covariance_crossmul(
    slc_stack: jnp.ndarray,
    half_window,
    strides,
    neighbor_arrays: jnp.ndarray | None = None,
    oversample: int = 2,
) -> jnp.ndarray:
    """Drop-in variant that uses range upsample→crossmul→lookdown inside each window."""
    if neighbor_arrays is None:
        rows, cols = slc_stack.shape[1:]
        full_window = (2 * half_window.y + 1, 2 * half_window.x + 1)
        neighbor_arrays = jnp.ones((rows, cols, *full_window), dtype=bool)

    nslc, rows, cols = slc_stack.shape
    row_strides, col_strides = strides.y, strides.x
    half_row, half_col = half_window.y, half_window.x

    out_rows = (rows + row_strides - 1) // row_strides
    out_cols = (cols + col_strides - 1) // col_strides
    in_r0 = row_strides // 2
    in_c0 = col_strides // 2

    def _get_win(r_out, c_out):
        r = in_r0 + r_out * row_strides
        c = in_c0 + c_out * col_strides
        r0 = lax.clamp(0, r - half_row, r)
        c0 = lax.clamp(0, c - half_col, c)
        # slice: (nslc, H, W)
        H = 2 * half_row + 1
        W = 2 * half_col + 1
        win = lax.dynamic_slice(slc_stack, (0, r0, c0), (nslc, H, W))
        mask2d = neighbor_arrays[r_out, c_out, :, :]
        return coh_mat_window_crossmul(win, mask2d, oversample=oversample)

    R, C = jnp.meshgrid(jnp.arange(out_rows), jnp.arange(out_cols), indexing="ij")
    f2 = vmap(vmap(_get_win, in_axes=(0, None)), in_axes=(None, 0))
    return f2(R, C)  # shape: (out_rows, out_cols, nslc, nslc)


def _fft_upsample_range(x: jnp.ndarray, k: int) -> jnp.ndarray:
    """Zero-pad complex FFT along last axis by integer factor k. Scales by k."""
    n = x.shape[-1]
    X = jnp.fft.fftshift(jnp.fft.fft(x, axis=-1), axes=-1)
    pad = (k - 1) * n
    left = pad // 2
    right = pad - left
    X_up = jnp.pad(X, [(0, 0)] * (x.ndim - 1) + [(left, right)], mode="constant")
    y = jnp.fft.ifft(jnp.fft.ifftshift(X_up, axes=-1), axis=-1) * k
    return y


def _lookdown_range(y: jnp.ndarray, k: int) -> jnp.ndarray:
    """Average consecutive k samples along last axis."""
    n_up = y.shape[-1]
    n = n_up // k
    y = y[..., : n * k]
    y = y.reshape(*y.shape[:-1], n, k).sum(axis=-1) / k
    return y


def _lookdown_shift_phase(n_cols: int, k: int) -> jnp.ndarray:
    """Linear-phase correction for upsample→lookdown shift (ISCE3-style).

    shift = (1 - 1/k)/2 samples. Phase ramp: exp(-i*2π*shift * f), implemented per-column index.
    """
    shift = (1.0 - 1.0 / k) / 2.0
    idx = jnp.arange(n_cols, dtype=jnp.float32)
    phase = -2.0 * jnp.pi * shift * idx / n_cols
    return jnp.exp(1j * phase)


@partial(jit, static_argnames=["oversample"])
def coh_mat_window_crossmul(
    slc_win: jnp.ndarray,
    neighbor_mask_2d: jnp.ndarray | None = None,
    oversample: int = 2,
) -> jnp.ndarray:
    """Compute (nslc,nslc) coherence from a window using upsample, crossmul, looks.

    slc_win: (nslc, H, W) complex
    neighbor_mask_2d: (H, W) bool (True = keep), optional
    """
    nslc, H, W = slc_win.shape
    k = oversample

    # optional window mask
    m = jnp.ones((H, W)) if neighbor_mask_2d is None else neighbor_mask_2d
    slc_win = jnp.where(m[None, ...], slc_win, 0)

    # upsample each SLC in range
    slc_up = _fft_upsample_range(slc_win, k)  # (nslc, H, W*k)

    # cross-multiply all pairs at upsampled rate
    # (nslc, nslc, H, W*k)
    numer_up = slc_up[:, None, :, :] * jnp.conj(slc_up[None, :, :, :])

    # look-down (average over k samples) back to native W
    numer = _lookdown_range(numer_up, k)  # (nslc, nslc, H, W)
    # apply the small linear-phase correction along range
    phase = _lookdown_shift_phase(W, k)  # (W,)
    numer = numer * phase[None, None, None, :]  # broadcast over (nslc, nslc, H, W)

    # average over window pixels to get covariance
    numer = numer.reshape(nslc, nslc, -1).sum(axis=-1)  # (nslc, nslc)

    # normalize to coherence
    power = jnp.sum(jnp.abs(slc_win.reshape(nslc, -1)) ** 2, axis=1)
    denom = jnp.sqrt(power[:, None] * power[None, :])
    coh = jnp.where(denom > 1e-6, numer / denom, 0.0 + 0.0j)
    return coh
