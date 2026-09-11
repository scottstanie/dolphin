from __future__ import annotations

import logging
import math
from enum import IntEnum
from functools import partial
from typing import NamedTuple, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax
from jax.scipy.linalg import cho_factor, cho_solve
from jax.typing import ArrayLike

from dolphin._types import HalfWindow, Strides
from dolphin.utils import take_looks

from . import covariance, crlb, metrics
from ._closure_phase import (
    closure_phase_coefficient,
    compute_nearest_closure_phases_batch,
    compute_two_hop_closure_phases_batch,
)
from ._eigenvalues import eigh_largest_stack, eigh_smallest_stack
from ._looks import CrlbLooksMethod, estimate_effective_looks_fraction
from ._multilooked_coherence import make_batch_extractor
from ._ps_filling import fill_ps_pixels

logger = logging.getLogger("dolphin")


DEFAULT_STRIDES = Strides(1, 1)


class PhaseLinkRuntimeError(Exception):
    """Exception raised while running the MLE solver."""


class EstimatorType(IntEnum):
    """Type of estimator used for phase linking."""

    EVD = 0
    EMI = 1


class PhaseLinkOutput(NamedTuple):
    """Output of the MLE solver."""

    cpx_phase: np.ndarray
    """Estimated linked phase."""

    temp_coh: np.ndarray
    """Temporal coherence of the optimization.
    A goodness of fit parameter from 0 to 1 at each pixel.
    """

    shp_counts: np.ndarray
    """Number of effective looks used in multilooking.

    For boolean SHP methods (GLRT, KS), this is the count of neighbor pixels.
    For float weights (Gaussian), this is the effective number of looks (ENL)
    via Kish's formula: ENL = (sum(w))^2 / sum(w^2).
    """

    eigenvalues: np.ndarray
    """The smallest (largest) eigenvalue resulting from EMI (EVD)."""

    estimator: np.ndarray  # dtype: np.int8
    """The estimator type used for phase linking at each pixel."""

    crlb_std_dev: np.ndarray
    """The CRLB standard deviation at each pixel."""

    closure_phases: np.ndarray
    """The closure phases at each pixel, for N-2 images."""

    closure_phase_coh: np.ndarray
    """Weighted closure-phase coefficient (gamma_CPw) at each pixel.

    A method-independent reliability indicator in [0, 1] computed directly from
    the sample coherence matrix (Heimpel et al. 2026, Eq. 3.16-3.17). Equals 1
    for an exact rank-1 linkage and 0 for fully decorrelated noise.
    """

    multilooked_coherence: np.ndarray
    """The nearest-N coherence magnitudes at each pixel."""

    two_hop_closure: np.ndarray = np.zeros((0, 0, 0), dtype=np.float32)
    """Mean two-hop closure phase (radians) at each pixel, one band per scale.

    Band ``j`` holds the angle of the mean closure phasor over all triplets
    ``(i, i+k, i+2k)`` of the real (non-compressed) SLCs for ``k = scales[j]``.
    Shape ``(rows, cols, len(scales))``; NaN where the stack is too short.
    """

    split_half_ratio: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    """Sampling-scatter check: RMS over dates of the phase disagreement between
    two disjoint halves of each window, divided by its CRLB prediction.

    About 1 when the reported CRLB matches the actual sampling scatter. Larger
    values flag within-window heterogeneity or texture that the Gaussian
    distributed-scatterer model does not describe. Zero where not computed.
    """

    effective_looks_fraction: float = 1.0
    """Ratio of effective to nominal looks used to scale the CRLB.
    1.0 unless `crlb_looks="effective"`.
    """


def run_phase_linking(
    slc_stack: ArrayLike,
    half_window: HalfWindow,
    strides: Strides = DEFAULT_STRIDES,
    use_evd: bool = False,
    beta: float = 0.0,
    zero_correlation_threshold: float = 0.0,
    reference_idx: int = 0,
    nodata_mask: ArrayLike | None = None,
    mask_input_ps: bool = False,
    ps_mask: ArrayLike | None = None,
    use_max_ps: bool = True,
    neighbor_arrays: ArrayLike | None = None,
    avg_mag: ArrayLike | None = None,
    use_slc_amp: bool = False,
    baseline_lag: Optional[int] = None,
    first_real_slc_idx: int = 0,
    compute_crlb: bool = True,
    flatten: bool = True,
    nearest_n_coherence: int = 0,
    two_hop_scales: Sequence[int] = (),
    crlb_looks: CrlbLooksMethod | str = CrlbLooksMethod.EFFECTIVE,
    effective_looks_fraction: float | None = None,
    split_half: bool = False,
) -> PhaseLinkOutput:
    """Estimate the linked phase for a stack of SLCs.

    If passing a `ps_mask`, will combine the PS phases with the
    estimated DS phases.

    Parameters
    ----------
    slc_stack : ArrayLike
        The SLC stack, with shape (n_images, n_rows, n_cols)
    half_window : HalfWindow, or tuple[int, int]
        A (named) tuple of (y, x) sizes for the half window.
        The full window size is 2 * half_window + 1 for x, y.
    strides : tuple[int, int], optional
        The (y, x) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm.
    beta : float, optional
        The regularization parameter, by default 0 (no regularization).
    zero_correlation_threshold : float, optional
        Snap correlation values in the coherence matrix below this value to 0.
        Default is 0 (no clipping).
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    nodata_mask : ArrayLike, optional
        A mask of bad/nodata pixels to ignore when estimating the covariance.
        Pixels with `True` (or 1) are ignored, by default None
        If None, all pixels are used, by default None.
    mask_input_ps : bool
        If True, pixels labeled as PS will get set to NaN during phase linking to
        avoid summing their phase. Default of False means that the SHP algorithm
        will decide if a pixel should be included, regardless of its PS label.
    ps_mask : ArrayLike, optional
        A mask of pixels marking persistent scatterers (PS) to
        skip when multilooking.
        Pixels with `True` (or 1) are PS and will be ignored
        (combined with `nodata_mask`).
        The phase from these pixels will be inserted back
        into the final estimate directly from `slc_stack`.
    use_max_ps : bool, optional
        Whether to use the maximum PS phase for the first pixel, or average all
        PS within the look window.
        By default True.
    neighbor_arrays : ArrayLike, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.
    avg_mag : ArrayLike, optional
        The average magnitude of the SLC stack, used to to find the brightest
        PS pixels to fill within each look window.
        If None, the average magnitude will be computed from `slc_stack`.
    use_slc_amp : bool, optional
        Whether to use the SLC amplitude when outputting the MLE estimate,
        or to set the SLC amplitude to 1.0. By default False.
    baseline_lag : int, optional, default=None
        lag for temporal baseline to do short temporal baseline inversion (STBAS)
    first_real_slc_idx : int, optional, default = 0
        The index of the first real SLC in the stack.
        This is only used for the CRLB computation.
        By default 0.
    compute_crlb : bool, optional
        Whether to compute the CRLB, by default True
    flatten : bool, optional
        If True, perform phase flattening before coherence estimation to
        improve accuracy of coherence magnitudes, by default True.
    nearest_n_coherence : int, optional
        Number of nearest coherence diagonals to extract and return.
        0 (default) means don't extract. 1 gives first off-diagonal
        (nearest neighbor coherences), 2 gives first 2 diagonals, etc.
    two_hop_scales : Sequence[int], optional
        Scales ``k`` at which to output the mean two-hop closure phase of
        triplets ``(i, i+k, i+2k)`` over the real SLCs. Default: none.
    crlb_looks : CrlbLooksMethod or str, optional
        How the CRLB counts looks: ``"effective"`` (default) scales the SHP count
        by an effective-looks fraction measured from the intensity
        autocorrelation of the stack, ``"shp_count"`` treats every neighbor as
        independent, ``"sqrt_half_window"`` uses the legacy constant.
    effective_looks_fraction : float, optional
        The fraction to use with ``crlb_looks="effective"``. Workflows estimate
        it once per stack with `estimate_stack_effective_looks_fraction` so the
        same value applies to every block. If None, it is estimated from this
        block alone, which varies with scene texture from block to block.
    split_half : bool, optional
        Also link two disjoint halves of every window and return the ratio of
        their phase disagreement to its CRLB prediction. Default False.

    Returns
    -------
    PhaseLinkOutput:
        A Named tuple with results from phase linking.

    """
    _, rows, cols = slc_stack.shape
    # Common pre-processing for both CPU and GPU versions:

    # Mask nodata pixels if given
    if nodata_mask is None:
        nodata_mask = np.zeros((rows, cols), dtype=bool)
    else:
        nodata_mask = nodata_mask.astype(bool)

    # Track the PS pixels, if given, and remove them from the stack
    # This will prevent the large amplitude PS pixels from dominating
    # the covariance estimation.
    if ps_mask is None:
        ps_mask = np.zeros((rows, cols), dtype=bool)
    else:
        ps_mask = ps_mask.astype(bool)
    _raise_if_all_nan(slc_stack)

    # Make sure we also are ignoring pixels which are nans for all SLCs
    if nodata_mask.shape != (rows, cols) or ps_mask.shape != (rows, cols):
        msg = (
            f"nodata_mask.shape={nodata_mask.shape}, ps_mask.shape={ps_mask.shape},"
            f" but != SLC (rows, cols) {rows, cols}"
        )
        raise ValueError(msg)
    # for any area that has nans in the SLC stack, mark it as nodata
    nodata_mask |= np.any(np.isnan(slc_stack), axis=0)
    # Make sure the PS mask didn't have extra burst borders that are nodata here
    ps_mask[nodata_mask] = False

    # Make a copy, and set the masked pixels to np.nan
    slc_stack_masked = slc_stack.copy()
    if mask_input_ps:
        ignore_mask = np.logical_or.reduce((nodata_mask, ps_mask))
        slc_stack_masked[:, ignore_mask] = np.nan
    else:
        slc_stack_masked[:, nodata_mask] = np.nan

    cpl_out = run_cpl(
        slc_stack=slc_stack_masked,
        half_window=half_window,
        strides=strides,
        use_evd=use_evd,
        beta=beta,
        zero_correlation_threshold=zero_correlation_threshold,
        reference_idx=reference_idx,
        neighbor_arrays=neighbor_arrays,
        baseline_lag=baseline_lag,
        first_real_slc_idx=first_real_slc_idx,
        compute_crlb=compute_crlb,
        flatten=flatten,
        nearest_n_coherence=nearest_n_coherence,
        two_hop_scales=two_hop_scales,
        crlb_looks=crlb_looks,
        effective_looks_fraction=effective_looks_fraction,
        split_half=split_half,
    )

    # Get the smaller, looked versions of the masks
    # We zero out nodata if all pixels within the window had nodata
    mask_looked = take_looks(nodata_mask, *strides, func_type="all")

    # Convert from jax array back to np
    temp_coh = np.array(cpl_out.temp_coh)

    # Set as unit-magnitude
    cpx_phase = np.exp(1j * np.angle(cpl_out.cpx_phase))
    # Fill in the PS pixels from the original SLC stack, if it was given
    crlb_std_dev = np.array(cpl_out.crlb_std_dev)
    if np.any(ps_mask):
        fill_ps_pixels(
            cpx_phase,
            temp_coh,
            slc_stack,
            ps_mask,
            strides,
            avg_mag,
            reference_idx,
            use_max_ps=use_max_ps,
            crlb_std_dev=crlb_std_dev if compute_crlb else None,
        )

    if use_slc_amp:
        # use the amplitude from the original SLCs
        # account for the strides when grabbing original data
        # we need to match `io.compute_out_shape` here
        slcs_decimated = decimate(slc_stack, strides)
        cpx_phase = np.exp(1j * np.angle(cpx_phase)) * np.abs(slcs_decimated)

    closure_phase_coh = np.array(cpl_out.closure_phase_coh)
    # Finally, ensure the nodata regions are 0
    cpx_phase[:, mask_looked] = np.nan
    temp_coh[mask_looked] = np.nan
    closure_phase_coh[mask_looked] = np.nan

    return PhaseLinkOutput(
        cpx_phase=cpx_phase,
        temp_coh=temp_coh,
        shp_counts=np.asarray(cpl_out.shp_counts),
        # Convert the rest to numpy for writing
        eigenvalues=np.asarray(cpl_out.eigenvalues),
        estimator=np.asarray(cpl_out.estimator),
        crlb_std_dev=crlb_std_dev,
        closure_phases=np.asarray(cpl_out.closure_phases),
        closure_phase_coh=closure_phase_coh,
        multilooked_coherence=np.asarray(cpl_out.multilooked_coherence),
        # Copies, so the workflow can fill NaNs in place
        two_hop_closure=np.array(cpl_out.two_hop_closure),
        split_half_ratio=np.array(cpl_out.split_half_ratio),
        effective_looks_fraction=float(cpl_out.effective_looks_fraction),
    )


def run_cpl(
    slc_stack: np.ndarray,
    half_window: HalfWindow,
    strides: Strides,
    use_evd: bool = False,
    beta: float = 0,
    zero_correlation_threshold: float = 0.0,
    reference_idx: int = 0,
    neighbor_arrays: Optional[np.ndarray] = None,
    baseline_lag: Optional[int] = None,
    flatten: bool = True,
    first_real_slc_idx: int = 0,
    compute_crlb: bool = True,
    nearest_n_coherence: int = 0,
    two_hop_scales: Sequence[int] = (),
    crlb_looks: CrlbLooksMethod | str = CrlbLooksMethod.EFFECTIVE,
    effective_looks_fraction: float | None = None,
    split_half: bool = False,
) -> PhaseLinkOutput:
    """Run the Combined Phase Linking (CPL) algorithm.

    Estimates a coherence matrix for each SLC pixel, then
    runs the EMI/EVD solver.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_slc, n_rows, n_cols)
    half_window : HalfWindow, or tuple[int, int]
        A (named) tuple of (y, x) sizes for the half window.
        The full window size is 2 * half_window + 1 for x, y.
    strides : tuple[int, int], optional
        The (y, x) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm.
    beta : float, optional
        The regularization parameter, by default 0 (no regularization).
    zero_correlation_threshold : float, optional
        Snap correlation values in the coherence matrix below this value to 0.
        Default is 0 (no clipping).
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    use_slc_amp : bool, optional
        Whether to use the SLC amplitude when outputting the MLE estimate,
        or to set the SLC amplitude to 1.0. By default False.
    neighbor_arrays : np.ndarray, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.
    baseline_lag : int, optional, default=None
        StBAS parameter to include only nearest-N interferograms for phase linking.
        A `baseline_lag` of `n` will only include the closest `n` interferograms.
        `baseline_line` must be positive.
    first_real_slc_idx : int, optional, default = 0
        The index of the first real SLC in the stack.
        This is only used for the CRLB computation.
        By default 0.
    compute_crlb : bool, optional
        Whether to compute the CRLB, by default True
    flatten : bool, optional
        If True, perform phase flattening before coherence estimation to
        improve accuracy of coherence magnitudes, by default True.
    nearest_n_coherence : int, optional
        Number of nearest coherence diagonals to extract and return.
        0 (default) means don't extract. 1 gives first off-diagonal
        (nearest neighbor coherences), 2 gives first 2 diagonals, etc.
    two_hop_scales : Sequence[int], optional
        Scales ``k`` for the mean two-hop closure phase of triplets
        ``(i, i+k, i+2k)`` over the real SLCs. Default: none.
    crlb_looks : CrlbLooksMethod or str, optional
        How the CRLB counts looks. See `run_phase_linking`.
    effective_looks_fraction : float, optional
        Stack-wide fraction for ``crlb_looks="effective"``; estimated from this
        block if None. See `run_phase_linking`.
    split_half : bool, optional
        Also link two disjoint halves of every window and return the ratio of
        their phase disagreement to its CRLB prediction. Default False.

    Returns
    -------
    cpx_phase : Array
        Optimized SLC phase, shape same as `slc_stack` unless Strides are requested.
    temp_coh : Array
        Temporal coherence of the optimization.
        A goodness of fit parameter from 0 to 1 at each pixel.
        shape = (out_rows, out_cols)
    eigenvalues : Array
        The eigenvalues of the coherence matrices.
        If `use_evd` is True, these are the largest eigenvalues;
        Otherwise, for EMI they are the smallest.
        shape = (out_rows, out_cols)
    estimator : Array
        The estimator used at each pixel.
        0 = EVD, 1 = EMI
        shape = (out_rows, out_cols)

    """
    from dolphin.utils import upsample_nearest

    if flatten:
        C_arrays_full = covariance.estimate_stack_covariance(
            slc_stack,
            half_window,
            # Strides(1, 1),
            strides,
            neighbor_arrays=neighbor_arrays,
        )
        cpx_phase0, _, _, _ = process_coherence_matrices(
            C_arrays_full,
            use_evd=True,
            compute_crlb=False,
        )
        cpx_phase0 = jnp.moveaxis(cpx_phase0, -1, 0)
        cpx_up = upsample_nearest(cpx_phase0, slc_stack.shape[1:], use_jax=True)
        C_arrays_flat = covariance.estimate_stack_covariance(
            slc_stack * cpx_up.conj(),
            half_window,
            strides,
            neighbor_arrays=neighbor_arrays,
        )
        # Just use the coherence values:
        C_arrays = jnp.abs(C_arrays_flat) * jnp.exp(1j * jnp.angle(C_arrays_full))
        del C_arrays_flat, C_arrays_full, cpx_phase0
    else:
        C_arrays = covariance.estimate_stack_covariance(
            slc_stack,
            half_window,
            strides,
            neighbor_arrays=neighbor_arrays,
        )

    ns = slc_stack.shape[0]
    if baseline_lag:
        u_rows, u_cols = jnp.triu_indices(ns, baseline_lag)
        l_rows, l_cols = jnp.tril_indices(ns, -baseline_lag)
        C_arrays = C_arrays.at[:, :, u_rows, u_cols].set(0.0 + 0j)
        C_arrays = C_arrays.at[:, :, l_rows, l_cols].set(0.0 + 0j)

    closure_phases = compute_nearest_closure_phases_batch(C_arrays)
    closure_phase_coh = closure_phase_coefficient(C_arrays)
    two_hop_closure = _mean_two_hop_closures(
        C_arrays, two_hop_scales, first_real_slc_idx
    )

    # Extract nearest-N coherence magnitudes if requested
    if nearest_n_coherence > 0:
        extract_coherences = make_batch_extractor(nearest_n_coherence)
        nearest_coherence = extract_coherences(C_arrays)
    else:
        # Return empty array if not requested
        rows, cols = C_arrays.shape[:2]
        nearest_coherence = jnp.zeros((rows, cols, 0), dtype=jnp.float32)

    crlb_looks = CrlbLooksMethod(crlb_looks)
    if crlb_looks == CrlbLooksMethod.SQRT_HALF_WINDOW:
        # Legacy behavior: one conservative constant for every pixel
        num_looks = math.sqrt(half_window[0] * half_window[1])
    else:
        # Solve for one look, then scale per pixel by the look count below
        num_looks = 1

    reference_idx = ns + reference_idx if reference_idx < 0 else reference_idx
    cpx_phase, eigenvalues, estimator, crlb_std_dev = process_coherence_matrices(
        C_arrays,
        use_evd=use_evd,
        beta=beta,
        zero_correlation_threshold=zero_correlation_threshold,
        reference_idx=reference_idx,
        num_looks=num_looks,
        first_real_slc_idx=first_real_slc_idx,
        compute_crlb=compute_crlb,
    )
    # Get the temporal coherence
    temp_coh = metrics.estimate_temp_coh(cpx_phase, C_arrays)
    out_shape = temp_coh.shape

    # Get the SHP counts for each pixel (if not using Rect window)
    if neighbor_arrays is None:
        shp_counts = jnp.zeros(out_shape, dtype=np.int16)
    else:
        # For boolean masks this is the neighbor count; for float weights (e.g.
        # Gaussian) it is Kish's effective sample size (sum w)^2 / sum(w^2).
        shp_counts = jnp.round(_count_looks(neighbor_arrays, out_shape, half_window))
        shp_counts = shp_counts.astype(jnp.int16)

    looks_fraction = 1.0
    if compute_crlb and crlb_looks != CrlbLooksMethod.SQRT_HALF_WINDOW:
        if crlb_looks == CrlbLooksMethod.EFFECTIVE:
            if effective_looks_fraction is None:
                looks_fraction = estimate_effective_looks_fraction(
                    slc_stack, half_window
                )
            else:
                looks_fraction = float(effective_looks_fraction)
        looks = _count_looks(neighbor_arrays, out_shape, half_window) * looks_fraction
        crlb_std_dev = crlb_std_dev / jnp.sqrt(jnp.maximum(looks, 1.0))[..., None]

    if split_half:
        split_half_ratio = _split_half_ratio(
            slc_stack,
            half_window,
            strides,
            neighbor_arrays,
            use_evd=use_evd,
            beta=beta,
            zero_correlation_threshold=zero_correlation_threshold,
            reference_idx=reference_idx,
            looks_fraction=looks_fraction,
            out_shape=out_shape,
        )
    else:
        split_half_ratio = jnp.zeros(out_shape, dtype=jnp.float32)

    # Reshape the (rows, cols, nslcs) output to be same as input stack
    cpx_phase_reshaped = jnp.moveaxis(cpx_phase, -1, 0)
    crlb_std_dev_reshaped = jnp.moveaxis(crlb_std_dev, -1, 0)

    return PhaseLinkOutput(
        cpx_phase=cpx_phase_reshaped,
        temp_coh=temp_coh,
        shp_counts=shp_counts,
        eigenvalues=eigenvalues,
        estimator=estimator,
        crlb_std_dev=crlb_std_dev_reshaped,
        closure_phases=closure_phases,
        closure_phase_coh=closure_phase_coh,
        multilooked_coherence=nearest_coherence,
        two_hop_closure=two_hop_closure,
        split_half_ratio=split_half_ratio,
        effective_looks_fraction=looks_fraction,
    )


def _count_looks(
    neighbor_arrays: ArrayLike | None, out_shape: tuple[int, int], half_window
) -> Array:
    """Per-pixel look count: window size, neighbor count, or Kish effective size."""
    if neighbor_arrays is None:
        n = (2 * half_window[0] + 1) * (2 * half_window[1] + 1)
        return jnp.full(out_shape, float(n), dtype=jnp.float32)
    na = jnp.asarray(neighbor_arrays)
    if na.dtype == jnp.bool_:
        return jnp.sum(na, axis=(-2, -1)).astype(jnp.float32)
    w_sum = jnp.sum(na, axis=(-2, -1))
    w_sq_sum = jnp.sum(na**2, axis=(-2, -1))
    enl = jnp.where(w_sq_sum > 0, w_sum**2 / jnp.maximum(w_sq_sum, 1e-12), 0.0)
    return enl.astype(jnp.float32)


def _mean_two_hop_closures(
    C_arrays: Array, scales: Sequence[int], first_real_slc_idx: int
) -> Array:
    """Angle of the mean two-hop closure phasor per pixel, one band per scale.

    Compressed SLCs are excluded so every triplet is made of real acquisitions.
    Scales the stack is too short for give NaN.
    """
    rows, cols = C_arrays.shape[:2]
    if not scales:
        return jnp.zeros((rows, cols, 0), dtype=jnp.float32)
    C_real = C_arrays[..., first_real_slc_idx:, first_real_slc_idx:]
    n_real = C_real.shape[-1]
    maps = []
    for scale in scales:
        k = int(scale)
        if k < 1 or n_real <= 2 * k:
            maps.append(jnp.full((rows, cols), jnp.nan, dtype=jnp.float32))
            continue
        xi = compute_two_hop_closure_phases_batch(C_real, scale=k)
        maps.append(jnp.angle(jnp.mean(jnp.exp(1j * xi), axis=-1)).astype(jnp.float32))
    return jnp.stack(maps, axis=-1)


def _split_half_ratio(
    slc_stack: ArrayLike,
    half_window: HalfWindow,
    strides: Strides,
    neighbor_arrays: ArrayLike | None,
    *,
    use_evd: bool,
    beta: float,
    zero_correlation_threshold: float,
    reference_idx: int,
    looks_fraction: float,
    out_shape: tuple[int, int],
) -> Array:
    """Link the left and right halves of every window and compare their scatter.

    The phase common to the window (deformation, atmosphere, any consistent
    nuisance) cancels in the difference of the two half-window estimates, so the
    difference is a direct, truth-free sample of the estimator's scatter. Under
    the Gaussian distributed-scatterer model its variance is the sum of the two
    halves' CRLB variances; the returned ratio is the RMS over non-reference
    dates of ``difference / predicted standard deviation``.
    """
    wy, wx = 2 * half_window[0] + 1, 2 * half_window[1] + 1
    n = jnp.asarray(slc_stack).shape[0]
    if wx < 3:
        return jnp.full(out_shape, jnp.nan, dtype=jnp.float32)
    if neighbor_arrays is None:
        base = np.ones((*out_shape, wy, wx), dtype=bool)
    else:
        base = np.asarray(neighbor_arrays)
    left = np.zeros((wy, wx), dtype=bool)
    left[:, : wx // 2] = True
    right = np.zeros((wy, wx), dtype=bool)
    right[:, wx // 2 + 1 :] = True

    halves = []
    for side in (left, right):
        mask = base * side
        C = covariance.estimate_stack_covariance(
            slc_stack, half_window, strides, neighbor_arrays=mask
        )
        cpx, _, _, sig1 = process_coherence_matrices(
            C,
            use_evd=use_evd,
            beta=beta,
            zero_correlation_threshold=zero_correlation_threshold,
            reference_idx=reference_idx,
            num_looks=1,
            compute_crlb=True,
        )
        looks = _count_looks(mask, out_shape, half_window) * looks_fraction
        halves.append((jnp.angle(cpx), sig1**2 / jnp.maximum(looks, 1.0)[..., None]))
    (phase_a, var_a), (phase_b, var_b) = halves
    delta = jnp.angle(jnp.exp(1j * (phase_a - phase_b)))
    z2 = delta**2 / (var_a + var_b)
    keep = jnp.arange(n) != (reference_idx % n)
    ratio = jnp.sqrt(jnp.nanmean(z2[..., keep], axis=-1))
    return ratio.astype(jnp.float32)


@partial(
    jit,
    static_argnames=(
        "use_evd",
        "beta",
        "reference_idx",
        "num_looks",
        "first_real_slc_idx",
        "compute_crlb",
    ),
)
def process_coherence_matrices(
    C_arrays,
    use_evd: bool = False,
    beta: float = 0.0,
    zero_correlation_threshold: float = 0.0,
    reference_idx: int = 0,
    num_looks: int = 1,
    first_real_slc_idx: int = 0,
    compute_crlb: bool = True,
) -> tuple[Array, Array, Array, Array]:
    """Estimate the linked phase for a stack of coherence matrices.

    This function is used after coherence estimation to estimate the
    optimized SLC phase.

    Parameters
    ----------
    C_arrays : ndarray, shape = (rows, cols, nslc, nslc)
        The sample coherence matrix at each pixel
        (e.g. from [dolphin.phase_link.covariance.estimate_stack_covariance][])
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm of [@Ansari2018EfficientPhaseEstimation].
    beta : float, optional
        The regularization parameter for inverting Gamma = |C|
        The regularization is applied as (1 - beta) * Gamma + beta * I
        Default is 0 (no regularization).
    zero_correlation_threshold : float, optional
        Snap correlation values in the coherence matrix below this value to 0.
        Default is 0 (no clipping).
    reference_idx : int, optional
        The index of the reference acquisition, by default 0
        All outputs are multiplied by the conjugate of the data at this index.
    num_looks : int, optional
        The number of looks used to form the input correlation data, used
        during CRLB computation.
    first_real_slc_idx : int, optional, default = 0
        The index of the first real SLC in the stack.
        Retained for API compatibility; the CRLB is referenced to `reference_idx`.
        By default 0.
    compute_crlb : bool, optional
        Whether to compute the CRLB
        Default is True.

    Returns
    -------
    eig_vecs : ndarray[float32], shape = (rows, cols, nslc)
        The phase resulting from the optimization at each output pixel.
        Shape is same as input slcs unless Strides > (1, 1)
    eig_vals : ndarray[float], shape = (rows, cols)
        The smallest (largest) eigenvalue as solved by EMI (EVD).
    estimator : Array
        The estimator used at each pixel.
        0 = EVD, 1 = EMI
    crlb_std_dev : ndarray[float32], shape = (rows, cols, nslc)
        The CRLB standard deviation at each pixel.

    """
    rows, cols, n, _ = C_arrays.shape
    # Retained for API compatibility; the CRLB now follows `reference_idx`.
    del first_real_slc_idx

    evd_eig_vals, evd_eig_vecs = eigh_largest_stack(C_arrays * jnp.abs(C_arrays))

    Gamma = jnp.abs(C_arrays)

    # Identity used for regularization and for solving
    Id = jnp.eye(n, dtype=Gamma.dtype)
    # repeat the identity matrix for each pixel
    Id = jnp.broadcast_to(Id, (rows, cols, n, n))

    if beta > 0:
        # Perform regularization
        Gamma = (1 - beta) * Gamma + beta * Id
    # Assume correlation below `zero_correlation_threshold` is 0
    Gamma = jnp.where(Gamma < zero_correlation_threshold, 0, Gamma)

    # Attempt to invert Gamma
    gamma_jitter = 1e-6
    cho, is_lower = cho_factor(Gamma + gamma_jitter * Id)

    # Check: If it fails the cholesky factor, it's close to singular and
    # we should just fall back to EVD
    # Use the already- factored |Gamma|^-1, solving Ax = I gives the inverse
    Gamma_inv = cho_solve((cho, is_lower), Id)
    if use_evd:
        # EVD
        eig_vals, eig_vecs = evd_eig_vals, evd_eig_vecs
        estimator = jnp.zeros(eig_vals.shape, dtype=bool)
    else:
        # EMI
        # estimate the wrapped phase based on the EMI paper
        # *smallest* eigenvalue decomposition of the (|Gamma|^-1  *  C) matrix
        # We're looking for the lambda nearest to 1. So shift by 0.99
        # Also, use the evd vectors as iteration starting point:
        mu = 0.99
        emi_eig_vals, emi_eig_vecs = eigh_smallest_stack(Gamma_inv * C_arrays, mu)
        # From the EMI paper, normalize the eigenvectors to have norm sqrt(n)
        emi_eig_vecs = (
            jnp.sqrt(n)
            * emi_eig_vecs
            / jnp.linalg.norm(emi_eig_vecs, axis=-1, keepdims=True)
        )
        # is the output is the inverse of the eigenvectors? or inverse conj?

        # Use https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.select.html
        # Note that `if` would fail the jit tracing
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#cond
        inv_has_nans = jnp.any(jnp.isnan(Gamma_inv), axis=(-1, -2))

        # Must broadcast the 2D boolean array so it's the same size as the outputs
        inv_has_nans_3d = jnp.tile(inv_has_nans[:, :, None], (1, 1, n))

        # For EVD, or places where inverting |Gamma| failed: fall back to computing EVD
        eig_vecs = lax.select(
            inv_has_nans_3d,
            # Run this on True: EVD, since we failed to invert:
            evd_eig_vecs,
            # Otherwise, on False, we're fine to use EMI
            emi_eig_vecs,
        )

        eig_vals = lax.select(inv_has_nans, evd_eig_vals, emi_eig_vals)
        # Make array of ints to indicate which estimator was used for each pixel
        # 0 means EVD, 1 mean EMI
        evd_used = jnp.zeros(emi_eig_vals.shape, dtype=jnp.int8)
        emi_used = jnp.ones(emi_eig_vals.shape, dtype=jnp.int8)
        estimator = lax.select(inv_has_nans, evd_used, emi_used)

    # Compute CRLB for each pixel
    if compute_crlb:
        # Build X once and do the inverse-free CRLB from X
        X = crlb._build_fisher_from_abs_gamma(Gamma, Gamma_inv, num_looks)
        # Reference the CRLB to the same acquisition as the output phases, so the
        # zero-uncertainty entry sits where the phase is zero.
        crlb_std_dev = crlb._crlb_from_x(X, reference_idx % n, 0, 1e-6)

    else:
        crlb_std_dev = jnp.zeros(C_arrays.shape[:-1], dtype=jnp.float32)

    # Now the shape of eig_vecs is (rows, cols, nslc)
    # at pixel (r, c), eig_vecs[r, c] is the largest (smallest) eigenvector if
    # we picked EVD (EMI)
    # The phase estimate on the reference day will be size (rows, cols)
    ref = eig_vecs[:, :, reference_idx]
    # Make sure each still has 3 dims, then reference all phases to `ref`
    evd_estimate = eig_vecs * jnp.exp(-1j * jnp.angle(ref[:, :, None]))

    return evd_estimate, eig_vals, estimator.astype("uint8"), crlb_std_dev


def decimate(arr: ArrayLike, strides: Strides) -> Array:
    """Decimate an array by strides in the x and y directions.

    Output will match [`io.compute_out_shape`][dolphin.io.compute_out_shape]

    Parameters
    ----------
    arr : ArrayLike
        2D or 3D array to decimate.
    strides : dict[str, int]
        The strides in the x and y directions.

    Returns
    -------
    ArrayLike
        The decimated array.

    """
    ys, xs = strides
    rows, cols = arr.shape[-2:]
    start_r = ys // 2
    start_c = xs // 2
    end_r = (rows // ys) * ys + 1
    end_c = (cols // xs) * xs + 1
    return arr[..., start_r:end_r:ys, start_c:end_c:xs]


def _raise_if_all_nan(slc_stack: np.ndarray):
    """Check for all NaNs in each SLC of the stack."""
    nans = np.isnan(slc_stack)
    # Check that there are no SLCS which are all nans:
    bad_slc_idxs = np.where(np.all(nans, axis=(1, 2)))[0]
    if bad_slc_idxs.size > 0:
        msg = f"slc_stack[{bad_slc_idxs}] out of {len(slc_stack)} are all NaNs."
        raise PhaseLinkRuntimeError(msg)
