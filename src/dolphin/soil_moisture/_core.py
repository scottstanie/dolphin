"""Core functions for soil moisture estimation from InSAR closure phases.

This module implements the InSAR Soil Moisture Index (ISMI) retrieval algorithm
based on cumulative closure phase analysis. The methodology is based on:

- Zheng & Fattahi (2025): "Modeling, prediction, and retrieval of surface soil
  moisture from InSAR closure phase", Remote Sensing of Environment.
- Wig et al. (2024): "Fine-Resolution Measurement of Soil Moisture from
  Cumulative InSAR Closure Phase", IEEE TGRS.

Theory
------
Non-zero closure phases arise from changes in the dielectric properties of the
scattering medium (e.g., soil moisture variations). The interference between
surface and subsurface reflections produces a closure phase signal that is
sensitive to soil moisture changes.

Key relationships:
- Positive asymmetric soil moisture anomalies produce positive closure phase
  step-changes
- Negative asymmetric soil moisture anomalies produce negative closure phase
  step-changes
- Low-frequency radar (L-band) exhibits heightened sensitivity to the vertical
  distribution of soil moisture

Algorithm
---------
1. Compute nearest-neighbor closure phases from covariance matrices
   (already done in phase_link._closure_phase)
2. Compute cumulative sum of closure phases over time
3. Remove linear trend to isolate soil moisture signal
4. The detrended cumulative closure phase serves as a relative soil moisture
   index (ISMI)

Caveats
-------
- The ISMI is a **relative** soil moisture product, not absolute volumetric
  soil moisture
- Calibration to absolute soil moisture requires external data (e.g., SMAP,
  Sentinel-1/SMAP fusion products, or in-situ measurements)
- Quality of fit is terrain-dependent; works best in areas with significant
  soil moisture variations
- Highly attenuating media (e.g., dense vegetation) may obscure the soil
  moisture signal
- The transfer function between closure phase and soil moisture depends on
  radar frequency, incidence angle, and soil texture (not implemented here)

"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger("dolphin")

__all__ = [
    "SoilMoistureOutput",
    "compute_cumulative_closure_phase",
    "detrend_cumulative_closure_phase",
    "compute_soil_moisture_index",
]


class SoilMoistureOutput(NamedTuple):
    """Output of soil moisture index computation.

    Attributes
    ----------
    ismi : NDArray
        InSAR Soil Moisture Index time series.
        Shape: (n_dates - 2, rows, cols) for nearest-neighbor triplets.
        This is a **relative** soil moisture product.
    cumulative_closure_phase : NDArray
        Cumulative closure phase before detrending.
        Shape: (n_dates - 2, rows, cols).
    trend : NDArray
        Linear trend that was removed (slope per time step).
        Shape: (rows, cols).

    """

    ismi: NDArray[np.floating]
    cumulative_closure_phase: NDArray[np.floating]
    trend: NDArray[np.floating]


def compute_cumulative_closure_phase(
    closure_phases: ArrayLike,
    *,
    axis: int = 0,
) -> NDArray[np.floating]:
    """Compute the cumulative sum of closure phases over time.

    The cumulative closure phase describes variation in the scattering
    properties of the ground over time, analogous to how cumulative InSAR
    phase describes surface displacement.

    Parameters
    ----------
    closure_phases : ArrayLike
        Array of closure phases with shape (n_triplets, rows, cols) or
        (n_triplets,) for a single pixel.
        Units should be radians.
    axis : int, optional
        Axis along which to compute the cumulative sum.
        Default is 0 (time axis).

    Returns
    -------
    NDArray
        Cumulative closure phase with the same shape as input.
        Units are radians.

    Notes
    -----
    The closure phase triplet for dates (i, i+1, i+2) is defined as:
        phi_closure(i) = phi(i,i+1) + phi(i+1,i+2) - phi(i,i+2)

    where phi(a,b) is the interferometric phase between dates a and b.

    For single-look pixels, closure phase is identically zero. Non-zero closure
    phases arise from multi-looking and indicate time-varying scattering
    properties such as soil moisture changes.

    References
    ----------
    .. [1] Zheng et al. (2022), "On Closure Phase and Systematic Bias in
       Multi-looked SAR Interferometry", IEEE TGRS.

    """
    closure_phases = np.asarray(closure_phases)
    return np.cumsum(closure_phases, axis=axis)


def detrend_cumulative_closure_phase(
    cumulative_closure_phase: ArrayLike,
    *,
    axis: int = 0,
    return_trend: bool = False,
) -> NDArray[np.floating] | tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Remove linear trend from cumulative closure phase time series.

    Detrending prevents the cumulative closure phase from being biased by a
    systematic increasing or decreasing trend, allowing it to better match
    the soil moisture signal.

    Parameters
    ----------
    cumulative_closure_phase : ArrayLike
        Cumulative closure phase array with shape (n_times, rows, cols) or
        (n_times,) for a single pixel.
    axis : int, optional
        Time axis along which to detrend. Default is 0.
    return_trend : bool, optional
        If True, also return the estimated linear trend (slope).
        Default is False.

    Returns
    -------
    detrended : NDArray
        Detrended cumulative closure phase with same shape as input.
    trend : NDArray, optional
        Only returned if `return_trend=True`.
        Linear trend (slope) that was removed.
        Shape is the input shape with the time axis removed.

    Notes
    -----
    The detrending is performed by fitting a linear model y = a*t + b
    and subtracting it from the data. This is equivalent to removing
    the best-fit line through the cumulative closure phase time series.

    """
    cumulative_closure_phase = np.asarray(cumulative_closure_phase)
    n_times = cumulative_closure_phase.shape[axis]

    # Create time index array
    t = np.arange(n_times, dtype=np.float64)

    # Move time axis to first position for easier computation
    arr = np.moveaxis(cumulative_closure_phase, axis, 0)
    original_shape = arr.shape
    arr_2d = arr.reshape(n_times, -1)

    # Fit linear trend: y = slope * t + intercept
    # Using numpy's polyfit for efficiency
    # We need to handle NaN values
    valid_mask = np.isfinite(arr_2d)
    has_valid = valid_mask.all(axis=0)

    slopes = np.zeros(arr_2d.shape[1], dtype=np.float64)
    intercepts = np.zeros(arr_2d.shape[1], dtype=np.float64)

    if has_valid.any():
        # For pixels with all valid data, use vectorized polyfit
        coeffs = np.polyfit(t, arr_2d[:, has_valid], deg=1)
        slopes[has_valid] = coeffs[0]
        intercepts[has_valid] = coeffs[1]

    # Handle pixels with some NaN values individually
    partial_valid = ~has_valid & valid_mask.any(axis=0)
    if partial_valid.any():
        for i in np.where(partial_valid)[0]:
            col_valid = valid_mask[:, i]
            if col_valid.sum() >= 2:  # Need at least 2 points for linear fit
                coeffs = np.polyfit(t[col_valid], arr_2d[col_valid, i], deg=1)
                slopes[i] = coeffs[0]
                intercepts[i] = coeffs[1]

    # Compute and subtract the trend
    trend_2d = slopes[np.newaxis, :] * t[:, np.newaxis] + intercepts[np.newaxis, :]
    detrended_2d = arr_2d - trend_2d

    # Reshape back to original shape
    detrended = np.moveaxis(detrended_2d.reshape(original_shape), 0, axis)

    if return_trend:
        trend_shape = list(original_shape)
        trend_shape.pop(0)  # Remove time dimension
        trend = slopes.reshape(trend_shape) if trend_shape else slopes[0]
        return detrended, trend
    return detrended


def compute_soil_moisture_index(
    closure_phases: ArrayLike,
    *,
    axis: int = 0,
    temporal_coherence: ArrayLike | None = None,
    coherence_threshold: float = 0.5,
) -> SoilMoistureOutput:
    """Compute the InSAR Soil Moisture Index from closure phases.

    This function implements the full workflow to derive a relative soil
    moisture product from closure phase measurements:
    1. Compute cumulative closure phase
    2. Remove linear trend
    3. Return the detrended result as the InSAR Soil Moisture Index (ISMI)

    Parameters
    ----------
    closure_phases : ArrayLike
        Array of nearest-neighbor closure phases.
        Shape: (n_triplets, rows, cols) or (n_triplets,) for single pixel.
        Units should be radians.
    axis : int, optional
        Time axis. Default is 0.
    temporal_coherence : ArrayLike, optional
        Temporal coherence array for quality masking.
        Shape: (rows, cols). Values should be in [0, 1].
        If provided, pixels below `coherence_threshold` will be masked (set to NaN).
    coherence_threshold : float, optional
        Threshold for temporal coherence masking. Default is 0.5.
        Only used if `temporal_coherence` is provided.

    Returns
    -------
    SoilMoistureOutput
        Named tuple containing:
        - ismi: InSAR Soil Moisture Index (detrended cumulative closure phase)
        - cumulative_closure_phase: Cumulative closure phase before detrending
        - trend: Linear trend that was removed

    Notes
    -----
    The ISMI is a **relative** soil moisture product. To convert to absolute
    volumetric soil moisture, external calibration data is required (e.g.,
    SMAP satellite measurements or in-situ soil moisture probes).

    The relationship between ISMI and actual soil moisture is:
    - Positive ISMI values indicate higher relative soil moisture
    - Negative ISMI values indicate lower relative soil moisture
    - The magnitude depends on radar frequency, incidence angle, and soil type

    For quantitative soil moisture retrieval, a transfer function or calibration
    model is needed, which depends on:
    - Radar wavelength (L-band is more sensitive than C-band)
    - Local soil texture and properties
    - Vegetation cover

    References
    ----------
    .. [1] Zheng & Fattahi (2025), "Modeling, prediction, and retrieval of
       surface soil moisture from InSAR closure phase", RSE.
    .. [2] Wig et al. (2024), "Fine-Resolution Measurement of Soil Moisture
       from Cumulative InSAR Closure Phase", IEEE TGRS.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulated closure phases (n_triplets=10, rows=100, cols=100)
    >>> closure_phases = np.random.randn(10, 100, 100) * 0.1
    >>> result = compute_soil_moisture_index(closure_phases)
    >>> ismi = result.ismi  # Shape: (10, 100, 100)

    """
    closure_phases = np.asarray(closure_phases, dtype=np.float64)

    # Apply temporal coherence mask if provided
    if temporal_coherence is not None:
        temporal_coherence = np.asarray(temporal_coherence)
        bad_pixels = temporal_coherence < coherence_threshold
        # Broadcast mask to closure phase dimensions
        if closure_phases.ndim > 1:
            # Expand mask to match closure phase time axis
            mask_expanded = np.broadcast_to(
                bad_pixels, closure_phases.shape[1:] if axis == 0 else closure_phases.shape[:-1]
            )
            if axis == 0:
                closure_phases = closure_phases.copy()
                closure_phases[:, mask_expanded] = np.nan
            else:
                closure_phases = closure_phases.copy()
                closure_phases[mask_expanded] = np.nan

    # Step 1: Compute cumulative closure phase
    cumulative = compute_cumulative_closure_phase(closure_phases, axis=axis)

    # Step 2: Remove linear trend
    detrended, trend = detrend_cumulative_closure_phase(
        cumulative, axis=axis, return_trend=True
    )

    return SoilMoistureOutput(
        ismi=detrended,
        cumulative_closure_phase=cumulative,
        trend=trend,
    )
