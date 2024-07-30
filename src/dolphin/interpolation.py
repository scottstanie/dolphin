import logging

import numba
import numpy as np
from numpy.typing import ArrayLike

from .similarity import get_circle_idxs

logger = logging.getLogger(__name__)

__all__ = [
    "interpolate",
    "gaussian_filter_nan",
    "mask_and_filter",
]


def interpolate(
    ifg: ArrayLike,
    weights: ArrayLike,
    weight_cutoff: float = 0.5,
    num_neighbors: int = 20,
    max_radius: int = 51,
    min_radius: int = 0,
    alpha: float = 0.75,
) -> np.ndarray:
    """Interpolate a complex interferogram based on pixel weights.

    Build upon persistent scatterer interpolation used in
    [@Chen2015PersistentScattererInterpolation] and
    [@Wang2022AccuratePersistentScatterer] by allowing floating-point weights
    instead of 0/1 PS weights.

    Parameters
    ----------
    ifg : np.ndarray, 2D complex array
        wrapped interferogram to interpolate
    weights : 2D float array
        Array of weights from 0 to 1 indicating how strongly to weigh
        the ifg values when interpolating.
        A special case of this is a PS mask where
            weights[i,j] = True if radar pixel (i,j) is a PS
            weights[i,j] = False if radar pixel (i,j) is not a PS
        Can also pass a coherence image to use as weights.
    weight_cutoff: float
        Threshold to use on `weights` so that pixels where
        `weight[i, j] < weight_cutoff` have phase values replaced by
        an interpolated value.
        The default is 0.5: pixels with weight less than 0.5 are replaced with a
        smoothed version of the surrounding pixels.
    num_neighbors: int (optional)
        number of nearest PS pixels used for interpolation
        num_neighbors = 20 by default
    max_radius : int (optional)
        maximum radius (in pixels) for PS searching
        max_radius = 51 by default
    min_radius : int (optional)
        minimum radius (in pixels) for PS searching
        max_radius = 0 by default
    alpha : float (optional)
        hyperparameter controlling the weight of PS in interpolation: smaller
        alpha means more weight is assigned to PS closer to the center pixel.
        alpha = 0.75 by default

    Returns
    -------
    interpolated_ifg : 2D complex array
        interpolated interferogram with the same amplitude, but different
        wrapped phase at non-ps pixels.

    """
    nrow, ncol = weights.shape

    weights_float = np.clip(weights.astype(np.float32), 0, 1)
    # Ensure weights are between 0 and 1
    if np.any(weights_float > 1):
        logger.warning("weights array has values greater than 1. Clipping to 1.")
    if np.any(weights_float < 0):
        logger.warning("weights array has negative values. Clipping to 0.")
    weights_float = np.clip(weights_float, 0, 1)

    interpolated_ifg = np.zeros((nrow, ncol), dtype=np.complex64)

    indices = np.array(
        get_circle_idxs(max_radius, min_radius=min_radius, sort_output=False)
    )

    _interp_loop(
        ifg,
        weights_float,
        weight_cutoff,
        num_neighbors,
        alpha,
        indices,
        interpolated_ifg,
    )
    return interpolated_ifg


@numba.njit(parallel=True)
def _interp_loop(
    ifg, weights, weight_cutoff, num_neighbors, alpha, indices, interpolated_ifg
):
    nrow, ncol = weights.shape
    nindices = len(indices)
    for r0 in numba.prange(nrow):
        for c0 in range(ncol):
            if weights[r0, c0] >= weight_cutoff:
                interpolated_ifg[r0, c0] = ifg[r0, c0]
                continue

            csum = 0.0 + 0j
            counter = 0
            r2 = np.zeros(num_neighbors, dtype=np.float64)
            cphase = np.zeros(num_neighbors, dtype=np.complex128)

            for i in range(nindices):
                idx = indices[i]
                r = r0 + idx[0]
                c = c0 + idx[1]

                if (
                    (r >= 0)
                    and (r < nrow)
                    and (c >= 0)
                    and (c < ncol)
                    and weights[r, c] >= weight_cutoff
                ):
                    # calculate the square distance to the center pixel
                    r2[counter] = idx[0] ** 2 + idx[1] ** 2

                    cphase[counter] = np.exp(1j * np.angle(ifg[r, c]))
                    counter += 1
                    if counter >= num_neighbors:
                        break

            # `counter` got up to one more than the number of elements
            # The last one will be the largest radius
            r2_norm = (r2[counter - 1] ** alpha) / 2
            for i in range(counter):
                csum += np.exp(-r2[i] / r2_norm) * cphase[i]

            interpolated_ifg[r0, c0] = np.abs(ifg[r0, c0]) * np.exp(1j * np.angle(csum))


def gaussian_filter_nan(
    image: ArrayLike, sigma: float, mode="constant", **kwargs
) -> np.ndarray:
    """Apply a gaussian filter to an image with NaNs (avoiding all nans).

    The scipy.ndimage `gaussian_filter` will make the output all NaNs if
    any of the pixels in the input that touches the kernel is NaN

    Source:
    https://stackoverflow.com/a/36307291

    Parameters
    ----------
    image : ndarray
        Image with nans to filter
    sigma : float
        Size of filter kernel. passed into `gaussian_filter`
    mode : str, default = "constant"
        Boundary mode for `[scipy.ndimage.gaussian_filter][]`
    **kwargs : Any
        Passed into `[scipy.ndimage.gaussian_filter][]`

    Returns
    -------
    ndarray
        Filtered version of `image`.

    """
    from scipy.ndimage import gaussian_filter

    if np.sum(np.isnan(image)) == 0:
        return gaussian_filter(image, sigma=sigma, mode=mode, **kwargs)

    V = image.copy()
    nan_idxs = np.isnan(image)
    V[nan_idxs] = 0
    V_filt = gaussian_filter(V, sigma, **kwargs)

    W = np.ones(image.shape)
    W[nan_idxs] = 0
    W_filt = gaussian_filter(W, sigma, **kwargs)

    return V_filt / W_filt


def mask_and_filter(ifg: ArrayLike, sigma: float, mask: ArrayLike) -> np.ndarray:
    """Apply a Gaussian filter to masked areas of an interferogram.

    This function replaces the masked areas of an interferogram with NaN values,
    applies a Gaussian filter, and then restores the original values in the
    unmasked areas.

    Parameters
    ----------
    ifg : ArrayLike
        The input interferogram (complex or real-valued 2D array).
    sigma : float
        The standard deviation for the Gaussian kernel.
    mask : ArrayLike
        Boolean mask where True indicates pixels to be filtered.

    Returns
    -------
    np.ndarray
        The filtered interferogram with original values preserved in unmasked areas.

    Notes
    -----
    This function uses the `gaussian_filter_nan` function to apply the filter
    while handling NaN values properly.

    """
    # Create a copy of the interferogram with masked areas set to NaN
    ifg_nanned = ifg.copy()
    ifg_nanned[mask] = np.nan

    # Apply Gaussian filter to the NaN-filled interferogram
    ifg_filt = gaussian_filter_nan(ifg_nanned, sigma=sigma)

    # Restore original values in unmasked areas
    ifg_filt[~mask] = ifg[~mask]

    return ifg_filt
