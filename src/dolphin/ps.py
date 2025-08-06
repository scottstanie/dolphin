"""Find the persistent scatterers in a stack of SLCs.

This module now supports two dispersion metrics:

* **NAD** - Normalised **A**mplitude **D**ispersion (sigma / mu)
* **NMAD** - Normalised **M**edian **A**bsolute **D**eviation (MAD / median)

The NMAD implementation follows Brouwer & Hanssen (2025) and includes the
empirical mapping from NMAD to the expected single-look phase standard
deviation (Eq. 20) so that future processing stages can exploit it.
"""

from __future__ import annotations

import logging
import shutil
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from osgeo import gdal

from dolphin import io, utils
from dolphin._types import Filename
from dolphin.io import EagerLoader, StackReader, repack_raster

gdal.UseExceptions()

logger = logging.getLogger("dolphin")


class DispersionMetric(str, Enum):
    """Enum of supported dispersion metrics used for PS selection."""

    NAD = "nad"  #: Normalised Amplitude Dispersion (sigma/mu)
    NMAD = "nmad"  #: Normalised Median Absolute Deviation (MAD/median)

    @classmethod
    def from_any(cls, value: str | "DispersionMetric") -> "DispersionMetric":
        """Convert string or DispersionMetric to DispersionMetric enum."""
        if isinstance(value, cls):
            return value
        value = value.lower()
        if value in {cls.NAD.value, "nad", "amp_dispersion"}:
            return cls.NAD
        if value in {cls.NMAD.value, "nmad"}:
            return cls.NMAD
        raise ValueError(f"Unknown dispersion metric: {value}")


# Default nodata / dtype for the outputs - valid for both NAD & NMAD
NODATA_VALUES = {"ps": 255, "dispersion": 0.0, "amp_mean": 0.0}
FILE_DTYPES = {"ps": np.uint8, "dispersion": np.float32, "amp_mean": np.float32}

# Backwards compatibility
NODATA_VALUES["amp_dispersion"] = NODATA_VALUES["dispersion"]
FILE_DTYPES["amp_dispersion"] = FILE_DTYPES["dispersion"]
_EXTRA_COMPRESSION = {
    "keep_bits": 10,
    "predictor": 3,
}
REPACK_OPTIONS = {
    "ps": {},
    "dispersion": _EXTRA_COMPRESSION,
    "amp_mean": _EXTRA_COMPRESSION,
}
# Backwards compatibility
REPACK_OPTIONS["amp_dispersion"] = REPACK_OPTIONS["dispersion"]


def create_ps(
    *,
    reader: StackReader,
    output_file: Filename,
    output_amp_mean_file: Filename,
    output_dispersion_file: Filename,
    like_filename: Filename,
    dispersion_threshold: float = 0.25,
    dispersion_metric: str | DispersionMetric = DispersionMetric.NAD,
    existing_amp_mean_file: Optional[Filename] = None,
    existing_dispersion_file: Optional[Filename] = None,
    nodata_mask: Optional[np.ndarray] = None,
    update_existing: bool = False,
    block_shape: tuple[int, int] = (512, 512),
    # Backwards compatibility parameters
    output_amp_dispersion_file: Optional[Filename] = None,
    amp_dispersion_threshold: Optional[float] = None,
    existing_amp_dispersion_file: Optional[Filename] = None,
    **tqdm_kwargs,
):
    """Create the dispersion (NAD or NMAD), mean, and PS rasters.

    Parameters
    ----------
    reader
        Dataset reader for the 3-D SLC stack (complex64).
    output_file
        Path to the resulting persistent-scatterer (byte) mask.
    output_dispersion_file
        Raster containing either NAD (sigma/mu) or NMAD (MAD/median), depending on
        *dispersion_metric*.
    output_amp_mean_file
        Raster containing the mean (or median when *dispersion_metric* is
        NMAD) amplitude.
    like_filename
        Template raster for geotransform / projection.
    dispersion_threshold
        Threshold below which a pixel is labelled PS.  Empirically, the same
        value (≈ 0.25) works reasonably for NMAD too, but adjust as needed.
    dispersion_metric
        Either ``"nad"`` (default) or ``"nmad"``.
    existing_amp_mean_file, existing_dispersion_file
        If provided, these rasters are reused (optionally merged with new
        information when *update_existing* is ``True``).
    nodata_mask
        Boolean mask of valid pixels.  When provided, accelerates processing by
        skipping invalid blocks.
    update_existing
        Whether to merge existing rasters with the current SLC stack.  When
        ``False`` (default) the existing rasters are *only* used for PS mask
        creation.
    block_shape
        2-D block size (row, col) for chunked I/O.
    output_amp_dispersion_file, amp_dispersion_threshold, existing_amp_dispersion_file
        Backwards compatibility parameters. Use output_dispersion_file,
        dispersion_threshold, and existing_dispersion_file instead.
    **tqdm_kwargs
        Extra keyword args forwarded to :pyclass:`tqdm.tqdm` inside
        :class:`dolphin.io.EagerLoader`.

    """
    # Handle backwards compatibility
    if output_amp_dispersion_file is not None:
        output_dispersion_file = output_amp_dispersion_file
        warnings.warn(
            "output_amp_dispersion_file is deprecated, use output_dispersion_file",
            DeprecationWarning,
            stacklevel=2,
        )
    if amp_dispersion_threshold is not None:
        dispersion_threshold = amp_dispersion_threshold
        warnings.warn(
            "amp_dispersion_threshold is deprecated, use dispersion_threshold",
            DeprecationWarning,
            stacklevel=2,
        )
    if existing_amp_dispersion_file is not None:
        existing_dispersion_file = existing_amp_dispersion_file
        warnings.warn(
            "existing_amp_dispersion_file is deprecated, use existing_dispersion_file",
            DeprecationWarning,
            stacklevel=2,
        )

    metric = DispersionMetric.from_any(dispersion_metric)

    # Short-circuit if the user only wants a mask from existing rasters
    if existing_dispersion_file and existing_amp_mean_file and not update_existing:
        logger.info("Using existing dispersion rasters - skipping calculation.")
        _use_existing_files(
            existing_amp_mean_file=existing_amp_mean_file,
            existing_dispersion_file=existing_dispersion_file,
            output_file=output_file,
            output_amp_mean_file=output_amp_mean_file,
            output_dispersion_file=output_dispersion_file,
            dispersion_threshold=dispersion_threshold,
        )
        return

    # Otherwise, we need to calculate the PS files from the SLC stack
    # Initialize the output files with zeros
    file_list = [output_file, output_dispersion_file, output_amp_mean_file]
    for fn, dtype, nodata in zip(
        file_list, FILE_DTYPES.values(), NODATA_VALUES.values(), strict=False
    ):
        io.write_arr(
            arr=None,
            like_filename=like_filename,
            output_name=fn,
            nbands=1,
            dtype=dtype,
            nodata=nodata,
        )
    # Re-usable buffer for magnitudes of the current block
    magnitude = np.zeros((reader.shape[0], *block_shape), dtype=np.float32)
    writer = io.BackgroundBlockWriter()

    block_gen = EagerLoader(reader, block_shape=block_shape, nodata_mask=nodata_mask)
    for cur_data, (rows, cols) in block_gen.iter_blocks(**tqdm_kwargs):
        cur_rows, cur_cols = cur_data.shape[-2:]

        if np.all(cur_data == 0) or np.all(np.isnan(cur_data)):
            # Fill with nodata directly - avoids extra work
            ps = (
                np.ones((cur_rows, cur_cols), dtype=FILE_DTYPES["ps"])
                * NODATA_VALUES["ps"]
            )
            mean_or_med = np.full(
                (cur_rows, cur_cols),
                NODATA_VALUES["amp_mean"],
                dtype=FILE_DTYPES["amp_mean"],
            )
            disp = np.full(
                (cur_rows, cur_cols),
                NODATA_VALUES["dispersion"],
                dtype=FILE_DTYPES["dispersion"],
            )
        else:
            magnitude_cur = np.abs(cur_data, out=magnitude[:, :cur_rows, :cur_cols])
            mean_or_med, disp, ps = calc_ps_block(
                magnitude_cur,
                dispersion_threshold=dispersion_threshold,
                dispersion_metric=metric,
                min_count=len(magnitude_cur),
            )
            ps = ps.astype(FILE_DTYPES["ps"])
            ps[disp == 0] = NODATA_VALUES["ps"]

        # Queue writes
        writer.queue_write(mean_or_med, output_amp_mean_file, rows.start, cols.start)
        writer.queue_write(disp, output_dispersion_file, rows.start, cols.start)
        writer.queue_write(ps, output_file, rows.start, cols.start)

    logger.info("Waiting to flush %d queued blocks...", writer.num_queued)
    writer.notify_finished()

    # Repack for better compression
    logger.info("Repacking PS rasters for better compression...")
    for fn, opt in zip(file_list, REPACK_OPTIONS.values(), strict=False):
        repack_raster(Path(fn), output_dir=None, **opt)

    logger.info("Finished writing out PS rasters.")


def _mad(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median absolute deviation that tolerates *NaN*s."""
    median = np.nanmedian(arr, axis=axis)
    return np.nanmedian(np.abs(arr - np.expand_dims(median, axis=axis)), axis=axis)


def calc_ps_block(
    stack_mag: ArrayLike,
    *,
    dispersion_threshold: float = 0.25,
    dispersion_metric: DispersionMetric = DispersionMetric.NAD,
    min_count: Optional[int] = None,
    # Backwards compatibility
    amp_dispersion_threshold: Optional[float] = None,
):
    """Compute dispersion (NAD or NMAD) for one image block.

    The logic is shared for both metrics - we only vary the estimator.

    Returns
    -------
    central_tendency : np.ndarray
        Mean (for NAD) **or** median (for NMAD) amplitude.
    dispersion : np.ndarray
        NAD or NMAD according to *dispersion_metric*.
    ps_mask : np.ndarray[bool]
        Pixels whose dispersion is below *dispersion_threshold*.

    """
    # Handle backwards compatibility
    if amp_dispersion_threshold is not None:
        dispersion_threshold = amp_dispersion_threshold
        warnings.warn(
            "amp_dispersion_threshold is deprecated, use dispersion_threshold",
            DeprecationWarning,
            stacklevel=2,
        )

    if np.iscomplexobj(stack_mag):
        raise ValueError("`stack_mag` must be real-valued (magnitude of SLCs).")

    if min_count is None:
        min_count = int(0.9 * stack_mag.shape[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if dispersion_metric is DispersionMetric.NAD:
            central = np.nanmean(stack_mag, axis=0)
            disp = np.nanstd(stack_mag, axis=0) / central
        elif dispersion_metric is DispersionMetric.NMAD:
            central = np.nanmedian(stack_mag, axis=0)
            mad = _mad(stack_mag, axis=0)
            disp = mad / central  # Eq. 17
        else:  # should never occur
            raise RuntimeError("Unhandled dispersion metric: " + str(dispersion_metric))

        count = np.count_nonzero(~np.isnan(stack_mag), axis=0)

    # Mask out pixels lacking sufficient looks
    disp[count < min_count] = np.nan

    # Replace NaNs/infs with 0 ➜ nodata
    central = np.nan_to_num(central, nan=0, posinf=0, neginf=0, copy=False)
    disp = np.nan_to_num(disp, nan=0, posinf=0, neginf=0, copy=False)

    ps_mask = disp < dispersion_threshold
    ps_mask[disp == 0] = False

    # Optional: compute sigma_psi from NMAD (Eq. 20) and attach as attribute for
    # downstream processing.  Returned here as a side-effect via numpy array
    # meta - consumers can pick it up with `.sigma_psi`.
    if dispersion_metric is DispersionMetric.NMAD:
        sigma_psi = 1.3 * disp + 1.9 * disp**2 + 11.6 * disp**3  # Eq. 20

        # Convert to a simple numpy array with custom attribute
        class DispersionArray(np.ndarray):
            def __new__(cls, input_array, sigma_psi=None):
                obj = np.asarray(input_array).view(cls)
                obj.sigma_psi = sigma_psi
                return obj

        disp = DispersionArray(disp.astype(np.float32), sigma_psi=sigma_psi)

    return (
        central.astype(np.float32),
        disp if dispersion_metric is DispersionMetric.NMAD else disp.astype(np.float32),
        ps_mask,
    )


def _use_existing_files(
    *,
    existing_amp_mean_file: Filename,
    existing_dispersion_file: Filename,
    output_file: Filename,
    output_amp_mean_file: Filename,
    output_dispersion_file: Filename,
    dispersion_threshold: float,
) -> None:
    disp = io.load_gdal(existing_dispersion_file, masked=True)
    ps = (disp < dispersion_threshold).astype(np.uint8)

    # Encode nodata
    ps[(disp == 0) | disp.mask] = NODATA_VALUES["ps"]

    io.write_arr(
        arr=ps,
        like_filename=existing_dispersion_file,
        output_name=output_file,
        nodata=NODATA_VALUES["ps"],
    )

    # Shallow copy rasters (fast, as they are typically tiled & compressed)
    shutil.copy(existing_dispersion_file, output_dispersion_file)
    shutil.copy(existing_amp_mean_file, output_amp_mean_file)


def multilook_ps_files(
    strides: dict[str, int],
    ps_mask_file: Filename,
    dispersion_file: Filename,
    # Backwards compatibility
    amp_dispersion_file: Optional[Filename] = None,
) -> tuple[Path, Path]:
    """Create a multilooked version of the full-res PS mask/dispersion.

    Parameters
    ----------
    strides : dict[str, int]
        Decimation factor for 'x', 'y'
    ps_mask_file : Filename
        Name of input full-res uint8 PS mask file
    dispersion_file : Filename
        Name of input full-res float32 dispersion file
    amp_dispersion_file : Optional[Filename]
        Backwards compatibility parameter. Use dispersion_file instead.

    Returns
    -------
    output_ps_file : Path
        Multilooked PS mask file
        Will be same as `ps_mask_file`, but with "_looked" added before suffix.
    output_disp_file : Path
        Multilooked dispersion file
        Similar naming scheme to `output_ps_file`

    """
    # Handle backwards compatibility
    if amp_dispersion_file is not None:
        dispersion_file = amp_dispersion_file
        warnings.warn(
            "amp_dispersion_file is deprecated, use dispersion_file",
            DeprecationWarning,
            stacklevel=2,
        )

    if strides == {"x": 1, "y": 1}:
        logger.info("No striding request, skipping multilook.")
        return Path(ps_mask_file), Path(dispersion_file)
    full_cols, full_rows = io.get_raster_xysize(ps_mask_file)
    out_rows, out_cols = full_rows // strides["y"], full_cols // strides["x"]

    ps_suffix = Path(ps_mask_file).suffix
    ps_out_path = Path(str(ps_mask_file).replace(ps_suffix, f"_looked{ps_suffix}"))
    logger.info(f"Saving a looked PS mask to {ps_out_path}")

    if Path(ps_out_path).exists():
        logger.info(f"{ps_out_path} exists, skipping.")
    else:
        ps_mask = io.load_gdal(ps_mask_file, masked=True).astype(bool)
        ps_mask_looked = utils.take_looks(
            ps_mask, strides["y"], strides["x"], func_type="any", edge_strategy="pad"
        )
        # make sure it's the same size as the MLE result/temp_coh after padding
        ps_mask_looked = ps_mask_looked[:out_rows, :out_cols]
        ps_mask_looked = ps_mask_looked.astype("uint8").filled(NODATA_VALUES["ps"])
        io.write_arr(
            arr=ps_mask_looked,
            like_filename=ps_mask_file,
            output_name=ps_out_path,
            strides=strides,
            nodata=NODATA_VALUES["ps"],
        )

    disp_suffix = Path(dispersion_file).suffix
    disp_out_path = Path(
        str(dispersion_file).replace(disp_suffix, f"_looked{disp_suffix}")
    )
    if disp_out_path.exists():
        logger.info(f"{disp_out_path} exists, skipping.")
    else:
        disp = io.load_gdal(dispersion_file, masked=True)
        # We use `nanmin` assuming that the multilooked PS is using
        # the strongest PS (the one with the lowest dispersion)
        disp_looked = utils.take_looks(
            disp,
            strides["y"],
            strides["x"],
            func_type="nanmin",
            edge_strategy="pad",
        )
        disp_looked = disp_looked[:out_rows, :out_cols]
        disp_looked = disp_looked.filled(NODATA_VALUES["dispersion"])
        io.write_arr(
            arr=disp_looked,
            like_filename=dispersion_file,
            output_name=disp_out_path,
            strides=strides,
            nodata=NODATA_VALUES["dispersion"],
        )
    return ps_out_path, disp_out_path


def combine_means(means: ArrayLike, N: ArrayLike) -> np.ndarray:
    r"""Compute the combined mean from multiple `mu_i` values.

    This function calculates the weighted average of amplitudes based on the
    number of original data points (N) that went into each mean.

    Parameters
    ----------
    means : ArrayLike
        A 3D array of mean values.
        Shape: (n_images, rows, cols)
    N : np.ndarray
        A list/array of weights indicating the number of original images.
        Shape: (depth,)

    Returns
    -------
    np.ndarray
        The combined mean.
        Shape: (height, width)

    Notes
    -----
    Both input arrays are expected to have the same shape.
    The operation is performed along axis=0.

    The combined mean is calculated as

    \begin{equation}
        E[X] = \frac{\sum_i N_i\mu_i}{\sum_i N_i}
    \end{equation}

    """
    N = np.asarray(N)
    if N.shape[0] != means.shape[0]:
        raise ValueError("Size of N must match the number of images in means.")
    if N.ndim == 1:
        N = N[:, None, None]

    weighted_sum = np.sum(means * N, axis=0)
    total_N = np.sum(N, axis=0)

    return weighted_sum / total_N


def combine_amplitude_dispersions(
    dispersions: np.ndarray, means: np.ndarray, N: ArrayLike | Sequence
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the combined amplitude dispersion from multiple groups.

    Given several ADs where difference numbers of images, N, went in,
    the function computes a weighted mean/variance to calculate the combined AD.

    Parameters
    ----------
    dispersions : np.ndarray
        A 3D array of amplitude dispersion values for each group.
        Shape: (depth, height, width)
    means : np.ndarray
        A 3D array of mean values for each group.
        Shape: (depth, height, width)
    N : np.ndarray
        An array sample sizes for each group.
        Shape: (depth, )

    Returns
    -------
    np.ndarray
        The combined amplitude dispersion.
        Shape: (height, width)
    np.ndarray
        The combined amplitude mean.
        Shape: (height, width)

    Notes
    -----
    All input arrays are expected to have the same shape.
    The operation is performed along `axis=0`.

    Let $X_i$ be the random variable for group $i$, with mean $\mu_i$ and variance
    $\sigma_i^2$, and $N_i$ be the number of samples in group $i$.

    The combined variance $\sigma^2$ uses the formula

    \begin{equation}
        \sigma^2 = E[X^2] - (E[X])^2
    \end{equation}

    where $E[X]$ is the combined mean, and $E[X^2]$ is the expected value of
    the squared random variable.

    The combined mean is calculated as:

    \begin{equation}
        E[X] = \frac{\sum_i N_i\mu_i}{\sum_i N_i}
    \end{equation}

    For $E[X^2]$, we use the property $E[X^2] = \sigma^2 + \mu^2$:

    \begin{equation}
        E[X^2] = \frac{\sum_i N_i(\sigma_i^2 + \mu_i^2)}{\sum_i N_i}
    \end{equation}

    Substituting these into the variance formula gives:

    \begin{equation}
        \sigma^2 = \frac{\sum_i N_i(\sigma_i^2 + \mu_i^2)}{\sum_i N_i} -
        \left(\frac{\sum_i N_i\mu_i}{\sum_i N_i}\right)^2
    \end{equation}

    """
    N = np.asarray(N)
    if N.ndim == 1:
        N = N[:, None, None]
    if not (means.shape == dispersions.shape):
        raise ValueError("Input arrays must have the same shape.")
    if means.shape[0] != N.shape[0]:
        raise ValueError("Size of N must match the number of groups in means.")

    combined_mean = combine_means(means, N)

    # Compute combined variance
    variances = (dispersions * means) ** 2
    total_N = np.sum(N, axis=0).squeeze()
    sum_N_var_meansq = np.sum(N * (variances + means**2), axis=0)
    combined_variance = (sum_N_var_meansq / total_N) - combined_mean**2

    return np.sqrt(combined_variance) / combined_mean, combined_mean
