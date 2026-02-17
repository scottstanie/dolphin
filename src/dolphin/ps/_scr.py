"""Signal-to-Clutter Ratio (SCR) estimation for PS selection.

Implements the SCR-based PS selection method described in:
- Agram & Simons (2015), "Efficient Persistent Scatterer Identification using
  Signal-to-Clutter Ratio", IEEE GRSL.
- Agram (2010), "Persistent Scatterer Interferometry in Natural Terrain", PhD thesis.

The SCR is estimated from the phase residues of interferograms formed from an
SLC stack. The phase residue is computed by subtracting the local average phase
(via a boxcar filter) from the original interferometric phase. The SCR is then
estimated via maximum likelihood over a grid of candidate SCR values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike
from osgeo import gdal
from scipy.ndimage import uniform_filter
from scipy.special import erfc

from dolphin import io
from dolphin._types import Filename
from dolphin.io import EagerLoader, StackReader, repack_raster

from ._amp_dispersion import FILE_DTYPES, NODATA_VALUES, REPACK_OPTIONS, calc_ps_block

gdal.UseExceptions()

logger = logging.getLogger("dolphin")

SCR_NODATA = 0.0
SCR_DTYPE = np.float32
_SCR_REPACK_OPTIONS = {"keep_bits": 10, "predictor": 3}


def create_scr(
    *,
    reader: StackReader,
    output_file: Filename,
    like_filename: Filename,
    window_size: int = 11,
    model: Literal["constant", "gaussian"] = "constant",
    nodata_mask: Optional[np.ndarray] = None,
    block_shape: tuple[int, int] = (512, 512),
    **tqdm_kwargs,
) -> None:
    """Create a signal-to-clutter ratio file from an SLC stack.

    Parameters
    ----------
    reader : StackReader
        A dataset reader for the 3D SLC stack.
    output_file : Filename
        The output SCR file (dtype: float32).
    like_filename : Filename
        The filename to use for the output file's spatial reference.
    window_size : int, optional
        Box-car filter window size for computing phase residues. Default is 11.
    model : {"constant", "gaussian"}, optional
        Phase distribution model for SCR estimation. Default is "constant".
    nodata_mask : Optional[np.ndarray]
        If provided, skips computing SCR over areas where the mask is False.
    block_shape : tuple[int, int], optional
        The 2D block size to load all bands at a time. Default is (512, 512).
    **tqdm_kwargs : optional
        Arguments to pass to `tqdm`.

    """
    io.write_arr(
        arr=None,
        like_filename=like_filename,
        output_name=output_file,
        nbands=1,
        dtype=SCR_DTYPE,
        nodata=SCR_NODATA,
    )

    writer = io.BackgroundBlockWriter()
    block_gen = EagerLoader(reader, block_shape=block_shape, nodata_mask=nodata_mask)
    for cur_data, (rows, cols) in block_gen.iter_blocks(**tqdm_kwargs):
        cur_rows, cur_cols = cur_data.shape[-2:]

        if cur_data.shape[0] < 2:
            # Need at least 2 SLCs to form an interferogram
            scr_block = np.full((cur_rows, cur_cols), SCR_NODATA, dtype=SCR_DTYPE)
        elif not (np.all(cur_data == 0) or np.all(np.isnan(cur_data))):
            scr_block = calc_scr_block(
                cur_data,
                window_size=window_size,
                model=model,
            )
        else:
            scr_block = np.full((cur_rows, cur_cols), SCR_NODATA, dtype=SCR_DTYPE)

        writer.queue_write(scr_block, output_file, rows.start, cols.start)

    logger.info(f"Waiting to write {writer.num_queued} blocks of SCR data.")
    writer.notify_finished()
    # Repack for better compression
    repack_raster(Path(output_file), output_dir=None, **_SCR_REPACK_OPTIONS)
    logger.info("Finished writing SCR file")


def calc_scr_block(
    slc_block: ArrayLike,
    window_size: int = 11,
    model: Literal["constant", "gaussian"] = "constant",
) -> np.ndarray:
    """Calculate signal-to-clutter ratio for a block of SLC data.

    Forms consecutive interferograms from the SLC stack, computes phase
    residues by subtracting a boxcar-filtered local average, then estimates
    SCR per pixel via maximum likelihood.

    Parameters
    ----------
    slc_block : ArrayLike
        Complex SLC data, shape (n_slc, rows, cols).
    window_size : int, optional
        Box-car filter window size for local phase estimation. Default is 11.
    model : {"constant", "gaussian"}, optional
        Phase distribution model for SCR estimation. Default is "constant".

    Returns
    -------
    scr : np.ndarray
        Signal-to-clutter ratio per pixel, shape (rows, cols), dtype float32.

    Notes
    -----
    Edge pixels (within `window_size // 2` of the block boundary) may have
    slightly less accurate SCR estimates due to the boxcar filter edge handling.

    """
    slc_block = np.asarray(slc_block)
    if slc_block.ndim != 3:
        msg = f"Expected 3D SLC block (n_slc, rows, cols), got shape {slc_block.shape}"
        raise ValueError(msg)

    n_slc, _nrow, _ncol = slc_block.shape
    if n_slc < 2:
        msg = "Need at least 2 SLCs to compute SCR"
        raise ValueError(msg)

    # Compute phase residues from consecutive interferograms
    phase_residues = _compute_phase_residues(slc_block, window_size)

    # Remove mean phase bias across interferograms
    cmean = np.mean(np.exp(1j * phase_residues), axis=0)
    phase_residues = np.angle(np.exp(1j * phase_residues) * np.conj(cmean)[np.newaxis])

    # Estimate SCR per pixel via MLE
    scr = _estimate_scr_mle(phase_residues, model=model)
    return scr.astype(np.float32)


def _compute_phase_residues(
    slc_block: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Compute interferometric phase residues from an SLC stack.

    Forms consecutive interferograms and subtracts the local average phase
    (estimated via boxcar filtering) to obtain phase residues.

    Parameters
    ----------
    slc_block : np.ndarray
        Complex SLC data, shape (n_slc, rows, cols).
    window_size : int
        Box-car filter window size.

    Returns
    -------
    phase_residues : np.ndarray
        Phase residues, shape (n_ifg, rows, cols), dtype float32.

    """
    n_slc, nrow, ncol = slc_block.shape
    n_ifg = n_slc - 1

    phase_residues = np.empty((n_ifg, nrow, ncol), dtype=np.float32)
    for i in range(n_ifg):
        ifg = slc_block[i + 1] * np.conj(slc_block[i])

        # Boxcar filter on real and imaginary parts separately
        filtered_real = uniform_filter(ifg.real.astype(np.float64), size=window_size)
        filtered_imag = uniform_filter(ifg.imag.astype(np.float64), size=window_size)
        ifg_filtered = filtered_real + 1j * filtered_imag

        # Phase residue: subtract filtered phase from original
        amp_filtered = np.abs(ifg_filtered)
        residue = ifg * np.conj(ifg_filtered) / (1e-5 + amp_filtered)
        phase_residues[i] = np.angle(residue)

    return phase_residues


def _estimate_scr_mle(
    phase_residues: np.ndarray,
    model: Literal["constant", "gaussian"] = "constant",
) -> np.ndarray:
    """Estimate SCR per pixel via maximum likelihood estimation.

    Tests a grid of candidate SCR values and selects the one that maximizes
    the log-likelihood of the observed phase residues.

    Parameters
    ----------
    phase_residues : np.ndarray
        Phase residues, shape (n_ifg, rows, cols) or (n_ifg, n_pixels).
    model : {"constant", "gaussian"}, optional
        Phase distribution model. Default is "constant".

    Returns
    -------
    scr : np.ndarray
        Estimated SCR values.

    """
    # Define candidate SCR grid
    rho = np.linspace(0.0, 0.99, 50)
    scr_candidates = rho / (1 - rho)

    if phase_residues.ndim == 3:
        n_ifg, nrow, ncol = phase_residues.shape
        phi = phase_residues.reshape(n_ifg, -1)
    else:
        phi = phase_residues

    n_pixels = phi.shape[1]

    # Pre-compute the PDF lookup table for the constant model
    if model == "constant":
        pdf_lookup = _build_constant_pdf_lookup(scr_candidates)

    log_likelihood = np.full((len(scr_candidates), n_pixels), -np.inf, dtype=np.float64)
    for i, scr_val in enumerate(scr_candidates):
        if model == "gaussian":
            p = _phase_pdf_gaussian(scr_val, phi)
        elif model == "constant":
            p = _phase_pdf_constant_lookup(phi, pdf_lookup[i])
        else:
            msg = f"Unknown model: {model!r}. Use 'constant' or 'gaussian'."
            raise ValueError(msg)

        # Sum log-likelihood across interferograms
        log_likelihood[i] = np.sum(np.log(np.maximum(p, 1e-30)), axis=0)

    best_idx = np.argmax(log_likelihood, axis=0)
    scr = scr_candidates[best_idx]

    if phase_residues.ndim == 3:
        scr = scr.reshape(nrow, ncol)
    return scr


def _phase_pdf_gaussian(gamma: float, phi: np.ndarray) -> np.ndarray:
    r"""PDF of interferometric phase assuming Gaussian signal and noise.

    Parameters
    ----------
    gamma : float
        Signal-to-clutter ratio.
    phi : np.ndarray
        Interferometric phase values.

    Returns
    -------
    np.ndarray
        Probability density at each phase value.

    Notes
    -----
    The PDF is:

    .. math::

        f(\phi | \gamma) = \frac{1-\rho^2}{2\pi(1-\beta^2)}
        \left(1 + \frac{\beta \arccos(-\beta)}{\sqrt{1-\beta^2}}\right)

    where :math:`\rho = \gamma / (1+\gamma)` and :math:`\beta = \rho \cos(\phi)`.

    """
    rho = gamma / (1 + gamma)
    beta = rho * np.cos(phi)
    f = (
        (1 - rho**2)
        / (2 * np.pi * (1 - beta**2))
        * (1 + beta * np.arccos(-beta) / np.sqrt(1 - beta**2))
    )
    return f


def _phase_pdf_single_look(gamma: float, theta: np.ndarray) -> np.ndarray:
    r"""PDF of SAR phase for a constant signal with Gaussian noise.

    Parameters
    ----------
    gamma : float
        Signal-to-clutter ratio.
    theta : np.ndarray
        SAR phase values.

    Returns
    -------
    np.ndarray
        Probability density at each phase value.

    Notes
    -----
    The PDF is from Agram (2015):

    .. math::

        p(\theta | \gamma) = \frac{1}{2\pi} e^{-\gamma \sin^2\theta}
        \left( e^{-\gamma \cos^2\theta}
        + \sqrt{\pi \gamma} \cos\theta \, \mathrm{erfc}(-\sqrt{\gamma}\cos\theta)
        \right)

    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    sqrt_gamma = np.sqrt(gamma)

    p = (
        1.0
        / (2 * np.pi)
        * np.exp(-gamma * sin_theta**2)
        * (
            np.exp(-gamma * cos_theta**2)
            + np.sqrt(np.pi * gamma) * cos_theta * erfc(-sqrt_gamma * cos_theta)
        )
    )
    return p


def _build_constant_pdf_lookup(
    scr_candidates: np.ndarray,
    n_phase_bins: int = 100,
) -> list[np.ndarray]:
    """Pre-compute PDF lookup tables for the constant signal model.

    For each candidate SCR, numerically integrates the joint phase distribution
    and stores a lookup table indexed by phase.

    Parameters
    ----------
    scr_candidates : np.ndarray
        Array of candidate SCR values.
    n_phase_bins : int, optional
        Number of phase bins for the lookup table. Default is 100.

    Returns
    -------
    list[np.ndarray]
        List of lookup tables, one per candidate SCR.

    """
    lookup_tables = []
    phi_test = np.linspace(-np.pi, np.pi, n_phase_bins + 1)

    for scr_val in scr_candidates:
        # Numerically integrate the joint phase distribution
        lut = _int_phase_pdf_constant(scr_val, phi_test, n_phase_bins)
        lookup_tables.append(lut)

    return lookup_tables


def _int_phase_pdf_constant(
    gamma: float,
    phi: np.ndarray,
    n_integration: int = 100,
) -> np.ndarray:
    """Compute interferometric phase PDF for constant signal model.

    Numerically integrates the joint phase distribution over the sum phase.

    Parameters
    ----------
    gamma : float
        Signal-to-clutter ratio.
    phi : np.ndarray
        Phase values at which to evaluate the PDF.
    n_integration : int
        Number of integration points.

    Returns
    -------
    np.ndarray
        PDF values at each phase value.

    """
    f = np.zeros_like(phi, dtype=np.float64)
    for i in range(n_integration):
        phi_sum = 2 * i * np.pi / n_integration - np.pi
        # Joint phase distribution for constant signal
        theta1 = (phi + phi_sum) / 2
        theta2 = (phi_sum - phi) / 2

        p1 = _phase_pdf_single_look(gamma, theta1)
        p2 = _phase_pdf_single_look(gamma, theta2)
        p1_shifted = _phase_pdf_single_look(gamma, theta1 + np.pi)
        p2_shifted = _phase_pdf_single_look(gamma, theta2 + np.pi)

        joint = 0.5 * (p1 * p2 + p1_shifted * p2_shifted)
        f += (2 * np.pi / n_integration) * joint

    return f


def _phase_pdf_constant_lookup(
    phi: np.ndarray,
    lookup_table: np.ndarray,
) -> np.ndarray:
    """Evaluate PDF using a pre-computed lookup table.

    Parameters
    ----------
    phi : np.ndarray
        Phase values at which to evaluate the PDF.
    lookup_table : np.ndarray
        Pre-computed PDF lookup table.

    Returns
    -------
    np.ndarray
        PDF values at each phase value.

    """
    n_bins = len(lookup_table) - 1
    idx = np.round((phi + np.pi) / (2 * np.pi / n_bins)).astype(int)
    idx = np.clip(idx, 0, n_bins)
    return lookup_table[idx]


def create_ps_scr(
    *,
    reader: StackReader,
    output_file: Filename,
    output_amp_mean_file: Filename,
    output_amp_dispersion_file: Filename,
    output_scr_file: Filename,
    like_filename: Filename,
    scr_threshold: float = 2.0,
    window_size: int = 11,
    model: Literal["constant", "gaussian"] = "constant",
    nodata_mask: Optional[np.ndarray] = None,
    block_shape: tuple[int, int] = (512, 512),
    **tqdm_kwargs,
) -> None:
    """Create PS mask using signal-to-clutter ratio thresholding.

    Performs a single pass through the SLC stack, computing both amplitude
    statistics and SCR per block. The PS mask is determined by thresholding
    the SCR values rather than amplitude dispersion.

    Parameters
    ----------
    reader : StackReader
        A dataset reader for the 3D SLC stack.
    output_file : Filename
        The output PS file (dtype: Byte).
    output_amp_mean_file : Filename
        The output mean amplitude file.
    output_amp_dispersion_file : Filename
        The output amplitude dispersion file.
    output_scr_file : Filename
        The output SCR file (dtype: float32).
    like_filename : Filename
        The filename to use for the output files' spatial reference.
    scr_threshold : float, optional
        SCR threshold to consider a pixel a PS. Default is 2.0.
    window_size : int, optional
        Box-car filter window size for computing phase residues. Default is 11.
    model : {"constant", "gaussian"}, optional
        Phase distribution model for SCR estimation. Default is "constant".
    nodata_mask : Optional[np.ndarray]
        If provided, skips computing over areas where the mask is False.
    block_shape : tuple[int, int], optional
        The 2D block size to load all bands at a time. Default is (512, 512).
    **tqdm_kwargs : optional
        Arguments to pass to `tqdm`.

    """
    # Initialize output files
    output_info = {
        "ps": (output_file, FILE_DTYPES["ps"], NODATA_VALUES["ps"]),
        "amp_dispersion": (
            output_amp_dispersion_file,
            FILE_DTYPES["amp_dispersion"],
            NODATA_VALUES["amp_dispersion"],
        ),
        "amp_mean": (
            output_amp_mean_file,
            FILE_DTYPES["amp_mean"],
            NODATA_VALUES["amp_mean"],
        ),
        "scr": (output_scr_file, SCR_DTYPE, SCR_NODATA),
    }
    for fn, dtype, nodata in output_info.values():
        io.write_arr(
            arr=None,
            like_filename=like_filename,
            output_name=fn,
            nbands=1,
            dtype=dtype,
            nodata=nodata,
        )

    magnitude = np.zeros((reader.shape[0], *block_shape), dtype=np.float32)

    writer = io.BackgroundBlockWriter()
    block_gen = EagerLoader(reader, block_shape=block_shape, nodata_mask=nodata_mask)
    for cur_data, (rows, cols) in block_gen.iter_blocks(**tqdm_kwargs):
        cur_rows, cur_cols = cur_data.shape[-2:]

        is_all_nodata = np.all(cur_data == 0) or np.all(np.isnan(cur_data))
        if not is_all_nodata and cur_data.shape[0] >= 2:
            # Amplitude statistics (reuse existing calc_ps_block)
            magnitude_cur = np.abs(cur_data, out=magnitude[:, :cur_rows, :cur_cols])
            mean, amp_disp, _ = calc_ps_block(
                magnitude_cur,
                amp_dispersion_threshold=0.25,  # not used for PS decision
                min_count=len(magnitude_cur),
            )

            # SCR estimation (reuse existing calc_scr_block)
            scr_block = calc_scr_block(cur_data, window_size=window_size, model=model)

            # PS from SCR threshold
            ps = (scr_block > scr_threshold).astype(FILE_DTYPES["ps"])
            ps[scr_block == SCR_NODATA] = NODATA_VALUES["ps"]
        else:
            ps = np.full(
                (cur_rows, cur_cols), NODATA_VALUES["ps"], dtype=FILE_DTYPES["ps"]
            )
            mean = np.full(
                (cur_rows, cur_cols),
                NODATA_VALUES["amp_mean"],
                dtype=FILE_DTYPES["amp_mean"],
            )
            amp_disp = np.full(
                (cur_rows, cur_cols),
                NODATA_VALUES["amp_dispersion"],
                dtype=FILE_DTYPES["amp_dispersion"],
            )
            scr_block = np.full((cur_rows, cur_cols), SCR_NODATA, dtype=SCR_DTYPE)

        writer.queue_write(mean, output_amp_mean_file, rows.start, cols.start)
        writer.queue_write(amp_disp, output_amp_dispersion_file, rows.start, cols.start)
        writer.queue_write(scr_block, output_scr_file, rows.start, cols.start)
        writer.queue_write(ps, output_file, rows.start, cols.start)

    logger.info(f"Waiting to write {writer.num_queued} blocks of data.")
    writer.notify_finished()

    # Repack for better compression
    logger.info("Repacking PS/SCR rasters for better compression")
    file_list = [output_file, output_amp_dispersion_file, output_amp_mean_file]
    for fn, opt in zip(file_list, REPACK_OPTIONS.values(), strict=False):
        repack_raster(Path(fn), output_dir=None, **opt)
    repack_raster(Path(output_scr_file), output_dir=None, **_SCR_REPACK_OPTIONS)
    logger.info("Finished writing PS files (SCR method)")
