"""Signal-to-Clutter Ratio (SCR) estimation for PS selection.

Implements the SCR-based PS selection method described in:
- Shanker & Zebker (2007), "Persistent scatterer selection using maximum
  likelihood estimation", GRL, doi:10.1029/2007GL030806.
- Agram & Simons (2015), "Efficient Persistent Scatterer Identification using
  Signal-to-Clutter Ratio", IEEE GRSL.

Single-master interferograms are formed from the SLC stack (using a configurable
reference SLC, default middle of stack). Each interferogram is spatially filtered
(boxcar) to estimate and remove the correlated phase (atmosphere, deformation,
orbit, DEM errors). The residual phase at each pixel follows a distribution
parameterized by the SCR, which is estimated via maximum likelihood over a grid
of candidate values.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from jax.scipy.special import erfc as _jax_erfc
from numpy.typing import ArrayLike
from osgeo import gdal

from dolphin import io
from dolphin._types import Filename
from dolphin.io import EagerLoader, StackReader, repack_raster

from ._amp_dispersion import FILE_DTYPES, NODATA_VALUES, REPACK_OPTIONS, calc_ps_block

gdal.UseExceptions()

logger = logging.getLogger("dolphin")

SCR_NODATA = 0.0
SCR_DTYPE = np.float32
_SCR_REPACK_OPTIONS = {"keep_bits": 10, "predictor": 3}

# Number of candidate SCR values tested in the MLE grid search
_N_SCR_CANDIDATES = 50
# Number of phase bins for the constant-model lookup table
_N_PHASE_BINS = 100


# ---------------------------------------------------------------------------
# JAX helper functions (not individually JIT'd; composed into _calc_scr_jax)
# ---------------------------------------------------------------------------


def _boxcar_filter_2d(arr, window_size):
    """2D uniform (boxcar) filter via ``lax.reduce_window``."""
    half = window_size // 2
    padding = [(half, half), (half, half)]
    dims = (window_size, window_size)
    strides = (1, 1)
    out = lax.reduce_window(arr, jnp.float32(0), lax.add, dims, strides, padding)
    return out / (window_size * window_size)


def _compute_phase_residues(slc_block, reference_idx, window_size):
    """Compute phase residues from single-master interferograms.

    Forms interferograms between each SLC and the reference SLC, spatially
    filters each interferogram with a boxcar, then extracts phase residues.
    """
    master = slc_block[reference_idx]
    ifgs = slc_block * jnp.conj(master)[None]

    def _residue_one(ifg):
        filtered_real = _boxcar_filter_2d(ifg.real, window_size)
        filtered_imag = _boxcar_filter_2d(ifg.imag, window_size)
        ifg_filtered = filtered_real + 1j * filtered_imag
        amp_filtered = jnp.abs(ifg_filtered)
        residue = ifg * jnp.conj(ifg_filtered) / (1e-5 + amp_filtered)
        return jnp.angle(residue)

    all_residues = vmap(_residue_one)(ifgs)
    # Remove the self-interferogram at reference_idx
    return jnp.concatenate(
        [all_residues[:reference_idx], all_residues[reference_idx + 1 :]], axis=0
    )


def _phase_pdf_gaussian(gamma, phi):
    r"""PDF of interferometric phase (Gaussian model).

    Shanker & Zebker (2007), Eq. 1.

    .. math::

        f(\phi | \gamma) = \frac{1-\rho^2}{2\pi(1-\beta^2)}
        \left(1 + \frac{\beta \arccos(-\beta)}{\sqrt{1-\beta^2}}\right)

    where :math:`\rho = \gamma / (1+\gamma)` and :math:`\beta = \rho \cos\phi`.
    """
    rho = gamma / (1 + gamma)
    beta = rho * jnp.cos(phi)
    return (
        (1 - rho**2)
        / (2 * jnp.pi * (1 - beta**2))
        * (1 + beta * jnp.arccos(-beta) / jnp.sqrt(1 - beta**2))
    )


def _phase_pdf_single_look(gamma, theta):
    r"""PDF of SAR phase for a constant signal with Gaussian noise.

    Agram & Simons (2015).
    """
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    sqrt_g = jnp.sqrt(gamma)
    return (
        1.0
        / (2 * jnp.pi)
        * jnp.exp(-gamma * sin_t**2)
        * (
            jnp.exp(-gamma * cos_t**2)
            + jnp.sqrt(jnp.pi * gamma) * cos_t * _jax_erfc(-sqrt_g * cos_t)
        )
    )


def _int_phase_pdf_constant(gamma, phi, n_integration):
    """Numerically integrate joint phase PDF for the constant-signal model."""
    i_vals = jnp.arange(n_integration)
    phi_sums = 2 * i_vals * jnp.pi / n_integration - jnp.pi
    # (n_integration, n_bins)
    theta1 = (phi[None, :] + phi_sums[:, None]) / 2
    theta2 = (phi_sums[:, None] - phi[None, :]) / 2

    p1 = _phase_pdf_single_look(gamma, theta1)
    p2 = _phase_pdf_single_look(gamma, theta2)
    p1s = _phase_pdf_single_look(gamma, theta1 + jnp.pi)
    p2s = _phase_pdf_single_look(gamma, theta2 + jnp.pi)
    joint = 0.5 * (p1 * p2 + p1s * p2s)
    return jnp.sum(joint, axis=0) * (2 * jnp.pi / n_integration)


def _estimate_scr(phase_residues, model):
    """Estimate SCR per pixel from phase residues.

    For ``model="coherence"``, uses the method-of-moments estimator
    (temporal coherence magnitude mapped to SCR). For ``"gaussian"``
    or ``"constant"``, performs a maximum-likelihood grid search over
    candidate SCR values.
    """
    nrow, ncol = phase_residues.shape[1], phase_residues.shape[2]
    phi = phase_residues.reshape(phase_residues.shape[0], -1)

    if model == "coherence":
        # Method-of-moments: temporal coherence → SCR
        phasors = jnp.exp(1j * phi)
        coherence = jnp.abs(jnp.mean(phasors, axis=0))
        scr = coherence / jnp.maximum(1 - coherence, 1e-6)
        return scr.reshape(nrow, ncol)

    # MLE grid search
    rho = jnp.linspace(0.0, 0.99, _N_SCR_CANDIDATES)
    scr_candidates = rho / (1 - rho)

    if model == "gaussian":

        def _ll_gaussian(scr_val):
            p = _phase_pdf_gaussian(scr_val, phi)
            return jnp.sum(jnp.log(jnp.maximum(p, 1e-30)), axis=0)

        all_ll = vmap(_ll_gaussian)(scr_candidates)
    else:
        # Build lookup tables for the constant-signal model
        phi_test = jnp.linspace(-jnp.pi, jnp.pi, _N_PHASE_BINS + 1)
        lookup = vmap(lambda g: _int_phase_pdf_constant(g, phi_test, _N_PHASE_BINS))(
            scr_candidates
        )
        idx = jnp.round((phi + jnp.pi) / (2 * jnp.pi / _N_PHASE_BINS)).astype(jnp.int32)
        idx = jnp.clip(idx, 0, _N_PHASE_BINS)

        def _ll_constant(lut):
            p = lut[idx]
            return jnp.sum(jnp.log(jnp.maximum(p, 1e-30)), axis=0)

        all_ll = vmap(_ll_constant)(lookup)

    best_idx = jnp.argmax(all_ll, axis=0)
    return scr_candidates[best_idx].reshape(nrow, ncol)


# ---------------------------------------------------------------------------
# JIT-compiled entry point (one per model due to static_argnames)
# ---------------------------------------------------------------------------


@partial(jit, static_argnames=["reference_idx", "window_size", "model"])
def _calc_scr_jax(slc_block, reference_idx, window_size, model):
    """JIT-compiled SCR computation for a single spatial block."""
    phase_residues = _compute_phase_residues(slc_block, reference_idx, window_size)
    return _estimate_scr(phase_residues, model)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_scr(
    *,
    reader: StackReader,
    output_file: Filename,
    like_filename: Filename,
    window_size: int = 11,
    model: Literal["constant", "gaussian", "coherence"] = "gaussian",
    reference_idx: int | None = None,
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
        Phase distribution model for SCR estimation. Default is "gaussian".
    reference_idx : int or None, optional
        Index of the reference SLC for single-master interferogram formation.
        If None, uses N // 2 (middle of stack). Default is None.
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
            scr_block = np.full((cur_rows, cur_cols), SCR_NODATA, dtype=SCR_DTYPE)
        elif not (np.all(cur_data == 0) or np.all(np.isnan(cur_data))):
            scr_block = calc_scr_block(
                cur_data,
                window_size=window_size,
                model=model,
                reference_idx=reference_idx,
            )
        else:
            scr_block = np.full((cur_rows, cur_cols), SCR_NODATA, dtype=SCR_DTYPE)

        writer.queue_write(scr_block, output_file, rows.start, cols.start)

    logger.info(f"Waiting to write {writer.num_queued} blocks of SCR data.")
    writer.notify_finished()
    repack_raster(Path(output_file), output_dir=None, **_SCR_REPACK_OPTIONS)
    logger.info("Finished writing SCR file")


def calc_scr_block(
    slc_block: ArrayLike,
    window_size: int = 11,
    model: Literal["constant", "gaussian", "coherence"] = "gaussian",
    reference_idx: int | None = None,
) -> np.ndarray:
    """Calculate signal-to-clutter ratio for a block of SLC data.

    Forms single-master interferograms, computes phase residues by subtracting
    a boxcar-filtered local average, then estimates SCR per pixel via MLE.

    Parameters
    ----------
    slc_block : ArrayLike
        Complex SLC data, shape (n_slc, rows, cols).
    window_size : int, optional
        Box-car filter window size for local phase estimation. Default is 11.
    model : {"constant", "gaussian"}, optional
        Phase distribution model for SCR estimation. Default is "gaussian"
        (Shanker & Zebker, 2007, Eq. 1).
    reference_idx : int or None, optional
        Index of the reference SLC. If None, uses N // 2. Default is None.

    Returns
    -------
    scr : np.ndarray
        Signal-to-clutter ratio per pixel, shape (rows, cols), dtype float32.

    """
    slc_block = np.asarray(slc_block)
    assert (
        slc_block.ndim == 3
    ), f"Expected 3D SLC block (n_slc, rows, cols), got shape {slc_block.shape}"
    n_slc = slc_block.shape[0]
    assert n_slc >= 2, "Need at least 2 SLCs to compute SCR"

    if reference_idx is None:
        reference_idx = n_slc // 2

    slc_jax = jnp.asarray(slc_block, dtype=jnp.complex64)
    scr = _calc_scr_jax(slc_jax, reference_idx, window_size, model)
    return np.asarray(scr, dtype=np.float32)


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
    model: Literal["constant", "gaussian", "coherence"] = "gaussian",
    reference_idx: int | None = None,
    min_count: int | None = None,
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
        Phase distribution model for SCR estimation. Default is "gaussian".
    reference_idx : int or None, optional
        Index of the reference SLC. If None, uses N // 2. Default is None.
    min_count : int or None, optional
        Minimum number of valid (non-zero) SLCs required per pixel. Pixels
        with fewer valid acquisitions have their SCR set to nodata.
        Default is ``int(0.9 * n_slc)``.
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

    n_slc = reader.shape[0]
    if min_count is None:
        min_count = int(0.9 * n_slc)

    magnitude = np.zeros((n_slc, *block_shape), dtype=np.float32)

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
                min_count=min_count,
            )

            # SCR estimation
            scr_block = calc_scr_block(
                cur_data,
                window_size=window_size,
                model=model,
                reference_idx=reference_idx,
            )

            # Mask pixels with too few valid SLCs
            count = np.count_nonzero(magnitude_cur > 0, axis=0)
            insufficient = count < min_count
            if np.any(insufficient):
                scr_block = scr_block.copy()
                scr_block[insufficient] = SCR_NODATA

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
