r"""Point Coherence Estimation (PCE) for PS selection.

Estimates per-pixel phase coherence from arc (double-difference) coherences,
following Costantini et al. (2014) [1]_.

Theory
------
For two pixels :math:`p` and :math:`p'`, the coherence of the arc
(double difference) :math:`\delta_a` between them satisfies

.. math::

    |\gamma(\epsilon_{p'})|^2 \, |\gamma(\epsilon_p)|^2
        = |\gamma(\delta_a)|^2

Taking the log and defining :math:`\Gamma(\epsilon_p) = \log|\gamma(\epsilon_p)|^2`,
this becomes the linear system

.. math::

    \Gamma(\epsilon_p) + \Gamma(\epsilon_{p'}) = \Gamma(\delta_a)

which is an overdetermined system solvable via least squares (L2) or
least absolute deviations (L1).

If the phase noise is Gaussian, then :math:`|\gamma(\epsilon_p)|^2 =
e^{-\sigma^2(\epsilon_p)}` and the system reduces to

.. math::

    \sigma^2(\epsilon_p) + \sigma^2(\epsilon_{p'}) = \sigma^2(\delta_a)


References
----------
.. [1] Costantini, M., Falco, S., Malvarosa, F., Minati, F., Trillo, F.,
   and Vecchioli, F. "Persistent Scatterer Pair Interferometry: Approach and
   Application to COSMO-SkyMed SAR Data." IEEE Journal of Selected Topics in
   Applied Earth Observations and Remote Sensing, 2014.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse import linalg as splinalg

from dolphin import io
from dolphin._types import Filename
from dolphin.io import StackReader

logger = logging.getLogger("dolphin")

PCE_NODATA = 0.0


def create_pce(
    *,
    reader: StackReader,
    output_file: Filename,
    amp_dispersion_file: Filename,
    like_filename: Filename,
    amp_dispersion_prefilter: float = 0.5,
    max_radius: int = 3,
    coherence_threshold: float = 0.5,
    block_shape: tuple[int, int] = (512, 512),
    nodata_mask: Optional[np.ndarray] = None,
) -> Path:
    """Create a point coherence raster from an SLC stack.

    Forms consecutive interferograms from the SLC stack, then estimates
    per-pixel phase coherence using arc-based double differences.

    Parameters
    ----------
    reader : StackReader
        Dataset reader for the 3D SLC stack.
    output_file : Filename
        Path for the output point coherence file (dtype: float32).
    amp_dispersion_file : Filename
        Path to the existing amplitude dispersion raster. Used to create
        the loose candidate mask via ``amp_dispersion_prefilter``.
    like_filename : Filename
        Filename to use for the output file's spatial reference.
    amp_dispersion_prefilter : float
        Loose threshold on amplitude dispersion for candidate selection.
    max_radius : int
        Max Chebyshev distance for arc neighbor search.
    coherence_threshold : float
        Threshold on estimated point coherence for PS labeling.
    block_shape : tuple[int, int]
        Block size for reading the SLC stack.
    nodata_mask : np.ndarray, optional
        If provided, further restricts candidates to masked-in areas.

    Returns
    -------
    Path
        Path to the output point coherence file.

    """
    output_file = Path(output_file)
    amp_disp = io.load_gdal(amp_dispersion_file, masked=True)

    # Build loose candidate mask from amplitude dispersion
    candidate_mask = (amp_disp > 0) & (amp_disp < amp_dispersion_prefilter)
    if hasattr(candidate_mask, "filled"):
        candidate_mask = candidate_mask.filled(False)
    candidate_mask = candidate_mask.astype(bool)
    if nodata_mask is not None:
        candidate_mask &= nodata_mask

    n_candidates = int(candidate_mask.sum())
    logger.info(
        f"PCE: {n_candidates} candidate pixels"
        f" (amp_dispersion < {amp_dispersion_prefilter})"
    )
    if n_candidates == 0:
        logger.warning("PCE: No candidate pixels found, writing zeros.")
        io.write_arr(
            arr=np.zeros(candidate_mask.shape, dtype=np.float32),
            like_filename=like_filename,
            output_name=output_file,
            nodata=PCE_NODATA,
        )
        return output_file

    # Read the full SLC stack and form consecutive interferograms
    logger.info("PCE: Reading SLC stack and forming consecutive interferograms")
    slc_stack = reader[:, :, :]
    ifg_stack = _form_consecutive_ifgs(slc_stack)
    del slc_stack

    logger.info(f"PCE: Formed {ifg_stack.shape[0]} consecutive interferograms")

    # Run the PCE estimation
    point_coherence = estimate_point_coherence(
        ifg_stack=ifg_stack,
        candidate_mask=candidate_mask,
        max_radius=max_radius,
        coherence_threshold=coherence_threshold,
    )

    # Write output
    io.write_arr(
        arr=point_coherence.astype(np.float32),
        like_filename=like_filename,
        output_name=output_file,
        nodata=PCE_NODATA,
    )
    logger.info(f"PCE: Wrote point coherence to {output_file}")

    return output_file


def _form_consecutive_ifgs(slc_stack: np.ndarray) -> np.ndarray:
    """Form consecutive interferograms from an SLC stack.

    Parameters
    ----------
    slc_stack : np.ndarray
        Complex SLC stack, shape ``(n_slc, rows, cols)``.

    Returns
    -------
    ifg_stack : np.ndarray
        Complex interferogram stack, shape ``(n_slc - 1, rows, cols)``.
        Each interferogram is ``slc[i+1] * conj(slc[i])``.

    """
    return slc_stack[1:] * np.conj(slc_stack[:-1])


def estimate_point_coherence(
    ifg_stack: ArrayLike,
    candidate_mask: np.ndarray,
    max_radius: int = 3,
    coherence_threshold: float = 0.5,
) -> np.ndarray:
    r"""Estimate per-pixel phase coherence using arc double differences.

    Builds a spatial graph of pixel pairs ("arcs") from candidate pixels,
    estimates the temporal coherence of each arc, then solves the overdetermined
    linear system to recover individual pixel coherences.

    Parameters
    ----------
    ifg_stack : ArrayLike
        3D stack of complex interferograms, shape ``(n_ifg, rows, cols)``.
        Each 2D slice is one interferogram (complex valued).
    candidate_mask : np.ndarray
        2D boolean mask, shape ``(rows, cols)``.
        True where a pixel is a candidate (e.g. from a loose amplitude
        dispersion threshold).
    max_radius : int, optional
        Maximum distance (in pixels, Chebyshev/chessboard metric) to search
        for neighbors when forming arcs. Default is 3.
    coherence_threshold : float, optional
        Threshold on the estimated point coherence to label a pixel as a PS.
        Pixels with coherence >= threshold are selected. Default is 0.5.

    Returns
    -------
    point_coherence : np.ndarray
        2D array of estimated point coherence, shape ``(rows, cols)``.
        Values in [0, 1]. Pixels outside `candidate_mask` are set to 0.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Identify candidate pixel indices from ``candidate_mask``.
    2. For each candidate, find other candidates within ``max_radius``
       to form arcs (pixel pairs).
    3. For each arc, compute the temporal coherence of the double-difference
       phase across the interferogram stack.
    4. Build the incidence matrix ``A`` where each row is an arc and each
       column is a candidate pixel, with +1 entries for the two pixels in
       the arc.
    5. Take log of the squared arc coherences to form the observation vector
       ``b = log(|gamma_arc|^2)``.
    6. Solve ``A @ x = b`` for ``x`` (the per-pixel log-coherence-squared)
       via least-squares.
    7. Convert back: ``point_coherence = exp(x / 2)``.

    """
    ifg_stack = np.asarray(ifg_stack)
    if ifg_stack.ndim != 3:
        msg = f"ifg_stack must be 3D (n_ifg, rows, cols), got shape {ifg_stack.shape}"
        raise ValueError(msg)
    if not np.iscomplexobj(ifg_stack):
        msg = "ifg_stack must be complex-valued."
        raise ValueError(msg)

    n_ifg, rows, cols = ifg_stack.shape

    # 1. Get candidate pixel indices
    candidate_rows, candidate_cols = np.nonzero(candidate_mask)
    n_candidates = len(candidate_rows)
    if n_candidates == 0:
        return np.zeros((rows, cols), dtype=np.float64)

    logger.info(f"PCE: {n_candidates} candidate pixels from mask")

    # Build a lookup from (row, col) -> candidate index
    # Use a 2D array for O(1) lookup
    pixel_to_idx = np.full((rows, cols), -1, dtype=np.int32)
    for i, (r, c) in enumerate(zip(candidate_rows, candidate_cols)):
        pixel_to_idx[r, c] = i

    # 2. Build arcs: find neighbor pairs within max_radius
    arc_rows, arc_cols, arc_idx_p, arc_idx_q = _build_arcs(
        candidate_rows, candidate_cols, pixel_to_idx, max_radius, rows, cols
    )
    n_arcs = len(arc_idx_p)
    if n_arcs == 0:
        logger.warning("PCE: No arcs found. Returning zeros.")
        return np.zeros((rows, cols), dtype=np.float64)

    logger.info(f"PCE: {n_arcs} arcs formed between candidate pairs")

    # 3. Compute temporal coherence for each arc
    arc_coherence = _compute_arc_coherences(
        ifg_stack, arc_rows, arc_cols, arc_idx_p, arc_idx_q
    )

    # 4 & 5. Build the system A @ x = b and solve
    point_log_coh_sq = _solve_coherence_system(
        arc_idx_p, arc_idx_q, arc_coherence, n_candidates
    )

    # 7. Convert back to coherence: gamma = exp(Gamma / 2)
    #    Gamma = log(|gamma|^2), so |gamma| = exp(Gamma / 2)
    point_coh = np.exp(point_log_coh_sq / 2.0)
    # Clip to [0, 1]
    np.clip(point_coh, 0.0, 1.0, out=point_coh)

    # Map back to 2D
    point_coherence = np.zeros((rows, cols), dtype=np.float64)
    point_coherence[candidate_rows, candidate_cols] = point_coh

    return point_coherence


def _build_arcs(
    candidate_rows: np.ndarray,
    candidate_cols: np.ndarray,
    pixel_to_idx: np.ndarray,
    max_radius: int,
    nrows: int,
    ncols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build arcs between candidate pixel pairs within a Chebyshev radius.

    Only half of the pairs are formed (p < q by index) to avoid duplicates.

    Returns
    -------
    arc_rows : np.ndarray
        Shape ``(n_arcs, 2)``: row coords of (pixel_p, pixel_q) per arc.
    arc_cols : np.ndarray
        Shape ``(n_arcs, 2)``: col coords of (pixel_p, pixel_q) per arc.
    arc_idx_p : np.ndarray
        Shape ``(n_arcs,)``: candidate index of pixel p.
    arc_idx_q : np.ndarray
        Shape ``(n_arcs,)``: candidate index of pixel q.

    """
    arc_idx_p_list = []
    arc_idx_q_list = []
    arc_rows_list = []
    arc_cols_list = []

    n_candidates = len(candidate_rows)
    for i in range(n_candidates):
        r_i, c_i = candidate_rows[i], candidate_cols[i]
        # Search within the bounding box
        r_min = max(0, r_i - max_radius)
        r_max = min(nrows - 1, r_i + max_radius)
        c_min = max(0, c_i - max_radius)
        c_max = min(ncols - 1, c_i + max_radius)

        for r_j in range(r_min, r_max + 1):
            for c_j in range(c_min, c_max + 1):
                j = pixel_to_idx[r_j, c_j]
                # Only take j > i to avoid duplicate arcs
                if j <= i:
                    continue
                arc_idx_p_list.append(i)
                arc_idx_q_list.append(j)
                arc_rows_list.append((r_i, r_j))
                arc_cols_list.append((c_i, c_j))

    if not arc_idx_p_list:
        return (
            np.empty((0, 2), dtype=np.intp),
            np.empty((0, 2), dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
        )

    arc_rows = np.array(arc_rows_list, dtype=np.intp)
    arc_cols = np.array(arc_cols_list, dtype=np.intp)
    arc_idx_p = np.array(arc_idx_p_list, dtype=np.intp)
    arc_idx_q = np.array(arc_idx_q_list, dtype=np.intp)
    return arc_rows, arc_cols, arc_idx_p, arc_idx_q


def _compute_arc_coherences(
    ifg_stack: np.ndarray,
    arc_rows: np.ndarray,
    arc_cols: np.ndarray,
    arc_idx_p: np.ndarray,
    arc_idx_q: np.ndarray,
) -> np.ndarray:
    r"""Compute temporal coherence for each arc (double-difference).

    For an arc between pixels p and q, the double-difference phase is

    .. math::
        \delta\phi_k = \phi_k(p) - \phi_k(q)

    for each interferogram k. The temporal coherence of this arc is

    .. math::
        |\gamma_{\mathrm{arc}}| =
        \left|\frac{1}{N}\sum_{k=1}^{N} e^{j\delta\phi_k}\right|

    Parameters
    ----------
    ifg_stack : np.ndarray
        Complex interferogram stack, shape ``(n_ifg, rows, cols)``.
    arc_rows : np.ndarray
        Shape ``(n_arcs, 2)``.
    arc_cols : np.ndarray
        Shape ``(n_arcs, 2)``.
    arc_idx_p, arc_idx_q : np.ndarray
        Shape ``(n_arcs,)``, candidate indices (unused here, but kept for API
        consistency).

    Returns
    -------
    arc_coherence : np.ndarray
        Shape ``(n_arcs,)``, absolute temporal coherence in [0, 1].

    """
    n_arcs = len(arc_rows)
    n_ifg = ifg_stack.shape[0]

    # Extract phase values for both ends of each arc across all interferograms
    # Shape: (n_ifg, n_arcs) for each pixel
    phase_p = ifg_stack[:, arc_rows[:, 0], arc_cols[:, 0]]
    phase_q = ifg_stack[:, arc_rows[:, 1], arc_cols[:, 1]]

    # Double difference: multiply p by conjugate of q
    # This gives exp(j * (phi_p - phi_q))
    dd = phase_p * np.conj(phase_q)
    # Normalize to unit magnitude so we only keep phase information
    dd_mag = np.abs(dd)
    # Avoid division by zero
    dd_mag = np.where(dd_mag > 0, dd_mag, 1.0)
    dd_unit = dd / dd_mag

    # Temporal coherence = |mean(exp(j * delta_phi))|
    arc_coherence = np.abs(np.mean(dd_unit, axis=0))

    return arc_coherence


def _solve_coherence_system(
    arc_idx_p: np.ndarray,
    arc_idx_q: np.ndarray,
    arc_coherence: np.ndarray,
    n_candidates: int,
    min_arc_coherence: float = 0.01,
) -> np.ndarray:
    r"""Solve the linear system for per-pixel coherence.

    Builds and solves

    .. math::
        A \mathbf{x} = \mathbf{b}

    where ``A`` is the incidence matrix (each row has two +1 entries for the
    pixels in the arc), ``b = log(|gamma_arc|^2)``, and ``x`` is the vector
    of per-pixel :math:`\Gamma = \log|\gamma|^2`.

    Uses sparse least-squares (LSQR) for efficiency.

    Parameters
    ----------
    arc_idx_p, arc_idx_q : np.ndarray
        Candidate pixel indices for each arc, shape ``(n_arcs,)``.
    arc_coherence : np.ndarray
        Arc temporal coherence, shape ``(n_arcs,)``.
    n_candidates : int
        Total number of candidate pixels (columns in A).
    min_arc_coherence : float
        Floor on arc coherence before taking log, to avoid log(0).

    Returns
    -------
    x : np.ndarray
        Per-pixel :math:`\Gamma = \log|\gamma|^2`, shape ``(n_candidates,)``.

    """
    n_arcs = len(arc_idx_p)

    # Clip coherence to avoid log(0)
    coh_clipped = np.clip(arc_coherence, min_arc_coherence, 1.0)

    # b = log(|gamma_arc|^2) = 2 * log(|gamma_arc|)
    # Since gamma_arc is already the absolute coherence:
    b = 2.0 * np.log(coh_clipped)

    # Build sparse incidence matrix: each row has +1 at columns p and q
    row_indices = np.concatenate([np.arange(n_arcs), np.arange(n_arcs)])
    col_indices = np.concatenate([arc_idx_p, arc_idx_q])
    data = np.ones(2 * n_arcs, dtype=np.float64)

    A = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_arcs, n_candidates),
    )

    # Solve via sparse LSQR
    result = splinalg.lsqr(A, b)
    x = result[0]

    return x
