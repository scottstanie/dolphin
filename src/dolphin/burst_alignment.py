"""Estimate and remove inter-burst phase artifacts from per-burst rasters.

Per-burst interferograms (or unwrapped phase) often disagree across the
seam where two bursts overlap because of calibration, azimuth-FM-rate
mis-registration, or ionospheric path differences. Mosaicking with a
"last-one-wins" rule (as `dolphin.stitching.merge_images` does) freezes
those discontinuities into the stitched product.

This module estimates a per-burst correction polynomial from the overlap
regions and writes corrected rasters that can then be stitched cleanly.
Two model degrees are supported:

- ``degree=0`` — a single constant offset per burst. Robust, cheap, and
  almost always worth running.
- ``degree=1`` — a planar ramp: ``offset + cx*(x-x_ref) + cy*(y-y_ref)``
  per burst, in the data CRS. This captures the typical S1 azimuth
  gradient (which projects onto both x and y in geocoded coordinates
  because of the platform heading), as well as a range gradient.

Both wrapped (complex) and unwrapped (real) inputs are handled. The
correction is multiplied as ``exp(-1j * polynomial)`` for wrapped data
and subtracted for unwrapped.

Notes
-----
The fit is least-squares on differences across overlaps, which leaves
the absolute level free (gauge). The first burst per connected component
is anchored to zero, so corrections are relative.

For ``degree=1`` the implementation iterates offset → ramp → offset
so that residual DC bias does not contaminate the slope estimate and vice versa.

For wrapped inputs the LSQ residual is computed on raw values, which is
only a good approximation when individual offsets are well below pi. In
typical S1 IW burst overlaps, offsets are tens of milliradians to a few
hundred milliradians — well within range. For unwrapped data there is no
such ambiguity.

Running this on wrapped phase, before unwrapping, is the recommended
order: it removes the seam discontinuity that would otherwise force the
unwrapper to bridge a 2*pi step at every burst boundary.

"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

from dolphin import io
from dolphin._types import Filename

logger = logging.getLogger("dolphin")

__all__ = [
    "BurstCorrection",
    "align_bursts",
    "apply_burst_offsets",
    "estimate_burst_offsets",
]


_MIN_OVERLAP_PIXELS = 40
_MIN_PATCH_PIXELS = 20
_MIN_VALID_PATCHES = 3
_MIN_INLIER_PATCHES = 6
_MAD_OUTLIER_THRESHOLD = 2.5


@dataclass(frozen=True)
class BurstCorrection:
    """Correction polynomial estimated for one burst.

    The phase artifact, evaluated at world coordinates ``(x, y)`` in the
    data CRS, is::

        offset + cx * (x - x_ref) + cy * (y - y_ref)

    For ``degree=0`` only ``offset`` is non-zero. ``x_ref``/``y_ref``
    are shared across all bursts in a single estimation run; they exist
    so the LSQ stays well-conditioned and so ``offset`` keeps its meaning
    as the value of the polynomial at the reference point.

    Subtracting this polynomial from unwrapped phase, or multiplying the
    complex data by ``exp(-1j * polynomial)`` for wrapped phase, removes
    the artifact.
    """

    offset: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    x_ref: float = 0.0
    y_ref: float = 0.0

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the polynomial value at world coordinates ``(x, y)``."""
        return self.offset + self.cx * (x - self.x_ref) + self.cy * (y - self.y_ref)

    @property
    def is_planar(self) -> bool:
        """True if either slope is non-zero (i.e. degree-1 correction)."""
        return self.cx != 0.0 or self.cy != 0.0


def estimate_burst_offsets(
    burst_files: Sequence[Filename],
    *,
    degree: int = 0,
    method: Literal["median", "mean"] = "median",
    anchor_index: int = 0,
    n_patches: tuple[int, int] = (4, 8),
    min_overlap_pixels: int = _MIN_OVERLAP_PIXELS,
    mad_threshold: float = _MAD_OUTLIER_THRESHOLD,
    max_fringes_per_burst: float | None = 0.5,
) -> dict[Path, BurstCorrection]:
    """Estimate per-burst phase correction from overlap regions.

    Parameters
    ----------
    burst_files : Sequence of paths
        Per-burst rasters that share a CRS and resolution. They may be
        wrapped (complex dtype) or unwrapped (real dtype); the data type
        of the first file determines the mode.
    degree : {0, 1}, optional
        Polynomial degree:
        - 0 (default): constant offset per burst.
        - 1: offset + planar ramp (cx, cy) in the data CRS. Uses a
          three-pass offset → ramp → offset iteration so the offset and
          ramp solves do not contaminate each other.
    method : {"median", "mean"}, optional
        Reduction over per-patch offsets within each overlap. Default
        "median" (robust).
    anchor_index : int, optional
        Index of the burst whose correction is pinned to zero in each
        connected component containing it. Default 0.
    n_patches : (int, int), optional
        Number of (azimuth, range) patches each overlap is split into for
        the planar fit (``degree=1``). Auto-capped per overlap if pixel
        counts are small. Ignored for ``degree=0``.
    min_overlap_pixels : int, optional
        Skip overlap pairs with fewer jointly-valid pixels.
    mad_threshold : float, optional
        Per-patch MAD outlier threshold (in units of MAD).
    max_fringes_per_burst : float, optional
        Tikhonov prior on the planar slope (``degree=1`` only). Acts as a
        Gaussian prior with std equal to the slope that would produce
        ``max_fringes_per_burst`` cycles of phase across the typical
        burst extent. Pulls slopes toward zero unless overlap data has
        SNR to justify them. Default 0.5 (half a fringe per burst), which
        is conservative for S1 with restituted orbits. Pass ``None`` to
        disable.

    Returns
    -------
    dict[Path, BurstCorrection]
        Map from input path to the per-burst correction. Bursts that
        share no overlap with any other burst (or are alone in a
        connected component) get ``BurstCorrection()`` (all zeros).

    """
    if degree not in (0, 1):
        raise ValueError(f"degree must be 0 or 1, got {degree}")

    paths = [Path(f) for f in burst_files]
    if len(paths) < 2:
        return {p: BurstCorrection() for p in paths}

    is_complex = _is_complex_raster(paths[0])
    logger.debug(
        "estimate_burst_offsets: %d bursts, degree=%d, %s mode",
        len(paths),
        degree,
        "wrapped (complex)" if is_complex else "unwrapped (real)",
    )

    # Per-burst extents in the data CRS, used to scale the Tikhonov prior.
    burst_extents = _read_burst_extents(paths)

    # Read every overlap once and cache the aligned arrays + world coords.
    cache: dict[tuple[int, int], _OverlapCache] = {}
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            entry = _load_overlap(paths[i], paths[j])
            if entry is None:
                continue
            if int(entry.valid.sum()) < min_overlap_pixels:
                continue
            cache[(i, j)] = entry

    if not cache:
        logger.warning("estimate_burst_offsets: no overlapping bursts found")
        return {p: BurstCorrection() for p in paths}

    # Choose a shared reference point so the planar LSQ stays well-
    # conditioned. Use the centroid of all valid overlap patches.
    x_ref, y_ref = _choose_reference(cache.values())

    # Step 1: offset-only fit.
    corrections = _initialize_corrections(paths, x_ref=x_ref, y_ref=y_ref)
    corrections = _fit_offset_step(
        cache=cache,
        corrections=corrections,
        paths=paths,
        is_complex=is_complex,
        method=method,
        n_patches=n_patches,
        mad_threshold=mad_threshold,
        anchor_index=anchor_index,
    )

    if degree == 1:
        corrections = _fit_planar_step(
            cache=cache,
            corrections=corrections,
            paths=paths,
            is_complex=is_complex,
            method=method,
            n_patches=n_patches,
            mad_threshold=mad_threshold,
            anchor_index=anchor_index,
            burst_extents=burst_extents,
            max_fringes_per_burst=max_fringes_per_burst,
        )
        corrections = _fit_offset_step(
            cache=cache,
            corrections=corrections,
            paths=paths,
            is_complex=is_complex,
            method=method,
            n_patches=n_patches,
            mad_threshold=mad_threshold,
            anchor_index=anchor_index,
        )

    return corrections


def apply_burst_offsets(
    burst_files: Sequence[Filename],
    corrections: Mapping[Path, BurstCorrection],
    output_dir: Filename,
    *,
    suffix: str = ".aligned",
) -> list[Path]:
    """Write corrected per-burst rasters with the polynomial removed.

    For complex (wrapped) inputs each pixel is multiplied by
    ``exp(-1j * polynomial(x, y))``. For real (unwrapped) inputs the
    polynomial is subtracted. Nodata pixels are passed through.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_paths = [Path(f) for f in burst_files]
    out_paths = _build_output_paths(in_paths, out_dir, suffix=suffix)

    for in_path, out_path in zip(in_paths, out_paths, strict=False):
        corr = corrections[in_path]

        with rasterio.open(in_path) as src:
            arr = src.read(1)
            nodata = src.nodata
            transform = src.transform

        corrected = _apply_correction(arr, corr, nodata=nodata, transform=transform)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        io.write_arr(arr=corrected, output_name=out_path, like_filename=in_path)
        logger.debug(
            "wrote %s (offset=%+.4f cx=%+.3e cy=%+.3e)",
            out_path,
            corr.offset,
            corr.cx,
            corr.cy,
        )

    return out_paths


def align_bursts(
    burst_files: Sequence[Filename],
    output_dir: Filename,
    *,
    degree: int = 0,
    method: Literal["median", "mean"] = "median",
    anchor_index: int = 0,
    suffix: str = ".aligned",
    max_fringes_per_burst: float | None = 0.5,
) -> tuple[list[Path], dict[Path, BurstCorrection]]:
    """Estimate corrections, apply them, and return the corrected paths."""
    corrections = estimate_burst_offsets(
        burst_files,
        degree=degree,
        method=method,
        anchor_index=anchor_index,
        max_fringes_per_burst=max_fringes_per_burst,
    )
    out_paths = apply_burst_offsets(
        burst_files, corrections, output_dir=output_dir, suffix=suffix
    )
    return out_paths, corrections


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


@dataclass
class _OverlapCache:
    """Aligned overlap arrays plus per-pixel world coordinates."""

    arr_a: np.ndarray
    arr_b: np.ndarray
    x_world: np.ndarray  # 1D, len = arr.shape[1]
    y_world: np.ndarray  # 1D, len = arr.shape[0]
    valid: np.ndarray  # bool 2D mask of jointly-valid pixels


def _initialize_corrections(
    paths: Sequence[Path], *, x_ref: float, y_ref: float
) -> dict[Path, BurstCorrection]:
    return {p: BurstCorrection(x_ref=x_ref, y_ref=y_ref) for p in paths}


def _read_burst_extents(paths: Sequence[Path]) -> dict[Path, tuple[float, float]]:
    """Return ``{path: (x_extent, y_extent)}`` in CRS units."""
    extents: dict[Path, tuple[float, float]] = {}
    for p in paths:
        with rasterio.open(p) as src:
            extents[p] = (
                float(src.bounds.right - src.bounds.left),
                float(src.bounds.top - src.bounds.bottom),
            )
    return extents


def _load_overlap(path_a: Path, path_b: Path) -> _OverlapCache | None:
    with rasterio.open(path_a) as ra, rasterio.open(path_b) as rb:
        if ra.crs != rb.crs:
            logger.warning(
                "skipping overlap %s vs %s: CRS mismatch", path_a.name, path_b.name
            )
            return None
        if ra.res != rb.res:
            logger.warning(
                "skipping overlap %s vs %s: resolution mismatch",
                path_a.name,
                path_b.name,
            )
            return None

        xmin = max(ra.bounds.left, rb.bounds.left)
        xmax = min(ra.bounds.right, rb.bounds.right)
        ymin = max(ra.bounds.bottom, rb.bounds.bottom)
        ymax = min(ra.bounds.top, rb.bounds.top)
        if xmax <= xmin or ymax <= ymin:
            return None

        win = from_bounds(xmin, ymin, xmax, ymax, transform=ra.transform)
        win = win.round_offsets().round_lengths()
        if win.width <= 0 or win.height <= 0:
            return None

        arr_a = ra.read(1, window=win)
        nodata_a = ra.nodata
        with WarpedVRT(
            rb,
            crs=ra.crs,
            transform=ra.transform,
            width=ra.width,
            height=ra.height,
            resampling=Resampling.nearest,
        ) as vrt:
            arr_b = vrt.read(1, window=win)
            nodata_b = vrt.nodata if vrt.nodata is not None else rb.nodata

        # World coords for pixel centers in the windowed read.
        col_off = int(win.col_off)
        row_off = int(win.row_off)
        cols = np.arange(win.width)
        rows = np.arange(win.height)
        x_world = ra.transform.a * (col_off + cols + 0.5) + ra.transform.c
        y_world = ra.transform.e * (row_off + rows + 0.5) + ra.transform.f

    valid = _valid_mask(arr_a, nodata_a) & _valid_mask(arr_b, nodata_b)
    return _OverlapCache(arr_a, arr_b, x_world, y_world, valid)


def _is_complex_raster(path: Path) -> bool:
    with rasterio.open(path) as src:
        return np.issubdtype(np.dtype(src.dtypes[0]), np.complexfloating)


def _valid_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    if np.iscomplexobj(arr):
        return np.isfinite(arr.real) & np.isfinite(arr.imag) & (arr != 0)
    valid = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        valid &= arr != nodata
    return valid


def _choose_reference(caches: "Iterable[_OverlapCache]") -> tuple[float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for c in caches:
        if not c.valid.any():
            continue
        # Just use the centroid of valid pixels in each overlap.
        rows, cols = np.nonzero(c.valid)
        xs.append(float(c.x_world[cols].mean()))
        ys.append(float(c.y_world[rows].mean()))
    if not xs:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def _residual_diff(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    valid: np.ndarray,
    corr_a: BurstCorrection,
    corr_b: BurstCorrection,
    x_world: np.ndarray,
    y_world: np.ndarray,
    is_complex: bool,
) -> np.ndarray:
    """Compute (b - a) - (poly_b - poly_a) at every overlap pixel.

    For wrapped data this is `arr_b * conj(arr_a) * exp(-1j*(poly_b - poly_a))`,
    still complex. Invalid pixels become NaN.
    """
    poly_a = corr_a.evaluate(x_world[None, :], y_world[:, None])
    poly_b = corr_b.evaluate(x_world[None, :], y_world[:, None])
    delta = poly_b - poly_a

    if is_complex:
        diff = arr_b * np.conj(arr_a) * np.exp(-1j * delta).astype(arr_a.dtype)
        out = np.full(diff.shape, np.nan + 0j, dtype=np.complex64)
        out[valid] = diff[valid]
        return out

    diff = (arr_b - arr_a) - delta
    out = np.full(diff.shape, np.nan, dtype=np.float64)
    out[valid] = diff[valid]
    return out


def _patch_stats(
    diff: np.ndarray,
    x_world: np.ndarray,
    y_world: np.ndarray,
    *,
    n_patches: tuple[int, int],
    method: str,
    mad_threshold: float,
    is_complex: bool,
) -> list[tuple[float, float, float, float]]:
    """Reduce a 2D overlap-difference array into a list of patch summaries.

    Returns
    -------
    list of (xc, yc, value, weight)
        ``value`` is a phase (radians) for complex inputs, a real
        difference otherwise. ``weight = sqrt(N_valid_in_patch)``.

    """
    ny, nx = diff.shape
    npy = max(1, min(n_patches[0], ny))
    npx = max(1, min(n_patches[1], nx))

    y_edges = np.linspace(0, ny, npy + 1, dtype=int)
    x_edges = np.linspace(0, nx, npx + 1, dtype=int)

    out: list[tuple[float, float, float, float]] = []
    for py in range(npy):
        y0, y1 = y_edges[py], y_edges[py + 1]
        if y1 <= y0:
            continue
        for px in range(npx):
            x0, x1 = x_edges[px], x_edges[px + 1]
            if x1 <= x0:
                continue
            block = diff[y0:y1, x0:x1]
            if is_complex:
                mask = np.isfinite(block.real) & np.isfinite(block.imag)
            else:
                mask = np.isfinite(block)
            n = int(mask.sum())
            if n < _MIN_PATCH_PIXELS:
                continue
            valid = block[mask]
            value = float(np.angle(valid.mean())) if is_complex else float(valid.mean())
            xs, ys = np.meshgrid(x_world[x0:x1], y_world[y0:y1])
            xc = float(xs[mask].mean())
            yc = float(ys[mask].mean())
            out.append((xc, yc, value, float(np.sqrt(n))))

    if len(out) < _MIN_VALID_PATCHES:
        return []

    # MAD outlier rejection on patch values.
    values = np.array([p[2] for p in out])
    if method == "median":
        center = float(np.median(values))
        resid = _wrap(values - center) if is_complex else (values - center)
        mad = float(np.median(np.abs(resid)))
        if mad > 0:
            inliers = np.abs(resid) <= mad_threshold * mad
            if int(inliers.sum()) >= _MIN_INLIER_PATCHES:
                out = [p for p, keep in zip(out, inliers, strict=False) if keep]
    return out


def _fit_offset_step(
    *,
    cache: dict[tuple[int, int], _OverlapCache],
    corrections: dict[Path, BurstCorrection],
    paths: Sequence[Path],
    is_complex: bool,
    method: str,
    n_patches: tuple[int, int],
    mad_threshold: float,
    anchor_index: int,
) -> dict[Path, BurstCorrection]:
    """One LSQ pass that updates only the constant `offset` per burst."""
    pair_eqs: list[tuple[int, int, float, float]] = []
    for (i, j), entry in cache.items():
        diff = _residual_diff(
            entry.arr_a,
            entry.arr_b,
            entry.valid,
            corrections[paths[i]],
            corrections[paths[j]],
            entry.x_world,
            entry.y_world,
            is_complex=is_complex,
        )
        patches = _patch_stats(
            diff,
            entry.x_world,
            entry.y_world,
            n_patches=n_patches,
            method=method,
            mad_threshold=mad_threshold,
            is_complex=is_complex,
        )
        if not patches:
            continue
        # Aggregate patch values into one (offset, weight) for this pair.
        vals = np.array([p[2] for p in patches])
        ws = np.array([p[3] for p in patches])
        offset = _aggregate_pair(vals, ws, method=method, is_complex=is_complex)
        pair_eqs.append((i, j, offset, float(np.sqrt((ws**2).sum()))))

    deltas = _solve_offset_lsq(
        pair_eqs,
        n_bursts=len(paths),
        anchor_index=anchor_index,
        is_complex=is_complex,
    )
    return _add_offsets(corrections, paths, deltas, is_complex=is_complex)


def _fit_planar_step(
    *,
    cache: dict[tuple[int, int], _OverlapCache],
    corrections: dict[Path, BurstCorrection],
    paths: Sequence[Path],
    is_complex: bool,
    method: str,
    n_patches: tuple[int, int],
    mad_threshold: float,
    anchor_index: int,
    burst_extents: dict[Path, tuple[float, float]],
    max_fringes_per_burst: float | None,
) -> dict[Path, BurstCorrection]:
    """One LSQ pass that fits planar (cx, cy) per burst on the residual."""
    # Each pair contributes one equation per surviving patch.
    pair_patches: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
    for (i, j), entry in cache.items():
        diff = _residual_diff(
            entry.arr_a,
            entry.arr_b,
            entry.valid,
            corrections[paths[i]],
            corrections[paths[j]],
            entry.x_world,
            entry.y_world,
            is_complex=is_complex,
        )
        patches = _patch_stats(
            diff,
            entry.x_world,
            entry.y_world,
            n_patches=n_patches,
            method=method,
            mad_threshold=mad_threshold,
            is_complex=is_complex,
        )
        if patches:
            pair_patches[(i, j)] = patches

    sample_corr = next(iter(corrections.values()))
    cx_arr, cy_arr = _solve_planar_lsq(
        pair_patches,
        n_bursts=len(paths),
        anchor_index=anchor_index,
        is_complex=is_complex,
        x_ref=sample_corr.x_ref,
        y_ref=sample_corr.y_ref,
        burst_extents=[burst_extents[p] for p in paths],
        max_fringes_per_burst=max_fringes_per_burst,
    )

    new_corr: dict[Path, BurstCorrection] = {}
    for k, p in enumerate(paths):
        c = corrections[p]
        new_corr[p] = BurstCorrection(
            offset=c.offset,
            cx=c.cx + float(cx_arr[k]),
            cy=c.cy + float(cy_arr[k]),
            x_ref=c.x_ref,
            y_ref=c.y_ref,
        )
    return new_corr


def _add_offsets(
    corrections: dict[Path, BurstCorrection],
    paths: Sequence[Path],
    deltas: np.ndarray,
    *,
    is_complex: bool,
) -> dict[Path, BurstCorrection]:
    new_corr: dict[Path, BurstCorrection] = {}
    for k, p in enumerate(paths):
        c = corrections[p]
        new_offset = c.offset + float(deltas[k])
        if is_complex:
            new_offset = _wrap_scalar(new_offset)
        new_corr[p] = BurstCorrection(
            offset=new_offset,
            cx=c.cx,
            cy=c.cy,
            x_ref=c.x_ref,
            y_ref=c.y_ref,
        )
    return new_corr


def _solve_offset_lsq(
    pair_eqs: list[tuple[int, int, float, float]],
    *,
    n_bursts: int,
    anchor_index: int,
    is_complex: bool,
) -> np.ndarray:
    """LSQ for `delta_j - delta_i = o_ij` per burst, anchored per component."""
    if not pair_eqs:
        return np.zeros(n_bursts, dtype=np.float64)

    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import lsqr

    adj = sparse.lil_matrix((n_bursts, n_bursts))
    for i, j, _, _ in pair_eqs:
        adj[i, j] = 1
        adj[j, i] = 1
    n_comp, labels = connected_components(adj.tocsr(), directed=False)

    out = np.zeros(n_bursts, dtype=np.float64)
    for comp in range(n_comp):
        comp_idx = [int(i) for i in np.where(labels == comp)[0]]
        if len(comp_idx) == 1:
            continue
        local = {gi: k for k, gi in enumerate(comp_idx)}
        comp_eqs = [
            (local[i], local[j], o, w)
            for i, j, o, w in pair_eqs
            if i in local and j in local
        ]
        if not comp_eqs:
            continue

        n_local = len(comp_idx)
        n_eq = len(comp_eqs) + 1
        A = sparse.lil_matrix((n_eq, n_local))
        b = np.zeros(n_eq)
        wt = np.zeros(n_eq)
        for k, (li, lj, o, w) in enumerate(comp_eqs):
            A[k, li] = -1.0
            A[k, lj] = +1.0
            b[k] = _wrap_scalar(o) if is_complex else o
            wt[k] = w

        anchor_local = local.get(anchor_index, 0)
        A[-1, anchor_local] = 1.0
        b[-1] = 0.0
        sum_w = wt[:-1].sum()
        wt[-1] = sum_w * 100 if sum_w > 0 else 1e6

        sqW = np.sqrt(wt)
        sol, *_ = lsqr(sparse.diags(sqW) @ A.tocsr(), sqW * b)
        sol = sol - sol[anchor_local]
        for k, gi in enumerate(comp_idx):
            out[gi] = _wrap_scalar(float(sol[k])) if is_complex else float(sol[k])
    return out


def _solve_planar_lsq(
    pair_patches: dict[tuple[int, int], list[tuple[float, float, float, float]]],
    *,
    n_bursts: int,
    anchor_index: int,
    is_complex: bool,
    x_ref: float,
    y_ref: float,
    burst_extents: Sequence[tuple[float, float]],
    max_fringes_per_burst: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """LSQ for per-burst (cx, cy) given planar-residual patch observations.

    Builds equations of the form::

        cx_j*(xc - x_ref) - cx_i*(xc - x_ref)
          + cy_j*(yc - y_ref) - cy_i*(yc - y_ref) = obs

    and anchors (cx, cy) of one burst per component to zero. Coordinates
    are scaled internally so the LSQ is well-conditioned, then the
    solution is rescaled back to per-meter slopes.

    If ``max_fringes_per_burst`` is not None, two Tikhonov prior rows are
    added per burst (``cx_scaled = 0`` and ``cy_scaled = 0``), with weight
    ``1/sigma**2`` where ``sigma`` is the per-burst slope (in scaled
    units) that would produce ``max_fringes_per_burst`` cycles of phase
    across that burst's extent.
    """
    if not pair_patches:
        return np.zeros(n_bursts), np.zeros(n_bursts)

    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import lsqr

    adj = sparse.lil_matrix((n_bursts, n_bursts))
    for (i, j), patches in pair_patches.items():
        if patches:
            adj[i, j] = 1
            adj[j, i] = 1
    n_comp, labels = connected_components(adj.tocsr(), directed=False)

    # Scale the coordinate range to keep the design matrix well-conditioned.
    all_xc = []
    all_yc = []
    for patches in pair_patches.values():
        for xc, yc, _, _ in patches:
            all_xc.append(xc - x_ref)
            all_yc.append(yc - y_ref)
    x_scale = max(float(np.std(all_xc)), 1.0)
    y_scale = max(float(np.std(all_yc)), 1.0)

    cx_out = np.zeros(n_bursts)
    cy_out = np.zeros(n_bursts)

    for comp in range(n_comp):
        comp_idx = [int(i) for i in np.where(labels == comp)[0]]
        if len(comp_idx) == 1:
            continue
        local = {gi: k for k, gi in enumerate(comp_idx)}
        comp_pairs = [
            (local[i], local[j], patches)
            for (i, j), patches in pair_patches.items()
            if i in local and j in local
        ]
        if not comp_pairs:
            continue

        n_local = len(comp_idx)
        # Two unknowns per burst (cx, cy) → columns 2k and 2k+1.
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        b_list: list[float] = []
        wt_list: list[float] = []
        eq = 0
        for li, lj, patches in comp_pairs:
            for xc, yc, val, w in patches:
                dx = (xc - x_ref) / x_scale
                dy = (yc - y_ref) / y_scale
                rows.extend([eq, eq, eq, eq])
                cols.extend([2 * li, 2 * li + 1, 2 * lj, 2 * lj + 1])
                data.extend([-dx, -dy, +dx, +dy])
                b_list.append(_wrap_scalar(val) if is_complex else val)
                wt_list.append(w)
                eq += 1

        if eq < 2 * n_local:
            # Under-determined; skip planar fit for this component.
            continue

        # Tikhonov prior: pull (cx, cy) toward zero unless overlap data
        # supports otherwise. The prior std is chosen so a slope producing
        # `max_fringes_per_burst` cycles across the burst's extent is one
        # standard deviation. Per-burst, since burst extents may differ.
        if max_fringes_per_burst is not None and max_fringes_per_burst > 0:
            for li_local, gi in enumerate(comp_idx):
                ext_x, ext_y = burst_extents[gi]
                # Convert max_fringes -> prior std on the *scaled* slope.
                # cx_scaled = cx * x_scale (so cx = cx_scaled / x_scale).
                # cx is in rad/m, ext_x in m -> phase across burst = cx * ext_x.
                # Set sigma_cx_scaled so that |cx_scaled / sigma| = 1 means
                # |cx * ext_x| = 2*pi*max_fringes.
                sigma_cx_scaled = (
                    2 * np.pi * max_fringes_per_burst * x_scale / max(ext_x, 1.0)
                )
                sigma_cy_scaled = (
                    2 * np.pi * max_fringes_per_burst * y_scale / max(ext_y, 1.0)
                )
                # Row: 1.0 * cx_scaled = 0 with wt = 1/sigma**2 (so the
                # weighted residual equals cx_scaled / sigma, contributing
                # (cx_scaled/sigma)**2 to the LSQ objective — canonical
                # Gaussian prior).
                for kc, sigma in (
                    (2 * li_local, sigma_cx_scaled),
                    (2 * li_local + 1, sigma_cy_scaled),
                ):
                    rows.append(eq)
                    cols.append(kc)
                    data.append(1.0)
                    b_list.append(0.0)
                    wt_list.append(1.0 / (sigma * sigma))
                    eq += 1

        # Anchor: pin cx_anchor = cy_anchor = 0 with high weight.
        anchor_local = local.get(anchor_index, 0)
        for kc in (2 * anchor_local, 2 * anchor_local + 1):
            rows.append(eq)
            cols.append(kc)
            data.append(1.0)
            b_list.append(0.0)
            wt_list.append(max(sum(wt_list) * 100, 1e6))
            eq += 1

        A = sparse.coo_matrix((data, (rows, cols)), shape=(eq, 2 * n_local)).tocsr()
        b_arr = np.array(b_list)
        w_arr = np.array(wt_list)
        sqW = np.sqrt(w_arr)
        sol, *_ = lsqr(sparse.diags(sqW) @ A, sqW * b_arr)

        # Re-fix gauge exactly.
        sol = sol.copy()
        sol[0::2] -= sol[2 * anchor_local]
        sol[1::2] -= sol[2 * anchor_local + 1]

        for k, gi in enumerate(comp_idx):
            cx_out[gi] = sol[2 * k] / x_scale
            cy_out[gi] = sol[2 * k + 1] / y_scale

    return cx_out, cy_out


def _wrap_scalar(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def _wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Continuous weighted median via cumulative-weight interpolation.

    Standard discrete weighted median picks the value at the cumulative
    half-weight crossing. This interpolates between adjacent samples,
    which is more accurate when ``len(values)`` is small.
    """
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order]
    cw = np.cumsum(sw) / sw.sum()
    return float(np.interp(0.5, cw, sv))


def _aggregate_pair(
    values: np.ndarray, weights: np.ndarray, *, method: str, is_complex: bool
) -> float:
    """Aggregate per-patch offsets into a single offset for one overlap pair.

    For ``method="median"``, computes a weighted median (after centering
    on the weighted circular mean for wrapped inputs, so the wrap
    convention does not bias the median). For ``method="mean"``, returns
    the weighted mean.
    """
    if method == "median":
        if is_complex:
            center = float(np.angle((np.exp(1j * values) * weights).sum()))
            resid = _wrap(values - center)
            return _wrap_scalar(center + _weighted_median(resid, weights))
        return _weighted_median(values, weights)
    if method == "mean":
        if is_complex:
            return _wrap_scalar(float(np.angle((np.exp(1j * values) * weights).sum())))
        return float(np.average(values, weights=weights))
    raise ValueError(f"unknown method {method!r}")


def _apply_correction(
    arr: np.ndarray,
    corr: BurstCorrection,
    *,
    nodata: float | None,
    transform,
) -> np.ndarray:
    valid = _valid_mask(arr, nodata)
    out = arr.copy()
    if not corr.is_planar:
        if np.iscomplexobj(arr):
            out[valid] = arr[valid] * np.exp(-1j * corr.offset).astype(arr.dtype)
        else:
            out[valid] = arr[valid] - corr.offset
        return out

    h, w = arr.shape
    cols = np.arange(w)
    rows = np.arange(h)
    x_world = transform.a * (cols + 0.5) + transform.c
    y_world = transform.e * (rows + 0.5) + transform.f
    polynomial = corr.evaluate(x_world[None, :], y_world[:, None])
    if np.iscomplexobj(arr):
        out[valid] = arr[valid] * np.exp(-1j * polynomial[valid]).astype(arr.dtype)
    else:
        out[valid] = arr[valid] - polynomial[valid]
    return out


def _build_output_paths(
    in_paths: Sequence[Path], out_dir: Path, *, suffix: str
) -> list[Path]:
    """Pick output paths that don't collide.

    If the input stems are already unique, writes ``<stem><suffix><ext>``
    flat in ``out_dir``. Otherwise — common when several bursts share a
    filename like ``20250725_20250806.int.vrt`` under different burst
    directories — each input's path relative to the inputs' common
    ancestor is used as a prefix, with separators replaced by underscores
    so the output stays a single flat directory.

    The output extension is ``.tif`` for ``.vrt`` inputs (since we write
    a real raster, not a VRT), and the input extension otherwise.
    """
    import os.path

    stems = [p.stem for p in in_paths]
    if len(set(stems)) == len(stems):
        out: list[Path] = []
        for p in in_paths:
            ext = ".tif" if p.suffix.lower() == ".vrt" else p.suffix
            out.append(out_dir / f"{p.stem}{suffix}{ext}")
        return out

    abs_parents = [str(p.resolve().parent) for p in in_paths]
    common = os.path.commonpath(abs_parents) if len(abs_parents) > 1 else ""
    out_paths: list[Path] = []
    for p, parent in zip(in_paths, abs_parents, strict=False):
        rel = os.path.relpath(parent, common) if common else parent
        prefix = rel.replace(os.sep, "_").strip("_") or "burst"
        ext = ".tif" if p.suffix.lower() == ".vrt" else p.suffix
        out_paths.append(out_dir / f"{prefix}__{p.stem}{suffix}{ext}")
    return out_paths


# Used in a type hint above.
from collections.abc import Iterable  # noqa: E402
