"""Module for computing phase similarity between complex interferogram pixels.

Uses metric from [@Wang2022AccuratePersistentScatterer] for similarity.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Literal, Sequence

import numba
import numpy as np
from numba import cuda
from numpy.typing import ArrayLike

from dolphin._types import PathOrStr

logger = logging.getLogger("dolphin")


@numba.njit(nogil=True)
def phase_similarity(x1: ArrayLike, x2: ArrayLike):
    """Compute the similarity between two complex 1D vectors."""
    n = len(x1)
    out = 0.0
    for i in range(n):
        out += np.real(x1[i] * np.conj(x2[i]))
    return out / n


@cuda.jit(device=True)
def _phase_similarity_gpu(ifg_stack, n_ifg, r1, c1, r2, c2):
    """Compute phase similarity between two pixels on the GPU.

    Uses direct indexing into `ifg_stack` rather than slicing, since CUDA
    device functions cannot create new array views.
    """
    out = 0.0
    for i in range(n_ifg):
        # real(x1 * conj(x2)) = real_1 * real_2 + imag_1 * imag_2
        v1 = ifg_stack[i, r1, c1]
        v2 = ifg_stack[i, r2, c2]
        out += v1.real * v2.real + v1.imag * v2.imag
    return out / n_ifg


@cuda.jit(device=True)
def _gpu_insertion_sort(arr, n):
    """In-place insertion sort for a small local array on the GPU."""
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


@cuda.jit(device=True)
def _gpu_nanmedian(arr, n):
    """Compute median of the first `n` elements, skipping NaN values."""
    if n == 0:
        return math.nan
    # Insertion sort the valid values
    _gpu_insertion_sort(arr, n)
    if n % 2 == 1:
        return arr[n // 2]
    else:
        return (arr[n // 2 - 1] + arr[n // 2]) / 2.0


@cuda.jit(device=True)
def _gpu_nanmax(arr, n):
    """Compute max of the first `n` elements."""
    if n == 0:
        return math.nan
    cur_max = arr[0]
    for i in range(1, n):
        if arr[i] > cur_max:
            cur_max = arr[i]
    return cur_max


# Use a constant for the summary type: 0 = median, 1 = max
_SUMMARY_MEDIAN = 0
_SUMMARY_MAX = 1


@cuda.jit
def _similarity_gpu_kernel(
    ifg_stack,
    idxs,
    mask,
    out_similarity,
    sim_buffer,
    summary_type,
):
    """GPU kernel to compute phase similarity for each pixel.

    Each CUDA thread processes one (row, col) pixel.

    Parameters
    ----------
    ifg_stack : 3D complex array (n_ifg, rows, cols)
        Unit interferograms.
    idxs : 2D int array (num_compare_pixels, 2)
        Relative row/col offsets for the circular neighborhood.
    mask : 2D bool array (rows, cols)
        True for valid pixels.
    out_similarity : 2D float32 array (rows, cols)
        Output similarity values (initialized to NaN).
    sim_buffer : 3D float64 array (rows, cols, num_compare_pixels)
        Pre-allocated workspace for per-pixel similarity vectors.
    summary_type : int
        0 for median, 1 for max.
    """
    r0, c0 = cuda.grid(2)
    n_ifg, rows, cols = ifg_stack.shape
    if r0 >= rows or c0 >= cols:
        return
    if not mask[r0, c0]:
        return

    num_compare_pixels = idxs.shape[0]
    count = 0

    for i_idx in range(num_compare_pixels):
        ir = idxs[i_idx, 0]
        ic = idxs[i_idx, 1]
        # Clip to image bounds
        r = max(min(r0 + ir, rows - 1), 0)
        c = max(min(c0 + ic, cols - 1), 0)
        if r == r0 and c == c0:
            continue
        if not mask[r, c]:
            continue

        sim_buffer[r0, c0, count] = _phase_similarity_gpu(
            ifg_stack, n_ifg, r0, c0, r, c
        )
        count += 1

    if count > 0:
        if summary_type == _SUMMARY_MEDIAN:
            out_similarity[r0, c0] = _gpu_nanmedian(sim_buffer[r0, c0], count)
        else:
            out_similarity[r0, c0] = _gpu_nanmax(sim_buffer[r0, c0], count)


def median_similarity(
    ifg_stack: ArrayLike, search_radius: int, mask: ArrayLike | None = None
):
    """Compute the median similarity of each pixel and its neighbors.

    Resulting similarity matches Equation (5) of [@Wang2022AccuratePersistentScatterer]

    Parameters
    ----------
    ifg_stack : ArrayLike
        3D stack of complex interferograms, or floating point phase.
        Shape is (n_ifg, rows, cols)
    search_radius: int
        maximum radius (in pixels) to search for neighbors when comparing each pixel.
    mask: ArrayLike (optional)
        Array of mask from True/False indicating whether to include the pixel (True)
        or ignore it (False).

    Returns
    -------
    np.ndarray
        2D array (shape (rows, cols)) of the median similarity at each pixel.

    """
    return _create_loop_and_run(
        ifg_stack=ifg_stack,
        search_radius=search_radius,
        mask=mask,
        func=np.nanmedian,
    )


def max_similarity(
    ifg_stack: ArrayLike, search_radius: int, mask: ArrayLike | None = None
):
    """Compute the maximum similarity of each pixel and its neighbors.

    Resulting similarity matches Equation (6) of [@Wang2022AccuratePersistentScatterer]

    Parameters
    ----------
    ifg_stack : ArrayLike
        3D stack of complex interferograms, or floating point phase.
        Shape is (n_ifg, rows, cols)
    search_radius: int
        maximum radius (in pixels) to search for neighbors when comparing each pixel.
    mask: ArrayLike (optional)
        Array of mask from True/False indicating whether to include the pixel (True)
        or ignore it (False).

    Returns
    -------
    np.ndarray
        2D array (shape (rows, cols)) of the maximum similarity for any neighbor
        at a pixel.

    """
    return _create_loop_and_run(
        ifg_stack=ifg_stack,
        search_radius=search_radius,
        mask=mask,
        func=np.nanmax,
    )


def _create_loop_and_run(
    ifg_stack: ArrayLike,
    search_radius: int,
    mask: ArrayLike | None,
    func: Callable[[ArrayLike], np.ndarray],
):
    _n_ifg, rows, cols = ifg_stack.shape
    # Mark any nans/all zeros as invalid
    invalid_mask = np.nan_to_num(ifg_stack).sum(axis=0) == 0
    if not np.iscomplexobj(ifg_stack):
        unit_ifgs = np.exp(1j * ifg_stack)
    else:
        unit_ifgs = np.exp(1j * np.angle(ifg_stack))
    out_similarity = np.full((rows, cols), fill_value=np.nan, dtype="float32")
    if mask is None:
        mask = np.ones((rows, cols), dtype="bool")
    mask[invalid_mask] = False

    if mask.shape != (rows, cols):
        raise ValueError(f"{ifg_stack.shape = }, but {mask.shape = }")

    idxs = get_circle_idxs(search_radius)

    from dolphin.utils import gpu_is_available

    if gpu_is_available():
        return _run_gpu(unit_ifgs, idxs, mask, out_similarity, func)
    loop_func = _make_loop_function(func)
    return loop_func(unit_ifgs, idxs, mask, out_similarity)


def _run_gpu(
    unit_ifgs: np.ndarray,
    idxs: np.ndarray,
    mask: np.ndarray,
    out_similarity: np.ndarray,
    func: Callable[[ArrayLike], np.ndarray],
) -> np.ndarray:
    """Run the similarity computation on the GPU."""
    _n_ifg, rows, cols = unit_ifgs.shape
    num_compare_pixels = len(idxs)

    # Determine summary type from the function
    if func is np.nanmedian:
        summary_type = _SUMMARY_MEDIAN
    elif func is np.nanmax:
        summary_type = _SUMMARY_MAX
    else:
        raise ValueError(f"Unsupported GPU summary function: {func}")

    # Ensure complex128 for the GPU kernel (numba cuda supports complex128)
    unit_ifgs_gpu = cuda.to_device(np.ascontiguousarray(unit_ifgs.astype(np.complex128)))
    idxs_gpu = cuda.to_device(np.ascontiguousarray(idxs.astype(np.int32)))
    mask_gpu = cuda.to_device(np.ascontiguousarray(mask))
    out_gpu = cuda.to_device(out_similarity)
    sim_buffer = cuda.to_device(
        np.zeros((rows, cols, num_compare_pixels), dtype=np.float64)
    )

    threads_per_block = (16, 16)
    blocks_per_grid = (
        math.ceil(rows / threads_per_block[0]),
        math.ceil(cols / threads_per_block[1]),
    )
    _similarity_gpu_kernel[blocks_per_grid, threads_per_block](
        unit_ifgs_gpu, idxs_gpu, mask_gpu, out_gpu, sim_buffer, summary_type
    )
    return out_gpu.copy_to_host()


def _make_loop_function(
    summary_func: Callable[[ArrayLike], np.ndarray],
):
    """Create a JIT-ed function for some summary of the neighbors's similarity.

    E.g.: for median similarity, call

        median_sim = _make_loop_function(np.median)
    """

    @numba.njit(nogil=True, parallel=True)
    def _masked_sim_loop(
        ifg_stack: np.ndarray,
        idxs: np.ndarray,
        mask: np.ndarray,
        out_similarity: np.ndarray,
    ) -> np.ndarray:
        """Loop over each pixel, make a masked phase similarity to its neighbors."""
        _, rows, cols = ifg_stack.shape

        num_compare_pixels = len(idxs)
        # Buffer to hold all comparison during the parallel loop
        cur_sim = np.zeros((rows, cols, num_compare_pixels))

        for r0 in numba.prange(rows):
            for c0 in range(cols):
                # Get the current pixel
                m0 = mask[r0, c0]
                if not m0:
                    continue
                x0 = ifg_stack[:, r0, c0]

                cur_sim_vec = cur_sim[r0, c0]
                count = 0

                # compare to all pixels in the circle around it
                for i_idx in range(num_compare_pixels):
                    ir, ic = idxs[i_idx]
                    # Clip to the image bounds
                    r = max(min(r0 + ir, rows - 1), 0)
                    c = max(min(c0 + ic, cols - 1), 0)
                    if r == r0 and c == c0:
                        continue

                    # Check for a pixel to ignore
                    if not mask[r, c]:
                        continue

                    x = ifg_stack[:, r, c]
                    # cur_sim_vec[count] = w * phase_similarity(x0, x)
                    cur_sim_vec[count] = phase_similarity(x0, x)
                    count += 1
                    # Assuming `summary_func` is nan-aware
                if count > 0:  # a 0 count will fail for `max`
                    out_similarity[r0, c0] = summary_func(cur_sim_vec[:count])
        return out_similarity

    return _masked_sim_loop


def get_circle_idxs(
    max_radius: int, min_radius: int = 0, sort_output: bool = True
) -> np.ndarray:
    """Get the relative indices of neighboring pixels in a circle.

    Adapted from c++ version of `psps` package:
    https://github.com/UT-Radar-Interferometry-Group/psps/blob/a15d458817fe7d06a6edaa0b3208ea78bc4782e7/src/cpp/similarity.cpp#L16
    """
    # using the mid-point circle drawing algorithm to search for neighboring PS pixels
    # # code adapted from "https://www.geeksforgeeks.org/mid-point-circle-drawing-algorithm/"
    visited = np.zeros((max_radius, max_radius), dtype=bool)
    visited[0][0] = True

    indices = []
    for r in range(1, max_radius):
        x = r
        y = 0
        p = 1 - r
        if r > min_radius:
            indices.append([r, 0])
            indices.append([-r, 0])
            indices.append([0, r])
            indices.append([0, -r])

        visited[r][0] = True
        visited[0][r] = True
        # flag > 0 means there are holes between concentric circles
        flag = 0
        while x > y:
            # do not need to fill holes
            if flag == 0:
                y += 1
                if p <= 0:
                    # Mid-point is inside or on the perimeter
                    p += 2 * y + 1
                else:
                    # Mid-point is outside the perimeter
                    x -= 1
                    p += 2 * y - 2 * x + 1

            else:
                flag -= 1

            # All the perimeter points have already been visited
            if x < y:
                break

            while not visited[x - 1][y]:
                x -= 1
                flag += 1

            visited[x][y] = True
            visited[y][x] = True
            if r > min_radius:
                indices.append([x, y])
                indices.append([-x, -y])
                indices.append([x, -y])
                indices.append([-x, y])

                if x != y:
                    indices.append([y, x])
                    indices.append([-y, -x])
                    indices.append([y, -x])
                    indices.append([-y, x])

            if flag > 0:
                x += 1

    if sort_output:
        # Sorting makes it run faster, better data access patterns
        return np.array(sorted(indices))
    else:
        # Indices run from middle outward
        return np.array(indices)


def create_similarities(
    ifg_file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    search_radius: int = 7,
    sim_type: Literal["median", "max"] = "median",
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
    add_overviews: bool = True,
    nearest_n: int | None = None,
):
    """Create a similarity raster from as stack of ifg files.

    Parameters
    ----------
    ifg_file_list : Sequence[PathOrStr]
        Paths to input interferograms
    output_file : PathOrStr
        Output raster path
    search_radius : int, optional
        Maximum radius to search for pixels, by default 7
    sim_type : str, optional
        Type of similarity function to run, by default "median"
        Choices: "median", "max"
    block_shape : tuple[int, int], optional
        Size of blocks to process at one time from `ifg_file_list`
        by default (512, 512)
    num_threads : int, optional
        Number of parallel blocks to process, by default 5
    add_overviews : bool, optional
        Whether to create overviews in `output_file` by default True
    nearest_n : int, optional
        If provided, reform the nearest N interferograms before computing similarity.

    """
    from dolphin._overviews import Resampling, create_image_overviews
    from dolphin.io import BackgroundRasterWriter, VRTStack, process_blocks
    from dolphin.timeseries import get_incidence_matrix

    if Path(output_file).exists():
        logger.info(f"{output_file} exists, skipping")
        return

    if sim_type == "median":
        sim_function = median_similarity
    elif sim_type == "max":
        sim_function = max_similarity
    else:
        raise ValueError(f"Unrecognized {sim_type = }")

    nodata_block = np.full(block_shape, fill_value=np.nan, dtype="float32")

    if nearest_n is not None:
        incidence_matrix = get_incidence_matrix(
            _create_nearest_n_pairs(len(ifg_file_list) + 1, n=nearest_n)
        )
        assert incidence_matrix.shape[1] == len(ifg_file_list)
    else:
        incidence_matrix = None

    def calc_sim(readers, rows, cols):
        block = readers[0][:, rows, cols]
        if np.sum(block) == 0 or np.isnan(block).all():
            return nodata_block[rows, cols], rows, cols

        if incidence_matrix is not None:
            block = _calc_nearest_diffs(block, incidence_matrix)

        out_avg = sim_function(ifg_stack=block, search_radius=search_radius)
        logger.debug(f"{rows = }, {cols = }, {block.shape = }, {out_avg.shape = }")
        return out_avg, rows, cols

    out_dir = Path(output_file).parent
    reader = VRTStack(ifg_file_list, outfile=out_dir / "sim_inputs.vrt")

    writer = BackgroundRasterWriter(
        output_file,
        like_filename=ifg_file_list[0],
        dtype="float32",
        driver="GTiff",
        nodata=np.nan,
    )
    process_blocks(
        [reader],
        writer,
        func=calc_sim,
        block_shape=block_shape,
        overlaps=(search_radius, search_radius),
        num_threads=num_threads,
    )
    writer.notify_finished()

    if add_overviews:
        logger.info("Creating overviews for unwrapped images")
        create_image_overviews(Path(output_file), resampling=Resampling.AVERAGE)


def _calc_nearest_diffs(block, incidence_matrix) -> np.ndarray:
    # Multiply the single-ref data by tall and skinny A matrix
    # to give the nearest-n differences
    num_imgs, rows, cols = block.shape
    block_mask = np.nan_to_num(block).sum(axis=0) == 0
    m, num_imgs = incidence_matrix.shape
    phase = np.angle(block) if np.iscomplexobj(block) else block
    columns = np.dot(incidence_matrix, phase.reshape(num_imgs, -1))
    block = columns.reshape(m, rows, cols)
    block[:, block_mask] = np.nan
    return np.exp(1j * block)


def _create_nearest_n_pairs(num_files: int, n: int = 3) -> list[tuple[int, int]]:
    """Create nearest-n interferogram pair indices for a list of `num_files` inputs."""
    ijs = []
    for i in range(num_files):
        for j in range(i + 1, i + n + 1):
            if j >= num_files:
                continue
            ijs.append((i, j))
    return ijs
