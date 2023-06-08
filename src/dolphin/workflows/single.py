"""Estimate wrapped phase for one updated SLC using the online algorithm.

References
----------
    .. [1] Mirzaee, Sara, Falk Amelung, and Heresh Fattahi. "Non-linear phase
    linking using joined distributed and persistent scatterers." Computers &
    Geosciences (2022): 105291.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from dolphin import io, shp
from dolphin._log import get_log
from dolphin._types import Filename
from dolphin.phase_link import PhaseLinkRuntimeError, compress, run_mle
from dolphin.stack import VRTStack

from ._enums import ShpMethod
from ._utils import setup_output_folder

logger = get_log(__name__)

__all__ = ["run_wrapped_phase_single"]


def run_wrapped_phase_single(
    *,
    slc_vrt_file: Filename,
    output_folder: Filename,
    half_window: dict,
    strides: dict = {"x": 1, "y": 1},
    reference_idx: int = 0,
    beta: float = 0.01,
    mask_file: Optional[Filename] = None,
    ps_mask_file: Optional[Filename] = None,
    amp_mean_file: Optional[Filename] = None,
    amp_dispersion_file: Optional[Filename] = None,
    shp_method: ShpMethod = ShpMethod.NONE,
    shp_alpha: float = 0.05,
    shp_nslc: Optional[int],
    max_bytes: float = 32e6,
    n_workers: int = 1,
    gpu_enabled: bool = True,
) -> tuple[list[Path], Path, Path]:
    """Estimate wrapped phase for one ministack."""
    # TODO: extract common stuff between here and sequential
    output_folder = Path(output_folder)
    vrt = VRTStack.from_vrt_file(slc_vrt_file)
    file_list_all = vrt.file_list
    date_list_all = vrt.dates

    # If we are using a different number of SLCs for the amplitude data,
    # we should note that for the SHP finding algorithms
    if shp_nslc is None:
        shp_nslc = len(file_list_all)

    logger.info(f"{vrt}: from {vrt.file_list[0]} to {vrt.file_list[-1]}")

    nrows, ncols = vrt.shape[-2:]
    if mask_file is not None:
        # The mask file will by 0s at invalid data, 1s at good
        nodata_mask = io.load_gdal(mask_file, masked=True).astype(bool).filled(False)
        # invert the mask so 1s are the missing data pixels
        nodata_mask = ~nodata_mask
        # check middle pixel
        if nodata_mask[nrows // 2, ncols // 2]:
            logger.warning(f"{mask_file} is True at {nrows//2, ncols//2}")
            logger.warning("Proceeding without the nodata mask.")
            nodata_mask = np.zeros((nrows, ncols), dtype=bool)
    else:
        nodata_mask = np.zeros((nrows, ncols), dtype=bool)

    if ps_mask_file is not None:
        ps_mask = io.load_gdal(ps_mask_file, masked=True)
        # Fill the nodata values with false
        ps_mask = ps_mask.astype(bool).filled(False)
    else:
        ps_mask = np.zeros_like(nodata_mask)

    if amp_mean_file is not None and amp_dispersion_file is not None:
        # Note: have to fill, since numba (as of 0.57) can't do masked arrays
        amp_mean = io.load_gdal(amp_mean_file, masked=True).filled(np.nan)
        amp_dispersion = io.load_gdal(amp_dispersion_file, masked=True).filled(np.nan)
        # convert back to variance from dispersion: amp_disp = std_dev / mean
        amp_variance = (amp_dispersion * amp_mean) ** 2
    else:
        amp_mean = amp_variance = None
    amp_stack = None

    xhalf, yhalf = half_window["x"], half_window["y"]
    xs, ys = strides["x"], strides["y"]

    # If we were passed any compressed SLCs in `file_list_all`,
    # then we want that index for when we create new compressed SLCs.
    # We skip the old compressed SLCs to create new ones
    first_non_comp_idx = 0
    for filename in file_list_all:
        if not Path(filename).name.startswith("compressed"):
            break
        first_non_comp_idx += 1

    # Make the output folder using the start/end dates
    d0 = date_list_all[first_non_comp_idx][0]
    d1 = date_list_all[-1][0]
    start_end = io._format_date_pair(d0, d1)

    msg = (
        f"Processing {len(file_list_all) - first_non_comp_idx} SLCs +"
        f" {first_non_comp_idx} compressed SLCs. "
    )
    logger.info(msg)

    # Create the background writer for this ministack
    writer = io.Writer()

    logger.info(
        f"{vrt}: from {Path(vrt.file_list[first_non_comp_idx]).name} to"
        f" {Path(vrt.file_list[-1]).name}"
    )
    logger.info(f"Total stack size (in pixels): {vrt.shape}")
    # Set up the output folder with empty files to write into
    output_slc_files = setup_output_folder(
        vrt,
        driver="GTiff",
        start_idx=first_non_comp_idx,
        strides=strides,
        output_folder=output_folder,
        nodata=0,
    )

    # Create the empty compressed SLC file
    comp_slc_file = output_folder / f"compressed_{start_end}.tif"
    io.write_arr(
        arr=None,
        like_filename=vrt.outfile,
        output_name=comp_slc_file,
        nbands=1,
        nodata=0,
        # Note that the compressed SLC is the same size as the original SLC
        # so we skip the `strides` argument
    )

    # Create the empty compressed temporal coherence file
    tcorr_file = output_folder / f"tcorr_{start_end}.tif"
    io.write_arr(
        arr=None,
        like_filename=vrt.outfile,
        output_name=tcorr_file,
        nbands=1,
        dtype=np.float32,
        strides=strides,
        nodata=0,
    )

    avg_coh_file = output_folder / f"avg_coh_{start_end}.tif"
    io.write_arr(
        arr=None,
        like_filename=vrt.outfile,
        output_name=avg_coh_file,
        nbands=1,
        dtype=np.uint16,
        strides=strides,  # output at SLC resolution
        nodata=0,
    )

    # Create the empty compressed temporal coherence file
    shp_counts_file = output_folder / f"shp_counts_{start_end}.tif"
    io.write_arr(
        arr=None,
        like_filename=vrt.outfile,
        output_name=shp_counts_file,
        nbands=1,
        dtype=np.uint16,
        strides=strides,
        nodata=0,
    )

    # Iterate over the stack in blocks
    # Note the overlap to redo the edge effects
    # TODO: adjust the writing to avoid the overlap

    # Note: dividing by len(stack) since cov is shape (rows, cols, nslc, nslc)
    # so we need to load less to not overflow memory
    stack_max_bytes = max_bytes / len(vrt)
    overlaps = (yhalf, xhalf)
    block_gen = vrt.iter_blocks(
        overlaps=overlaps,
        max_bytes=stack_max_bytes,
        skip_empty=True,
        nodata_mask=nodata_mask,
        show_progress=False,
    )
    for cur_data, (rows, cols) in block_gen:
        if np.all(cur_data == 0):
            continue
        cur_data = cur_data.astype(np.complex64)

        if shp_method == "ks":
            # Only actually compute if we need this one
            amp_stack = np.abs(cur_data)

        # Compute the neighbor_arrays for this block
        neighbor_arrays = shp.estimate_neighbors(
            halfwin_rowcol=(yhalf, xhalf),
            alpha=shp_alpha,
            strides=strides,
            mean=amp_mean[rows, cols] if amp_mean is not None else None,
            var=amp_variance[rows, cols] if amp_variance is not None else None,
            nslc=shp_nslc,
            amp_stack=amp_stack,
            method=shp_method,
        )
        # # Run the phase linking process on the current ministack
        # # TESTING TODO
        reference_idx = max(0, first_non_comp_idx - 1)
        try:
            cur_mle_stack, tcorr, avg_coh = run_mle(
                cur_data,
                half_window=half_window,
                strides=strides,
                beta=beta,
                reference_idx=reference_idx,
                nodata_mask=nodata_mask[rows, cols],
                ps_mask=ps_mask[rows, cols],
                neighbor_arrays=neighbor_arrays,
                avg_mag=amp_mean[rows, cols] if amp_mean is not None else None,
                n_workers=n_workers,
                gpu_enabled=gpu_enabled,
            )
        except PhaseLinkRuntimeError as e:
            # note: this is a warning instead of info, since it should
            # get caught at the "skip_empty" step
            msg = f"At block {rows.start}, {cols.start}: {e}"
            if "are all NaNs" in e.args[0]:
                # Some SLCs in the ministack are all NaNs
                # This happens from a shifting burst window near the edges,
                # and seems to cause no issues
                logger.debug(msg)
            else:
                logger.warning(msg)
            continue

        # Fill in the nan values with 0
        np.nan_to_num(cur_mle_stack, copy=False)
        np.nan_to_num(tcorr, copy=False)

        # Save each of the MLE estimates (ignoring the compressed SLCs)
        assert len(cur_mle_stack[first_non_comp_idx:]) == len(output_slc_files)
        # Get the location within the output file, shrinking down the slices
        out_row_start = rows.start // ys
        out_col_start = cols.start // xs
        for img, f in zip(cur_mle_stack[first_non_comp_idx:], output_slc_files):
            writer.queue_write(img, f, out_row_start, out_col_start)

        # Save the temporal coherence blocks
        writer.queue_write(tcorr, tcorr_file, out_row_start, out_col_start)

        # Save avg coh index
        if avg_coh is not None:
            writer.queue_write(avg_coh, avg_coh_file, out_row_start, out_col_start)
            writer.queue_write(avg_coh, avg_coh_file, out_row_start, out_col_start)
        # Save the SHP counts for each pixel (if not using Rect window)
        shp_counts = np.sum(neighbor_arrays[rows, cols], axis=(-2, -1))
        writer.queue_write(shp_counts, shp_counts_file, out_row_start, out_col_start)

        # Compress the ministack using only the non-compressed SLCs
        cur_comp_slc = compress(
            cur_data[first_non_comp_idx:],
            cur_mle_stack[first_non_comp_idx:],
        )
        # Save the compressed SLC block
        # TODO: make a flag? We don't always need to save the compressed SLCs
        writer.queue_write(cur_comp_slc, comp_slc_file, rows.start, cols.start)
        # logger.debug(f"Saved compressed block SLC to {cur_comp_slc_file}")

    # Block until all the writers for this ministack have finished
    logger.info(f"Waiting to write {writer.num_queued} blocks of data.")
    writer.notify_finished()
    logger.info(f"Finished ministack of size {vrt.shape}.")

    return output_slc_files, comp_slc_file, tcorr_file