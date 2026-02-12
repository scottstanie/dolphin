"""Utilities for planning and merging overlapping temporal segments."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from opera_utils import get_dates

from dolphin import io

logger = logging.getLogger("dolphin")


@dataclass
class SegmentInfo:
    """Metadata for a single temporal segment of the SLC stack."""

    index: int
    """Segment number (0-based)."""

    file_list: list[Path]
    """SLC files in this segment."""

    dates: list[datetime]
    """Acquisition dates in this segment."""

    overlap_with_previous: list[datetime] = field(default_factory=list)
    """Dates shared with the previous segment (empty for the first segment)."""

    @property
    def start_date(self) -> datetime:
        return self.dates[0]

    @property
    def end_date(self) -> datetime:
        return self.dates[-1]


def plan_segments(
    cslc_file_list: Sequence[Path],
    segment_size: int,
    overlap_size: int,
    file_date_fmt: str = "%Y%m%d",
) -> list[SegmentInfo]:
    """Split a list of SLC files into overlapping temporal segments.

    Parameters
    ----------
    cslc_file_list : Sequence[Path]
        List of CSLC files, sorted by date.
    segment_size : int
        Number of SLC acquisitions per segment.
    overlap_size : int
        Number of overlapping acquisitions between adjacent segments.
    file_date_fmt : str
        Date format in filenames.

    Returns
    -------
    list[SegmentInfo]
        List of segment metadata, one per planned segment.

    Raises
    ------
    ValueError
        If overlap_size >= segment_size, or the stack is too small.

    """
    if overlap_size >= segment_size:
        msg = (
            f"overlap_size ({overlap_size}) must be < segment_size ({segment_size})"
        )
        raise ValueError(msg)

    n_files = len(cslc_file_list)
    if n_files < 2:
        msg = f"Need at least 2 SLC files, got {n_files}"
        raise ValueError(msg)

    step = segment_size - overlap_size
    dates = [get_dates(f, fmt=file_date_fmt)[0] for f in cslc_file_list]

    segments: list[SegmentInfo] = []
    start = 0
    seg_idx = 0

    while start < n_files:
        end = min(start + segment_size, n_files)
        seg_files = list(cslc_file_list[start:end])
        seg_dates = dates[start:end]

        # Compute overlap with the previous segment
        if seg_idx > 0:
            prev_seg = segments[seg_idx - 1]
            overlap_dates = sorted(set(seg_dates) & set(prev_seg.dates))
        else:
            overlap_dates = []

        segments.append(
            SegmentInfo(
                index=seg_idx,
                file_list=seg_files,
                dates=seg_dates,
                overlap_with_previous=overlap_dates,
            )
        )

        seg_idx += 1
        start += step

        # If the remaining files would form a segment smaller than
        # overlap_size, absorb them into the last segment
        if start < n_files and (n_files - start) <= overlap_size:
            # Extend the last segment to include the remaining files
            remaining_files = list(cslc_file_list[start:n_files])
            remaining_dates = dates[start:n_files]
            last = segments[-1]
            # Only add files that are not already in the segment
            for f, d in zip(remaining_files, remaining_dates):
                if d not in last.dates:
                    last.file_list.append(f)
                    last.dates.append(d)
            break

    logger.info(
        "Planned %d segments from %d SLCs (segment_size=%d, overlap=%d)",
        len(segments),
        n_files,
        segment_size,
        overlap_size,
    )
    for seg in segments:
        logger.info(
            "  Segment %d: %d SLCs, dates %s to %s (overlap with prev: %d dates)",
            seg.index,
            len(seg.file_list),
            seg.dates[0].strftime(file_date_fmt),
            seg.dates[-1].strftime(file_date_fmt),
            len(seg.overlap_with_previous),
        )

    return segments


def merge_timeseries(
    segment_ts_paths: list[list[Path]],
    segments: list[SegmentInfo],
    output_dir: Path,
    merge_method: str = "linear_blend",
    file_date_fmt: str = "%Y%m%d",
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
) -> list[Path]:
    """Merge per-segment timeseries into a single consistent timeseries.

    Each segment has its own reference date (the first date of that segment).
    This function:
      1. Estimates a bulk offset between adjacent segments using the overlap.
      2. Applies the accumulated offsets so all segments are relative to the
         first segment's reference date.
      3. In the overlap regions, blends the two estimates (simple mean or
         linear feathering).

    Parameters
    ----------
    segment_ts_paths : list[list[Path]]
        Per-segment timeseries displacement files.  Each inner list is
        sorted by date and contains one raster per date (excluding the
        reference date, whose displacement is zero by definition).
    segments : list[SegmentInfo]
        Segment metadata (must match ``segment_ts_paths`` in order).
    output_dir : Path
        Directory to write the merged timeseries rasters.
    merge_method : str
        ``"mean"`` or ``"linear_blend"``.
    file_date_fmt : str
        Date format in the filenames.
    block_shape : tuple[int, int]
        Block shape for reading/writing.
    num_threads : int
        Number of parallel blocks.

    Returns
    -------
    list[Path]
        Paths to the merged timeseries rasters (one per unique date,
        excluding the global reference date).

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a unified date list across all segments
    all_dates: list[datetime] = []
    for seg in segments:
        for d in seg.dates:
            if d not in all_dates:
                all_dates.append(d)
    all_dates.sort()

    global_ref_date = all_dates[0]
    output_dates = all_dates[1:]  # skip the global reference (displacement = 0)

    logger.info(
        "Merging %d segments into unified timeseries with %d dates"
        " (global ref: %s)",
        len(segments),
        len(output_dates),
        global_ref_date.strftime(file_date_fmt),
    )

    # --- Parse per-segment timeseries into {date -> path} maps ---
    seg_date_to_path: list[dict[datetime, Path]] = []
    for i, (ts_paths, seg) in enumerate(zip(segment_ts_paths, segments)):
        date_map: dict[datetime, Path] = {}
        for p in ts_paths:
            file_dates = get_dates(p, fmt=file_date_fmt)
            # timeseries files are named <ref_date>_<secondary_date>.tif
            if len(file_dates) >= 2:
                sec_date = file_dates[1]
            else:
                sec_date = file_dates[0]
            date_map[sec_date] = p
        seg_date_to_path.append(date_map)

    # --- Estimate bulk offsets between adjacent segments ---
    # Each segment's timeseries is relative to its own reference date (its
    # first date).  We need to chain them to the global reference.
    #
    # For segments i and i+1, the overlap dates have displacement estimates
    # from both segments.  The bulk offset is estimated as the median
    # difference of the displacement values in the overlap.
    cumulative_offsets: list[NDArray | None] = [None] * len(segments)
    cumulative_offsets[0] = None  # first segment: no offset needed

    # Get raster dimensions from the first timeseries file
    first_ts_file = next(iter(seg_date_to_path[0].values()))
    cols, rows = io.get_raster_xysize(first_ts_file)

    for i in range(1, len(segments)):
        seg = segments[i]
        prev_seg = segments[i - 1]
        overlap_dates = seg.overlap_with_previous

        if not overlap_dates:
            logger.warning(
                "Segment %d has no overlap with segment %d — "
                "setting bulk offset to 0",
                i,
                i - 1,
            )
            cumulative_offsets[i] = np.zeros((rows, cols), dtype=np.float32)
            continue

        # For each overlap date, compute the difference between the two
        # segment estimates.  We then take the pixel-wise median.
        diffs: list[NDArray] = []
        for d in overlap_dates:
            path_prev = seg_date_to_path[i - 1].get(d)
            path_cur = seg_date_to_path[i].get(d)
            if path_prev is None or path_cur is None:
                continue

            arr_prev = io.load_gdal(path_prev, masked=True)
            arr_cur = io.load_gdal(path_cur, masked=True)

            # The previous segment's value is already offset-corrected
            # (relative to global ref).  The current segment's value is
            # relative to its own reference.  We also need to re-reference
            # the previous segment's dates to account for the fact that
            # the overlap dates in the *previous* segment measure
            # displacement relative to the previous segment's start,
            # which has already been chained.
            diff = np.asarray(arr_prev, dtype=np.float64) - np.asarray(
                arr_cur, dtype=np.float64
            )
            diffs.append(diff)

        if diffs:
            # Pixel-wise median of the differences
            diff_stack = np.stack(diffs, axis=0)
            bulk_offset = np.nanmedian(diff_stack, axis=0).astype(np.float32)
        else:
            bulk_offset = np.zeros((rows, cols), dtype=np.float32)

        cumulative_offsets[i] = bulk_offset
        logger.info(
            "Segment %d bulk offset: median=%.4f rad",
            i,
            np.nanmedian(bulk_offset),
        )

    # --- Write the merged timeseries ---
    from dolphin.utils import format_dates

    merged_paths: list[Path] = []
    for out_date in output_dates:
        out_name = output_dir / f"{format_dates(global_ref_date, out_date)}.tif"
        merged_paths.append(out_name)

        if out_name.exists():
            continue

        # Collect contributions from all segments that cover this date
        contributions: list[tuple[int, Path]] = []
        for seg_i, date_map in enumerate(seg_date_to_path):
            if out_date in date_map:
                contributions.append((seg_i, date_map[out_date]))

        if not contributions:
            logger.warning("No segment covers date %s", out_date)
            continue

        if len(contributions) == 1:
            # Only one segment covers this date: apply offset and write
            seg_i, src_path = contributions[0]
            arr = io.load_gdal(src_path, masked=True)
            out_arr = np.asarray(arr, dtype=np.float32)
            if cumulative_offsets[seg_i] is not None:
                out_arr = out_arr + cumulative_offsets[seg_i]
            nodata = io.get_raster_nodata(src_path)
            io.write_arr(
                arr=out_arr,
                output_name=out_name,
                like_filename=src_path,
                nodata=nodata,
            )
        else:
            # Multiple segments cover this date (overlap region) -> blend
            arrays = []
            nodata_val = None
            like_file = None
            for seg_i, src_path in contributions:
                arr = np.asarray(
                    io.load_gdal(src_path, masked=True), dtype=np.float32
                )
                if cumulative_offsets[seg_i] is not None:
                    arr = arr + cumulative_offsets[seg_i]
                arrays.append((seg_i, arr))
                if nodata_val is None:
                    nodata_val = io.get_raster_nodata(src_path)
                    like_file = src_path

            out_arr = _blend_overlap(
                arrays=arrays,
                segments=segments,
                out_date=out_date,
                merge_method=merge_method,
            )
            io.write_arr(
                arr=out_arr,
                output_name=out_name,
                like_filename=like_file,
                nodata=nodata_val,
            )

    logger.info("Wrote %d merged timeseries rasters to %s", len(merged_paths), output_dir)
    return merged_paths


def _blend_overlap(
    arrays: list[tuple[int, NDArray]],
    segments: list[SegmentInfo],
    out_date: datetime,
    merge_method: str,
) -> NDArray:
    """Blend displacement arrays from overlapping segments for one date.

    Parameters
    ----------
    arrays : list[tuple[int, NDArray]]
        List of (segment_index, displacement_array) tuples.
    segments : list[SegmentInfo]
        Full segment metadata.
    out_date : datetime
        The date being blended.
    merge_method : str
        ``"mean"`` or ``"linear_blend"``.

    Returns
    -------
    NDArray
        Blended displacement array.

    """
    if len(arrays) == 1:
        return arrays[0][1]

    if merge_method == "mean":
        stack = np.stack([a for _, a in arrays], axis=0)
        return np.nanmean(stack, axis=0).astype(np.float32)

    # linear_blend: for two overlapping segments, ramp weights linearly
    # across the overlap dates.
    if len(arrays) == 2:
        seg_i_early, arr_early = arrays[0]
        seg_i_late, arr_late = arrays[1]
        # Make sure early is actually the earlier segment
        if seg_i_early > seg_i_late:
            seg_i_early, arr_early, seg_i_late, arr_late = (
                seg_i_late,
                arr_late,
                seg_i_early,
                arr_early,
            )
        overlap_dates = segments[seg_i_late].overlap_with_previous
        if not overlap_dates:
            # fallback to mean
            return np.nanmean(
                np.stack([arr_early, arr_late], axis=0), axis=0
            ).astype(np.float32)

        n_overlap = len(overlap_dates)
        overlap_dates_sorted = sorted(overlap_dates)
        try:
            pos = overlap_dates_sorted.index(out_date)
        except ValueError:
            # out_date not in the overlap list (shouldn't happen)
            return np.nanmean(
                np.stack([arr_early, arr_late], axis=0), axis=0
            ).astype(np.float32)

        # Weight for the later segment increases linearly from 0 to 1
        w_late = (pos + 1) / (n_overlap + 1)
        w_early = 1.0 - w_late

        blended = (w_early * arr_early + w_late * arr_late).astype(np.float32)
        return blended

    # For > 2 overlapping segments (unlikely but possible), fall back to mean
    stack = np.stack([a for _, a in arrays], axis=0)
    return np.nanmean(stack, axis=0).astype(np.float32)
