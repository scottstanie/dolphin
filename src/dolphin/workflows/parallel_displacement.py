#!/usr/bin/env python
"""Parallel (overlapping-segment) displacement workflow.

Splits a full SLC stack into overlapping temporal segments, processes each
independently, and merges the per-segment timeseries.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from dolphin import __version__, timeseries, utils
from dolphin._log import log_runtime, setup_logging
from dolphin.timeseries import ReferencePoint

from . import displacement
from .config._parallel_displacement import ParallelDisplacementWorkflow
from .displacement import OutputPaths
from .segment import SegmentInfo, merge_timeseries, plan_segments

logger = logging.getLogger("dolphin")


@dataclass
class ParallelOutputPaths:
    """Output files of the `ParallelDisplacementWorkflow`."""

    segment_outputs: list[OutputPaths]
    """Per-segment outputs from the displacement workflow."""

    segments: list[SegmentInfo]
    """Segment planning metadata."""

    merged_timeseries_paths: list[Path] | None
    """Paths to the merged (unified) timeseries rasters."""

    reference_point: ReferencePoint | None
    """Reference point selected during timeseries inversion."""


def _create_segment_cfg(
    cfg: ParallelDisplacementWorkflow,
    segment: SegmentInfo,
) -> ParallelDisplacementWorkflow:
    """Create a per-segment config derived from the parent config.

    The segment config uses the same processing options but has:
    - A subset of the SLC file list
    - A per-segment work directory
    - ministack_size set to the segment length (process in one shot)

    """
    cfg_dict = cfg.model_dump(exclude={"cslc_file_list", "segment_options"})

    # Per-segment work directory: work_dir / segment_<idx>_<start>_<end>
    seg_label = (
        f"segment_{segment.index:03d}"
        f"_{segment.start_date.strftime('%Y%m%d')}"
        f"_{segment.end_date.strftime('%Y%m%d')}"
    )
    top_work = cfg_dict["work_directory"]
    cfg_dict["work_directory"] = Path(top_work) / seg_label
    cfg_dict["cslc_file_list"] = segment.file_list

    # Set ministack_size to cover the whole segment in one pass
    cfg_dict["phase_linking"]["ministack_size"] = len(segment.file_list)

    # We return a plain DisplacementWorkflow for running — the
    # ParallelDisplacementWorkflow validator would re-validate segment
    # options, which are not needed at the segment level.
    from .config import DisplacementWorkflow

    return DisplacementWorkflow(**cfg_dict)


@log_runtime
def run(
    cfg: ParallelDisplacementWorkflow,
    debug: bool = False,
) -> ParallelOutputPaths:
    """Run the parallel displacement workflow.

    Parameters
    ----------
    cfg : ParallelDisplacementWorkflow
        Configuration for the parallel workflow.
    debug : bool, optional
        Enable debug logging, by default False.

    Returns
    -------
    ParallelOutputPaths
        Combined output paths from all segments and the merged timeseries.

    """
    if cfg.log_file is None:
        cfg.log_file = cfg.work_directory / "dolphin.log"
    setup_logging(logger_name="dolphin", debug=debug, filename=cfg.log_file)
    logger.debug(cfg.model_dump())

    if not cfg.worker_settings.gpu_enabled:
        utils.disable_gpu()
    utils.set_num_threads(cfg.worker_settings.threads_per_worker)

    # ------------------------------------------------------------------
    # 1. Plan segments
    # ------------------------------------------------------------------
    seg_opts = cfg.segment_options
    segments = plan_segments(
        cslc_file_list=cfg.cslc_file_list,
        segment_size=seg_opts.segment_size,
        overlap_size=seg_opts.overlap_size,
        file_date_fmt=cfg.input_options.cslc_date_fmt,
    )

    if len(segments) < 2:
        logger.info(
            "Only 1 segment planned — falling back to standard"
            " DisplacementWorkflow"
        )
        from .config import DisplacementWorkflow

        std_cfg = DisplacementWorkflow(**cfg.model_dump(exclude={"segment_options"}))
        std_output = displacement.run(std_cfg, debug=debug)
        return ParallelOutputPaths(
            segment_outputs=[std_output],
            segments=segments,
            merged_timeseries_paths=std_output.timeseries_paths,
            reference_point=std_output.reference_point,
        )

    # ------------------------------------------------------------------
    # 2. Create per-segment configs
    # ------------------------------------------------------------------
    seg_cfgs = [_create_segment_cfg(cfg, seg) for seg in segments]
    for seg_cfg in seg_cfgs:
        seg_cfg.create_dir_tree()

    # ------------------------------------------------------------------
    # 3. Run each segment (in parallel if n_parallel_bursts > 1)
    # ------------------------------------------------------------------
    num_workers = cfg.worker_settings.n_parallel_bursts
    num_parallel = min(num_workers, len(segments))

    Executor = (
        ProcessPoolExecutor if num_parallel > 1 else utils.DummyProcessPoolExecutor
    )
    logger.info(
        "Processing %d segments with %d parallel workers",
        len(segments),
        num_parallel,
    )

    segment_outputs: list[OutputPaths] = [None] * len(segments)  # type: ignore[list-item]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with Executor(max_workers=num_parallel, mp_context=ctx) as exc:
        fut_to_idx = {
            exc.submit(displacement.run, seg_cfg, debug): i
            for i, seg_cfg in enumerate(seg_cfgs)
        }
        for fut in fut_to_idx:
            idx = fut_to_idx[fut]
            segment_outputs[idx] = fut.result()

    # ------------------------------------------------------------------
    # 4. Merge per-segment timeseries
    # ------------------------------------------------------------------
    ts_opts = cfg.timeseries_options
    if not (ts_opts.run_inversion or ts_opts.run_velocity):
        logger.info("Timeseries inversion disabled — skipping merge step")
        _print_summary(cfg)
        return ParallelOutputPaths(
            segment_outputs=segment_outputs,
            segments=segments,
            merged_timeseries_paths=None,
            reference_point=None,
        )

    # Collect timeseries paths from each segment
    segment_ts_paths: list[list[Path]] = []
    for out in segment_outputs:
        if out.timeseries_paths:
            segment_ts_paths.append(out.timeseries_paths)
        else:
            segment_ts_paths.append([])

    if all(len(p) == 0 for p in segment_ts_paths):
        logger.warning("No timeseries outputs found in any segment")
        _print_summary(cfg)
        return ParallelOutputPaths(
            segment_outputs=segment_outputs,
            segments=segments,
            merged_timeseries_paths=None,
            reference_point=None,
        )

    merged_dir = cfg.work_directory / "merged_timeseries"
    merged_paths = merge_timeseries(
        segment_ts_paths=segment_ts_paths,
        segments=segments,
        output_dir=merged_dir,
        merge_method=seg_opts.merge_method.value,
        file_date_fmt=cfg.input_options.cslc_date_fmt,
        block_shape=ts_opts.block_shape,
        num_threads=ts_opts.num_parallel_blocks,
    )

    # Use the reference point from the first segment
    ref_point = segment_outputs[0].reference_point

    # ------------------------------------------------------------------
    # 5. Optionally estimate velocity on the merged timeseries
    # ------------------------------------------------------------------
    if ts_opts.run_velocity and merged_paths:
        velocity_file = merged_dir / "velocity.tif"
        timeseries.create_velocity(
            unw_file_list=merged_paths,
            output_file=velocity_file,
            reference=ref_point,
            block_shape=ts_opts.block_shape,
            num_threads=ts_opts.num_parallel_blocks,
            file_date_fmt=cfg.input_options.cslc_date_fmt,
        )

    _print_summary(cfg)
    return ParallelOutputPaths(
        segment_outputs=segment_outputs,
        segments=segments,
        merged_timeseries_paths=merged_paths,
        reference_point=ref_point,
    )


def _print_summary(cfg):
    """Print the maximum memory usage and version info."""
    max_mem = utils.get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Config file dolphin version: {cfg._dolphin_version}")
    logger.info(f"Current running dolphin version: {__version__}")
