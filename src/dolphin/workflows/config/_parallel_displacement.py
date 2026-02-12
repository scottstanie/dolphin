from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated

import tyro
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from ._displacement import DisplacementWorkflow

__all__ = [
    "MergeMethod",
    "SegmentOptions",
    "ParallelDisplacementWorkflow",
]

logger = logging.getLogger("dolphin")


class MergeMethod(str, Enum):
    """Method for merging overlapping segment timeseries."""

    MEAN = "mean"
    """Simple average of displacement in the overlap region."""

    LINEAR_BLEND = "linear_blend"
    """Linear feathering: ramp weights from 1->0 and 0->1 across the overlap."""


class SegmentOptions(BaseModel, extra="forbid"):
    """Options controlling how the SLC stack is split into overlapping segments."""

    segment_size: Annotated[
        int,
        tyro.conf.arg(aliases=("--segment-size",)),
    ] = Field(
        28,
        gt=2,
        description=(
            "Number of SLC acquisitions per temporal segment. Each segment is"
            " processed independently (phase linking through timeseries), enabling"
            " full parallelism. A typical value is ~1.25 years of acquisitions"
            " (e.g. 28 for a 12-day repeat)."
        ),
    )

    overlap_size: Annotated[
        int,
        tyro.conf.arg(aliases=("--overlap-size",)),
    ] = Field(
        7,
        gt=0,
        description=(
            "Number of SLC acquisitions that overlap between adjacent segments."
            " This overlap is used to estimate bulk offsets and feather/merge the"
            " per-segment timeseries. A typical value is ~0.25 years of"
            " acquisitions (e.g. 7 for a 12-day repeat)."
        ),
    )

    merge_method: MergeMethod = Field(
        MergeMethod.LINEAR_BLEND,
        description=(
            "Method for combining the overlap region between segments."
            " 'mean' takes the simple average; 'linear_blend' ramps weights"
            " linearly across the overlap for smooth feathering."
        ),
    )

    @model_validator(mode="after")
    def _check_overlap_less_than_segment(self: Self) -> Self:
        if self.overlap_size >= self.segment_size:
            msg = (
                f"overlap_size ({self.overlap_size}) must be strictly less than"
                f" segment_size ({self.segment_size})."
            )
            raise ValueError(msg)
        return self


class ParallelDisplacementWorkflow(DisplacementWorkflow):
    """Configuration for the parallel (overlapping-segment) displacement workflow.

    Instead of processing SLCs sequentially with compressed SLCs carrying
    forward, this workflow splits the full SLC stack into overlapping temporal
    segments.  Each segment is processed independently (and can run in
    parallel), then the per-segment timeseries are merged via a bulk-offset
    estimation and feathering step in the overlap regions.

    This approach is inspired by Descartes Labs' processing pipeline
    (Calef, Olsen, Agram 2024; arXiv:2405.06838) and is well-suited for
    batch reprocessing where the full SLC stack is already available.
    """

    segment_options: SegmentOptions = Field(default_factory=SegmentOptions)

    # The per-segment ministack_size should normally equal the segment_size
    # so that each segment is processed as a single ministack (no sequential
    # linking within the segment).  We override the default ministack_size
    # in the validator below.
    @model_validator(mode="after")
    def _sync_ministack_to_segment(self: Self) -> Self:
        # By default, set ministack_size to segment_size so each segment is
        # processed in one shot (no sequential compressed SLC linking).
        # Users may override to a smaller value for very large segments.
        seg_size = self.segment_options.segment_size
        if self.phase_linking.ministack_size == 15:  # still at default
            self.phase_linking.ministack_size = seg_size
        return self
