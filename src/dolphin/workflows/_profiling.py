"""Lightweight wall-time profiling for the phase-linking block loop.

Temporary instrumentation used to decide whether
[`run_wrapped_phase_single`][dolphin.workflows.single.run_wrapped_phase_single]
is I/O-bound or compute-bound before reworking the parallelism strategy.

The per-stage timers are thread-safe so they can wrap sections of
``_process_block``, which runs concurrently across a ``ThreadPoolExecutor``.
Reads in that loop are serialized by a lock, so ``sum(read)`` is real
wall-clock critical-path time; the compute stages overlap across workers, so
their summed time can legitimately exceed the loop wall-clock.

[`log_input_read_profile`][dolphin.workflows._profiling.log_input_read_profile]
reports how an input granule is stored (chunking, compression). It is
metadata-only and effectively free; actual read wall-time is measured by
``StageTimer``.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dolphin._types import Filename

logger = logging.getLogger("dolphin")

__all__ = ["StageTimer", "log_input_read_profile"]


@dataclass
class StageTimer:
    """Thread-safe accumulator of wall time spent in named processing stages."""

    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: Lock = field(default_factory=Lock)

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        """Record wall time spent inside the ``with`` block under ``stage``."""
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        with self._lock:
            self.totals[stage] += elapsed
            self.counts[stage] += 1

    def log_summary(
        self,
        *,
        label: str,
        wall_seconds: float,
        n_workers: int,
        n_blocks: int,
        drain_seconds: float,
    ) -> None:
        """Log per-stage totals against the block-loop wall-clock.

        Parameters
        ----------
        label
            Identifier for the ministack being summarized.
        wall_seconds
            Wall-clock duration of the whole block loop.
        n_workers
            Number of concurrent workers the loop ran with.
        n_blocks
            Number of blocks dispatched to the loop.
        drain_seconds
            Time the loop waited on the background writer after the last
            block finished computing (the write-bound tail).

        """
        tag = "[block-profile]"
        logger.info(
            "%s %s: wall=%.1fs, %d block(s), %d worker(s)",
            tag,
            label,
            wall_seconds,
            n_blocks,
            n_workers,
        )
        with self._lock:
            ordered = sorted(self.totals.items(), key=lambda kv: -kv[1])
            for stage, total in ordered:
                count = self.counts[stage]
                per_ms = 1e3 * total / count if count else 0.0
                pct = 100 * total / wall_seconds if wall_seconds else 0.0
                logger.info(
                    "%s   %-11s sum=%8.1fs  %5.1f%% of wall  (%d calls, %.0f ms each)",
                    tag,
                    stage,
                    total,
                    pct,
                    count,
                    per_ms,
                )
            read_total = self.totals.get("read", 0.0)
        logger.info(
            "%s   write-drain  %8.1fs  (wait on background writer)", tag, drain_seconds
        )
        # Reads are lock-serialized, so their summed time is on the critical path.
        read_frac = read_total / wall_seconds if wall_seconds else 0.0
        verdict = "I/O-bound" if read_frac > 0.5 else "compute-bound (or mixed)"
        logger.info(
            "%s   serialized reads are %.0f%% of wall -> likely %s",
            tag,
            100 * read_frac,
            verdict,
        )


def log_input_read_profile(
    file_path: Filename,
    *,
    subdataset: str | None = None,
) -> None:
    """Log how an input granule is stored, to explain its read cost.

    Reports the GDAL driver, compression, on-disk block (chunk) size, raster
    dimensions, and dtype. GDAL must read and decompress every storage block
    overlapping a requested window, so a coarsely chunked granule inflates
    read cost: a ``block_shape`` workflow read can touch many more bytes than
    it returns. In the worst case (one chunk spanning the whole band) any
    sub-window read decompresses the entire band.

    This is metadata-only -- no pixels are read -- so it is effectively free.
    Actual read wall-time is measured by the ``read`` stage of
    [`StageTimer`][dolphin.workflows._profiling.StageTimer] inside the block
    loop.

    Parameters
    ----------
    file_path
        A representative (non-compressed) input granule.
    subdataset
        HDF5/NetCDF subdataset path, if the input is an HDF5/NetCDF file.

    """
    from osgeo import gdal

    gdal.UseExceptions()
    path = f'NETCDF:"{file_path}":{subdataset}' if subdataset else str(file_path)
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)
    # GetBlockSize returns [x, y]; for the HDF5 driver this is the chunk size.
    block_x, block_y = band.GetBlockSize()
    compression = band.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") or "NONE"
    driver = ds.GetDriver().ShortName
    dtype = gdal.GetDataTypeName(band.DataType)
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ds = None  # release GDAL handle
    logger.info(
        "[read-profile] %s: driver=%s dtype=%s size=%dx%d block=%dx%d "
        "compression=%s (sizes are rows x cols)",
        file_path,
        driver,
        dtype,
        ny,
        nx,
        block_y,
        block_x,
        compression,
    )
