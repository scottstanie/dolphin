"""Lightweight wall-time profiling for the phase-linking block loop.

Temporary instrumentation used to decide whether
[`run_wrapped_phase_single`][dolphin.workflows.single.run_wrapped_phase_single]
is I/O-bound or compute-bound before reworking the parallelism strategy.

The per-stage timers are thread-safe so they can wrap sections of
``_process_block``, which runs concurrently across a ``ThreadPoolExecutor``.
Reads in that loop are serialized by a lock, so ``sum(read)`` is real
wall-clock critical-path time; the compute stages overlap across workers, so
their summed time can legitimately exceed the loop wall-clock.
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

__all__ = ["StageTimer", "benchmark_read"]


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


def benchmark_read(
    file_path: Filename,
    *,
    subdataset: str | None = None,
    block: tuple[int, int] = (1024, 1024),
) -> None:
    """Log the read cost and compression of one input granule.

    Reads a ``block``-sized window from the first band of ``file_path``
    twice. The first read pays disk I/O plus decompression; the second is
    typically served from the OS page cache, so the cold-minus-warm gap is a
    rough proxy for raw disk-I/O time, and the warm read for
    decompression-plus-copy time.

    Parameters
    ----------
    file_path
        A representative (non-compressed) input granule.
    subdataset
        HDF5/NetCDF subdataset path, if the input is an HDF5/NetCDF file.
    block
        ``(rows, cols)`` window size to read.

    """
    from osgeo import gdal

    gdal.UseExceptions()
    path = f'NETCDF:"{file_path}":{subdataset}' if subdataset else str(file_path)
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)
    ny = min(block[0], ds.RasterYSize)
    nx = min(block[1], ds.RasterXSize)
    compression = band.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") or "NONE"
    driver = ds.GetDriver().ShortName

    t0 = time.perf_counter()
    band.ReadAsArray(0, 0, nx, ny)
    cold_ms = 1e3 * (time.perf_counter() - t0)

    t0 = time.perf_counter()
    band.ReadAsArray(0, 0, nx, ny)
    warm_ms = 1e3 * (time.perf_counter() - t0)

    ds = None  # release GDAL handle
    logger.info(
        "[read-benchmark] %s: driver=%s compression=%s block=%dx%d "
        "cold=%.0fms warm=%.0fms (cold-warm ~= disk I/O, warm ~= decompress+copy)",
        file_path,
        driver,
        compression,
        ny,
        nx,
        cold_ms,
        warm_ms,
    )
