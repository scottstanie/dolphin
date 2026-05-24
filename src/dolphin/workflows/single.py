"""Estimate wrapped phase for one ministack of SLCs."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
from numpy.typing import DTypeLike
from tqdm.auto import tqdm

from dolphin import io, shp, similarity
from dolphin._decorators import atomic_output
from dolphin._types import Filename, HalfWindow, Strides
from dolphin.io import BlockIndices, EagerLoader, StridedBlockManager, VRTStack
from dolphin.phase_link import PhaseLinkRuntimeError, compress, run_phase_linking
from dolphin.ps import calc_ps_block
from dolphin.stack import MiniStackInfo
from dolphin.utils import DummyProcessPoolExecutor, grow_nodata_region

from .config import OutputFormat, ShpMethod

logger = logging.getLogger("dolphin")

__all__ = ["run_wrapped_phase_single"]


@dataclass
class OutputFile:
    filename: Path
    dtype: DTypeLike
    strides: Optional[dict[str, int]] = None
    nbands: int = 1
    nodata: float = 0


@atomic_output(output_arg="output_folder", is_dir=True)
def run_wrapped_phase_single(
    *,
    vrt_stack: VRTStack,
    ministack: MiniStackInfo,
    output_folder: Filename,
    half_window: dict,
    strides: Optional[dict] = None,
    beta: float = 0.00,
    zero_correlation_threshold: float = 0.0,
    use_evd: bool = False,
    mask_file: Optional[Filename] = None,
    ps_mask_file: Optional[Filename] = None,
    amp_mean_file: Optional[Filename] = None,
    amp_dispersion_file: Optional[Filename] = None,
    shp_method: ShpMethod = ShpMethod.NONE,
    shp_alpha: float = 0.05,
    shp_nslc: Optional[int] = None,
    similarity_nearest_n: int | None = None,
    write_closure_phase: bool = True,
    write_crlb: bool = True,
    block_shape: tuple[int, int] = (512, 512),
    baseline_lag: Optional[int] = None,
    max_workers: int = 1,
    output_format: OutputFormat = OutputFormat.GEOTIFF,
    **tqdm_kwargs,
):
    """Estimate wrapped phase for one ministack.

    Output files will all be placed in the provided `output_folder`.
    """
    if strides is None:
        strides = {"x": 1, "y": 1}
    # TODO: extract common stuff between here and sequential
    strides_tup = Strides(y=strides["y"], x=strides["x"])
    half_window_tup = HalfWindow(y=half_window["y"], x=half_window["x"])
    output_folder = Path(output_folder)
    input_slc_files = ministack.file_list
    if len(input_slc_files) != vrt_stack.shape[0]:
        raise ValueError(f"{len(ministack.file_list) = }, but {vrt_stack.shape = }")

    # If we are using a different number of SLCs for the amplitude data,
    # we should note that for the SHP finding algorithms
    if shp_nslc is None:
        shp_nslc = len(input_slc_files)

    logger.info(f"{vrt_stack}: from {ministack.dates[0]} to {ministack.dates[-1]}")

    nrows, ncols = vrt_stack.shape[-2:]

    nodata_mask = _get_nodata_mask(mask_file, nrows, ncols)
    ps_mask = _get_ps_mask(ps_mask_file, nrows, ncols)
    amp_mean, amp_variance = _get_amp_mean_variance(amp_mean_file, amp_dispersion_file)

    xhalf, yhalf = half_window["x"], half_window["y"]

    # If we were passed any compressed SLCs in `input_slc_files`,
    # then we want that index for when we create new compressed SLCs.
    # We skip the old compressed SLCs to create new ones
    first_real_slc_idx = ministack.first_real_slc_idx

    msg = (
        f"Processing {len(input_slc_files) - first_real_slc_idx} SLCs +"
        f" {first_real_slc_idx} compressed SLCs. "
    )
    logger.info(msg)

    # Create the background writer for this ministack
    writer = io.BackgroundBlockWriter()

    logger.info(f"Total stack size (in pixels): {vrt_stack.shape}")
    # Set up the output folder with empty files to write into
    like_filename = vrt_stack.outfile

    # Stack outputs (one layer per date / triplet) are written either as
    # per-layer GeoTIFFs (default) or as 3D cubes inside a shared GeoZarr
    # store. The block loop calls ``.write_layer(idx, data, row, col)`` on
    # each stack output, which abstracts over the two implementations.
    slc_stack = _make_stack_output(
        kind="slcs",
        output_format=output_format,
        ministack=ministack,
        output_folder=output_folder,
        like_filename=like_filename,
        strides=strides,
        dtype=np.complex64,
        name_generator=_name_slcs,
        shared_tiff_writer=writer,
        keep_bits=12,
    )
    phase_linked_slc_files: list[Path] = slc_stack.files

    crlb_stack: _LayerStackOutput | None = None
    closure_stack: _LayerStackOutput | None = None
    if write_crlb:
        crlb_stack = _make_stack_output(
            kind="crlb",
            output_format=output_format,
            ministack=ministack,
            output_folder=output_folder,
            like_filename=like_filename,
            strides=strides,
            dtype=np.float32,
            name_generator=_name_crlbs,
            sub_dir="crlb",
            shared_tiff_writer=writer,
            keep_bits=10,
        )
    if write_closure_phase:
        closure_stack = _make_stack_output(
            kind="closure_phases",
            output_format=output_format,
            ministack=ministack,
            output_folder=output_folder,
            like_filename=like_filename,
            strides=strides,
            dtype=np.float32,
            name_generator=_name_closure_phases,
            sub_dir="closure_phases",
            shared_tiff_writer=writer,
            keep_bits=10,
        )
    phase_linked_crlb_files: list[Path] = (
        crlb_stack.files if crlb_stack is not None else []
    )
    closure_phase_files: list[Path] = (
        closure_stack.files if closure_stack is not None else []
    )

    comp_slc_info = ministack.get_compressed_slc_info()

    # Use the real-SLC date range for output file naming
    start_end = ministack.real_slc_date_range_str
    output_files: dict[str, OutputFile] = {
        # The compressed SLC does not used strides, but has extra band for dispersion
        "compressed_slc": OutputFile(
            output_folder / comp_slc_info.filename, np.complex64, nbands=2
        ),
        # but all the rest do:
        "temporal_coherence": OutputFile(
            output_folder / f"temporal_coherence_{start_end}.tif", np.float32, strides
        ),
        "shp_counts": OutputFile(
            output_folder / f"shp_counts_{start_end}.tif", np.uint16, strides
        ),
        "eigenvalues": OutputFile(
            output_folder / f"eigenvalues_{start_end}.tif", np.float32, strides
        ),
        "estimator": OutputFile(
            output_folder / f"estimator_{start_end}.tif",
            np.int8,
            strides,
            nodata=255,
        ),
    }

    for op in output_files.values():
        io.write_arr(
            arr=None,
            like_filename=like_filename,
            output_name=op.filename,
            dtype=op.dtype,
            strides=op.strides,
            nbands=op.nbands,
            nodata=op.nodata,
        )

    # Iterate over the output grid
    block_manager = StridedBlockManager(
        arr_shape=(nrows, ncols),
        block_shape=block_shape,
        strides=strides_tup,
        half_window=half_window_tup,
    )
    # Set up the background loader
    loader = EagerLoader(reader=vrt_stack, block_shape=block_shape)
    # Queue all input slices, skip ones that are all nodata
    blocks: list[
        tuple[BlockIndices, BlockIndices, BlockIndices, BlockIndices, BlockIndices]
    ] = []
    # Queue all input slices, skip ones that are all nodata
    for b in block_manager.iter_blocks():
        in_rows, in_cols = b[2]
        # nodata_mask is numpy convention: True for bad (masked).
        if nodata_mask[in_rows, in_cols].all():
            continue
        # loader.queue_read(in_rows, in_cols)
        blocks.append(b)
    ###########################
    write_lock = Lock()
    read_lock = Lock()

    Executor = ThreadPoolExecutor if max_workers > 1 else DummyProcessPoolExecutor
    pbar = tqdm(total=len(blocks), **tqdm_kwargs)

    def _process_block(
        block: tuple[
            BlockIndices, BlockIndices, BlockIndices, BlockIndices, BlockIndices
        ],
    ):
        (
            (out_rows, out_cols),
            (out_trim_rows, out_trim_cols),
            (in_rows, in_cols),
            (in_no_pad_rows, in_no_pad_cols),
            (in_trim_rows, in_trim_cols),
        ) = block
        with read_lock:
            cur_data, _ = loader.read(in_rows, in_cols)
        if np.all(cur_data == 0) or np.isnan(cur_data).all():
            return block, None, None, None

        cur_data = cur_data.astype(np.complex64)

        # Only actually compute if we need this one
        amp_stack = np.abs(cur_data) if shp_method == "ks" else None

        # Compute the neighbor_arrays for this block
        neighbor_arrays = shp.estimate_neighbors(
            halfwin_rowcol=(yhalf, xhalf),
            alpha=shp_alpha,
            strides=Strides(y=strides_tup[0], x=strides_tup[1]),
            mean=amp_mean[in_rows, in_cols] if amp_mean is not None else None,
            var=amp_variance[in_rows, in_cols] if amp_variance is not None else None,
            nslc=shp_nslc,
            amp_stack=amp_stack,
            method=shp_method,
        )
        try:
            pl_output = run_phase_linking(
                cur_data,
                half_window=half_window_tup,
                strides=strides_tup,
                use_evd=use_evd,
                beta=beta,
                zero_correlation_threshold=zero_correlation_threshold,
                reference_idx=ministack.output_reference_idx,
                nodata_mask=nodata_mask[in_rows, in_cols],
                ps_mask=ps_mask[in_rows, in_cols],
                neighbor_arrays=neighbor_arrays,
                baseline_lag=baseline_lag,
                avg_mag=amp_mean[in_rows, in_cols] if amp_mean is not None else None,
                first_real_slc_idx=ministack.first_real_slc_idx,
                compute_crlb=write_crlb,
            )
        except PhaseLinkRuntimeError as e:
            # note: this is a warning instead of info, since it should
            # get caught at the "skip_empty" step
            msg = f"At block {in_rows.start}, {in_cols.start}: {e}"
            if "are all NaNs" in e.args[0]:
                # Some SLCs in the ministack are all NaNs
                # This happens from a shifting burst window near the edges,
                # and seems to cause no issues
                logger.debug(msg)
            else:
                logger.warning(msg)
            return block, None, None, None

        # Fill in the nan values with 0
        np.nan_to_num(pl_output.cpx_phase, copy=False)
        np.nan_to_num(pl_output.crlb_std_dev, copy=False)
        np.nan_to_num(pl_output.temp_coh, copy=False)

        # Compress the ministack using only the non-compressed SLCs
        # Get the mean to set as pixel magnitudes
        abs_stack = np.abs(cur_data[first_real_slc_idx:, in_trim_rows, in_trim_cols])
        cur_data_mean, cur_amp_dispersion, _ = calc_ps_block(abs_stack)
        cur_comp_slc = compress(
            # Get the inner portion of the full-res SLC data
            cur_data[:, in_trim_rows, in_trim_cols],
            pl_output.cpx_phase[:, out_trim_rows, out_trim_cols],
            first_real_slc_idx=first_real_slc_idx,
            slc_mean=cur_data_mean,
            reference_idx=ministack.compressed_reference_idx,
        )

        # Save each of the MLE estimates (ignoring those corresponding to
        # compressed SLCs indexes)
        assert len(pl_output.cpx_phase[first_real_slc_idx:]) == len(
            phase_linked_slc_files
        )
        # ### Save results ###
        with write_lock:
            # ### Save results ###
            for i, img in enumerate(
                pl_output.cpx_phase[first_real_slc_idx:, out_trim_rows, out_trim_cols]
            ):
                slc_stack.write_layer(i, img, out_rows.start, out_cols.start)

            if write_crlb and crlb_stack is not None:
                for i, img in enumerate(
                    pl_output.crlb_std_dev[
                        first_real_slc_idx:, out_trim_rows, out_trim_cols
                    ]
                ):
                    crlb_stack.write_layer(i, img, out_rows.start, out_cols.start)

            if write_closure_phase and closure_stack is not None:
                # Save closure phases (N-2 images for N dates)
                for i in range(closure_stack.n_layers):
                    closure_img = pl_output.closure_phases[
                        out_trim_rows, out_trim_cols, i
                    ]
                    closure_stack.write_layer(
                        i, closure_img, out_rows.start, out_cols.start
                    )

            # Save the compressed SLC block
            writer.queue_write(
                cur_comp_slc,
                output_files["compressed_slc"].filename,
                in_no_pad_rows.start,
                in_no_pad_cols.start,
                band=1,
            )
            # Save the amplitude dispersion of the real SLC data
            writer.queue_write(
                cur_amp_dispersion,
                output_files["compressed_slc"].filename,
                in_no_pad_rows.start,
                in_no_pad_cols.start,
                band=2,
            )

            # All other outputs are strided (smaller in size)
            out_datas: dict[str, np.ndarray] = {
                "temporal_coherence": pl_output.temp_coh,
                "shp_counts": pl_output.shp_counts,
                "eigenvalues": pl_output.eigenvalues,
                "estimator": pl_output.estimator,
            }
            for key, data in out_datas.items():
                output_file = output_files[key]
                trimmed_data = data[out_trim_rows, out_trim_cols]

                writer.queue_write(
                    # Erode the edge pixels before writing:
                    grow_nodata_region(
                        trimmed_data, nodata=output_file.nodata, n_pixels=2, copy=True
                    ),
                    output_file.filename,
                    out_rows.start,
                    out_cols.start,
                )
            pbar.update()

    with Executor(max_workers) as exc:
        # Consume all blocks from the `.map` call
        deque(exc.map(_process_block, blocks))

    # Block until all the writers for this ministack have finished
    logger.info(f"Waiting to write {writer.num_queued} blocks of data.")
    writer.notify_finished()
    slc_stack.close()
    if crlb_stack is not None:
        crlb_stack.close()
    if closure_stack is not None:
        closure_stack.close()
    logger.info(f"Finished ministack of size {vrt_stack.shape}.")
    loader.notify_finished()

    # GEOZARR: emit thin VRTs that expose each cube layer as a 2D raster
    # so the rest of the tif-based pipeline keeps working with no data
    # duplication. (For GEOTIFF mode this is a no-op.)
    if output_format == OutputFormat.GEOZARR:
        logger.info("Emitting per-layer VRT shims pointing into GeoZarr cube")
        slc_stack.export_tifs(like_filename=like_filename)
        if crlb_stack is not None:
            crlb_stack.export_tifs(like_filename=like_filename)
        if closure_stack is not None:
            closure_stack.export_tifs(like_filename=like_filename)

    # ``repack_rasters`` is a GeoTIFF-only post-pass that re-tiles + tightens
    # compression on the per-layer tifs. The zarr cube is already chunked
    # and compressed at write time (and the writer's ``keep_bits`` knob
    # already applied the mantissa truncation), so we skip it in GEOZARR
    # mode — repack would create one tif per VRT, undoing the no-duplication
    # win.
    if output_format == OutputFormat.GEOTIFF:
        logger.info("Repacking phase linking outputs for more compression")
        io.repack_rasters(phase_linked_slc_files, keep_bits=12)

    logger.info("Creating similarity raster on outputs")
    similarity.create_similarities(
        phase_linked_slc_files,
        output_file=output_folder / f"similarity_{start_end}.tif",
        num_threads=1,
        add_overviews=False,
        nearest_n=similarity_nearest_n,
        block_shape=block_shape,
    )

    if output_format == OutputFormat.GEOTIFF:
        if write_crlb:
            logger.info("Repacking CRLB files for more compression")
            # CRLB needs only low precision output
            io.repack_rasters(phase_linked_crlb_files, use_16_bits=True)
        if write_closure_phase:
            logger.info("Repacking closure phase files for more compression")
            io.repack_rasters(closure_phase_files, keep_bits=10)

    written_comp_slc = output_files["compressed_slc"]
    ccslc_info = ministack.get_compressed_slc_info()
    ccslc_info.write_metadata(output_file=written_comp_slc.filename)
    # TODO: Does it make sense to return anything from this?
    # or just allow user to search through the `output_folder` they provided?


class _MappedRaster:
    """Thin wrapper around RasterReader with optional post-processing on read.

    Delegates all GDAL reading and nodata detection to RasterReader (which uses
    rasterio.windows.Window.from_slices and handles open-ended slices correctly),
    then optionally fills nodata, casts, or inverts the result.
    """

    def __init__(
        self,
        filename: Filename,
        *,
        nodata_fill=None,
        out_dtype: Optional[np.dtype] = None,
        invert: bool = False,
    ):
        self._reader = io.RasterReader.from_file(filename)
        self.shape = self._reader.shape
        self._nodata_fill = nodata_fill
        self._out_dtype = out_dtype
        self._invert = invert

    def __getitem__(self, key):
        block = self._reader[key]
        # RasterReader returns np.ma.MaskedArray when nodata is set.
        if self._nodata_fill is not None and isinstance(block, np.ma.MaskedArray):
            block = block.filled(self._nodata_fill)
        if self._out_dtype is not None:
            block = block.astype(self._out_dtype)
        if self._invert:
            block = ~block
        return block


class _LazyAmpVariance:
    """Lazily computes amplitude variance from mean and dispersion files."""

    def __init__(self, amp_mean_file: Filename, amp_dispersion_file: Filename):
        self._mean = _MappedRaster(
            amp_mean_file, nodata_fill=np.nan, out_dtype=np.float32
        )
        self._disp = _MappedRaster(
            amp_dispersion_file, nodata_fill=np.nan, out_dtype=np.float32
        )
        self.shape = self._mean.shape

    def __getitem__(self, key):
        mean_block = self._mean[key]
        disp_block = self._disp[key]
        return (disp_block * mean_block) ** 2


def _get_nodata_mask(
    mask_file: Optional[Filename],
    nrows: int,
    ncols: int,
) -> _MappedRaster | np.ndarray:
    if mask_file is not None:
        # Return a lazy reader: loads blocks on demand and inverts
        # (mask file has 1=good, 0=bad; numpy convention is True=bad).
        return _MappedRaster(mask_file, nodata_fill=0, out_dtype=bool, invert=True)
    else:
        return np.zeros((nrows, ncols), dtype=bool)


def _get_ps_mask(
    ps_mask_file: Optional[Filename], nrows: int, ncols: int
) -> _MappedRaster | np.ndarray:
    if ps_mask_file is not None:
        return _MappedRaster(ps_mask_file, nodata_fill=0, out_dtype=bool)
    else:
        return np.zeros((nrows, ncols), dtype=bool)


def _get_amp_mean_variance(
    amp_mean_file: Optional[Filename],
    amp_dispersion_file: Optional[Filename],
) -> tuple[Optional[_MappedRaster], Optional[_LazyAmpVariance]]:
    if amp_mean_file is not None and amp_dispersion_file is not None:
        return (
            _MappedRaster(amp_mean_file, nodata_fill=np.nan, out_dtype=np.float32),
            _LazyAmpVariance(amp_mean_file, amp_dispersion_file),
        )
    return None, None


def _name_slcs(ministack: MiniStackInfo) -> list[str]:
    """Generate SLC filenames for the ministack."""
    start_idx = ministack.first_real_slc_idx
    date_strs = ministack.get_date_str_list()[start_idx:]
    return [f"{Path(d).stem}.slc.tif" for d in date_strs]


def _name_crlbs(ministack: MiniStackInfo) -> list[str]:
    """Generate CRLB filenames for the ministack."""
    start_idx = ministack.first_real_slc_idx
    date_strs = ministack.get_date_str_list()[start_idx:]
    return [f"crlb_{Path(d).stem}.tif" for d in date_strs]


def _name_closure_phases(ministack: MiniStackInfo) -> list[str]:
    """Generate closure phase triplet filenames for the ministack."""
    date_strs = ministack.get_date_str_list()
    # Get only the first date in case of compressed
    date_strs = [d.split("_")[0] for d in date_strs]
    # Create triplets
    num_closure_phases = len(date_strs) - 2
    date_triplets = ["_".join(date_strs[i : i + 3]) for i in range(num_closure_phases)]
    return [f"closure_phase_{triplet}.tif" for triplet in date_triplets]


def setup_output_folder(
    ministack: MiniStackInfo,
    name_generator: Callable[[MiniStackInfo], list[str]],
    driver: str = "GTiff",
    dtype="complex64",
    like_filename: Optional[Filename] = None,
    strides: Optional[dict[str, int]] = None,
    nodata: Optional[float] = 0,
    output_folder: Optional[Path] = None,
) -> list[Path]:
    """Create empty raster files in the output folder.

    Used to prepare outputs for phase linking, CRLB estimates,
    and closure phase triplets.

    Parameters
    ----------
    ministack : MiniStackInfo
        [dolphin.stack.MiniStackInfo][] object for the current batch of SLCs
    name_generator : Callable[[MiniStackInfo], list[str]]
        Function that generates the names of the output files for a given
        ministack.
    driver : str, optional
        Name of GDAL driver, by default "GTiff"
    dtype : str, optional
        Numpy datatype of output files, by default "complex64"
    like_filename : Filename, optional
        Filename to use for getting the shape/GDAL metadata of the output files.
        If None, will use the first SLC in `vrt_stack`
    strides : dict[str, int], optional
        Strides to use when creating the empty files, by default {"y": 1, "x": 1}
        Larger strides will create smaller output files, computed using
        [dolphin.io.compute_out_shape][]
    nodata : float, optional
        Nodata value to use for the output files, by default 0.
    output_folder : Path, optional
        Path to output folder, by default None
        If None, will use the same folder as the first SLC in `vrt_stack`

    Returns
    -------
    list[Path]
        list of saved empty files for the outputs of phase linking

    """
    """Create empty output files using custom filename generation logic."""
    if strides is None:
        strides = {"y": 1, "x": 1}
    if output_folder is None:
        output_folder = ministack.output_folder
    # Note: during the workflow, the ministack.output_folder is different than
    # the `run_wrapped_phase_single` argument `output_folder`.
    # The latter is the tempdir made by @atomic_output
    output_folder.mkdir(exist_ok=True, parents=True)

    filenames = name_generator(ministack)
    output_files = []
    for filename in filenames:
        output_path = output_folder / filename

        io.write_arr(
            arr=None,
            like_filename=like_filename,
            output_name=output_path,
            driver=driver,
            nbands=1,
            dtype=dtype,
            strides=strides,
            nodata=nodata,
        )
        output_files.append(output_path)

    return output_files


# ----------------------------------------------------------------------
# Layer-stack outputs: one writer per "kind" of per-date layer.
#
# Both implementations expose the same minimal interface:
#   .files      -> list[Path] handed to downstream consumers
#                  (interferogram formation, stitching, ...) — GeoTIFFs for
#                  the tif backend, VRT shims for the GeoZarr backend.
#   .n_layers   -> int
#   .write_layer(layer_idx, data, row_start, col_start)
#   .close()
#   .export_tifs(like_filename) -> finalise the ``.files`` for this stack.
#                                   No-op for GeoTIFF; emits per-layer VRTs
#                                   for GeoZarr.
#
# When ``output_format=GEOZARR``, the parallel block loop writes natively
# into a single ``cube.zarr`` per ministack (no per-block tif writes during
# the loop). Once the cube is closed, ``export_tifs`` writes one tiny VRT
# per layer; each VRT carries the same shape/SRS/geotransform and points
# at ``ZARR:cube.zarr:/<variable>:i`` as its source. GDAL/rasterio open the
# VRT and pull pixels from the cube on demand, so the existing tif-based
# downstream pipeline (interferogram formation, stitching, unwrap,
# timeseries) sees normal 2D rasters without any data being copied.
# ----------------------------------------------------------------------


class _LayerStackOutput:
    """Base / minimal interface; concrete subclasses below."""

    files: list[Path]
    n_layers: int

    def write_layer(
        self, layer_idx: int, data: np.ndarray, row_start: int, col_start: int
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def export_tifs(self, like_filename: Filename) -> None:  # noqa: ARG002
        """Materialize per-layer tifs from the cube. No-op for TIFF stacks."""
        return None


class _TiffLayerStack(_LayerStackOutput):
    """Per-layer GeoTIFFs written via a shared ``BackgroundBlockWriter``."""

    def __init__(
        self,
        files: list[Path],
        writer: "io.BackgroundBlockWriter",
    ):
        self.files = files
        self.n_layers = len(files)
        self._writer = writer

    def write_layer(
        self, layer_idx: int, data: np.ndarray, row_start: int, col_start: int
    ) -> None:
        self._writer.queue_write(data, self.files[layer_idx], row_start, col_start)

    def close(self) -> None:
        # Writer is shared across stacks; closed once by the caller.
        return None


class _ZarrLayerStack(_LayerStackOutput):
    """3D ``(n_layers, y, x)`` cube inside a shared GeoZarr store.

    Downstream consumers (interferogram formation, stitching, ...) read the
    cube layers through one VRT shim per layer rather than through
    duplicated tif data. Each VRT is a few-hundred-byte XML file pointing
    at ``ZARR:cube.zarr:/<variable>:i`` and exposes that layer to GDAL as
    a normal 2D raster.
    """

    def __init__(
        self,
        store_path: Path,
        name: str,
        n_layers: int,
        tif_paths: list[Path],
        like_filename: Filename,
        dtype: DTypeLike,
        strides: dict[str, int],
        keep_bits: int | None,
    ):
        from dolphin.io._geozarr import (
            BackgroundGeoZarrStackWriter,
            layer_vrt_path,
        )

        self.store_path = store_path
        self.name = name
        self.n_layers = n_layers
        self._dtype = np.dtype(dtype)
        self._strides = strides
        self._keep_bits = keep_bits

        # Per-layer "files" presented to downstream code are VRT shims at
        # the same stem as the planned tif paths; downstream globs that
        # expanded to ``2*.slc.tif`` now match ``2*.slc.vrt`` instead.
        self._vrt_paths = [layer_vrt_path(p) for p in tif_paths]
        self.files = list(self._vrt_paths)

        # Strided "like" geo for the output cube. The block loop emits
        # data at strided resolution (e.g. multi-look output).
        h_out, w_out, gt_out, crs_wkt = _strided_geo(like_filename, strides)
        self._out_height = h_out
        self._out_width = w_out
        self._geotransform = gt_out
        self._crs_wkt = crs_wkt

        self._writer = BackgroundGeoZarrStackWriter(
            store_path=store_path,
            name=name,
            n_layers=n_layers,
            shape=(h_out, w_out),
            dtype=self._dtype,
            crs_wkt=crs_wkt,
            geotransform=gt_out,
            keep_bits=keep_bits,
            layer_dim_name="time",
        )

    def write_layer(
        self, layer_idx: int, data: np.ndarray, row_start: int, col_start: int
    ) -> None:
        self._writer.queue_write(data, row_start, col_start, layer=layer_idx)

    def close(self) -> None:
        self._writer.close()

    def export_tifs(self, like_filename: Filename) -> None:  # noqa: ARG002
        """Emit one VRT per cube layer; no pixel data is copied."""
        from dolphin.io._geozarr import emit_layer_vrts

        emit_layer_vrts(
            store_path=self.store_path,
            variable=self.name,
            layer_paths=self._vrt_paths,
            height=self._out_height,
            width=self._out_width,
            dtype=self._dtype,
            crs_wkt=self._crs_wkt,
            geotransform=self._geotransform,
        )


def _strided_geo(
    like_filename: Filename, strides: dict[str, int]
) -> tuple[int, int, list[float], str]:
    """Return ``(height, width, geotransform, crs_wkt)`` after applying strides."""
    from dolphin.utils import compute_out_shape

    h0, w0 = io.get_raster_xysize(like_filename)[::-1]
    h, w = compute_out_shape(
        (h0, w0), Strides(y=strides["y"], x=strides["x"])
    )
    gt = list(io.get_raster_gt(like_filename))
    gt[1] *= strides["x"]
    gt[5] *= strides["y"]
    crs_wkt = io.get_raster_crs(like_filename).to_wkt()
    return h, w, gt, crs_wkt


def _make_stack_output(
    *,
    kind: str,
    output_format: OutputFormat,
    ministack: MiniStackInfo,
    output_folder: Path,
    like_filename: Filename,
    strides: dict[str, int],
    dtype: DTypeLike,
    name_generator: Callable[[MiniStackInfo], list[str]],
    shared_tiff_writer: "io.BackgroundBlockWriter",
    keep_bits: int | None = None,
    sub_dir: str | None = None,
) -> _LayerStackOutput:
    """Build a stack output of the requested format.

    Both formats compute the planned per-layer tif paths up-front (so
    ``.files`` is always populated for downstream consumers), but only the
    TIFF format writes empty tifs to disk here. The ZARR format defers tif
    materialization to ``export_tifs`` after the block loop finishes.
    """
    target_dir = output_folder / sub_dir if sub_dir else output_folder
    target_dir.mkdir(exist_ok=True, parents=True)
    filenames = name_generator(ministack)
    tif_paths = [target_dir / f for f in filenames]

    if output_format == OutputFormat.GEOTIFF:
        # Allocate empty tifs up-front so the BackgroundBlockWriter can write
        # into them block-by-block.
        for p in tif_paths:
            io.write_arr(
                arr=None,
                like_filename=like_filename,
                output_name=p,
                driver="GTiff",
                nbands=1,
                dtype=np.dtype(dtype),
                strides=strides,
                nodata=0,
            )
        return _TiffLayerStack(files=tif_paths, writer=shared_tiff_writer)

    if output_format == OutputFormat.GEOZARR:
        store_path = output_folder / "cube.zarr"
        return _ZarrLayerStack(
            store_path=store_path,
            name=kind,
            n_layers=len(tif_paths),
            tif_paths=tif_paths,
            like_filename=like_filename,
            dtype=dtype,
            strides=strides,
            keep_bits=keep_bits,
        )

    raise ValueError(f"Unknown output_format {output_format!r}")
