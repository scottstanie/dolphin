"""GeoZarr cube writers.

Provides a 3D-cube alternative to the per-layer GeoTIFF stack writers used in
the phase-linking workflow. Where ``BackgroundStackWriter`` creates one
GeoTIFF per layer (one per date, triplet, etc.) and threads block writes
across N files, ``GeoZarrStackWriter`` creates a single 3D array
``(n_layers, height, width)`` inside a Zarr v3 store and writes to it as a
unified cube.

The on-disk layout follows the GeoZarr convention enough to be consumed by
GeoZarr-aware tools (rioxarray, geozarr-toolkit, bowser):

- a ``y`` / ``x`` coordinate dimension at pixel centers
- a ``spatial_ref`` scalar variable carrying ``crs_wkt`` + ``GeoTransform``
- ``grid_mapping = "spatial_ref"`` on each data array (CF convention)
- root ``proj:wkt2`` attribute so non-rioxarray readers can find the CRS

By default the store is written in **zarr v2** format. This is the version
GDAL's ``ZARR`` driver (3.10+) reads transparently via the
``ZARR:"path/cube.zarr":/var:i`` subdataset syntax, which lets rasterio /
GDAL-based downstream code consume cube layers as if they were ordinary 2D
rasters — no tif export required. Pass ``zarr_format=3`` to opt into the
newer spec at the cost of dropping GDAL-side read compatibility.

This module imports ``zarr`` lazily; ``import dolphin.io._geozarr`` does not
itself require zarr. The optional extra ``dolphin[geozarr]`` pulls in the
runtime stack (zarr, xarray, rioxarray).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import rasterio
from numpy.typing import ArrayLike, DTypeLike

from dolphin._types import Filename

from ._background import BackgroundWriter
from ._utils import _unpack_3d_slices, round_mantissa
from ._writers import DatasetStackWriter, DatasetWriter

if TYPE_CHECKING:
    from dolphin._types import Index

__all__ = [
    "BackgroundGeoZarrStackWriter",
    "GeoZarrStackWriter",
    "GeoZarrWriter",
    "create_geozarr_skeleton",
    "emit_layer_vrts",
    "layer_vrt_path",
    "zarr_subdataset_uri",
]

GEOZARR_EXTRAS_ERROR = (
    "GeoZarr output requires the `geozarr` extra. Install with"
    " `pip install dolphin[geozarr]` (pulls in zarr, xarray, rioxarray)."
)
# Zarr v2 is the on-disk format GDAL's ZARR driver supports best as of
# GDAL 3.10 (the ``bytes`` codec required by v3 isn't implemented on the
# read side until later). Stick with v2 by default so downstream GDAL /
# rasterio consumers can open ``ZARR:cube.zarr:/slcs:i`` directly without
# any tif materialization.
DEFAULT_ZARR_FORMAT = 2


def _import_zarr():
    """Import zarr lazily, raising a friendly error if missing."""
    try:
        import zarr
    except ImportError as e:  # pragma: no cover - tested via integration
        raise ImportError(GEOZARR_EXTRAS_ERROR) from e
    return zarr


def _spatial_ref_attrs(crs_wkt: str, geotransform: Sequence[float]) -> dict[str, Any]:
    """Build CF ``spatial_ref`` attrs matching rioxarray's convention.

    ``GeoTransform`` is the 6-element GDAL geotransform serialized as a
    space-separated string so GDAL/rasterio/rioxarray can round-trip it.
    """
    return {
        "crs_wkt": crs_wkt,
        "spatial_ref": crs_wkt,
        "GeoTransform": " ".join(str(float(v)) for v in geotransform),
        "grid_mapping_name": "transverse_mercator",  # placeholder; readers re-derive
    }


def _pixel_center_coords(
    height: int, width: int, geotransform: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel-center ``(y, x)`` 1D coordinate arrays.

    Mirrors the convention rioxarray uses on ``open_rasterio``: each
    coordinate is the geotransform applied to ``index + 0.5``.
    """
    gt = list(geotransform)
    x = (np.arange(width, dtype=np.float64) + 0.5) * gt[1] + gt[0]
    y = (np.arange(height, dtype=np.float64) + 0.5) * gt[5] + gt[3]
    return y, x


def _read_geo_metadata(
    like_filename: Filename,
) -> tuple[int, int, str, tuple[float, ...]]:
    """Pull ``(height, width, crs_wkt, geotransform)`` from a GDAL-readable raster."""
    with rasterio.open(like_filename) as src:
        crs_wkt = src.crs.to_wkt() if src.crs else ""
        gt = src.transform.to_gdal()
        return src.height, src.width, crs_wkt, tuple(gt)


def create_geozarr_skeleton(
    store_path: Filename,
    *,
    height: int,
    width: int,
    crs_wkt: str,
    geotransform: Sequence[float],
    group: str | None = None,
    mode: str = "a",
    zarr_format: int = DEFAULT_ZARR_FORMAT,
) -> None:
    """Initialise a zarr store with ``y``/``x``/``spatial_ref`` coordinates.

    Subsequent calls to ``GeoZarrStackWriter`` / ``GeoZarrWriter`` write 2D
    or 3D data arrays into the same group; they assume y/x/spatial_ref are
    already present.

    Parameters
    ----------
    store_path : Filename
        Path to the zarr store (a directory).
    height, width : int
        Full-resolution raster dimensions.
    crs_wkt : str
        Coordinate reference system as a WKT string.
    geotransform : Sequence[float]
        6-element GDAL geotransform ``(ulx, dx, 0, uly, 0, dy)``.
    group : str, optional
        Sub-group within the store to write to. ``None`` writes at the root.
    mode : str
        Mode for opening the store: ``"w"`` overwrites, ``"a"`` appends.
    zarr_format : int
        Zarr spec version, 2 or 3. Defaults to 2 because GDAL's ZARR driver
        reads v2 cleanly; v3 read support depends on having a recent enough
        GDAL build.

    """
    zarr = _import_zarr()

    y, x = _pixel_center_coords(height, width, geotransform)
    root = zarr.open_group(
        str(store_path), mode=mode, path=group, zarr_format=zarr_format
    )

    # Write coord arrays only if they don't already exist (idempotent).
    if "y" not in root:
        ya = root.create_array("y", shape=y.shape, dtype=y.dtype)
        ya[:] = y
        ya.attrs.update({"_ARRAY_DIMENSIONS": ["y"], "axis": "Y", "units": "m"})
    if "x" not in root:
        xa = root.create_array("x", shape=x.shape, dtype=x.dtype)
        xa[:] = x
        xa.attrs.update({"_ARRAY_DIMENSIONS": ["x"], "axis": "X", "units": "m"})
    if "spatial_ref" not in root:
        sr = root.create_array("spatial_ref", shape=(), dtype="int32")
        sr[...] = 0
        sr.attrs.update(_spatial_ref_attrs(crs_wkt, geotransform))

    # Top-level GeoZarr ``proj:`` hint so non-CF readers can resolve the CRS
    # without having to descend into ``spatial_ref``.
    if crs_wkt and "proj:wkt2" not in root.attrs:
        root.attrs["proj:wkt2"] = crs_wkt


_RESERVED_COORD_NAMES = frozenset({"x", "y", "spatial_ref"})


def _ensure_data_array(
    store_path: Filename,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: DTypeLike,
    chunks: tuple[int, ...] | None,
    shards: tuple[int, ...] | None,
    fill_value: float | None,
    extra_dim_names: Sequence[str] = (),
    group: str | None = None,
    zarr_format: int = DEFAULT_ZARR_FORMAT,
):
    """Create-or-open one data array inside the store and stamp CF attrs."""
    if name in _RESERVED_COORD_NAMES:
        # Quietly returning the existing 1-D coord array would silently
        # corrupt cube writes (the 3-D set would land in a 1-D coordinate),
        # so reject these names up front.
        msg = (
            f"{name!r} is reserved for the spatial coordinate scaffold and"
            " cannot be used as a data variable name."
        )
        raise ValueError(msg)

    zarr = _import_zarr()
    root = zarr.open_group(
        str(store_path), mode="a", path=group, zarr_format=zarr_format
    )

    if name in root:
        existing = root[name]
        if tuple(existing.shape) != tuple(shape):
            msg = (
                f"Array {name!r} already exists in {store_path} with shape"
                f" {tuple(existing.shape)}; requested shape {tuple(shape)}."
            )
            raise ValueError(msg)
        return existing

    create_kwargs: dict[str, Any] = {
        "name": name,
        "shape": shape,
        "dtype": np.dtype(dtype),
    }
    if chunks is not None:
        create_kwargs["chunks"] = chunks
    # Sharding is a v3-only feature; silently drop on v2 stores.
    if shards is not None and zarr_format >= 3:
        create_kwargs["shards"] = shards
    if fill_value is not None:
        create_kwargs["fill_value"] = fill_value

    arr = root.create_array(**create_kwargs)

    dims = [*list(extra_dim_names), "y", "x"]
    arr.attrs["_ARRAY_DIMENSIONS"] = dims
    arr.attrs["grid_mapping"] = "spatial_ref"
    return arr


def _resolve_chunks(
    shape: tuple[int, ...], chunk_yx: int = 256
) -> tuple[int, ...]:
    """Pick sensible zarr chunks for a (..., y, x) array."""
    h, w = shape[-2:]
    yx = (min(chunk_yx, h), min(chunk_yx, w))
    if len(shape) == 2:
        return yx
    # For 3D+, chunk one layer at a time so block writes don't have to align
    # to a multi-layer boundary.
    return (1,) * (len(shape) - 2) + yx


class GeoZarrWriter(DatasetWriter):
    """Block-writable 2D Zarr array inside a GeoZarr store.

    Implements the ``DatasetWriter`` protocol so it can stand in for a
    ``RasterWriter`` in code that writes a single 2D layer.

    Parameters
    ----------
    store_path : Filename
        Path to the parent zarr store.
    name : str
        Variable name inside the store.
    shape : tuple[int, int]
        ``(height, width)`` of the output array.
    dtype : DTypeLike
        Element dtype.
    like_filename : Filename, optional
        Raster to pull CRS/geotransform from for the spatial_ref coord. If
        ``None``, the store must already have ``spatial_ref`` initialised.
    chunks, shards : tuple, optional
        Override chunk / shard shapes. Defaults to ``(256, 256)`` chunks.
    fill_value : float or int, optional
        Fill value (also exposed as the nodata convention to readers).
    keep_bits : int, optional
        Truncate float mantissa to this many bits before write, for better
        compression downstream.
    group : str, optional
        Subgroup within ``store_path``.

    """

    def __init__(
        self,
        store_path: Filename,
        *,
        name: str,
        shape: tuple[int, int],
        dtype: DTypeLike,
        like_filename: Filename | None = None,
        chunks: tuple[int, ...] | None = None,
        shards: tuple[int, ...] | None = None,
        fill_value: float | None = None,
        keep_bits: int | None = None,
        group: str | None = None,
        zarr_format: int = DEFAULT_ZARR_FORMAT,
    ):
        if like_filename is not None:
            h, w, crs_wkt, gt = _read_geo_metadata(like_filename)
            # Shape sanity: a partial mismatch silently corrupts the cube.
            if (h, w) != tuple(shape):
                msg = (
                    f"{like_filename}: raster shape {(h, w)} differs from"
                    f" requested {shape}"
                )
                raise ValueError(msg)
            create_geozarr_skeleton(
                store_path,
                height=h,
                width=w,
                crs_wkt=crs_wkt,
                geotransform=gt,
                group=group,
                zarr_format=zarr_format,
            )

        self.store_path = Path(store_path)
        self.name = name
        self.group = group
        self.zarr_format = zarr_format
        self.keep_bits = keep_bits
        self._shape = tuple(shape)
        self._dtype = np.dtype(dtype)
        self.ndim = 2

        if chunks is None:
            chunks = _resolve_chunks(self._shape)

        self._array = _ensure_data_array(
            store_path,
            name=name,
            shape=self._shape,
            dtype=self._dtype,
            chunks=chunks,
            shards=shards,
            fill_value=fill_value,
            group=group,
            zarr_format=zarr_format,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __setitem__(self, key: tuple[Index, ...], value: np.ndarray, /) -> None:
        if np.issubdtype(value.dtype, np.floating) and self.keep_bits is not None:
            round_mantissa(value, keep_bits=self.keep_bits)
        self._array[key] = value


class GeoZarrStackWriter(DatasetStackWriter):
    """Block-writable 3D Zarr cube inside a GeoZarr store.

    Drop-in replacement for ``BackgroundStackWriter``: where the GeoTIFF
    version creates one file per layer and writes to each, this writes a
    single ``(n_layers, height, width)`` cube. Per-layer attributes (e.g.
    dates) can be passed in via ``layer_coord`` and ``layer_coord_name``.

    Parameters
    ----------
    store_path : Filename
        Path to the parent zarr store (directory).
    name : str
        Variable name for the cube inside the store.
    n_layers : int
        Length of the leading (non-spatial) axis.
    like_filename : Filename, optional
        Raster to copy shape + CRS + geotransform from. If omitted, the
        ``shape`` and ``crs_wkt``/``geotransform`` kwargs must be supplied.
    shape : tuple[int, int], optional
        ``(height, width)`` override; required if ``like_filename`` is None.
    dtype : DTypeLike, optional
        Element dtype. Defaults to the dtype of ``like_filename``.
    chunks, shards : tuple, optional
        Override zarr chunk / shard shapes.
    fill_value : float or int, optional
        Fill value written to the cube before any block writes land.
    keep_bits : int, optional
        Truncate float mantissa to this many bits before write.
    layer_dim_name : str
        Dimension name for the leading axis (e.g. ``"time"`` or ``"pair"``).
    layer_coord_name : str, optional
        Name for a 1D coordinate array on the leading axis (e.g.
        ``"time"`` for dates). ``None`` skips writing the coord.
    layer_coord : array-like, optional
        Values for the layer coordinate. Length must equal ``n_layers``.
    group : str, optional
        Subgroup within ``store_path``.

    """

    ndim = 3

    def __init__(
        self,
        store_path: Filename,
        *,
        name: str,
        n_layers: int,
        like_filename: Filename | None = None,
        shape: tuple[int, int] | None = None,
        dtype: DTypeLike | None = None,
        crs_wkt: str | None = None,
        geotransform: Sequence[float] | None = None,
        chunks: tuple[int, ...] | None = None,
        shards: tuple[int, ...] | None = None,
        fill_value: float | None = None,
        keep_bits: int | None = None,
        layer_dim_name: str = "time",
        layer_coord_name: str | None = None,
        layer_coord: ArrayLike | None = None,
        group: str | None = None,
        zarr_format: int = DEFAULT_ZARR_FORMAT,
    ):
        if like_filename is not None:
            h, w, src_crs, src_gt = _read_geo_metadata(like_filename)
            if shape is None:
                shape = (h, w)
            if crs_wkt is None:
                crs_wkt = src_crs
            if geotransform is None:
                geotransform = src_gt
            if dtype is None:
                with rasterio.open(like_filename) as ds:
                    dtype = np.dtype(ds.dtypes[0])

        if shape is None or dtype is None:
            msg = "Must provide `like_filename` or both `shape` and `dtype`."
            raise ValueError(msg)
        if crs_wkt is None or geotransform is None:
            msg = (
                "Must provide `like_filename` or both `crs_wkt` and `geotransform`."
            )
            raise ValueError(msg)

        create_geozarr_skeleton(
            store_path,
            height=shape[0],
            width=shape[1],
            crs_wkt=crs_wkt,
            geotransform=geotransform,
            group=group,
            zarr_format=zarr_format,
        )

        self.store_path = Path(store_path)
        self.name = name
        self.group = group
        self.zarr_format = zarr_format
        self.crs_wkt = crs_wkt
        self.geotransform = tuple(geotransform)
        self.keep_bits = keep_bits
        self._dtype = np.dtype(dtype)
        self._shape: tuple[int, ...] = (n_layers, shape[0], shape[1])
        self.layer_dim_name = layer_dim_name

        if chunks is None:
            chunks = _resolve_chunks(self._shape)

        self._array = _ensure_data_array(
            store_path,
            name=name,
            shape=self._shape,
            dtype=self._dtype,
            chunks=chunks,
            shards=shards,
            fill_value=fill_value,
            extra_dim_names=(layer_dim_name,),
            group=group,
            zarr_format=zarr_format,
        )

        if layer_coord is not None and layer_coord_name is not None:
            self._write_layer_coord(layer_coord_name, layer_coord)

    def _write_layer_coord(self, coord_name: str, values: ArrayLike) -> None:
        zarr = _import_zarr()
        arr = np.asarray(values)
        if arr.shape != (self._shape[0],):
            msg = (
                f"layer_coord length {arr.shape} doesn't match n_layers"
                f" {self._shape[0]}"
            )
            raise ValueError(msg)
        root = zarr.open_group(
            str(self.store_path),
            mode="a",
            path=self.group,
            zarr_format=self.zarr_format,
        )
        if coord_name in root:
            return

        # zarr v3 has no datetime dtype yet; serialize datetimes as ISO
        # strings via the variable-length UTF-8 codec so xarray can parse
        # them back on read.
        if np.issubdtype(arr.dtype, np.datetime64):
            strs = [np.datetime_as_string(x, unit="s") for x in arr]
            ca = root.create_array(coord_name, shape=arr.shape, dtype="str")
            ca[:] = strs
        elif arr.dtype.kind in ("U", "S", "O"):
            ca = root.create_array(coord_name, shape=arr.shape, dtype="str")
            ca[:] = [str(x) for x in arr]
        else:
            ca = root.create_array(coord_name, shape=arr.shape, dtype=arr.dtype)
            ca[:] = arr
        ca.attrs["_ARRAY_DIMENSIONS"] = [self.layer_dim_name]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def write_block(
        self,
        data: ArrayLike,
        row_start: int,
        col_start: int,
        layer: int | slice | None = None,
    ) -> None:
        """Write a 2D or 3D block at ``(row_start, col_start)``.

        ``layer`` selects the leading-axis slice: ``None`` writes the full
        cube (data must be 3D with ``n_layers`` along axis 0), an int writes
        one layer, a slice writes contiguous layers.
        """
        arr = np.asarray(data)
        if (
            np.issubdtype(arr.dtype, np.floating)
            and self.keep_bits is not None
        ):
            # Cast to writable in case caller passed a view of a frozen array.
            if not arr.flags.writeable:
                arr = arr.copy()
            round_mantissa(arr, keep_bits=self.keep_bits)

        if arr.ndim == 2:
            nrows, ncols = arr.shape
            rows = slice(row_start, row_start + nrows)
            cols = slice(col_start, col_start + ncols)
            band: Any = slice(None) if layer is None else layer
            self._array[band, rows, cols] = arr
            return

        # 3D path
        _, nrows, ncols = arr.shape
        rows = slice(row_start, row_start + nrows)
        cols = slice(col_start, col_start + ncols)
        if layer is None:
            self._array[:, rows, cols] = arr
        else:
            self._array[layer, rows, cols] = arr

    def __setitem__(self, key: tuple[Index, ...], value: np.ndarray, /) -> None:
        bands, rows, cols = _unpack_3d_slices(key)
        arr = np.asarray(value)
        if (
            np.issubdtype(arr.dtype, np.floating)
            and self.keep_bits is not None
        ):
            if not arr.flags.writeable:
                arr = arr.copy()
            round_mantissa(arr, keep_bits=self.keep_bits)
        self._array[bands, rows, cols] = arr


class BackgroundGeoZarrStackWriter(BackgroundWriter, DatasetStackWriter):
    """Background-thread wrapper around ``GeoZarrStackWriter``.

    Mirrors ``BackgroundStackWriter`` so workflow code can swap the GeoTIFF
    stack writer out for a Zarr cube writer with no other call-site changes.

    The constructor accepts everything ``GeoZarrStackWriter`` does, plus
    ``max_queue`` / ``debug`` for the background-thread plumbing.
    """

    def __init__(
        self,
        store_path: Filename,
        *,
        name: str,
        n_layers: int,
        max_queue: int = 0,
        debug: bool = False,
        **stack_kwargs: Any,
    ):
        super().__init__(nq=max_queue, name="GeoZarrStackWriter")
        if debug:
            # Synchronous mode: short-circuit the background thread so test
            # failures surface with a useful stack trace.
            self.notify_finished()
            self.queue_write = self.write  # type: ignore[assignment]

        self._writer = GeoZarrStackWriter(
            store_path, name=name, n_layers=n_layers, **stack_kwargs
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._writer.shape

    @property
    def dtype(self) -> np.dtype:
        return self._writer.dtype

    @property
    def closed(self) -> bool:
        return self._thread.is_alive() is False

    def write(
        self,
        data: ArrayLike,
        row_start: int,
        col_start: int,
        layer: int | slice | None = None,
    ) -> None:
        """Foreground write; called from the background thread by the queue."""
        self._writer.write_block(data, row_start, col_start, layer=layer)

    def __setitem__(self, key: tuple[Index, ...], value: np.ndarray, /) -> None:
        bands, rows, cols = _unpack_3d_slices(key)
        # Same constraint as BackgroundStackWriter: we only support writes
        # that span all layers, since per-layer queueing would require the
        # caller to drive the layer index.
        if bands not in (slice(None), slice(None, None, None), ...):
            self.notify_finished()
            raise NotImplementedError("Can only write to all layers at once.")
        self.queue_write(value, rows.start, cols.start)

    def close(self) -> None:
        """Drain the background queue and release the writer."""
        self.notify_finished()


# ----------------------------------------------------------------------
# VRT shims: thin XML files that expose a single 2D layer of a cube as
# an ordinary GDAL-readable raster. This is how the rest of the
# tif-based pipeline (interferogram formation, stitching, unwrap)
# consumes GeoZarr outputs without any code changes: a downstream caller
# opens the .vrt with rasterio/GDAL, and GDAL's ZARR driver pulls the
# actual pixel data from the cube on demand. No data is duplicated.
# ----------------------------------------------------------------------

_VRT_TEMPLATE = """\
<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
  <SRS>{crs_wkt}</SRS>
  <GeoTransform>{gt0}, {gt1}, {gt2}, {gt3}, {gt4}, {gt5}</GeoTransform>
  <VRTRasterBand dataType="{gdal_dtype}" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{source}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
      <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""

# Numpy dtype name -> GDAL data type string used in VRT XML.
_NUMPY_TO_GDAL_VRT_DTYPE: dict[str, str] = {
    "uint8": "Byte",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "uint64": "UInt64",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
    "complex64": "CFloat32",
    "complex128": "CFloat64",
}


def zarr_subdataset_uri(
    store_path: Filename, variable: str, layer: int, group: str | None = None
) -> str:
    """Build a GDAL ``ZARR:`` connection string for one 2D slice of a cube.

    The result can be opened by ``gdal.Open`` / ``rasterio.open`` directly.
    ``layer`` selects the index along the leading (non-spatial) dimension.
    The store path is resolved to an absolute path so VRTs referencing it
    stay valid if they're later moved between directories (e.g. the
    per-ministack → top-level shuffle in ``sequential.py``).
    """
    inner = f"/{group}/{variable}" if group else f"/{variable}"
    return f'ZARR:"{Path(store_path).resolve()}":{inner}:{layer}'


def layer_vrt_path(output_path: Path | str) -> Path:
    """Return the canonical VRT path corresponding to a planned tif path.

    Replaces a trailing ``.tif`` with ``.vrt`` so downstream globs that look
    for stack outputs by basename keep working with minimal pattern tweaks.
    """
    p = Path(output_path)
    if p.suffix == ".tif":
        return p.with_suffix(".vrt")
    return p.with_suffix(p.suffix + ".vrt")


def emit_layer_vrts(
    *,
    store_path: Filename,
    variable: str,
    layer_paths: Sequence[Path],
    height: int,
    width: int,
    dtype: DTypeLike,
    crs_wkt: str,
    geotransform: Sequence[float],
    group: str | None = None,
) -> list[Path]:
    """Write one VRT per layer of ``variable`` inside the cube at ``store_path``.

    Each VRT references ``ZARR:store_path:/[group/]variable:i`` as its source
    filename, so GDAL/rasterio readers see a normal 2D raster with the cube
    layer's pixel data, transform, and CRS. No data is copied — the VRT is a
    handful of bytes per layer.

    Parameters
    ----------
    store_path : Filename
        Path to the zarr store directory.
    variable : str
        Array name inside the store (e.g. ``"slcs"``).
    layer_paths : Sequence[Path]
        Output VRT paths, one per layer along the cube's leading axis. The
        number of paths must match the cube's leading-axis length, but that
        is checked by the caller.
    height, width : int
        Pixel dimensions of each layer.
    dtype : DTypeLike
        Numpy dtype of cube elements; mapped to a GDAL VRT data type name.
    crs_wkt : str
        Spatial reference written into the VRT's ``<SRS>`` element.
    geotransform : Sequence[float]
        6-element GDAL geotransform.
    group : str, optional
        Subgroup within the store, if the variable lives inside one.

    Returns
    -------
    list[Path]
        The same ``layer_paths`` (resolved to ``Path`` objects), now on disk.

    """
    np_dtype = np.dtype(dtype)
    gdal_dtype = _NUMPY_TO_GDAL_VRT_DTYPE.get(np_dtype.name)
    if gdal_dtype is None:
        msg = f"No GDAL VRT dtype mapping for numpy dtype {np_dtype.name!r}"
        raise ValueError(msg)

    gt = tuple(float(v) for v in geotransform)
    if len(gt) != 6:
        raise ValueError(f"geotransform must have 6 elements, got {len(gt)}")

    out: list[Path] = []
    for i, raw_path in enumerate(layer_paths):
        p = Path(raw_path)
        p.parent.mkdir(exist_ok=True, parents=True)
        source = zarr_subdataset_uri(store_path, variable, i, group=group)
        p.write_text(
            _VRT_TEMPLATE.format(
                xsize=width,
                ysize=height,
                crs_wkt=crs_wkt,
                gt0=gt[0],
                gt1=gt[1],
                gt2=gt[2],
                gt3=gt[3],
                gt4=gt[4],
                gt5=gt[5],
                gdal_dtype=gdal_dtype,
                source=source,
            )
        )
        out.append(p)
    return out
