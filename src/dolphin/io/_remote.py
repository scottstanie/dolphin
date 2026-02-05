"""Remote HDF5 file readers for streaming access to cloud-hosted data.

Uses ``opera_utils.disp._remote.open_h5`` (via *fsspec* + *h5py*) so that
large HDF5 products (e.g. NISAR GSLC, ~40 GB) can be read block-by-block
without downloading the entire file first.

The readers implement the :class:`~dolphin.io.StackReader` protocol and can
be used as drop-in replacements for :class:`VRTStack` in the displacement
workflow.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from os import fspath
from pathlib import Path
from typing import Sequence

import numpy as np
from osgeo import gdal, osr

from dolphin._types import Filename

logger = logging.getLogger("dolphin")

__all__ = [
    "RemoteHDF5Reader",
    "RemoteHDF5StackReader",
    "fix_url_scheme",
    "is_remote_url",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_remote_url(path: str | Path) -> bool:
    """Return *True* if *path* looks like a remote URL (http/https/s3).

    Also detects URLs that have been mangled by ``pathlib.Path`` which
    collapses ``//`` → ``/`` (e.g. ``https://host/…`` → ``https:/host/…``).
    """
    s = str(path)
    return s.startswith(("http://", "https://", "s3://", "http:/", "https:/", "s3:/"))


def fix_url_scheme(url: str) -> str:
    """Restore a URL scheme mangled by ``pathlib.Path``.

    ``Path("https://host/path")`` collapses ``//`` → ``/`` on POSIX,
    producing ``https:/host/path``.  This function restores the double slash.
    """
    for scheme in ("https:", "http:", "s3:"):
        if url.startswith(scheme + "/") and not url.startswith(scheme + "//"):
            return url.replace(scheme + "/", scheme + "//", 1)
    return url


# ---------------------------------------------------------------------------
# Single-file reader
# ---------------------------------------------------------------------------


class RemoteHDF5Reader:
    """Read a single dataset from a remote HDF5 file via ``open_h5``.

    The underlying *h5py.File* handle is opened once and cached for the
    lifetime of this object.  Closing can be done explicitly with
    :meth:`close` or via a context manager.

    Parameters
    ----------
    url : str
        Local path, HTTPS URL, or ``s3://`` URI.
    dset_name : str
        HDF5 dataset path inside the file (e.g. ``"/data/VV"``).
    page_size : int
        HDF5 page buffer size in bytes (must be a power of 2).
    rdcc_nbytes : int
        Raw-data chunk cache size in bytes.
    earthdata_username, earthdata_password : str, optional
        Earthdata Login credentials (falls back to ``~/.netrc``).
    asf_endpoint : str
        ASF credential endpoint name.
    fsspec_kwargs : dict, optional
        Extra keyword arguments forwarded to *fsspec*.
    """

    def __init__(
        self,
        url: str,
        dset_name: str,
        page_size: int = 4 * 1024 * 1024,
        rdcc_nbytes: int = 1024**3,
        earthdata_username: str | None = None,
        earthdata_password: str | None = None,
        asf_endpoint: str = "OPERA",
        fsspec_kwargs: dict | None = None,
    ):
        self.url = url
        self.dset_name = dset_name
        self._page_size = page_size
        self._rdcc_nbytes = rdcc_nbytes
        self._auth_kwargs = {
            "earthdata_username": earthdata_username,
            "earthdata_password": earthdata_password,
            "asf_endpoint": asf_endpoint,
        }
        self._fsspec_kwargs = fsspec_kwargs

        # Populated by _open()
        self._h5file = None
        self._dset = None
        self._open()

    # -- protocol attributes filled by _open() --
    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def ndim(self) -> int:  # type: ignore[override]
        return len(self.shape)

    # -- internal ----------------------------------------------------------

    def _open(self) -> None:
        from opera_utils.disp._remote import open_h5

        kw = dict(self._auth_kwargs)
        if self._fsspec_kwargs is not None:
            kw["fsspec_kwargs"] = self._fsspec_kwargs

        self._h5file = open_h5(
            self.url,
            page_size=self._page_size,
            rdcc_nbytes=self._rdcc_nbytes,
            **kw,
        )
        self._dset = self._h5file[self.dset_name]
        self.shape = self._dset.shape
        self.dtype = self._dset.dtype
        self.chunks = self._dset.chunks
        self.nodata: float | None = self._dset.attrs.get("_FillValue", None)
        if self.nodata is None:
            self.nodata = self._dset.attrs.get("missing_value", None)

    # -- DatasetReader interface -------------------------------------------

    def __getitem__(self, key, /) -> np.ndarray:
        if self._dset is None:
            msg = "Reader is closed."
            raise RuntimeError(msg)
        return np.asarray(self._dset[key])

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying h5py.File handle."""
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:  # noqa: BLE001
                pass
            self._h5file = None
            self._dset = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"RemoteHDF5Reader({self.url!r}, dset={self.dset_name!r})"


# ---------------------------------------------------------------------------
# Stack reader (drop-in replacement for VRTStack)
# ---------------------------------------------------------------------------


class RemoteHDF5StackReader:
    """Stack reader for remote HDF5 files.

    This reader bypasses GDAL's VRT mechanism and instead uses *fsspec* +
    *h5py* (via ``opera_utils.open_h5``) for efficient cloud-optimised
    access to remote HDF5 files such as NISAR GSLC products.

    It exposes the same attributes that :class:`VRTStack` does
    (``file_list``, ``outfile``, ``dates``, ``subdataset``, ``nodata``,
    ``shape``, ``dtype``, ``ndim``) so it can be used as a drop-in
    replacement in the wrapped-phase and sequential workflows.

    Parameters
    ----------
    file_list : sequence of str
        URLs (``https://…`` or ``s3://…``) or local paths to HDF5 files.
    subdataset : str
        HDF5 dataset path inside each file (e.g. ``"/data/VV"``).
    outfile : str or Path
        Path where a small reference GeoTIFF will be written.  Downstream
        code uses this file as a ``like_filename`` when creating outputs.
    reference_file : Path, optional
        An existing local raster whose geotransform/projection will be
        copied to the reference GeoTIFF.  If *None*, the reader attempts
        to extract geospatial metadata from the first HDF5 file.
    sort_files : bool
        Whether to sort files by date (default *True*).
    file_date_fmt : str
        strftime format for parsing dates from filenames.
    page_size, rdcc_nbytes : int
        Cloud-optimised HDF5 parameters forwarded to ``open_h5``.
    earthdata_username, earthdata_password, asf_endpoint : str, optional
        Earthdata credentials forwarded to ``open_h5``.
    fsspec_kwargs : dict, optional
        Extra *fsspec* configuration.
    num_threads : int
        Number of threads for parallel band reads.
    """

    def __init__(
        self,
        file_list: Sequence[str | Filename],
        subdataset: str,
        outfile: str | Path = "slc_stack_reference.tif",
        reference_file: Path | None = None,
        sort_files: bool = True,
        file_date_fmt: str = "%Y%m%d",
        page_size: int = 4 * 1024 * 1024,
        rdcc_nbytes: int = 1024**3,
        earthdata_username: str | None = None,
        earthdata_password: str | None = None,
        asf_endpoint: str = "OPERA",
        fsspec_kwargs: dict | None = None,
        num_threads: int = 1,
    ):
        from opera_utils import get_dates, sort_files_by_date

        self.subdataset = subdataset
        self._page_size = page_size
        self._rdcc_nbytes = rdcc_nbytes
        self._auth_kwargs = {
            "earthdata_username": earthdata_username,
            "earthdata_password": earthdata_password,
            "asf_endpoint": asf_endpoint,
        }
        self._fsspec_kwargs = fsspec_kwargs
        self.num_threads = num_threads

        # ----- sort files by date -----------------------------------------
        files: list[str] = [str(f) for f in file_list]
        dates_list = [get_dates(f, fmt=file_date_fmt) for f in files]
        if sort_files:
            files, dates_list = sort_files_by_date(
                files, file_date_fmt=file_date_fmt
            )

        self.file_list: list[str] = files
        self.dates = dates_list

        # ----- open the first reader to determine shape/dtype -------------
        self._readers: dict[int, RemoteHDF5Reader] = {}
        first_reader = self._get_reader(0)
        self._shape_2d: tuple[int, ...] = first_reader.shape
        self._dtype: np.dtype = first_reader.dtype
        self.nodata: float | None = first_reader.nodata

        # ----- create a small reference GeoTIFF for like_filename ---------
        self.outfile = Path(outfile).resolve()
        if reference_file is not None:
            _copy_reference_raster(reference_file, self.outfile, first_reader.shape)
        else:
            _create_reference_raster(
                self.outfile,
                h5file=first_reader._h5file,
                dset_name=subdataset,
                shape_2d=first_reader.shape,
                dtype=first_reader.dtype,
                nodata=first_reader.nodata,
            )

    # ----- reader cache ---------------------------------------------------

    def _get_reader(self, idx: int) -> RemoteHDF5Reader:
        """Return a (possibly cached) reader for index *idx*."""
        if idx not in self._readers:
            self._readers[idx] = RemoteHDF5Reader(
                url=self.file_list[idx],
                dset_name=self.subdataset,
                page_size=self._page_size,
                rdcc_nbytes=self._rdcc_nbytes,
                fsspec_kwargs=self._fsspec_kwargs,
                **self._auth_kwargs,
            )
        return self._readers[idx]

    # ----- StackReader protocol -------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.file_list), *self._shape_2d)  # type: ignore[return-value]

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index) -> np.ndarray:
        # Simple integer index → read a full 2-D slice
        if isinstance(index, int):
            if index < 0:
                index = len(self) + index
            return self._get_reader(index)[...]

        n, rows, cols = index
        if isinstance(rows, int):
            rows = slice(rows, rows + 1)
        if isinstance(cols, int):
            cols = slice(cols, cols + 1)

        if isinstance(n, int):
            if n < 0:
                n = len(self) + n
            return self._get_reader(n)[rows, cols]

        if n is ...:
            n = slice(None)

        bands = list(range(len(self)))[n]

        def _read_band(i: int) -> np.ndarray:
            return self._get_reader(i)[rows, cols]

        if self.num_threads <= 1 or len(bands) <= 1:
            data = np.stack([_read_band(i) for i in bands], axis=0)
        else:
            with ThreadPoolExecutor(max_workers=self.num_threads) as exc:
                data = np.stack(list(exc.map(_read_band, bands)), axis=0)

        return data

    # ----- convenience for creating sub-stacks ----------------------------

    def create_subset(
        self,
        file_list: Sequence[str | Filename],
        outfile: str | Path,
        sort_files: bool = False,
        file_date_fmt: str = "%Y%m%d",
    ) -> "RemoteHDF5StackReader":
        """Create a new reader backed by a subset of files.

        The spatial reference raster is reused from the parent reader.
        """
        return RemoteHDF5StackReader(
            file_list=file_list,
            subdataset=self.subdataset,
            outfile=outfile,
            reference_file=self.outfile,
            sort_files=sort_files,
            file_date_fmt=file_date_fmt,
            page_size=self._page_size,
            rdcc_nbytes=self._rdcc_nbytes,
            fsspec_kwargs=self._fsspec_kwargs,
            num_threads=self.num_threads,
            **self._auth_kwargs,
        )

    # ----- VRTStack compat ------------------------------------------------

    def __fspath__(self) -> str:
        return fspath(self.outfile)

    def __repr__(self) -> str:
        return (
            f"RemoteHDF5StackReader({len(self.file_list)} files,"
            f" outfile={self.outfile})"
        )

    # ----- lifecycle ------------------------------------------------------

    def close(self) -> None:
        for r in self._readers.values():
            r.close()
        self._readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reference-raster helpers
# ---------------------------------------------------------------------------


def _extract_h5_geo_info(
    h5file,
    dset_name: str,
) -> tuple[tuple[float, ...] | None, str | None]:
    """Try to extract a GDAL-style geotransform and WKT projection.

    Looks in several common locations used by NISAR GSLC and OPERA CSLC
    products.  Returns ``(geotransform, projection_wkt)`` or
    ``(None, None)`` when nothing is found.
    """
    import h5py

    # 1. Check for grid_mapping / spatial_ref on the dataset itself
    dset = h5file[dset_name]
    gm_name = dset.attrs.get("grid_mapping", None)
    if gm_name is not None:
        if isinstance(gm_name, bytes):
            gm_name = gm_name.decode()
        # Resolve relative to the dataset's parent group
        parent = "/".join(dset_name.rstrip("/").split("/")[:-1]) or "/"
        gm_path = f"{parent}/{gm_name}" if parent != "/" else f"/{gm_name}"
        if gm_path in h5file:
            gm_ds = h5file[gm_path]
            proj_wkt = _read_attr(gm_ds, "spatial_ref") or _read_attr(
                gm_ds, "crs_wkt"
            )
            gt = _read_attr(gm_ds, "GeoTransform")
            if gt is not None:
                if isinstance(gt, str):
                    gt = tuple(float(x) for x in gt.split())
                elif hasattr(gt, "__iter__"):
                    gt = tuple(float(x) for x in gt)
                return gt, proj_wkt

    # 2. Look for NISAR-style coordinate arrays (xCoordinates / yCoordinates)
    parent_group_path = "/".join(dset_name.rstrip("/").split("/")[:-1]) or "/"
    if parent_group_path in h5file:
        group = h5file[parent_group_path]
        x_coord = y_coord = None

        for xname in ("xCoordinates", "x_coordinates", "x"):
            if xname in group:
                x_coord = group[xname]
                break
        for yname in ("yCoordinates", "y_coordinates", "y"):
            if yname in group:
                y_coord = group[yname]
                break

        if x_coord is not None and y_coord is not None:
            # Read just the first two values of each coordinate array
            x0 = float(x_coord[0])
            x1 = float(x_coord[1]) if x_coord.shape[0] > 1 else x0 + 1.0
            y0 = float(y_coord[0])
            y1 = float(y_coord[1]) if y_coord.shape[0] > 1 else y0 - 1.0
            dx = x1 - x0
            dy = y1 - y0

            gt = (x0 - dx / 2, dx, 0.0, y0 - dy / 2, 0.0, dy)

            # Look for projection info
            proj_wkt = None
            for proj_name in ("projection", "spatial_ref", "crs"):
                if proj_name in group:
                    pds = group[proj_name]
                    proj_wkt = _read_attr(pds, "spatial_ref") or _read_attr(
                        pds, "crs_wkt"
                    )
                    if proj_wkt is None and isinstance(pds, h5py.Dataset):
                        try:
                            val = pds[()]
                            if isinstance(val, (bytes, np.bytes_)):
                                val = val.decode()
                            if isinstance(val, str) and (
                                "PROJCS" in val or "GEOGCS" in val or "EPSG" in val
                            ):
                                proj_wkt = val
                        except Exception:  # noqa: BLE001
                            pass
                    if proj_wkt is not None:
                        break

            return gt, proj_wkt

    # 3. Check the dataset's own attributes (some producers embed these)
    for attr_name in ("GeoTransform", "geotransform", "geo_transform"):
        gt_raw = _read_attr(dset, attr_name)
        if gt_raw is not None:
            if isinstance(gt_raw, str):
                gt = tuple(float(x) for x in gt_raw.split())
            elif hasattr(gt_raw, "__iter__"):
                gt = tuple(float(x) for x in gt_raw)
            else:
                continue
            proj_wkt = _read_attr(dset, "spatial_ref") or _read_attr(
                dset, "crs_wkt"
            )
            return gt, proj_wkt

    return None, None


def _read_attr(obj, name: str):
    """Read an HDF5 attribute, decoding bytes to str if needed."""
    val = obj.attrs.get(name, None)
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode()
    return val


def _create_reference_raster(
    outfile: Path,
    *,
    h5file,
    dset_name: str,
    shape_2d: tuple[int, ...],
    dtype: np.dtype,
    nodata: float | None,
) -> None:
    """Write a minimal 1-band GeoTIFF that downstream code can use as a
    ``like_filename`` for output file creation.
    """
    gt, proj_wkt = _extract_h5_geo_info(h5file, dset_name)
    nrows, ncols = shape_2d[-2], shape_2d[-1]

    if gt is None:
        # Fall back to a simple pixel-coordinate transform
        logger.warning(
            "Could not extract geotransform from remote HDF5 file."
            " Using identity pixel-coordinate transform for reference raster."
            " Provide a `reference_file` for correct geolocation."
        )
        gt = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    gdal_dtype = _numpy_to_gdal_type(dtype)
    ds = driver.Create(fspath(outfile), ncols, nrows, 1, gdal_dtype)
    ds.SetGeoTransform(gt)

    if proj_wkt:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj_wkt)
        ds.SetSpatialRef(srs)

    if nodata is not None:
        ds.GetRasterBand(1).SetNoDataValue(float(nodata))

    ds.FlushCache()
    ds = None  # close
    logger.info("Created reference raster: %s (%d x %d)", outfile, ncols, nrows)


def _copy_reference_raster(
    src_file: Path,
    dst_file: Path,
    shape_2d: tuple[int, ...],
) -> None:
    """Copy geospatial metadata from an existing raster into a new file."""
    src_ds = gdal.Open(fspath(src_file))
    if src_ds is None:
        msg = f"Cannot open reference file: {src_file}"
        raise FileNotFoundError(msg)

    nrows, ncols = shape_2d[-2], shape_2d[-1]
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(
        fspath(dst_file),
        ncols,
        nrows,
        1,
        src_ds.GetRasterBand(1).DataType,
    )
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())
    srs = src_ds.GetSpatialRef()
    if srs is not None:
        dst_ds.SetSpatialRef(srs)

    nd = src_ds.GetRasterBand(1).GetNoDataValue()
    if nd is not None:
        dst_ds.GetRasterBand(1).SetNoDataValue(nd)

    dst_ds.FlushCache()
    dst_ds = src_ds = None
    logger.info("Copied reference raster from %s → %s", src_file, dst_file)


def _numpy_to_gdal_type(dtype: np.dtype) -> int:
    """Map a NumPy dtype to a GDAL type code."""
    mapping = {
        np.dtype("float32"): gdal.GDT_Float32,
        np.dtype("float64"): gdal.GDT_Float64,
        np.dtype("complex64"): gdal.GDT_CFloat32,
        np.dtype("complex128"): gdal.GDT_CFloat64,
        np.dtype("int16"): gdal.GDT_Int16,
        np.dtype("int32"): gdal.GDT_Int32,
        np.dtype("uint8"): gdal.GDT_Byte,
        np.dtype("uint16"): gdal.GDT_UInt16,
        np.dtype("uint32"): gdal.GDT_UInt32,
    }
    return mapping.get(np.dtype(dtype), gdal.GDT_CFloat32)
