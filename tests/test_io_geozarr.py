"""Tests for the GeoZarr cube writers in ``dolphin.io._geozarr``.

Skipped automatically if the optional ``geozarr`` extra (zarr/xarray/rioxarray)
is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from dolphin.io._geozarr import (  # noqa: E402
    BackgroundGeoZarrStackWriter,
    GeoZarrStackWriter,
    GeoZarrWriter,
    create_geozarr_skeleton,
)


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "cube.zarr"


@pytest.fixture
def geo():
    """Plausible UTM-ish (height, width, crs_wkt, geotransform)."""
    crs_wkt = (
        'PROJCS["WGS 84 / UTM zone 33N",GEOGCS["WGS 84",'
        'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Transverse_Mercator"],'
        'PARAMETER["central_meridian",15],UNIT["metre",1],'
        'AUTHORITY["EPSG","32633"]]'
    )
    # ulx=500000, dx=10, uly=5000000, dy=-10
    gt = (500000.0, 10.0, 0.0, 5000000.0, 0.0, -10.0)
    return 32, 48, crs_wkt, gt


def test_create_skeleton(store_path, geo):
    h, w, crs_wkt, gt = geo
    create_geozarr_skeleton(
        store_path, height=h, width=w, crs_wkt=crs_wkt, geotransform=gt
    )

    root = zarr.open_group(str(store_path), mode="r")
    assert "y" in root and "x" in root and "spatial_ref" in root
    assert root["y"].shape == (h,)
    assert root["x"].shape == (w,)
    # Pixel-center convention: first pixel center = uly + 0.5 * dy
    assert root["y"][0] == pytest.approx(gt[3] + 0.5 * gt[5])
    assert root["x"][0] == pytest.approx(gt[0] + 0.5 * gt[1])
    assert "GeoTransform" in dict(root["spatial_ref"].attrs)
    assert root.attrs.get("proj:wkt2") == crs_wkt


def test_create_skeleton_idempotent(store_path, geo):
    """Calling twice with the same args must not error or overwrite coords."""
    h, w, crs_wkt, gt = geo
    create_geozarr_skeleton(
        store_path, height=h, width=w, crs_wkt=crs_wkt, geotransform=gt
    )
    create_geozarr_skeleton(
        store_path, height=h, width=w, crs_wkt=crs_wkt, geotransform=gt
    )


def test_2d_writer_roundtrip(store_path, geo):
    h, w, crs_wkt, gt = geo
    create_geozarr_skeleton(
        store_path, height=h, width=w, crs_wkt=crs_wkt, geotransform=gt
    )
    writer = GeoZarrWriter(
        store_path,
        name="temp_coh",
        shape=(h, w),
        dtype=np.float32,
        fill_value=0.0,
    )
    assert writer.shape == (h, w)
    assert writer.dtype == np.float32
    assert writer.ndim == 2

    data = np.random.rand(10, 12).astype(np.float32)
    writer[5:15, 0:12] = data

    root = zarr.open_group(str(store_path), mode="r")
    arr = root["temp_coh"]
    assert arr.attrs["grid_mapping"] == "spatial_ref"
    np.testing.assert_array_equal(np.asarray(arr[5:15, 0:12]), data)


def test_stack_writer_per_layer_blocks(store_path, geo):
    h, w, crs_wkt, gt = geo
    n_layers = 4
    writer = GeoZarrStackWriter(
        store_path,
        name="slcs",
        n_layers=n_layers,
        shape=(h, w),
        dtype=np.complex64,
        crs_wkt=crs_wkt,
        geotransform=gt,
    )
    assert writer.shape == (n_layers, h, w)
    assert writer.ndim == 3

    rng = np.random.default_rng(0)
    data = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))).astype(
        np.complex64
    )
    # Write each layer to a different spatial offset.
    for i in range(n_layers):
        writer.write_block(data * (i + 1), row_start=i, col_start=2 * i, layer=i)

    root = zarr.open_group(str(store_path), mode="r")
    arr = root["slcs"]
    for i in range(n_layers):
        got = np.asarray(arr[i, i : i + 4, 2 * i : 2 * i + 4])
        np.testing.assert_array_equal(got, (data * (i + 1)).astype(np.complex64))


def test_stack_writer_layer_coord(store_path, geo):
    h, w, crs_wkt, gt = geo
    n_layers = 3
    dates = np.array(
        ["2022-01-01", "2022-01-13", "2022-01-25"], dtype="datetime64[s]"
    )
    GeoZarrStackWriter(
        store_path,
        name="slcs",
        n_layers=n_layers,
        shape=(h, w),
        dtype=np.complex64,
        crs_wkt=crs_wkt,
        geotransform=gt,
        layer_coord_name="time",
        layer_coord=dates,
    )
    root = zarr.open_group(str(store_path), mode="r")
    assert "time" in root
    # Datetimes get serialized as ISO strings (object dtype) for zarr v3.
    stored = list(root["time"][:])
    assert stored[0] == "2022-01-01T00:00:00"


def test_background_stack_writer(store_path, geo):
    h, w, crs_wkt, gt = geo
    n_layers = 3
    writer = BackgroundGeoZarrStackWriter(
        store_path,
        name="closure_phases",
        n_layers=n_layers,
        shape=(h, w),
        dtype=np.float32,
        crs_wkt=crs_wkt,
        geotransform=gt,
        debug=True,  # synchronous so we can read back immediately
    )

    layer = np.full((h, w), 1.5, dtype=np.float32)
    writer.write(layer, row_start=0, col_start=0, layer=0)
    writer.close()

    root = zarr.open_group(str(store_path), mode="r")
    np.testing.assert_array_equal(np.asarray(root["closure_phases"][0]), layer)


def test_stack_writer_shape_mismatch_via_like_filename(tmp_path, slc_file_list, geo):
    _, _, _, _ = geo
    store_path = tmp_path / "cube.zarr"
    # like_filename is 5x10 (from conftest slc_stack); requesting a (3, 7)
    # shape should fail loudly via the GeoZarrWriter sanity check.
    with pytest.raises(ValueError, match="differs from"):
        GeoZarrWriter(
            store_path,
            name="x",
            shape=(3, 7),
            dtype=np.float32,
            like_filename=slc_file_list[0],
        )


def test_keep_bits_rounding(store_path, geo):
    h, w, crs_wkt, gt = geo
    writer = GeoZarrStackWriter(
        store_path,
        name="phase",
        n_layers=1,
        shape=(h, w),
        dtype=np.float32,
        crs_wkt=crs_wkt,
        geotransform=gt,
        keep_bits=8,
    )
    rng = np.random.default_rng(0)
    data = rng.random((h, w), dtype=np.float32)
    expected = data.copy()
    # Mantissa rounding mutates the input in place; pretend we passed a fresh
    # array by copying so we can recompute the expected.
    writer.write_block(data.copy(), row_start=0, col_start=0, layer=0)

    root = zarr.open_group(str(store_path), mode="r")
    got = np.asarray(root["phase"][0])
    # Result should differ from original (truncation actually happened), but
    # not by more than a small fraction.
    assert np.abs(got - expected).max() > 0
    np.testing.assert_allclose(got, expected, atol=1e-2)
