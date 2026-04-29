"""Tests for dolphin.burst_alignment."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from dolphin.burst_alignment import (
    BurstCorrection,
    align_bursts,
    apply_burst_offsets,
    estimate_burst_offsets,
)


def _write_raster(path, arr, transform, crs, nodata):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": arr.dtype,
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def _make_overlapping_pair(tmp_path, dtype, true_offset, suffix=".tif"):
    """Two abutting tiles with along-track overlap and a known phase offset.

    Tile A: rows 0..120, Tile B: rows 80..200, overlap rows 80..120.
    """
    crs = "EPSG:32611"
    res = 20.0
    cols = 100

    rng = np.random.default_rng(0)
    full_rows = 200
    phase = rng.uniform(-np.pi, np.pi, (full_rows, cols)).astype(np.float32)
    if np.issubdtype(dtype, np.complexfloating):
        scene = np.exp(1j * phase).astype(dtype)
    else:
        scene = phase.astype(dtype)

    a = scene[0:120].copy()
    b_src = scene[80:200].copy()
    if np.issubdtype(dtype, np.complexfloating):
        b = (b_src * np.exp(1j * true_offset)).astype(dtype)
    else:
        b = (b_src + true_offset).astype(dtype)

    top_y = full_rows * res
    transform_a = from_origin(0.0, top_y, res, res)
    transform_b = from_origin(0.0, top_y - 80 * res, res, res)

    path_a = tmp_path / f"a{suffix}"
    path_b = tmp_path / f"b{suffix}"
    _write_raster(path_a, a, transform_a, crs, nodata=0)
    _write_raster(path_b, b, transform_b, crs, nodata=0)
    return path_a, path_b


def test_estimate_offsets_complex(tmp_path):
    true_offset = 0.7
    a, b = _make_overlapping_pair(tmp_path, np.complex64, true_offset)

    corr = estimate_burst_offsets([a, b])

    assert isinstance(corr[a], BurstCorrection)
    assert corr[a].offset == 0.0
    assert abs(corr[b].offset - true_offset) < 1e-3


def test_estimate_offsets_unwrapped(tmp_path):
    true_offset = 1.234
    a, b = _make_overlapping_pair(tmp_path, np.float32, true_offset)

    corr = estimate_burst_offsets([a, b])

    assert corr[a].offset == 0.0
    assert abs(corr[b].offset - true_offset) < 1e-3


def test_estimate_offsets_wrapped_handles_pi_minus(tmp_path):
    true_offset = np.pi - 0.3
    a, b = _make_overlapping_pair(tmp_path, np.complex64, true_offset)

    corr = estimate_burst_offsets([a, b])

    assert abs(corr[b].offset - true_offset) < 5e-3


def test_apply_offsets_complex(tmp_path):
    true_offset = 0.5
    a, b = _make_overlapping_pair(tmp_path, np.complex64, true_offset)

    out_paths, _ = align_bursts([a, b], tmp_path / "out")

    residual = estimate_burst_offsets(out_paths)
    assert abs(residual[out_paths[1]].offset) < 1e-3


def test_apply_offsets_unwrapped(tmp_path):
    true_offset = 2.5
    a, b = _make_overlapping_pair(tmp_path, np.float32, true_offset)

    out_paths, _ = align_bursts([a, b], tmp_path / "out")

    residual = estimate_burst_offsets(out_paths)
    assert abs(residual[out_paths[1]].offset) < 1e-3


def test_three_bursts_chain(tmp_path):
    crs = "EPSG:32611"
    res = 20.0
    cols = 80
    full_rows = 240
    rng = np.random.default_rng(42)
    phase = rng.uniform(-np.pi, np.pi, (full_rows, cols)).astype(np.float32)
    scene = np.exp(1j * phase).astype(np.complex64)
    top_y = full_rows * res

    o_b, o_c = 0.4, -0.6
    a = scene[0:100].copy()
    b = (scene[80:180] * np.exp(1j * o_b)).astype(np.complex64)
    c = (scene[160:240] * np.exp(1j * o_c)).astype(np.complex64)

    pa = tmp_path / "a.tif"
    pb = tmp_path / "b.tif"
    pc = tmp_path / "c.tif"
    _write_raster(pa, a, from_origin(0.0, top_y, res, res), crs, 0)
    _write_raster(pb, b, from_origin(0.0, top_y - 80 * res, res, res), crs, 0)
    _write_raster(pc, c, from_origin(0.0, top_y - 160 * res, res, res), crs, 0)

    corr = estimate_burst_offsets([pa, pb, pc])
    assert corr[pa].offset == 0.0
    assert abs(corr[pb].offset - o_b) < 5e-3
    assert abs(corr[pc].offset - o_c) < 5e-3


def test_apply_offset_passthrough_value(tmp_path):
    a, _ = _make_overlapping_pair(tmp_path, np.complex64, 0.0)
    corr = estimate_burst_offsets([a])
    assert corr[a].offset == 0.0
    out = apply_burst_offsets([a], corr, tmp_path / "out")
    with rasterio.open(a) as ra, rasterio.open(out[0]) as rb:
        np.testing.assert_array_equal(ra.read(1), rb.read(1))


def _make_planar_pair(tmp_path, dtype, true_offset, true_cx, true_cy):
    """Two abutting tiles with a known planar artifact on burst B.

    Burst B's artifact, evaluated at world (x, y), is::
        offset + cx*(x - x_ref) + cy*(y - y_ref)
    where (x_ref, y_ref) is the centroid of the overlap.
    """
    crs = "EPSG:32611"
    res = 20.0
    cols = 200
    full_rows = 240

    rng = np.random.default_rng(7)
    phase = rng.uniform(-np.pi, np.pi, (full_rows, cols)).astype(np.float32)
    if np.issubdtype(dtype, np.complexfloating):
        scene = np.exp(1j * phase).astype(dtype)
    else:
        scene = phase.astype(dtype)

    top_y = full_rows * res

    # Tile A: rows 0..140, Tile B: rows 100..240, overlap rows 100..140.
    a = scene[0:140].copy()
    b_src = scene[100:240].copy()

    transform_a = from_origin(0.0, top_y, res, res)
    transform_b = from_origin(0.0, top_y - 100 * res, res, res)

    # Pick a reference point inside the overlap (use overlap centroid).
    overlap_y_top = top_y - 100 * res
    overlap_y_bot = top_y - 140 * res
    x_ref = cols * res / 2
    y_ref = (overlap_y_top + overlap_y_bot) / 2

    b_h, b_w = b_src.shape
    cols_b = np.arange(b_w)
    rows_b = np.arange(b_h)
    x_world = transform_b.a * (cols_b + 0.5) + transform_b.c
    y_world = transform_b.e * (rows_b + 0.5) + transform_b.f
    poly = (
        true_offset
        + true_cx * (x_world[None, :] - x_ref)
        + true_cy * (y_world[:, None] - y_ref)
    )
    if np.issubdtype(dtype, np.complexfloating):
        b = (b_src * np.exp(1j * poly)).astype(dtype)
    else:
        b = (b_src + poly).astype(dtype)

    path_a = tmp_path / "a.tif"
    path_b = tmp_path / "b.tif"
    _write_raster(path_a, a, transform_a, crs, nodata=0)
    _write_raster(path_b, b, transform_b, crs, nodata=0)
    return path_a, path_b


def test_planar_fit_unwrapped(tmp_path):
    # Real-valued planar artifact: offset + cx*x + cy*y on burst B.
    # Disable the Tikhonov prior to check the unbiased estimator.
    a, b = _make_planar_pair(
        tmp_path,
        np.float32,
        true_offset=0.5,
        true_cx=1e-4,  # rad / m
        true_cy=-2e-4,
    )
    corr = estimate_burst_offsets([a, b], degree=1, max_fringes_per_burst=None)

    out_paths, _ = align_bursts(
        [a, b], tmp_path / "out", degree=1, max_fringes_per_burst=None
    )
    residual = estimate_burst_offsets(out_paths, degree=0)
    assert abs(residual[out_paths[1]].offset) < 1e-3
    assert corr[a].cx == 0.0 and corr[a].cy == 0.0
    # Sign convention: corr.evaluate is the *artifact* model; B's
    # artifact has +cx, +cy, so corr[b].cx should match true_cx.
    assert abs(corr[b].cx - 1e-4) < 5e-6
    assert abs(corr[b].cy - (-2e-4)) < 5e-6


def test_planar_fit_wrapped(tmp_path):
    # Same but on wrapped phase. Use small slopes so the cross-overlap
    # ramp stays well below pi.
    a, b = _make_planar_pair(
        tmp_path,
        np.complex64,
        true_offset=0.3,
        true_cx=5e-5,
        true_cy=-5e-5,
    )
    corr = estimate_burst_offsets([a, b], degree=1, max_fringes_per_burst=None)
    out_paths, _ = align_bursts(
        [a, b], tmp_path / "out", degree=1, max_fringes_per_burst=None
    )
    residual = estimate_burst_offsets(out_paths, degree=0)
    assert abs(residual[out_paths[1]].offset) < 5e-3
    assert abs(corr[b].cx - 5e-5) < 5e-6
    assert abs(corr[b].cy - (-5e-5)) < 5e-6


def test_tikhonov_prior_shrinks_noise_only_ramp(tmp_path):
    # Two abutting bursts with no real ramp, just decorrelated phase in
    # the overlap (random noise). With the default prior the planar fit
    # should produce a slope much smaller than the no-prior fit.
    crs = "EPSG:32611"
    res = 20.0
    cols = 200
    full_rows = 240
    rng = np.random.default_rng(123)

    # Identical underlying scene so true ramp is zero.
    phase = rng.uniform(-np.pi, np.pi, (full_rows, cols)).astype(np.float32)

    # Tile A: rows 0..140; Tile B: rows 100..240; overlap rows 100..140.
    a = phase[0:140].copy()
    # Replace overlap rows on B with totally fresh noise so the LSQ has
    # nothing real to lock onto.
    b = phase[100:240].copy()
    b[:40] = rng.uniform(-np.pi, np.pi, (40, cols)).astype(np.float32)

    top_y = full_rows * res
    pa = tmp_path / "a.tif"
    pb = tmp_path / "b.tif"
    _write_raster(pa, a, from_origin(0.0, top_y, res, res), crs, 0)
    _write_raster(pb, b, from_origin(0.0, top_y - 100 * res, res, res), crs, 0)

    corr_no_prior = estimate_burst_offsets(
        [pa, pb], degree=1, max_fringes_per_burst=None
    )
    corr_prior = estimate_burst_offsets([pa, pb], degree=1, max_fringes_per_burst=0.5)

    # The prior should pull cx, cy of burst B much closer to 0.
    no_prior_norm = abs(corr_no_prior[pb].cx) + abs(corr_no_prior[pb].cy)
    prior_norm = abs(corr_prior[pb].cx) + abs(corr_prior[pb].cy)
    assert prior_norm < no_prior_norm
    # Prior-shrunk slope should also be physically tame.
    burst_extent = 240 * res
    max_phase_swing = (abs(corr_prior[pb].cx) + abs(corr_prior[pb].cy)) * burst_extent
    assert max_phase_swing < 2 * np.pi  # less than one fringe


def test_tikhonov_prior_does_not_overshrink_real_ramp(tmp_path):
    # If the data does contain a real ramp, the prior should bend toward
    # it but not erase it. With max_fringes=0.5 (the default), a real
    # quarter-fringe ramp should still come through within ~30%.
    a, b = _make_planar_pair(
        tmp_path,
        np.float32,
        true_offset=0.0,
        true_cx=8e-5,  # ~0.4 fringe across 200 px (4 km) burst
        true_cy=0.0,
    )
    corr = estimate_burst_offsets([a, b], degree=1, max_fringes_per_burst=0.5)
    # The prior introduces some shrinkage, but the slope should still be
    # within 30% of truth on this synthetic scene.
    assert abs(corr[b].cx - 8e-5) / 8e-5 < 0.3
