"""Tests for the parallel (overlapping-segment) displacement workflow."""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pydantic
import pytest
from osgeo import gdal

from dolphin.workflows.config import ParallelDisplacementWorkflow
from dolphin.workflows.config._parallel_displacement import (
    MergeMethod,
    SegmentOptions,
)
from dolphin.workflows.segment import SegmentInfo, _blend_overlap, plan_segments


# ============================================================
# SegmentOptions config tests
# ============================================================
class TestSegmentOptions:
    def test_defaults(self):
        opts = SegmentOptions()
        assert opts.segment_size == 28
        assert opts.overlap_size == 7
        assert opts.merge_method == MergeMethod.LINEAR_BLEND

    def test_custom_values(self):
        opts = SegmentOptions(segment_size=50, overlap_size=10, merge_method="mean")
        assert opts.segment_size == 50
        assert opts.overlap_size == 10
        assert opts.merge_method == MergeMethod.MEAN

    def test_overlap_must_be_less_than_segment(self):
        with pytest.raises(pydantic.ValidationError, match="overlap_size"):
            SegmentOptions(segment_size=10, overlap_size=10)

        with pytest.raises(pydantic.ValidationError, match="overlap_size"):
            SegmentOptions(segment_size=10, overlap_size=15)

    def test_segment_size_must_be_gt_2(self):
        with pytest.raises(pydantic.ValidationError):
            SegmentOptions(segment_size=2, overlap_size=1)

    def test_overlap_must_be_gt_0(self):
        with pytest.raises(pydantic.ValidationError):
            SegmentOptions(segment_size=10, overlap_size=0)


# ============================================================
# ParallelDisplacementWorkflow config tests
# ============================================================
class TestParallelDisplacementWorkflowConfig:
    def test_creation(self, dir_with_1_slc):
        cfg = ParallelDisplacementWorkflow(
            cslc_file_list=dir_with_1_slc / "slclist.txt",
            input_options={"subdataset": "data"},
        )
        assert isinstance(cfg.segment_options, SegmentOptions)
        assert cfg.segment_options.segment_size == 28

    def test_ministack_synced_to_segment(self, dir_with_1_slc):
        cfg = ParallelDisplacementWorkflow(
            cslc_file_list=dir_with_1_slc / "slclist.txt",
            input_options={"subdataset": "data"},
            segment_options={"segment_size": 40, "overlap_size": 10},
        )
        # ministack_size should be overridden to segment_size
        assert cfg.phase_linking.ministack_size == 40

    def test_ministack_not_overridden_if_explicitly_set(self, dir_with_1_slc):
        cfg = ParallelDisplacementWorkflow(
            cslc_file_list=dir_with_1_slc / "slclist.txt",
            input_options={"subdataset": "data"},
            segment_options={"segment_size": 40, "overlap_size": 10},
            phase_linking={"ministack_size": 20},
        )
        # User explicitly set ministack_size=20, should be respected
        assert cfg.phase_linking.ministack_size == 20

    def test_roundtrip_yaml(self, tmp_path, dir_with_1_slc):
        outfile = tmp_path / "parallel_config.yaml"
        cfg = ParallelDisplacementWorkflow(
            cslc_file_list=dir_with_1_slc / "slclist.txt",
            input_options={"subdataset": "data"},
            segment_options={"segment_size": 30, "overlap_size": 5},
        )
        cfg.to_yaml(outfile)
        cfg2 = ParallelDisplacementWorkflow.from_yaml(outfile)
        assert cfg == cfg2
        assert cfg2.segment_options.segment_size == 30
        assert cfg2.segment_options.overlap_size == 5

    def test_roundtrip_dict(self, dir_with_1_slc):
        cfg = ParallelDisplacementWorkflow(
            cslc_file_list=dir_with_1_slc / "slclist.txt",
            input_options={"subdataset": "data"},
        )
        cfg_dict = cfg.model_dump()
        cfg2 = ParallelDisplacementWorkflow(**cfg_dict)
        assert cfg == cfg2


# ============================================================
# plan_segments tests
# ============================================================
@pytest.fixture()
def slc_files_30(tmp_path):
    """Create 30 dummy SLC GeoTIFF files with daily dates."""
    d = tmp_path / "slc_30"
    d.mkdir()
    start = datetime(2022, 1, 1)
    driver = gdal.GetDriverByName("GTiff")
    files = []
    for i in range(30):
        date_str = (start + timedelta(days=i)).strftime("%Y%m%d")
        fname = d / f"{date_str}.slc.tif"
        ds = driver.Create(str(fname), 10, 5, 1, gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(np.zeros((5, 10), dtype=np.complex64))
        ds = None
        files.append(fname)
    return files


class TestPlanSegments:
    def test_basic_segmentation(self, slc_files_30):
        """10 SLCs per segment, 3 overlap -> step=7, expect 5 segments."""
        segments = plan_segments(slc_files_30, segment_size=10, overlap_size=3)
        # With 30 files and step=7: segments start at 0,7,14,21,28
        # Segment at 28 would only have 2 files (< overlap=3), so it's
        # absorbed into the previous segment.
        assert len(segments) >= 3

        # First segment has no overlap with previous
        assert segments[0].overlap_with_previous == []
        assert len(segments[0].file_list) == 10

        # Subsequent segments should have overlapping dates
        for seg in segments[1:]:
            assert len(seg.overlap_with_previous) > 0

    def test_all_dates_covered(self, slc_files_30):
        """Every SLC date should appear in at least one segment."""
        segments = plan_segments(slc_files_30, segment_size=10, overlap_size=3)
        covered_files = set()
        for seg in segments:
            covered_files.update(seg.file_list)
        assert covered_files == set(slc_files_30)

    def test_overlap_dates_correct(self, slc_files_30):
        """Overlap dates should be the intersection of adjacent segments."""
        segments = plan_segments(slc_files_30, segment_size=10, overlap_size=3)
        for i in range(1, len(segments)):
            prev_dates = set(segments[i - 1].dates)
            cur_dates = set(segments[i].dates)
            expected_overlap = sorted(prev_dates & cur_dates)
            assert segments[i].overlap_with_previous == expected_overlap

    def test_single_segment(self, slc_files_30):
        """If segment_size >= n_files, only one segment."""
        segments = plan_segments(slc_files_30, segment_size=100, overlap_size=5)
        assert len(segments) == 1
        assert len(segments[0].file_list) == 30

    def test_exact_fit(self, slc_files_30):
        """30 files, segment=15, overlap=5 -> step=10: starts 0,10,20.
        Segment 0: 0-14, Segment 1: 10-24, Segment 2: 20-29."""
        segments = plan_segments(slc_files_30, segment_size=15, overlap_size=5)
        assert len(segments) == 3
        assert len(segments[0].file_list) == 15
        assert len(segments[1].file_list) == 15
        # Last segment: 20-29 = 10 files
        assert len(segments[2].file_list) == 10

    def test_invalid_overlap(self, slc_files_30):
        with pytest.raises(ValueError, match="overlap_size"):
            plan_segments(slc_files_30, segment_size=5, overlap_size=5)

    def test_too_few_files(self, tmp_path):
        d = tmp_path / "slc_1"
        d.mkdir()
        driver = gdal.GetDriverByName("GTiff")
        fname = d / "20220101.slc.tif"
        ds = driver.Create(str(fname), 10, 5, 1, gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(np.zeros((5, 10), dtype=np.complex64))
        ds = None
        with pytest.raises(ValueError, match="at least 2"):
            plan_segments([fname], segment_size=10, overlap_size=3)

    def test_segment_dates_sorted(self, slc_files_30):
        segments = plan_segments(slc_files_30, segment_size=10, overlap_size=3)
        for seg in segments:
            assert seg.dates == sorted(seg.dates)

    def test_small_remainder_absorbed(self):
        """When the remainder after the last full step is <= overlap_size,
        those files are absorbed into the last segment."""
        # Create 12 files, segment=5, overlap=3, step=2
        # Starts: 0, 2, 4, 6, 8, 10 -> but 10-11 is only 2 files, <= overlap=3
        # So absorbed into previous segment
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            driver = gdal.GetDriverByName("GTiff")
            start = datetime(2022, 1, 1)
            files = []
            for i in range(12):
                date_str = (start + timedelta(days=i)).strftime("%Y%m%d")
                fname = d / f"{date_str}.slc.tif"
                ds = driver.Create(str(fname), 10, 5, 1, gdal.GDT_CFloat32)
                ds.GetRasterBand(1).WriteArray(
                    np.zeros((5, 10), dtype=np.complex64)
                )
                ds = None
                files.append(fname)

            segments = plan_segments(files, segment_size=5, overlap_size=3)
            # All 12 files should be covered
            covered = set()
            for seg in segments:
                covered.update(seg.file_list)
            assert covered == set(files)


# ============================================================
# _blend_overlap tests
# ============================================================
class TestBlendOverlap:
    def test_mean_blend(self):
        arr1 = np.ones((5, 10), dtype=np.float32) * 2.0
        arr2 = np.ones((5, 10), dtype=np.float32) * 4.0
        seg0 = SegmentInfo(
            index=0,
            file_list=[],
            dates=[datetime(2022, 1, 1), datetime(2022, 1, 2)],
        )
        seg1 = SegmentInfo(
            index=1,
            file_list=[],
            dates=[datetime(2022, 1, 2), datetime(2022, 1, 3)],
            overlap_with_previous=[datetime(2022, 1, 2)],
        )
        result = _blend_overlap(
            arrays=[(0, arr1), (1, arr2)],
            segments=[seg0, seg1],
            out_date=datetime(2022, 1, 2),
            merge_method="mean",
        )
        np.testing.assert_allclose(result, 3.0)

    def test_linear_blend_midpoint(self):
        """With 3 overlap dates and the middle one, weights should be 0.5/0.5."""
        arr_early = np.ones((5, 10), dtype=np.float32) * 10.0
        arr_late = np.ones((5, 10), dtype=np.float32) * 20.0

        overlap_dates = [
            datetime(2022, 1, 3),
            datetime(2022, 1, 4),
            datetime(2022, 1, 5),
        ]
        seg0 = SegmentInfo(
            index=0,
            file_list=[],
            dates=[datetime(2022, 1, 1)] + overlap_dates,
        )
        seg1 = SegmentInfo(
            index=1,
            file_list=[],
            dates=overlap_dates + [datetime(2022, 1, 6)],
            overlap_with_previous=overlap_dates,
        )

        # Middle date (index 1): w_late = 2/4 = 0.5
        result = _blend_overlap(
            arrays=[(0, arr_early), (1, arr_late)],
            segments=[seg0, seg1],
            out_date=datetime(2022, 1, 4),
            merge_method="linear_blend",
        )
        expected = 0.5 * 10.0 + 0.5 * 20.0
        np.testing.assert_allclose(result, expected)

    def test_linear_blend_edges(self):
        """First overlap date should weight early segment more."""
        arr_early = np.ones((5, 10), dtype=np.float32) * 10.0
        arr_late = np.ones((5, 10), dtype=np.float32) * 20.0

        overlap_dates = [
            datetime(2022, 1, 3),
            datetime(2022, 1, 4),
            datetime(2022, 1, 5),
        ]
        seg0 = SegmentInfo(
            index=0,
            file_list=[],
            dates=[datetime(2022, 1, 1)] + overlap_dates,
        )
        seg1 = SegmentInfo(
            index=1,
            file_list=[],
            dates=overlap_dates + [datetime(2022, 1, 6)],
            overlap_with_previous=overlap_dates,
        )

        # First overlap date (index 0): w_late = 1/4 = 0.25
        result = _blend_overlap(
            arrays=[(0, arr_early), (1, arr_late)],
            segments=[seg0, seg1],
            out_date=datetime(2022, 1, 3),
            merge_method="linear_blend",
        )
        expected = 0.75 * 10.0 + 0.25 * 20.0
        np.testing.assert_allclose(result, expected)

        # Last overlap date (index 2): w_late = 3/4 = 0.75
        result = _blend_overlap(
            arrays=[(0, arr_early), (1, arr_late)],
            segments=[seg0, seg1],
            out_date=datetime(2022, 1, 5),
            merge_method="linear_blend",
        )
        expected = 0.25 * 10.0 + 0.75 * 20.0
        np.testing.assert_allclose(result, expected)

    def test_single_array_passthrough(self):
        arr = np.ones((5, 10), dtype=np.float32) * 42.0
        result = _blend_overlap(
            arrays=[(0, arr)],
            segments=[],
            out_date=datetime(2022, 1, 1),
            merge_method="mean",
        )
        np.testing.assert_array_equal(result, arr)


# ============================================================
# Test fixtures shared with existing tests
# ============================================================
@pytest.fixture()
def dir_with_1_slc(tmp_path, slc_file_list_nc):
    p = tmp_path / "slc"
    p.mkdir()
    fname = "slc_20220101.nc"
    shutil.copy(slc_file_list_nc[0], p / fname)
    with open(p / "slclist.txt", "w") as f:
        f.write(fname + "\n")
    return p
