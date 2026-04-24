import datetime
from pathlib import Path

import pytest

from dolphin import stack
from dolphin.io import _readers
from dolphin.phase_link import simulate
from dolphin.utils import gpu_is_available
from dolphin.workflows import single

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)


@pytest.mark.parametrize("write_extra", [False, True])
def test_sequential_gtiff(tmp_path, slc_file_list, write_extra: bool):
    """Run through the sequential estimation with a GeoTIFF stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:3]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)
    is_compressed = [False] * len(files)
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=is_compressed,
    )

    hy, hx = 1, 2
    half_window = {"x": hx, "y": hy}
    strides = {"x": 1, "y": 1}
    output_folder = tmp_path / "single"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window=half_window,
        strides=strides,
        shp_method="rect",
        write_crlb=write_extra,
        write_closure_phase=write_extra,
    )

    assert output_folder.exists()
    # Check that all the expected outputs are there
    assert len(list(output_folder.glob("2*.slc.tif"))) == 3
    assert len(list(output_folder.glob("compressed_*tif"))) == 1
    assert len(list(output_folder.glob("temporal_coherence*tif"))) == 1


def _duplicate_compressed_base_ministack() -> stack.MiniStackInfo:
    """Build a ministack with two compressed SLCs sharing base date 20220101.

    Mirrors the ``CompressedSlcPlan.ALWAYS_FIRST`` sequential case where every
    compressed SLC carries the same base (reference) phase date.
    """
    ref = datetime.datetime(2022, 1, 1)
    return stack.MiniStackInfo(
        file_list=[Path(f"f{i}.tif") for i in range(5)],
        dates=[
            [ref, datetime.datetime(2022, 1, 1), datetime.datetime(2022, 1, 2)],
            [ref, datetime.datetime(2022, 1, 2), datetime.datetime(2022, 1, 3)],
            [datetime.datetime(2022, 1, 3)],
            [datetime.datetime(2022, 1, 4)],
            [datetime.datetime(2022, 1, 5)],
        ],
        is_compressed=[True, True, False, False, False],
    )


def test_get_multilooked_coherence_output_info_all_real():
    ms = stack.MiniStackInfo(
        file_list=[Path(f"f{i}.tif") for i in range(4)],
        dates=[[datetime.datetime(2022, 1, i + 1)] for i in range(4)],
        is_compressed=[False] * 4,
    )
    fns, bands = single._get_multilooked_coherence_output_info(ms, n=2)
    # For 4 SLCs, n=2: pairs (0,1),(1,2),(2,3),(0,2),(1,3) — all have distinct dates.
    assert bands == [0, 1, 2, 3, 4]
    assert fns == [
        "multilooked_coherence_20220101_20220102.tif",
        "multilooked_coherence_20220102_20220103.tif",
        "multilooked_coherence_20220103_20220104.tif",
        "multilooked_coherence_20220101_20220103.tif",
        "multilooked_coherence_20220102_20220104.tif",
    ]


def test_get_multilooked_coherence_output_info_duplicate_compressed_base():
    ms = _duplicate_compressed_base_ministack()
    fns, bands = single._get_multilooked_coherence_output_info(ms, n=2)
    # Raw pairs for 5 SLCs, n=2: (0,1),(1,2),(2,3),(3,4),(0,2),(1,3),(2,4).
    # (0,1) is dropped: both compressed SLCs carry base date 20220101 → self-pair.
    # (0,2) is dropped: same filename as (1,2) — both yield 20220101_20220103.
    assert bands == [1, 2, 3, 5, 6]
    assert fns == [
        "multilooked_coherence_20220101_20220103.tif",
        "multilooked_coherence_20220103_20220104.tif",
        "multilooked_coherence_20220104_20220105.tif",
        "multilooked_coherence_20220101_20220104.tif",
        "multilooked_coherence_20220103_20220105.tif",
    ]
    assert len(fns) == len(set(fns))


def test_get_closure_phase_output_info_all_real():
    ms = stack.MiniStackInfo(
        file_list=[Path(f"f{i}.tif") for i in range(5)],
        dates=[[datetime.datetime(2022, 1, i + 1)] for i in range(5)],
        is_compressed=[False] * 5,
    )
    fns, bands = single._get_closure_phase_output_info(ms)
    # 5 SLCs → 3 triplets (0,1,2), (1,2,3), (2,3,4), all with distinct dates.
    assert bands == [0, 1, 2]
    assert fns == [
        "closure_phase_20220101_20220102_20220103.tif",
        "closure_phase_20220102_20220103_20220104.tif",
        "closure_phase_20220103_20220104_20220105.tif",
    ]


def test_get_closure_phase_output_info_duplicate_compressed_base():
    ms = _duplicate_compressed_base_ministack()
    fns, bands = single._get_closure_phase_output_info(ms)
    # Triplets (i, i+1, i+2) for 5 SLCs → i in {0, 1, 2}.
    # i=0 → (20220101, 20220101, 20220103) dropped: d0 == d1.
    # i=1 → (20220101, 20220103, 20220104) kept.
    # i=2 → (20220103, 20220104, 20220105) kept.
    assert bands == [1, 2]
    assert fns == [
        "closure_phase_20220101_20220103_20220104.tif",
        "closure_phase_20220103_20220104_20220105.tif",
    ]
    assert len(fns) == len(set(fns))


def test_run_single_closure_phase_duplicate_compressed_base(tmp_path, slc_file_list):
    """Two compressed SLCs sharing a base date must not yield degenerate triplets."""
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:5]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)

    ms = _duplicate_compressed_base_ministack()
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=ms.dates,
        is_compressed=ms.is_compressed,
    )

    output_folder = tmp_path / "single"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window={"x": 2, "y": 1},
        strides={"x": 1, "y": 1},
        shp_method="rect",
        write_crlb=False,
        write_closure_phase=True,
    )

    closure_files = sorted((output_folder / "closure_phases").glob("*.tif"))
    names = [p.name for p in closure_files]
    # No degenerate triplet (any two dates equal) was written.
    for name in names:
        # name looks like "closure_phase_<d0>_<d1>_<d2>.tif"
        d0, d1, d2 = name.removeprefix("closure_phase_").removesuffix(".tif").split("_")
        assert len({d0, d1, d2}) == 3, f"Degenerate triplet written: {name}"
    assert len(names) == len(set(names))
    assert set(names) == {
        "closure_phase_20220101_20220103_20220104.tif",
        "closure_phase_20220103_20220104_20220105.tif",
    }


def test_run_single_nearest_n_coherence_duplicate_compressed_base(
    tmp_path, slc_file_list
):
    """Two compressed SLCs sharing a base date must not yield colliding rasters."""
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:5]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)

    ms = _duplicate_compressed_base_ministack()
    # Swap in the real on-disk file paths (the VRT reader is what matters for I/O;
    # `dates`/`is_compressed` drive the filename generation we are testing).
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=ms.dates,
        is_compressed=ms.is_compressed,
    )

    output_folder = tmp_path / "single"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window={"x": 2, "y": 1},
        strides={"x": 1, "y": 1},
        shp_method="rect",
        write_crlb=False,
        write_closure_phase=False,
        nearest_n_coherence=2,
    )

    coh_files = sorted((output_folder / "multilooked_coherence").glob("*.tif"))
    names = [p.name for p in coh_files]
    # No self-date pair written, and every filename is unique on disk.
    assert "multilooked_coherence_20220101_20220101.tif" not in names
    assert len(names) == len(set(names))
    assert set(names) == {
        "multilooked_coherence_20220101_20220103.tif",
        "multilooked_coherence_20220103_20220104.tif",
        "multilooked_coherence_20220104_20220105.tif",
        "multilooked_coherence_20220101_20220104.tif",
        "multilooked_coherence_20220103_20220105.tif",
    }
