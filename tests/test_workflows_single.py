import numpy as np
import pytest

from dolphin import stack
from dolphin.io import _readers, load_gdal
from dolphin.phase_link import simulate
from dolphin.utils import gpu_is_available
from dolphin.workflows import single
from dolphin.workflows.config import OutputFormat

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


def test_sequential_geozarr(tmp_path, slc_file_list):
    """Single-ministack run with GeoZarr output.

    Asserts that:
    - a ``cube.zarr`` is produced with the expected per-kind 3D variables
    - one ``.slc.vrt`` is emitted per date (pointing into the cube)
    - no per-date ``.slc.tif`` files are written (no data duplication)
    - reading the VRT via GDAL/rasterio returns the same pixel data as
      the corresponding cube layer
    """
    pytest.importorskip("zarr")
    import zarr

    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:3]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)
    is_compressed = [False] * len(files)
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=is_compressed,
    )

    output_folder = tmp_path / "single_zarr"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window={"x": 2, "y": 1},
        strides={"x": 1, "y": 1},
        shp_method="rect",
        write_crlb=True,
        write_closure_phase=True,
        output_format=OutputFormat.GEOZARR,
    )

    cube_path = output_folder / "cube.zarr"
    assert cube_path.exists(), "GeoZarr cube directory not created"

    root = zarr.open_group(str(cube_path), mode="r")
    # Coord scaffolding required by GeoZarr / rioxarray readers.
    assert {"y", "x", "spatial_ref"}.issubset(set(root.keys()))
    # One 3D array per kind.
    assert "slcs" in root and root["slcs"].ndim == 3
    assert root["slcs"].shape[0] == 3  # 3 input dates
    assert "crlb" in root
    assert "closure_phases" in root
    assert root["closure_phases"].shape[0] == 1  # N-2 triplets for N=3

    # No tif duplication: VRTs are the new per-layer artifacts.
    slc_tifs = sorted(output_folder.glob("2*.slc.tif"))
    slc_vrts = sorted(output_folder.glob("2*.slc.vrt"))
    assert slc_tifs == [], (
        f"Per-date GeoTIFFs were written in GEOZARR mode (duplication): {slc_tifs}"
    )
    assert len(slc_vrts) == 3

    # Reading the VRT through GDAL must yield the same data as the cube layer.
    for i, vrt in enumerate(slc_vrts):
        cube_layer = np.asarray(root["slcs"][i])
        vrt_data = load_gdal(vrt)
        np.testing.assert_array_equal(cube_layer, vrt_data)
