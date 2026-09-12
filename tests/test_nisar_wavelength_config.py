"""Verify carrier selection at the displacement configuration boundary."""

import h5py
import numpy as np
import pytest
from osgeo import gdal

from dolphin.workflows.config import DisplacementWorkflow


def _product(tmp_path, date="20260101", frequency_a=1_229_000_000.0):
    path = tmp_path / f"NISAR_L2_GSLC_{date}.h5"
    with h5py.File(path, "w") as hf:
        for frequency, value in [("A", frequency_a), ("B", 1_293_500_000.0)]:
            root = f"/science/LSAR/GSLC/grids/frequency{frequency}"
            hf[f"{root}/centerFrequency"] = value
            hf[f"{root}/HH"] = np.ones((3, 4), dtype=np.complex64)
    return path


@pytest.mark.parametrize("frequency,hz", [("A", 1_229_000_000), ("B", 1_293_500_000)])
def test_selected_subdataset_sets_scale(tmp_path, frequency, hz):
    cfg = DisplacementWorkflow(
        cslc_file_list=[_product(tmp_path)],
        work_directory=tmp_path / "out",
        input_options={
            "subdataset": f"/science/LSAR/GSLC/grids/frequency{frequency}/HH"
        },
    )
    assert cfg.input_options.wavelength == pytest.approx(299792458 / hz)


def test_mixed_carriers_rejected(tmp_path):
    paths = [_product(tmp_path), _product(tmp_path, "20260113", 1_257_500_000)]
    with pytest.raises(ValueError, match="different radar wavelengths"):
        DisplacementWorkflow(
            cslc_file_list=paths,
            work_directory=tmp_path / "out",
            input_options={"subdataset": "/science/LSAR/GSLC/grids/frequencyA/HH"},
        )


def test_explicit_legacy_scale_preserved(tmp_path):
    cfg = DisplacementWorkflow(
        cslc_file_list=[_product(tmp_path)],
        work_directory=tmp_path / "out",
        input_options={
            "subdataset": "/science/LSAR/GSLC/grids/frequencyA/HH",
            "wavelength": 0.238408,
        },
    )
    assert cfg.input_options.wavelength == 0.238408


def test_subdataset_identifies_renamed_product(tmp_path):
    path = _product(tmp_path).rename(tmp_path / "input_20260101.h5")
    cfg = DisplacementWorkflow(
        cslc_file_list=[path],
        work_directory=tmp_path / "out",
        input_options={"subdataset": "/science/LSAR/GSLC/grids/frequencyB/HH"},
    )
    assert cfg.input_options.wavelength == pytest.approx(299792458 / 1_293_500_000)


@pytest.mark.parametrize("value", [None, "0", "nan", "0.24393202441008951"])
def test_exported_raster_requires_valid_carrier(tmp_path, value):
    path = tmp_path / "NISAR_GSLC_20260101.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(path), 4, 3, 1, gdal.GDT_CFloat32)
    if value is not None:
        ds.SetMetadataItem("WAVELENGTH_METERS", value)
    ds = None
    if value in {None, "0", "nan"}:
        with pytest.raises(ValueError):
            DisplacementWorkflow(cslc_file_list=[path], work_directory=tmp_path / "out")
    else:
        cfg = DisplacementWorkflow(
            cslc_file_list=[path], work_directory=tmp_path / "out"
        )
        assert cfg.input_options.wavelength == float(value)
