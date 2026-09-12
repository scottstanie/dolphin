"""Carrier metadata for NISAR displacement scaling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def nisar_stack_wavelength(files: Sequence[Path], subdataset: str | None) -> float:
    """Read the selected carrier for every input, rejecting mixed wavelengths.

    HDF5 inputs require opera-utils' NISAR reader (PR #225). Exported rasters
    carry WAVELENGTH_METERS metadata. A filename cannot identify the carrier;
    missing metadata requires an explicit input_options.wavelength override.
    """
    wavelengths = []
    for path in files:
        if path.suffix.lower() in {".h5", ".hdf5", ".nc"}:
            if subdataset is None:
                raise ValueError("NISAR HDF5 inputs require the selected subdataset")
            try:
                from opera_utils.nisar import get_nisar_wavelength
            except ImportError as exc:
                raise ValueError(
                    "NISAR wavelength detection requires opera-utils with PR #225;"
                    " upgrade it or set input_options.wavelength from centerFrequency"
                ) from exc
            wavelength = get_nisar_wavelength(path, subdataset)
        else:
            from osgeo import gdal

            ds = gdal.Open(str(path))
            value = ds.GetMetadataItem("WAVELENGTH_METERS")
            ds = None
            if value is None:
                raise ValueError(
                    f"Missing WAVELENGTH_METERS in {path}; re-export the GSLC or set"
                    " input_options.wavelength from the selected centerFrequency"
                )
            wavelength = float(value)
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError(f"Invalid radar wavelength in {path}: {wavelength}")
        wavelengths.append(wavelength)
    if not wavelengths:
        raise ValueError("Cannot read a wavelength from an empty NISAR stack")
    if not np.allclose(wavelengths, wavelengths[0], rtol=1e-8, atol=0):
        raise ValueError("NISAR stack contains different radar wavelengths")
    return wavelengths[0]
