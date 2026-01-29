"""Soil moisture estimation from InSAR closure phases.

This module implements relative soil moisture retrieval from InSAR closure phase
measurements, based on the methodology described in:

- Zheng & Fattahi (2025): "Modeling, prediction, and retrieval of surface soil
  moisture from InSAR closure phase", Remote Sensing of Environment.
- Wig et al. (2024): "Fine-Resolution Measurement of Soil Moisture from
  Cumulative InSAR Closure Phase", IEEE TGRS.

The algorithm computes the cumulative closure phase time series and removes a
linear trend to derive a relative soil moisture index (InSAR Soil Moisture Index,
or ISMI).
"""

from ._core import (
    SoilMoistureOutput,
    compute_cumulative_closure_phase,
    compute_soil_moisture_index,
    detrend_cumulative_closure_phase,
)
from ._raster import (
    create_soil_moisture_index,
    create_soil_moisture_index_from_arrays,
)

__all__ = [
    "SoilMoistureOutput",
    "compute_cumulative_closure_phase",
    "detrend_cumulative_closure_phase",
    "compute_soil_moisture_index",
    "create_soil_moisture_index",
    "create_soil_moisture_index_from_arrays",
]
