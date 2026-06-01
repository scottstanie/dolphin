from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import numpy as np

from dolphin._types import Filename
from dolphin.io._core import DEFAULT_TIFF_OPTIONS_RIO
from dolphin.utils import full_suffix

from ._constants import CONNCOMP_SUFFIX, DEFAULT_CCL_NODATA, DEFAULT_UNW_NODATA
from ._utils import _zero_from_mask

__all__ = [
    "unwrap_whirlwind",
]


logger = logging.getLogger("dolphin")


def unwrap_whirlwind(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    mask_file: Optional[Filename] = None,
    zero_where_masked: bool = False,
    unw_nodata: Optional[float] = DEFAULT_UNW_NODATA,
    ccl_nodata: Optional[int] = DEFAULT_CCL_NODATA,
) -> tuple[Path, Path]:
    """Unwrap an interferogram and grow conncomps using whirlwind.

    Uses ``whirlwind.unwrap``, which emits both the
    unwrapped phase and SNAPHU-style connected component labels from a
    single MCF solve.

    Parameters
    ----------
    ifg_filename : Filename
        Path to input interferogram.
    corr_filename : Filename
        Path to input correlation file.
    unw_filename : Filename
        Path to output unwrapped phase file.
    nlooks : float
        Effective number of looks used to form the input correlation data.
    mask_file : Filename, optional
        Path to binary byte mask file. Assumes 1 = valid, 0 = invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        Ignored if no mask is provided. Default False.
    unw_nodata : float, optional
        Nodata value for the output unwrapped phase raster.
    ccl_nodata : int, optional
        Nodata value for the connected component labels.

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.

    """
    import snaphu  # used here only for raster I/O
    import whirlwind as ww

    # Create a context manager that combines other context managers -- one for each
    # input raster file. Upon exiting the context block, each context manager in the
    # stack will be closed in LIFO order.
    with ExitStack() as stack:
        if zero_where_masked and (mask_file is not None):
            logger.info(f"Zeroing phase/corr of pixels masked in {mask_file}")
            zeroed_ifg_file, zeroed_corr_file = _zero_from_mask(
                ifg_filename, corr_filename, mask_file
            )
            igram = stack.enter_context(snaphu.io.Raster(zeroed_ifg_file))
            corr = stack.enter_context(snaphu.io.Raster(zeroed_corr_file))
        else:
            igram = stack.enter_context(snaphu.io.Raster(ifg_filename))
            corr = stack.enter_context(snaphu.io.Raster(corr_filename))

        if mask_file is None:
            mask_arr = None
        else:
            mask = stack.enter_context(snaphu.io.Raster(mask_file))
            mask_arr = np.ascontiguousarray(mask[:, :], dtype=bool)

        logger.info("Unwrapping using whirlwind")
        igram_arr = np.ascontiguousarray(igram[:, :], dtype=np.complex64)
        corr_arr = np.ascontiguousarray(corr[:, :], dtype=np.float32)
        # ww.unwrap returns (phase, conncomp) from the robust tiled pipeline.
        # Goldstein is off by default (pass goldstein_alpha>0 to enable; under
        # evaluation upstream). Renamed from the old ww.unwrap_with_conncomp.
        unw, conncomp_arr = ww.unwrap(igram_arr, corr_arr, float(nlooks), mask=mask_arr)

        logger.info("Writing unwrapped phase to raster file")
        with snaphu.io.Raster.create(
            unw_filename,
            like=igram,
            nodata=unw_nodata,
            dtype=np.float32,
            **DEFAULT_TIFF_OPTIONS_RIO,
        ) as unw_raster:
            unw_raster[:, :] = unw

        unw_suffix = full_suffix(unw_filename)
        cc_filename = str(unw_filename).replace(unw_suffix, CONNCOMP_SUFFIX)

        logger.info("Writing whirlwind connected component labels")
        with snaphu.io.Raster.create(
            cc_filename,
            like=igram,
            nodata=ccl_nodata,
            dtype=np.uint16,
            **DEFAULT_TIFF_OPTIONS_RIO,
        ) as conncomp_raster:
            conncomp_raster[:, :] = conncomp_arr.astype(np.uint16)

    if zero_where_masked and (mask_file is not None):
        logger.info(f"Zeroing unw/conncomp of pixels masked in {mask_file}")
        return _zero_from_mask(unw_filename, cc_filename, mask_file)

    return Path(unw_filename), Path(cc_filename)
