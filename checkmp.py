import sys
import time

import dask.array as da
import numpy as np

from dolphin import io
from dolphin.phase_link import covariance, mle


def _prep_memmaps(slc_stack):
    nslc, rows, cols = slc_stack.shape
    slc_stack.tofile("tmp_slc_stack")
    mem_slc_stack = np.memmap(
        "tmp_slc_stack", dtype=slc_stack.dtype, mode="r", shape=slc_stack.shape
    )
    mem_C_arrays = np.memmap(
        "tmp_C", dtype="complex64", mode="w+", shape=(rows, cols, nslc, nslc)
    )
    return mem_slc_stack, mem_C_arrays


if __name__ == "__main__":
    half_window = {"x": 11, "y": 5}
    slc_stack = io.load_gdal("scratch/slc_stack.vrt")
    slc_stack = np.tile(slc_stack, (2, 2, 2))
    print(f"slc_stack.shape: {slc_stack.shape}")

    nslc, rows, cols = slc_stack.shape

    mem_slc_stack, mem_C_arrays = _prep_memmaps(slc_stack)

    try:
        b, nw, s = list(map(int, sys.argv[1:]))
    except:
        b, nw, s = 200, 1, 1

    strides = {"x": s, "y": s}

    t0 = time.time()
    covariance.estimate_stack_covariance_cpu_shm(
        slc_stack[:, :b, :b],
        half_window=half_window,
        strides=strides,
        n_workers=nw,
    )
    print(f"SharedArray: {time.time() - t0:.2f} s")

    t0 = time.time()
    covariance.estimate_stack_covariance_cpu_mp(
        slc_stack[:, :b, :b],
        half_window=half_window,
        strides=strides,
        n_workers=nw,
    )
    print(f"memmaps: {time.time() - t0:.2f} s")

    t0 = time.time()
    covariance.estimate_stack_covariance_cpu_pymp(
        slc_stack[:, :b, :b],
        half_window=half_window,
        strides=strides,
        n_workers=nw,
    )
    print(f"Pymp: {time.time() - t0:.2f} s")
