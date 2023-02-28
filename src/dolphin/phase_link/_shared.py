import ctypes
from multiprocessing import Array

import numpy as np
from numpy.typing import NDArray

_CTYPES_TO_NUMPY = {
    ctypes.c_char: np.dtype(np.uint8),
    ctypes.c_wchar: np.dtype(np.int16),
    ctypes.c_byte: np.dtype(np.int8),
    ctypes.c_ubyte: np.dtype(np.uint8),
    ctypes.c_short: np.dtype(np.int16),
    ctypes.c_ushort: np.dtype(np.uint16),
    ctypes.c_int: np.dtype(np.int32),
    ctypes.c_uint: np.dtype(np.uint32),
    ctypes.c_long: np.dtype(np.int64),
    ctypes.c_ulong: np.dtype(np.uint64),
    ctypes.c_float: np.dtype(np.float32),
    ctypes.c_double: np.dtype(np.float64),
}

_NUMPY_TO_CTYPES = dict(zip(_CTYPES_TO_NUMPY.values(), _CTYPES_TO_NUMPY.keys()))

# Add complex types: will require doing 2 * arr.size for initializing
_CTYPES_TO_NUMPY[ctypes.c_float] = np.dtype(np.complex64)
_NUMPY_TO_CTYPES[np.dtype(np.complex64)] = ctypes.c_float


def shm_as_ndarray(mp_array: Array, shape=None, complex=False):
    """Given a multiprocessing.Array, returns an ndarray pointing to
    the same data."""

    # support SynchronizedArray:
    if not hasattr(mp_array, "_type_"):
        mp_array = mp_array.get_obj()

    dtype = _CTYPES_TO_NUMPY[mp_array._type_]
    if not complex:
        result = np.frombuffer(mp_array, dtype)
    else:
        result = np.frombuffer(mp_array, dtype).view(np.complex64)

    if shape is not None:
        result = result.reshape(shape)

    return np.asarray(result)


def ndarray_to_shm(arr: NDArray, lock=False):
    """Generate an 1D multiprocessing.Array containing the data from
    the passed ndarray.  The data will be *copied* into shared
    memory."""

    array1d = arr.ravel(order="A")

    c_type = _NUMPY_TO_CTYPES[array1d.dtype]

    is_complex = np.iscomplexobj(arr)
    size = array1d.size if not is_complex else 2 * array1d.size

    result = Array(c_type, size, lock=lock)
    shm_as_ndarray(result, complex=is_complex)[:] = array1d
    return result
