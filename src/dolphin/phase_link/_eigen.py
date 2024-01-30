"""Module for solving for one eigen-pair quickly in jax."""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from jax.typing import ArrayLike

__all__ = [
    "get_eigvecs",
    "eigh_largest",
    "eigh_largest_stack",
    "eigh_smallest",
    "eigh_smallest_stack",
]


@partial(jit, static_argnames=("use_evd",))
def get_eigvecs(C_arrays: ArrayLike, use_evd: bool = False) -> Array:
    # Subset index for scipy.eigh: larges eig for EVD. Smallest for EMI.
    subset_idx = C_arrays.shape[-1] - 1 if use_evd else 0

    def get_top_eigvecs(C: Array):
        # The eigenvalues in ascending order, each repeated according
        # The column ``eigenvectors[:, i]`` is the normalized eigenvector
        # corresponding to the eigenvalue ``eigenvalues[i]``.
        return jnp.linalg.eigh(C)[1][:, [subset_idx]]

    # vmap over the first 2 dimensions (rows, cols)
    get_eigvecs_block = vmap(vmap(get_top_eigvecs))
    return get_eigvecs_block(C_arrays)


@partial(jit, static_argnames=("is_unit_vector",))
def rayleigh(A: ArrayLike, v: ArrayLike, is_unit_vector: bool = True) -> Array:
    r"""Compute the Rayleigh quotient for a matrix A and unit vector v.

    https://en.wikipedia.org/wiki/Rayleigh_quotient

    \[
    R(A, v) = \\frac{v^{*} A v}{v^{*} v}
    \]

    If v is a unit vector, the Rayleigh quotient is just the numerator
    """
    if is_unit_vector:
        return v.T.conj() @ A @ v
    else:
        return v.T.conj() @ A @ v / v.T.conj() @ v


def eigh_largest(
    mat: ArrayLike, max_iters: int = 25, tol: float = 1e-4
) -> tuple[Array, float, float]:
    """Find only the largest eigen pair for Hermitian `mat`.

    Notes
    -----
    Uses the power iteration method ([1]_)

    Parameters
    ----------
    mat : ArrayLike
        The matrix to find the largest eigenvector of.
    max_iters : int, optional
        The maximum number of iterations to run, by default 25
    tol : float, optional
        The tolerance for the eigenvalue residual, by default 1e-4

    Returns
    -------
    eigenvalue : float
        The largest eigenvalue of `mat`
        Note: This returns a float, even if `mat` is complex, since it assumes
        `mat` is Hermitian.
    largest_eigenvector : Array
        The eigenvector corresponding to the largest eigenvalue of `mat`
    eigenvalue_residual : float
        The residual of the `eigenvalue` of the power iteration algorithm from the
        last iteration.

    References
    ----------
    .. [1] https://mathreview.uwaterloo.ca/archive/voli/1/panju.pdf
    .. [2] https://math.stackexchange.com/questions/271864/how-to-compute-the-smallest-eigenvalue-using-the-power-iteration-algorithm
    """
    # Format for a `lax.while_loop`
    # def while_loop(cond_fun, body_fun, init_val):
    #     val = init_val
    #     while cond_fun(val):
    #         val = body_fun(val)
    #     return val

    # The body function is another matrix multiply, followed by normalizing
    def body_fun(val: tuple[Array, Array, Array, Array, int]):
        prev_vec, old_vec, prev_eigenvalue, _old_eigenvalue, cur_iter = val
        # Do next matmul
        next_vec = mat @ prev_vec
        # Now divide by norm
        next_normed_vec = next_vec / jnp.linalg.norm(next_vec)

        # Get current eigenvalue for the loop cond
        new_eigenvalue = rayleigh(mat, next_normed_vec, is_unit_vector=True)
        return next_normed_vec, prev_vec, new_eigenvalue, prev_eigenvalue, cur_iter + 1

    # We will loop while the change in eigenvalue is greater than tol
    def cond_fun(val: tuple[Array, Array, Array, Array, int]):
        _new_vec, _old_vec, new_eigenvalue, old_eigenvalue, cur_iter = val
        eig_residual = jnp.abs(new_eigenvalue - old_eigenvalue)
        return (eig_residual > tol) & (cur_iter < max_iters)

    # Start with some unit vector. Make it zero phase:
    m, n = mat.shape
    v0 = jnp.ones(m, dtype=mat.dtype) / jnp.sqrt(m)
    # Dummy vals for starting vec (0) and eigenvalues
    eig0 = jnp.array(1.0, dtype=mat.dtype)
    v_final, _, last_eigenvalue, pre_eigenvalue, n_iters = lax.while_loop(
        cond_fun, body_fun, (v0, 0 * v0, eig0, 0 * eig0, 0)
    )
    # Get the residual for debug/quality information
    eig_residual = jnp.abs(last_eigenvalue - pre_eigenvalue)

    return jnp.real(last_eigenvalue), v_final[:, None], jnp.float32(eig_residual)


@partial(jit, static_argnames=("max_iters", "tol"))
def eigh_largest_stack(
    C_arrays: ArrayLike, max_iters: int = 60, tol: float = 1e-4
) -> tuple[Array, Array, Array]:
    """Find only the smallest largest pair for a stack of Hermitian matrices.

    See Also
    --------
    `eigh_largest`

    Parameters
    ----------
    C_arrays : ArrayLike
        The matrices to find the largest eigenvector of.
        Shape: (num_rows, num_cols, M, M)
    max_iters : int, optional
        The maximum number of iterations to run, by default 25
    tol : float, optional
        The tolerance for the eigenvalue residual, by default 1e-4

    Returns
    -------
    eigenvalues : Array
        The largest eigenvalues of `C_arrays`. Shape: (num_rows, num_cols)
        Note: This returns a float, even if `mat` is complex, since it assumes
        `mat` is Hermitian.
    largest_eigenvectors : Array
        The eigenvectors corresponding to the largest eigenvalues.
        shape = (num_rows, num_cols, M, 1)
    eigenvalue_residuals : Array
        The residual of the `eigenvalue` of the power iteration algorithm from the
        last iteration.
        Shape: (num_rows, num_cols)

    References
    ----------
    .. [1] https://mathreview.uwaterloo.ca/archive/voli/1/panju.pdf
    """
    func_cols = vmap(lambda x: eigh_largest(x, max_iters, tol))
    func_stack = vmap(func_cols)
    return func_stack(C_arrays)


def eigh_smallest(
    mat: ArrayLike, max_iters: int = 100, tol: float = 1e-5
) -> tuple[Array, float, float]:
    """Find only the smallest eigen pair for Hermitian `mat`.

    Notes
    -----
    This function uses `eigh_largest`, but first transforms `mat` so that
    the smallest (positive) eigenvalue is now the largest (negative) [1]_.

    This function assumes that `mat` is a *coherence* matrix so that the largest
    possible eigenvalue is N for an (N, N) coherence matrix.


    Parameters
    ----------
    mat : ArrayLike
        The matrix to find the eigenvector of.
    max_iters : int, optional
        The maximum number of iterations to run, by default 25
    tol : float, optional
        The tolerance for the eigenvalue residual, by default 1e-4

    Returns
    -------
    eigenvalue : float
        The smallest eigenvalue of `mat`
        Note: This returns a float, even if `mat` is complex, since it assumes
        `mat` is Hermitian.
    largest_eigenvector : Array
        The eigenvector corresponding to the smallest eigenvalue of `mat`
    eigenvalue_residual : float
        The residual of the `eigenvalue` of the power iteration algorithm from the
        last iteration.

    References
    ----------
    .. [1] https://mathreview.uwaterloo.ca/archive/voli/1/panju.pdf
    .. [2] https://math.stackexchange.com/questions/271864/how-to-compute-the-smallest-eigenvalue-using-the-power-iteration-algorithm
    """
    m, n = mat.shape
    assert m == n
    Id = jnp.eye(m, dtype=mat.dtype)
    # Subtract n*I
    mat_neg = mat - n * Id
    # Find the largest (in absolute value) eigen pair
    eig_neg, vec_neg, residual = eigh_largest(mat_neg, max_iters, tol)

    # Add back to the eigenvalue
    eig = eig_neg + n
    # Get the residual for debug/quality information
    return eig, vec_neg, residual

    # # repeat the identity matrix for each pixel
    # Id = jnp.tile(Id, (mat.shape[0], mat.shape[1], 1, 1))


@partial(jit, static_argnames=("max_iters", "tol"))
def eigh_smallest_stack(
    C_arrays: ArrayLike, max_iters: int = 50, tol: float = 1e-4
) -> tuple[Array, Array, Array]:
    """Find only the smallest eigen pair for a stack of Hermitian matrices.

    See Also
    --------
    `eigh_smallest`

    Parameters
    ----------
    C_arrays : ArrayLike
        The matrices to find the smallest eigenvector of.
        Shape: (num_rows, num_cols, M, M)
    max_iters : int, optional
        The maximum number of iterations to run, by default 25
    tol : float, optional
        The tolerance for the eigenvalue residual, by default 1e-4

    Returns
    -------
    eigenvalues : Array
        The smallest eigenvalues of `C_arrays`. Shape: (num_rows, num_cols)
        Note: This returns a float, even if `mat` is complex, since it assumes
        `mat` is Hermitian.
    smallest_eigenvectors : Array
        The eigenvectors corresponding to the smallest eigenvalues.
        shape = (num_rows, num_cols, M, 1)
    eigenvalue_residuals : Array
        The residual of the `eigenvalue` of the power iteration algorithm from the
        last iteration.
        Shape: (num_rows, num_cols)

    References
    ----------
    .. [1] https://mathreview.uwaterloo.ca/archive/voli/1/panju.pdf
    """
    func_cols = vmap(lambda x: eigh_smallest(x, max_iters, tol))
    func_stack = vmap(func_cols)
    return func_stack(C_arrays)
