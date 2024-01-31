"""Module for solving for one eigen-pair quickly in jax."""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from jax.typing import ArrayLike

__all__ = [
    "eigh_largest",
    "eigh_largest_stack",
    "eigh_smallest",
    "eigh_smallest_stack",
]


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
        return (v.T.conj() @ A @ v) / (v.T.conj() @ v)


import jax.debug


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
        The largest phase difference between the final eigenvector and the previous
        iterate's eigenvector.
        Default is 1e-4.

    Returns
    -------
    eigenvalue : float
        The largest eigenvalue of `mat`
        Note: This returns a float, even if `mat` is complex, since it assumes
        `mat` is Hermitian.
    largest_eigenvector : Array
        The eigenvector corresponding to the largest eigenvalue of `mat`
    residual_max_phase_diff : float
        The largest phase difference between the final eigenvector and the previous
        iterate's eigenvector.

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
        cur_vec, old_vec, prev_eigenvalue, _old_max_phase_diff, cur_iter = val
        # Do next matmul
        next_vec = mat @ cur_vec
        # Normalize to a unit vector
        next_normed_vec = next_vec / jnp.linalg.norm(next_vec)

        # Get current eigenvalue for the loop cond
        new_eigenvalue = rayleigh(mat, next_normed_vec, is_unit_vector=True)

        # Get the max element-wise phase-difference to check convergence
        phase_diff = jnp.angle(next_normed_vec * old_vec.conj())
        max_phase_diff = jnp.max(jnp.abs(phase_diff))
        return next_normed_vec, cur_vec, new_eigenvalue, max_phase_diff, cur_iter + 1

    def cond_fun(val: tuple[Array, Array, Array, Array, int]):
        new_vec, old_vec, _new_eigenvalue, max_phase_diff, cur_iter = val
        # We will loop while the max phase change is greater than `tol`
        return (max_phase_diff > tol) & (cur_iter < max_iters)

    # Start with some unit vector. Make it zero phase:
    m, n = mat.shape
    v0 = (jnp.ones(m) + 1j * jnp.ones(m)).astype(mat.dtype)
    v0 /= jnp.linalg.norm(v0)
    # Dummy vals for starting vec (0) and eigenvalues
    eig0 = jnp.array(1.0, dtype=mat.dtype)
    v_final, _v_prev, last_eigenvalue, last_max_phase_diff, _n_iters = lax.while_loop(
        cond_fun,
        body_fun,
        (v0, -v0, eig0, 1e3, 0),
    )

    jax.debug.print("{} iterations", (_n_iters))
    return jnp.real(last_eigenvalue), v_final, jnp.float32(last_max_phase_diff)


@partial(jit, static_argnames=("max_iters", "tol"))
def eigh_largest_stack(
    C_arrays: ArrayLike, max_iters: int = 25, tol: float = 1e-4
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
        The largest phase differences between the final eigenvector and the previous
        iterate's eigenvector.
        Default is 1e-4.

    Returns
    -------
    eigenvalues : Array
        The largest eigenvalues of `C_arrays`. Shape: (num_rows, num_cols)
        Note: This returns a float, even if `mat` is complex, since it assumes
        `mat` is Hermitian.
    largest_eigenvectors : Array
        The eigenvectors corresponding to the largest eigenvalues.
        shape = (num_rows, num_cols, M)
    residual_max_phase_diff : float
        The largest phase difference between the final eigenvector and the previous
        iterate's eigenvector.
        Shape: (num_rows, num_cols)

    References
    ----------
    .. [1] https://mathreview.uwaterloo.ca/archive/voli/1/panju.pdf
    """
    func_cols = vmap(lambda x: eigh_largest(x, max_iters, tol))
    func_stack = vmap(func_cols)
    return func_stack(C_arrays)


@partial(jit, static_argnames=("max_iters", "tol"))
def eigh_smallest(
    mat: ArrayLike, max_iters: int = 100, tol: float = 1e-4
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
        The maximum number of iterations to run, by default 100
    tol : float, optional
        Tolerance for largest differences between iterates for the eigenvector phase.
        Default: 1e-4

    Returns
    -------
    eigenvalue : float
        The smallest eigenvalue of `mat`
        Note: This returns a float, even if `mat` is complex, since it assumes
        `mat` is Hermitian.
    largest_eigenvector : Array
        The eigenvector corresponding to the smallest eigenvalue of `mat`
    residual_max_phase_diff : float
        The largest phase difference between the final eigenvector and the previous
        iterate's eigenvector.

    References
    ----------
    .. [1] https://mathreview.uwaterloo.ca/archive/voli/1/panju.pdf
    .. [2] https://math.stackexchange.com/questions/271864/how-to-compute-the-smallest-eigenvalue-using-the-power-iteration-algorithm
    """
    m, n = mat.shape
    Id = jnp.eye(m, dtype=mat.dtype)
    # Subtract n*I
    mat_neg = mat - n * Id
    # Find the largest (in absolute value) eigen pair
    eig_neg, vec_neg, residual = eigh_largest(mat_neg, max_iters, tol)

    # Add back to the eigenvalue
    eig = eig_neg + n
    # Get the residual for debug/quality information
    return eig, vec_neg, residual


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
        The maximum number of iterations to run, by default 100
    tol : float, optional
        Tolerance for largest differences between iterates for the eigenvector phase.
        Default: 1e-4

    Returns
    -------
    eigenvalues : Array
        The smallest eigenvalues of `C_arrays`. Shape: (num_rows, num_cols)
        Note: This returns a float, even if `mat` is complex, since it assumes
        `mat` is Hermitian.
    smallest_eigenvectors : Array
        The eigenvectors corresponding to the smallest eigenvalues.
        shape = (num_rows, num_cols, M)
    residual_max_phase_diff : float
        The largest phase difference between the final eigenvector and the previous
        iterate's eigenvector.
        Shape: (num_rows, num_cols)

    References
    ----------
    .. [1] https://mathreview.uwaterloo.ca/archive/voli/1/panju.pdf
    """
    func_cols = vmap(lambda x: eigh_smallest(x, max_iters, tol))
    func_stack = vmap(func_cols)
    return func_stack(C_arrays)


@partial(jit, static_argnames=("max_iters", "tol"))
def eigh_closest_to_mu(
    mat: ArrayLike, mu: float = 1.0, max_iters: int = 100, tol: float = 1e-4
) -> tuple[Array, float, float]:
    mat_minus_mu_i = mat - mu * jnp.eye(mat.shape[0], dtype=mat.dtype)
    eig, vec, residual = eigh_largest(jnp.linalg.inv(mat_minus_mu_i), max_iters, tol)

    # largest_of_inv = 1 / (lam - mu)
    # (lam - mu) = 1 / largest_of_inv
    # lam = (1 / largest_of_inv) + mu
    (1 / eig) + mu

    # The body function is another matrix multiply, followed by normalizing
    def body_fun(val: tuple[Array, Array, Array, Array, int]):
        cur_vec, old_vec, prev_eigenvalue, _old_max_phase_diff, cur_iter = val
        # Do next matmul
        next_vec = mat @ cur_vec
        # Normalize to a unit vector
        next_normed_vec = next_vec / jnp.linalg.norm(next_vec)

        # Get current eigenvalue for the loop cond
        new_eigenvalue = rayleigh(mat, next_normed_vec, is_unit_vector=True)

        # Get the max element-wise phase-difference to check convergence
        phase_diff = jnp.angle(next_normed_vec * old_vec.conj())
        max_phase_diff = jnp.max(jnp.abs(phase_diff))
        return next_normed_vec, cur_vec, new_eigenvalue, max_phase_diff, cur_iter + 1

    def cond_fun(val: tuple[Array, Array, Array, Array, int]):
        new_vec, old_vec, _new_eigenvalue, max_phase_diff, cur_iter = val
        # We will loop while the max phase change is greater than `tol`
        return (max_phase_diff > tol) & (cur_iter < max_iters)

    # Start with some unit vector. Make it zero phase:
    m, n = mat.shape
    v0 = (jnp.ones(m) + 1j * jnp.ones(m)).astype(mat.dtype)
    v0 /= jnp.linalg.norm(v0)
    # Dummy vals for starting vec (0) and eigenvalues
    eig0 = jnp.array(1.0, dtype=mat.dtype)
    v_final, _v_prev, last_eigenvalue, last_max_phase_diff, _n_iters = lax.while_loop(
        cond_fun,
        body_fun,
        (v0, -v0, eig0, 1e3, 0),
    )

    jax.debug.print("{} iterations", (_n_iters))
    return jnp.real(last_eigenvalue), v_final, jnp.float32(last_max_phase_diff)


# Section 2.1
# https://services.math.duke.edu/~jtwong/math361-2019/lectures/Lec10eigenvalues.pdf
@partial(jit, static_argnames=("mu", "max_iters", "tol"))
def eigh_closest_to_mu_rayleigh(
    mat: ArrayLike, mu: float, max_iters: int = 100, tol: float = 1e-4
):
    Id = jnp.eye(mat.shape[0], dtype=mat.dtype)

    # from jax.numpy.linalg import norm, solve
    from jax.numpy.linalg import norm
    from jax.scipy.linalg import lu_factor, lu_solve
    # from jax.scipy.linalg import lu_factor, lu_solve, solve

    A = mat - mu * Id
    A_lu = lu_factor(A)

    # The body function is another matrix multiply, followed by normalizing
    def body_fun(val: tuple[Array, Array, Array, Array, int]):
        cur_vec, old_vec, mu_k, cur_eig, _prev_eig, _old_max_phase_diff, cur_iter = val
        # Do next linear solve

        # Rayleigh iteration:
        # mu_k = rayleigh(mat, cur_vec, is_unit_vector=True)

        # Each iteration we are solving (A - mu_k * I) @ next_vec = cur_vec
        # next_vec = jax.scipy.linalg.solve(mat - mu_k * Id, cur_vec)
        # Non-rayleigh iteration
        # next_vec = jax.scipy.linalg.solve(mat - mu * Id, cur_vec)

        # jax.debug.print("mu_k: {}", mu_k)
        # A_k = mat - mu_k * Id
        # next_vec = solve(A_k, cur_vec)

        # Static A:
        # next_vec = solve(A, cur_vec)
        next_vec = lu_solve(A_lu, cur_vec)

        # Normalize to a unit vector
        next_vec /= norm(next_vec)

        # # Other way? slower?
        # next_vec = jnp.linalg.inv(mat - mu_k * Id) @ cur_vec
        # next_vec = jnp.linalg.inv(mat - mu * Id) @ cur_vec

        # mu_k = mu_k + rayleigh(A, cur_vec, is_unit_vector=False)

        # Get current eigenvalue for the loop cond
        new_eig = rayleigh(mat, next_vec, is_unit_vector=True)
        # mu_k1 = rayleigh(A_k, next_vec, is_unit_vector=True)
        # mu_k2 = rayleigh(A, next_vec, is_unit_vector=True)
        mu_k = new_eig
        # jax.debug.print("mu_k: {}, mu_k1: {}, mu_k2: {}", mu_k, mu_k1, mu_k2)
        # jax.debug.print("mu_k: {}, mu_k2: {}", mu_k, mu_k2)

        # Get the max element-wise phase-difference to check convergence
        phase_diff = jnp.angle(next_vec * old_vec.conj())
        max_diff = jnp.max(jnp.abs(phase_diff))
        return (
            next_vec,
            cur_vec,
            mu_k,
            new_eig,
            cur_eig,
            max_diff,
            cur_iter + 1,
        )

    def cond_fun(val: tuple[Array, Array, Array, Array, int]):
        (new_vec, old_vec, _mu_k, new_eig, old_eig, max_phase_diff, cur_iter) = val
        eig_diff = jnp.abs(new_eig - old_eig)
        # We will loop while the max phase change is greater than `tol`
        # return ((max_phase_diff > tol) | (eig_diff > tol)) & (cur_iter < max_iters)
        return (eig_diff > tol) & (cur_iter < max_iters)

    # Start with some unit vector. Make it zero phase:
    m, n = mat.shape
    v0 = (jnp.ones(m) + 1j * jnp.ones(m)).astype(mat.dtype)
    v0 /= jnp.linalg.norm(v0)
    # Dummy vals for starting vec (0) and eigenvalues
    eig0 = jnp.array(1.0, dtype=mat.dtype)
    (
        v_final,
        _v_prev,
        mu_last,
        last_eigenvalue,
        _prev_eig,
        last_max_phase_diff,
        _n_iters,
    ) = lax.while_loop(
        cond_fun,
        body_fun,
        (v0, -v0, mu, eig0, 0 * eig0, 1e3, 0),
    )
    # Now `last_eigenvalue` is the largest of (A - mu I)^{-1}
    # We want the value of lambda for A that's closes to mu
    # desired_lambda = mu + (1 / last_eigenvalue)
    # The eigenvector is still the same though
    desired_lambda = last_eigenvalue

    # jax.debug.print("{} iterations. mu: {}, {}", _n_iters, mu_last, last_eigenvalue)
    return jnp.real(desired_lambda), v_final, jnp.float32(last_max_phase_diff)
