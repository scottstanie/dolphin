import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def compute_nearest_closure_phases(
    cov_matrix: Array,
) -> Array:
    """Compute the nearest-neighbor closure phases for a single covariance matrix."""
    # Extract the diagonals we need
    # First super-diagonal: Used for (i, i+1), then (i+1, i+2)
    diag_1 = jnp.diag(cov_matrix, k=1)  # length N-1
    # Second super-diagonal (i, i+2). Used for the bandwidth-2 interferograms
    diag_2 = jnp.diag(cov_matrix, k=2)  # length N-2

    # Compute closure phases as complex numbers, then take the angle
    closure_complex = diag_1[:-1] * diag_1[1:] * jnp.conj(diag_2)
    return jnp.angle(closure_complex)


# Vectorized version for multiple covariance matrices (e.g., different pixels)
@jax.jit
def compute_nearest_closure_phases_batch(
    cov_matrices: Array,
) -> Array:
    """Compute nearest-neighbor closure phases for a batch of covariance matrices.

    Parameters
    ----------
    cov_matrices : Array
        Complex (..., R, C, N, N) array of M covariance matrices

    Returns
    -------
    Array
        Closure phases: (..., R, C, N-2) array of closure phases

    """
    return jax.vmap(jax.vmap(compute_nearest_closure_phases))(cov_matrices)


@jax.jit
def closure_phase_coefficient(C: Array) -> Array:
    r"""Weighted closure-phase coefficient :math:`\gamma_{CPw}`.

    A method-independent reliability indicator computed directly from the sample
    coherence matrix, defined in [@Heimpel2026Heuristic] as

    .. math::

        \gamma_{CPw} = \max\!\Bigl(
            \frac{\sum_{i<j<k} |T_{ij} T_{jk} T_{ki}| \cos\phi^{\Delta}_{ijk}}
                 {\sum_{i<j<k} |T_{ij} T_{jk} T_{ki}|},
            \; 0 \Bigr)

    where :math:`\phi^{\Delta}_{ijk} = \angle(T_{ij} T_{jk} T_{ki})` is the triplet
    closure phase. Equals 1 for a rank-1 outer product :math:`T = vv^H` (exact
    linkage) and 0 when :math:`T = I_N` (no coherence).

    The triplet sum is evaluated in closed form to avoid materializing the
    :math:`O(n^3)` tensor of triplet products per pixel. Because both the numerator
    integrand :math:`\mathrm{Re}(T_{ij} T_{jk} T_{ki})` and the denominator
    integrand :math:`|T_{ij} T_{jk} T_{ki}|` are invariant under permutations of
    :math:`(i,j,k)` (for Hermitian :math:`T` with real unit diagonal), summing
    over ordered triplets equals :math:`1/6` of the sum over all index tuples,
    and the degenerate (two or three equal indices) terms cancel cleanly:

    .. math::

        \sum_{i<j<k} \mathrm{Re}(T_{ij} T_{jk} T_{ki})
          = [\mathrm{Re}(\mathrm{tr}(T^3)) - 3 \|T\|_F^2 + 2n] / 6

        \sum_{i<j<k} |T_{ij} T_{jk} T_{ki}|
          = [\mathrm{tr}(|T|^3) - 3 \|T\|_F^2 + 2n] / 6

    The factor of :math:`1/6` cancels in the ratio, leaving two :math:`n \times n`
    matrix multiplications per pixel.

    Parameters
    ----------
    C : Array
        Complex Hermitian sample coherence matrix, shape (..., n, n), with
        unit-magnitude diagonal.

    Returns
    -------
    Array
        :math:`\gamma_{CPw}` values with shape equal to the leading dims of `C`,
        in [0, 1].

    References
    ----------
    .. [1] Heimpel et al., "Heuristic quality coefficients for interferometric
       phase linking", ISPRS J. Photogramm. Remote Sens. 237 (2026) 1-21.
       doi:10.1016/j.isprsjprs.2026.04.015

    """
    A = jnp.abs(C)
    n = C.shape[-1]
    T2 = C @ C
    A2 = A @ A
    # ||T||_F^2 = sum |T_ij|^2
    fro_sq = jnp.sum(A * A, axis=(-2, -1))
    # tr(T^3) = <T^2, T>_F = sum T2_ij * conj(T_ij); real for Hermitian T.
    tr_T3 = jnp.real(jnp.sum(T2 * jnp.conj(C), axis=(-2, -1)))
    # tr(|T|^3) = <A^2, A>_F for real symmetric A = |T|.
    tr_A3 = jnp.sum(A2 * A, axis=(-2, -1))
    num = tr_T3 - 3.0 * fro_sq + 2.0 * n
    den = tr_A3 - 3.0 * fro_sq + 2.0 * n
    # For n < 3 (no triplets) or an all-zero window, den == 0; return 0.
    safe_den = jnp.where(den > 0, den, 1.0)
    return jnp.maximum(jnp.where(den > 0, num / safe_den, 0.0), 0.0)
