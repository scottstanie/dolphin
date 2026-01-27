"""Extract nearest-N coherence magnitudes from coherence matrices."""

import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def extract_nearest_coherences(
    cov_matrix: Array,
    n: int,
) -> Array:
    """Extract the nearest-N coherence magnitudes from a single covariance matrix.

    For a coherence matrix C of size (N, N), returns the magnitudes of the
    first `n` super-diagonals stacked together.

    Parameters
    ----------
    cov_matrix : Array
        Complex (N, N) coherence/covariance matrix
    n : int
        Number of diagonals to extract. n=1 gives first off-diagonal (nearest
        neighbor coherences), n=2 gives first 2 diagonals, etc.

    Returns
    -------
    Array
        Coherence magnitudes with shape (n_interferograms,) where
        n_interferograms = sum_{k=1}^{n} (N - k) = n*N - n*(n+1)/2

    """
    diags = []
    for k in range(1, n + 1):
        diag_k = jnp.diag(cov_matrix, k=k)
        diags.append(jnp.abs(diag_k))
    return jnp.concatenate(diags)


def make_batch_extractor(n: int):
    """Create a JIT-compiled batch extractor for a specific n.

    Parameters
    ----------
    n : int
        Number of diagonals to extract.

    Returns
    -------
    Callable
        JIT-compiled function that takes (rows, cols, N, N) coherence matrices
        and returns (rows, cols, n_interferograms) coherence magnitudes.

    """

    @jax.jit
    def _extract_single(cov_matrix: Array) -> Array:
        diags = []
        for k in range(1, n + 1):
            diag_k = jnp.diag(cov_matrix, k=k)
            diags.append(jnp.abs(diag_k))
        return jnp.concatenate(diags)

    @jax.jit
    def extract_batch(cov_matrices: Array) -> Array:
        """Extract nearest-N coherences for a batch of covariance matrices.

        Parameters
        ----------
        cov_matrices : Array
            Complex (rows, cols, N, N) array of covariance matrices

        Returns
        -------
        Array
            Coherence magnitudes: (rows, cols, n_interferograms) array

        """
        return jax.vmap(jax.vmap(_extract_single))(cov_matrices)

    return extract_batch


def get_nearest_coherence_count(matrix_size: int, n: int) -> int:
    """Get the number of coherence values for a given matrix size and n.

    Parameters
    ----------
    matrix_size : int
        Size of the coherence matrix (number of SLCs)
    n : int
        Number of diagonals to extract

    Returns
    -------
    int
        Number of coherence values = sum_{k=1}^{n} (matrix_size - k)

    """
    return sum(matrix_size - k for k in range(1, n + 1))


def get_nearest_coherence_ifg_pairs(num_slcs: int, n: int) -> list[tuple[int, int]]:
    """Get the (reference_idx, secondary_idx) pairs for the nearest-N coherences.

    Useful for labeling outputs or understanding which interferogram each
    coherence value corresponds to.

    Parameters
    ----------
    num_slcs : int
        Number of SLCs in the stack
    n : int
        Number of diagonals (bandwidth)

    Returns
    -------
    list[tuple[int, int]]
        List of (ref_idx, sec_idx) pairs, e.g. for n=2 with 4 SLCs:
        [(0,1), (1,2), (2,3), (0,2), (1,3)]

    """
    pairs = []
    for k in range(1, n + 1):
        for i in range(num_slcs - k):
            pairs.append((i, i + k))
    return pairs
