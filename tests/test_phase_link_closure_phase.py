import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dolphin.phase_link._closure_phase import (
    closure_phase_coefficient,
    compute_nearest_closure_phases,
    compute_nearest_closure_phases_batch,
)


@pytest.mark.parametrize("n", [4, 10, 23])
def test_output_shape(n):
    """Output must be (n-2,) for a single covariance matrix."""
    C = jnp.eye(n, dtype=jnp.complex64)
    out = compute_nearest_closure_phases(C)
    assert out.shape == (n - 2,)


def test_rank1_outer_product_closure_is_zero():
    """For C = v v^H (rank-1), every closure phase is identically zero."""
    n = 12
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (n,)) + 1j * jax.random.normal(key, (n,))
    C = jnp.outer(v, jnp.conj(v))  # v v^H, Hermitian rank-1

    phi = compute_nearest_closure_phases(C)
    assert jnp.allclose(phi, 0.0, atol=1e-6)


def test_random_hermitian_manual_check():
    """Compare against a direct NumPy/JAX implementation for one matrix."""
    n = 7
    key = jax.random.PRNGKey(123)
    A = jax.random.normal(key, (n, n)) + 1j * jax.random.normal(key, (n, n))
    C = A + A.T.conj()  # make it Hermitian

    phi = compute_nearest_closure_phases(C)

    # Manual reference computation
    d1 = jnp.diag(C, k=1)  # C_{i,i+1}
    d2 = jnp.diag(C, k=2)  # C_{i,i+2}
    expected = jnp.angle(d1[:-1] * d1[1:] * jnp.conj(d2))

    assert jnp.allclose(phi, expected, atol=1e-6)


def test_batch_vectorization():
    """Sanity-check batched call and that [0,0] matches single-matrix path."""
    r, c, n = 3, 4, 9
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (r, c, n, n)) + 1j * jax.random.normal(key, (r, c, n, n))
    C = A + jnp.swapaxes(A, -2, -1).conj()  # Hermitian (r,c,n,n)

    batch_phi = compute_nearest_closure_phases_batch(C)
    assert batch_phi.shape == (r, c, n - 2)

    # spot-check one element against the scalar implementation
    assert jnp.allclose(
        batch_phi[0, 0],
        compute_nearest_closure_phases(C[0, 0]),
        atol=1e-6,
    )


class TestClosurePhaseCoefficient:
    """Tests for the weighted closure-phase coefficient gamma_CPw."""

    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_identity_is_zero(self, n):
        """gamma_CPw(T=I) = 0 (fully decorrelated)."""
        C = jnp.eye(n, dtype=jnp.complex64)
        assert float(closure_phase_coefficient(C)) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_rank1_outer_product_is_one(self, n):
        """gamma_CPw(T = v v^H) = 1 for unit-modulus v (exact linkage)."""
        rng = np.random.default_rng(n)
        phases = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
        v = jnp.exp(1j * jnp.asarray(phases))
        C = jnp.outer(v, jnp.conj(v))  # unit diag, rank-1
        assert float(closure_phase_coefficient(C)) == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("n", [4, 8])
    def test_bounded_0_to_1(self, n):
        """gamma_CPw is clamped to [0, 1] for Hermitian T with unit diagonal."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            # construct a sample coherence by averaging a few random rank-1s
            phases = rng.uniform(-np.pi, np.pi, (5, n)).astype(np.float32)
            vs = np.exp(1j * phases)
            C = np.mean([np.outer(v, v.conj()) for v in vs], axis=0)
            # normalize to unit diagonal (standard sample-coherence definition)
            d = np.sqrt(np.abs(np.diag(C)))
            C = C / np.outer(d, d)
            gamma = float(closure_phase_coefficient(jnp.asarray(C)))
            assert 0.0 <= gamma <= 1.0

    def test_batch_shape_and_values(self):
        """Batched (r, c, n, n) input returns matching (r, c) output."""
        r, cc, n = 3, 4, 7
        # mix identity and rank-1 matrices to get both 0 and 1 outputs
        rng = np.random.default_rng(99)
        v = jnp.exp(1j * jnp.asarray(rng.uniform(-np.pi, np.pi, n).astype(np.float32)))
        C_id = jnp.eye(n, dtype=jnp.complex64)
        C_r1 = jnp.outer(v, jnp.conj(v))
        batch = jnp.stack(
            [
                jnp.stack([C_id if (i + j) % 2 else C_r1 for j in range(cc)])
                for i in range(r)
            ]
        )
        out = closure_phase_coefficient(batch)
        assert out.shape == (r, cc)
        # Check per-cell: matches what each matrix gives on its own.
        for i in range(r):
            for j in range(cc):
                expected = 0.0 if (i + j) % 2 else 1.0
                assert float(out[i, j]) == pytest.approx(expected, abs=1e-5)

    def test_noisy_rank1_near_one(self):
        """gamma_CPw stays high when T is a sample estimate of a rank-1 signal."""
        n, W = 8, 200
        rng = np.random.default_rng(7)
        phases = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
        v = np.exp(1j * phases)
        # draw W realizations with small independent phase noise per sample
        noise = 0.1 * rng.standard_normal((W, n)).astype(np.float32)
        Omega = v[None, :] * np.exp(1j * noise)
        C = (Omega.conj().T @ Omega) / W
        d = np.sqrt(np.abs(np.diag(C)))
        C = C / np.outer(d, d)
        gamma = float(closure_phase_coefficient(jnp.asarray(C)))
        assert gamma > 0.95
