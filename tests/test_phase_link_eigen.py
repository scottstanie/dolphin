import numpy as np
import numpy.testing as npt
import pytest

from dolphin.phase_link import (
    eigh_largest,
    eigh_largest_stack,
    eigh_smallest,
    eigh_smallest_stack,
)
from dolphin.phase_link.simulate import simulate_coh

# Used for matrix size
N = 20


def get_eigvec_phase_difference(a, b):
    """Compare the phase difference of two eigenvectors (or stacks).

    Using this since there may be a relative offset in the phase of
    each starting, depending on the algorithm.
    """
    a_0 = a * a[..., [0]].conj()
    b_0 = b * b[..., [0]].conj()
    return np.angle(a_0 * b_0.conj())


class TestEighSingle:
    """Test the single pixel case."""

    @pytest.fixture
    def coh_matrix(self):
        C = simulate_coh(num_acq=N, add_signal=True)[0]

        npt.assert_allclose(C, C.T.conj())
        return C

    @pytest.fixture
    def expected_largest(self, coh_matrix):
        # Compare to numpy
        eig_vals, eig_vecs = np.linalg.eigh(coh_matrix)
        expected_eig = eig_vals[-1]
        # Check it's sorted in ascending order
        assert np.all(expected_eig >= eig_vals)
        # Should be positive definite
        assert np.all(eig_vals >= 0)
        expected_evec = eig_vecs[:, [-1]]
        assert expected_evec.shape == (N, 1)
        return expected_eig, expected_evec

    def test_eigh_largest(self, coh_matrix, expected_largest):
        expected_evalue, expected_evec = expected_largest
        evalue, evec, residual = eigh_largest(coh_matrix)
        assert evec.shape == (N, 1)

        npt.assert_allclose(expected_evalue, evalue, atol=1e-5)
        assert np.max(get_eigvec_phase_difference(expected_evec, evec)) < 2e-4
        assert residual < 1e-4

    def test_eigh_largest_tolerance(self, coh_matrix, expected_largest):
        expected_evalue, expected_evec = expected_largest
        # Test the single-pixel case
        tol = 1e-7
        evalue, evec, residual = eigh_largest(coh_matrix, max_iters=500, tol=tol)
        assert residual < tol

        npt.assert_allclose(expected_evalue, evalue, rtol=2 * tol)
        # Note: the max phase diff for one element may be higher than
        # the eigenvalue residual
        phase_diff_tol = 10 * tol
        assert np.max(get_eigvec_phase_difference(expected_evec, evec)) < phase_diff_tol

    @pytest.fixture
    def coh_gamma_inv(self, coh_matrix) -> np.ndarray:
        return coh_matrix * np.abs(np.linalg.inv(coh_matrix))

    @pytest.fixture
    def expected_smallest(self, coh_gamma_inv):
        # Compare to numpy
        eig_vals, eig_vecs = np.linalg.eigh(coh_gamma_inv)
        expected_eig = eig_vals[0]
        # Check it's sorted in ascending order
        assert np.all(expected_eig <= eig_vals)
        # Should be positive definite
        assert np.all(eig_vals >= 0)
        expected_evec = eig_vecs[:, [0]]
        assert expected_evec.shape == (N, 1)
        return expected_eig, expected_evec

    def test_eigh_smallest(self, coh_gamma_inv, expected_smallest):
        expected_evalue, expected_evec = expected_smallest
        evalue, evec, residual = eigh_smallest(coh_gamma_inv)
        assert evec.shape == (N, 1)
        # Note: the convergence is much worse for the smallest, where the eigenvalues
        # might looks something like
        # [0.07619964, 0.077591  , 0.07998301, ...
        npt.assert_allclose(expected_evalue, evalue, atol=0.2)
        # But we really care about the phase difference more
        assert np.max(get_eigvec_phase_difference(expected_evec, evec)) < 1e-4
        assert residual < 1e-3


class TestEighStack:
    """Test the stack functionality."""

    @pytest.fixture
    def coh_stack(self):
        num_rows, num_cols = 6, 7
        out = np.empty((num_rows, num_cols, N, N), dtype=np.complex64)
        for row in range(num_rows):
            for col in range(num_cols):
                out[row, col] = simulate_coh(num_acq=N, add_signal=True)[0]
        return out

    @pytest.fixture
    def expected_largest(self, coh_stack):
        # Compare to numpy
        eig_vals, eig_vecs = np.linalg.eigh(coh_stack)
        assert np.all(eig_vals >= 0)
        expected_eig = eig_vals[:, :, -1]
        expected_evec = eig_vecs[:, :, :, [-1]]
        assert expected_eig.shape == (6, 7)
        assert expected_evec.shape == (6, 7, N, 1)
        return expected_eig, expected_evec

    def test_eigh_largest_stack(self, coh_stack, expected_largest):
        expected_eig, expected_evec = expected_largest
        evalues, evecs, residuals = eigh_largest_stack(coh_stack)
        assert evalues.shape == (6, 7)
        assert evecs.shape == (6, 7, N, 1)
        assert residuals.shape == (6, 7)

        npt.assert_allclose(expected_eig, evalues, atol=2e-5)
        assert np.max(np.abs(get_eigvec_phase_difference(expected_evec, evecs))) < 1e-4

    @pytest.fixture
    def coh_gamma_inv_stack(self, coh_stack) -> np.ndarray:
        return coh_stack * np.abs(np.linalg.inv(coh_stack))

    @pytest.fixture
    def expected_smallest(self, coh_gamma_inv_stack):
        # Compare to numpy
        eig_vals, eig_vecs = np.linalg.eigh(coh_gamma_inv_stack)
        assert np.all(eig_vals >= 0)
        expected_eig = eig_vals[:, :, 0]
        expected_evec = eig_vecs[:, :, :, [0]]
        assert expected_eig.shape == (6, 7)
        assert expected_evec.shape == (6, 7, N, 1)
        return expected_eig, expected_evec

    def test_eigh_smallest_stack(self, coh_gamma_inv_stack, expected_smallest):
        expected_eig, expected_evec = expected_smallest
        evalues, evecs, residuals = eigh_smallest_stack(coh_gamma_inv_stack)
        assert evalues.shape == (6, 7)
        assert evecs.shape == (6, 7, N, 1)
        assert residuals.shape == (6, 7)
