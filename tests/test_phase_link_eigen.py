import numpy as np
import numpy.testing as npt
import pytest

from dolphin.phase_link import eigh_largest, eigh_largest_stack
from dolphin.phase_link.simulate import simulate_coh


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
        C = simulate_coh(num_acq=20, add_signal=True)[0]

        npt.assert_allclose(C, C.T.conj())
        return C

    @pytest.fixture
    def expected(self, coh_matrix):
        # Compare to numpy
        eig_vals, eig_vecs = np.linalg.eigh(coh_matrix)
        expected_eig = eig_vals[-1]
        # Check it's sorted in ascending order
        assert np.all(expected_eig >= eig_vals)
        # Should be positive definite
        assert np.all(eig_vals >= 0)
        expected_evec = eig_vecs[:, [-1]]
        assert expected_evec.shape == (20, 1)
        return expected_eig, expected_evec

    def test_eigh_largest(self, coh_matrix, expected):
        expected_evalue, expected_evec = expected
        evec, evalue, residual = eigh_largest(coh_matrix)
        assert evec.shape == (20, 1)

        npt.assert_allclose(expected_evalue, evalue, atol=1e-5)
        assert np.max(get_eigvec_phase_difference(expected_evec, evec)) < 1e-4

    def test_eigh_tolerance(self, coh_matrix, expected):
        expected_evalue, expected_evec = expected
        # Test the single-pixel case
        tol = 1e-7
        evec, evalue, residual = eigh_largest(coh_matrix, max_iters=500, tol=tol)
        assert residual < tol

        npt.assert_allclose(expected_evalue, evalue, rtol=2 * tol)
        # Note: the max phase diff for one element may be higher than
        # the eigenvalue residual
        phase_diff_tol = 10 * tol
        assert np.max(get_eigvec_phase_difference(expected_evec, evec)) < phase_diff_tol


class TestEighStack:
    """Test the stack functionality."""

    @pytest.fixture
    def coh_stack(self):
        num_rows, num_cols = 6, 7
        out = np.empty((num_rows, num_cols, 20, 20), dtype=np.complex64)
        for row in range(num_rows):
            for col in range(num_cols):
                out[row, col] = simulate_coh(num_acq=20, add_signal=True)[0]
        return out

    @pytest.fixture
    def expected(self, coh_stack):
        # Compare to numpy
        eig_vals, eig_vecs = np.linalg.eigh(coh_stack)
        assert np.all(eig_vals >= 0)
        expected_eig = eig_vals[:, :, -1]
        expected_evec = eig_vecs[:, :, :, [-1]]
        assert expected_eig.shape == (6, 7)
        assert expected_evec.shape == (6, 7, 20, 1)
        return expected_eig, expected_evec

    def test_eigh_largest_stack(self, coh_stack, expected):
        expected_eig, expected_evec = expected
        evecs, evalues, residuals = eigh_largest_stack(coh_stack)
        assert evecs.shape == (6, 7, 20, 1)
        assert evalues.shape == (6, 7)
        assert residuals.shape == (6, 7)

        npt.assert_allclose(expected_eig, evalues, atol=2e-5)
        assert np.max(np.abs(get_eigvec_phase_difference(expected_evec, evecs))) < 1e-4
