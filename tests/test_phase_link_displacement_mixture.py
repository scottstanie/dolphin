import numpy as np
import pytest

from dolphin.phase_link._closure_phase import compute_nearest_closure_phases
from dolphin.phase_link.simulate import (
    evd,
    mle,
    simulate_displacement_covariance,
)


def test_displacement_only_counterexample():
    # Phase advance of the moving component: 0, -1, -2 radians in the SLCs.
    d = np.outer([0.0, 1.0, 2.0], [0.0, 1.0])
    C = simulate_displacement_covariance(d, 4 * np.pi, [0.8, 0.2])
    z1, z2 = 0.8 + 0.2 * np.exp(1j), 0.8 + 0.2 * np.exp(2j)
    expected = np.angle(z1 * z1 * z2.conjugate())
    assert expected > 0.1
    np.testing.assert_allclose(compute_nearest_closure_phases(C), [expected], atol=1e-6)
    np.testing.assert_allclose(C[0, [1, 2]], [z1, z2], atol=1e-14)
    assert np.min(np.abs(C)) > 0.7


@pytest.mark.parametrize("weights", [[1, 0], [0, 1], [0.5, 0.5]])
def test_single_component_and_symmetric_positive_lobe_close(weights):
    d = np.outer([0.0, 1.0, 2.0], [0.0, 1.0])
    C = simulate_displacement_covariance(d, 4 * np.pi, weights)
    np.testing.assert_allclose(compute_nearest_closure_phases(C), 0, atol=1e-6)


def test_symmetric_distribution_can_have_pi_closure():
    # Symmetry makes the centered characteristic function real, not positive.
    d = np.outer([0.0, 2.0, 4.0], [0.0, 1.0])
    C = simulate_displacement_covariance(d, 4 * np.pi)
    np.testing.assert_allclose(
        np.abs(compute_nearest_closure_phases(C)), np.pi, atol=1e-6
    )


def test_physical_covariance_and_invariances():
    rng = np.random.default_rng(42)
    d = rng.normal(size=(7, 4))
    weights = [2.0, 1.0, 3.0, 0.0]
    C = simulate_displacement_covariance(d, 5.6, weights)
    np.testing.assert_allclose(C, C.conj().T, atol=1e-14)
    np.testing.assert_allclose(np.diag(C), 1, atol=1e-14)
    assert np.linalg.eigvalsh(C).min() > -1e-12
    np.testing.assert_allclose(
        simulate_displacement_covariance(d + np.array([3, 7, 1, 4]), 5.6, weights),
        C,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        simulate_displacement_covariance(d * 1000, 5600, np.array(weights) * 5),
        C,
        atol=1e-14,
    )
    common_motion = rng.normal(size=(7, 1))
    shifted = simulate_displacement_covariance(d + common_motion, 5.6, weights)
    np.testing.assert_allclose(
        compute_nearest_closure_phases(shifted),
        compute_nearest_closure_phases(C),
        atol=1e-6,
    )


def test_coherent_unresolved_scatterers_converge_to_mixture():
    # Complex scattering coefficients are random across looks but fixed in time.
    rng = np.random.default_rng(9)
    d = np.outer([0.0, 1.0, 2.0], [0.0, 1.0])
    weights = np.array([0.8, 0.2])
    fields = rng.normal(size=(2, 100_000)) + 1j * rng.normal(size=(2, 100_000))
    fields *= np.sqrt(weights[:, None] / 2)
    slc = np.exp(-1j * d) @ fields
    C = slc @ slc.conj().T / slc.shape[1]
    power = np.sqrt(np.diag(C).real)
    C /= power[:, None] * power[None, :]
    expected = simulate_displacement_covariance(d, 4 * np.pi, weights)
    np.testing.assert_allclose(C, expected, atol=0.005)
    np.testing.assert_allclose(
        compute_nearest_closure_phases(C),
        compute_nearest_closure_phases(expected),
        atol=0.005,
    )
    triplets = slc[0] * slc[1].conj() * slc[1] * slc[2].conj()
    triplets *= slc[2] * slc[0].conj()
    np.testing.assert_allclose(np.angle(triplets), 0, atol=1e-12)


@pytest.mark.parametrize(
    "d,wavelength,weights",
    [
        ([], 1, None),
        ([[1], [np.nan]], 1, None),
        ([[1j]], 1, None),
        ([[1]], 0, None),
        ([[1]], np.inf, None),
        ([[1]], 1j, None),
        ([[1]], [1], None),
        ([[1, 2]], 1, [1]),
        ([[1, 2]], 1, [-1, 2]),
        ([[1, 2]], 1, [0, 0]),
        ([[1, 2]], 1, [1, np.inf]),
        ([[1, 2]], 1, [1, 1j]),
    ],
)
def test_invalid_inputs(d, wavelength, weights):
    with pytest.raises(ValueError):
        simulate_displacement_covariance(d, wavelength, weights)


@pytest.mark.parametrize("estimator", [evd, mle])
def test_phase_linking_reports_dominant_component_when_motion_wraps(estimator):
    # Eight dates at 12-day spacing; 20% of the power moves, 80% is stationary.
    # A constant real coherence factor stands in for noise and leaves closure unchanged.
    wavelength, dt, p = 56.0, 12.0, 0.2
    q = 4 * np.pi / wavelength
    times = np.arange(8) * dt

    def linked_displacement(x):
        v = x / (q * dt)
        C = simulate_displacement_covariance(
            np.outer(times, [0.0, v]), wavelength, [1 - p, p]
        )
        C = 0.8 * C
        np.fill_diagonal(C, 1.0)
        # theta_k - theta_0 = -q (d_k - d_0) for the linked SLC phase.
        return -np.unwrap(np.angle(estimator(C))) / q, p * v * times

    # Small differential motion: the linked phase tracks the power-weighted mean.
    est, mean = linked_displacement(0.1)
    np.testing.assert_allclose(est, mean, atol=0.03 * mean[-1])
    # Motion wrapping over the stack: the linked phase follows the stationary 80%,
    # with only a small residual oscillation leaking from the moving component.
    est, mean = linked_displacement(2.0)
    rate = np.polyfit(times, est, 1)[0]
    assert abs(rate) < 0.03 * mean[-1] / times[-1]
    assert np.abs(est).max() < 0.05 * mean[-1]


def test_summed_nearest_closure_equals_twice_chain_difference():
    # sum_k closure(k, k+1, k+2) == 2 * (lag-1 chain) - (even + odd lag-2 chains),
    # exactly, whenever no wrapping occurs. A cumulative nearest-closure map is thus
    # twice the disagreement between the bandwidth-1 and bandwidth-2 time series.
    rng = np.random.default_rng(3)
    n = 12
    phases = rng.uniform(-0.3, 0.3, size=(n, n))
    C = 0.7 * np.exp(1j * (phases - phases.T))
    np.fill_diagonal(C, 1.0)
    phi1, phi2 = np.angle(np.diag(C, 1)), np.angle(np.diag(C, 2))
    summed = np.sum(np.asarray(compute_nearest_closure_phases(C)))
    expected = phi1[:-1].sum() + phi1[1:].sum() - phi2.sum()
    np.testing.assert_allclose(summed, expected, atol=1e-6)
