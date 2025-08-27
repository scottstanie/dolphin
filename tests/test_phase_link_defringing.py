import jax.numpy as jnp
import numpy as np
import pytest

from dolphin._types import HalfWindow, Strides
from dolphin.phase_linking._core import run_phase_linking
from dolphin.phase_linking.covariance import estimate_stack_covariance
from dolphin.phase_linking.defringing import deramp_window_for_cov


def _make_ramped_stack(
    nslc: int, H: int, W: int, fy_step: float, fx_step: float, noise: float = 0.0
):
    """Create an SLC stack with time-varying linear ramps across space."""
    # Time-integrated per-epoch slopes
    fys = np.cumsum(np.full(nslc - 1, fy_step))
    fxs = np.cumsum(np.full(nslc - 1, fx_step))
    fys = np.concatenate([[0.0], fys])
    fxs = np.concatenate([[0.0], fxs])

    yy = np.linspace(-(H // 2), H // 2, H)
    xx = np.linspace(-(W // 2), W // 2, W)
    Y, X = np.meshgrid(yy, xx, indexing="ij")

    slc = np.empty((nslc, H, W), dtype=np.complex64)
    for t in range(nslc):
        phase = fys[t] * Y + fxs[t] * X
        base = np.exp(1j * phase)
        if noise > 0:
            n = (
                noise
                * (np.random.randn(H, W) + 1j * np.random.randn(H, W))
                / np.sqrt(2)
            )
            base = base + n
            # Keep unit-ish magnitude for stability
            base = base / np.maximum(1e-6, np.abs(base))
        slc[t] = base
    return slc


def _offdiag_mean_abs(C: np.ndarray) -> float:
    """Mean |C_ij| for i != j."""
    n = C.shape[-1]
    iu = np.triu_indices(n, 1)
    il = np.tril_indices(n, -1)
    vals = np.concatenate(
        [np.abs(C[..., iu[0], iu[1]]).ravel(), np.abs(C[..., il[0], il[1]]).ravel()]
    )
    return float(vals.mean())


@pytest.mark.parametrize(
    "nslc,H,W,fy_step,fx_step,noise", [(5, 9, 9, 0.30, 0.25, 0.05)]
)
def test_defringing_improves_cov_offdiag(nslc, H, W, fy_step, fx_step, noise):
    """Slope-normalized covariance should have stronger off-diagonals."""
    np.random.seed(0)
    slc = _make_ramped_stack(nslc, H, W, fy_step, fx_step, noise=noise)

    half = HalfWindow(H // 2, W // 2)
    strides = Strides(1, 1)

    # Covariance at the center pixel window, without defringing
    C_off = estimate_stack_covariance(
        slc, half, strides=strides, neighbor_arrays=None, defringe_mode="off"
    )
    C_on = estimate_stack_covariance(
        slc, half, strides=strides, neighbor_arrays=None, defringe_mode="coh_only"
    )

    # Grab the center output pixel
    r = H // 2
    c = W // 2
    Cb = np.array(C_off[r, c])
    Ca = np.array(C_on[r, c])

    offdiag_before = _offdiag_mean_abs(Cb)
    offdiag_after = _offdiag_mean_abs(Ca)

    # Expect a noticeable improvement
    assert offdiag_after > offdiag_before + 0.10, (offdiag_before, offdiag_after)
    assert offdiag_after / max(offdiag_before, 1e-6) > 1.25, (
        offdiag_before,
        offdiag_after,
    )


def test_deramp_window_is_noop_when_no_ramp():
    """If there is no spatial ramp, deramping should be ~identity."""
    nslc, H, W = 4, 7, 7
    # Constant phase per time step (no spatial trend)
    rng = np.random.default_rng(123)
    t_phase = rng.uniform(-np.pi, np.pi, size=nslc)
    slc = np.exp(1j * t_phase[:, None, None]) * np.ones(
        (nslc, H, W), dtype=np.complex64
    )

    slc_j = jnp.asarray(slc)
    out = deramp_window_for_cov(slc_j)
    # Compare per-sample phases (magnitude already ~1)
    diff = np.angle(np.array(out) * np.conj(slc))
    # Wrap to [-pi, pi] and check small
    assert np.max(np.abs(diff)) < 1e-3


@pytest.mark.parametrize("use_evd", [False, True])
def test_phase_invariance_in_run_phase_linking(use_evd):
    """'coh_only' must not change the final linked phase (only the coherence)."""
    np.random.seed(0)
    nslc, H, W = 6, 15, 15
    slc = _make_ramped_stack(nslc, H, W, fy_step=0.20, fx_step=0.15, noise=0.03)

    half = HalfWindow(3, 3)
    strides = Strides(2, 2)

    out_off = run_phase_linking(
        slc_stack=slc,
        half_window=half,
        strides=strides,
        use_evd=use_evd,
        compute_crlb=False,
        defringe_mode="off",
    )
    out_on = run_phase_linking(
        slc_stack=slc,
        half_window=half,
        strides=strides,
        use_evd=use_evd,
        compute_crlb=False,
        defringe_mode="coh_only",
    )

    # Compare angles of the linked phase (unit magnitude)
    ang_off = np.angle(out_off.cpx_phase)
    ang_on = np.angle(out_on.cpx_phase)

    # Phase can differ by multiples of 2π due to numerical noise; compare wrapped difference
    diff = np.angle(np.exp(1j * (ang_on - ang_off)))
    assert np.nanmax(np.abs(diff)) < 1e-3, f"max phase diff: {np.nanmax(np.abs(diff))}"

    # (Optional) Demonstrate coherence improvement at least somewhere
    # Many pixels should see equal-or-better temp_coh with defringing
    better = np.nansum(out_on.temp_coh >= out_off.temp_coh)
    total = np.isfinite(out_off.temp_coh).sum()
    assert better >= 0.6 * total  # at least 60% non-worse
