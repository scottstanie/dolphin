"""Demonstrate sequential datum adjustment using in-memory phase linking.

This script simulates the sequential phase linking process with datum adjustment
between ministacks. It processes a stack of SLCs split into multiple ministacks,
runs phase linking on each with different reference indices, and then performs
a final adjustment phase linking on the compressed SLCs.
"""

from __future__ import annotations

import logging

import numpy as np

from dolphin import HalfWindow
from dolphin.phase_link._compress import compress
from dolphin.phase_link._core import run_phase_linking

logger = logging.getLogger(__name__)
import jax.random
from synth import covariance, crlb
from synth.config import CustomCoherence


def generate_synthetic_slc_stack(
    shape: tuple[int, int, int], num_looks: float = 5 * 5
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic complex SLC data.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the output stack (n_images, n_rows, n_cols)

    Returns
    -------
    noisy_stack : ndarray
        Complex SLC stack with correlated Gaussian random data
    crlb_std_devs : ndarray
        CRLB

    """
    coh = CustomCoherence(gamma_inf=0.1, tau0=36)
    days = np.arange(100) * 12
    C = coh.to_array(days)
    crlb_std_devs = crlb.compute_lower_bound_std(C, num_looks=int(num_looks))

    from synth.deformation import gaussian

    num_time_steps, rows, cols = shape
    shape2d = (rows, cols)
    from troposim import turbulence

    turb = turbulence.simulate(shape=shape, max_amp=2)
    # Get shape of deformation in final form (normalized to 1 max)
    final_defo = gaussian(shape=shape2d, sigma=30).reshape((1, *shape2d))
    final_defo *= 5 / np.max(final_defo)
    # Broadcast this shape with linear evolution
    np.linspace(0, 1, num=num_time_steps)[:, None, None]

    defo = turb
    # defo = turb + final_defo * time_evolution

    key = jax.random.key(1)
    noisy_stack = np.asarray(
        covariance.make_noisy_samples_jax(key, C=C, defo_stack=defo)
    )
    return np.asarray(noisy_stack), crlb_std_devs


def rmse(stack: np.ndarray) -> np.ndarray:
    """Find the root mean square error (from 0) of a 3D stack."""
    if np.iscomplexobj(stack):
        stack = np.angle(stack)
    return np.sqrt(np.mean(np.abs(stack) ** 2, axis=(1, 2)))


def run(
    total_shape: tuple[int, int, int] = (100, 200, 200),
    ministack_size: int = 20,
    half_window: tuple[int, int] = (5, 5),
) -> tuple[np.ndarray]:
    """Run sequential datum adjustment demonstration.

    Parameters
    ----------
    total_shape : tuple[int, int, int]
        Total shape of the SLC stack (n_images, n_rows, n_cols)
    ministack_size : int
        Number of images per ministack (M in the description)
    half_window : tuple[int, int]
        Half window size (y, x) for phase linking

    Returns
    -------
    rmse_out_phases : np.ndarray
        RMSE of the phases after datum adjustment
    rmse_out_phases_new : np.ndarray
        RMSE of the phases without datum adjustment (dolphin near-real-time version)

    """
    logging.basicConfig(level=logging.INFO)

    n_images, _n_rows, _n_cols = total_shape
    n_ministacks = n_images // ministack_size

    assert (
        n_images % ministack_size == 0
    ), f"n_images ({n_images}) must be divisible by ministack_size ({ministack_size})"

    logger.info(
        f"Generating synthetic SLC stack with shape {total_shape}, "
        f"{n_ministacks} ministacks of size {ministack_size}"
    )

    full_slc_stack, crlb_std_devs = generate_synthetic_slc_stack(
        total_shape, num_looks=4 * np.prod(half_window)
    )

    ministack_phases: list[np.ndarray] = []
    ministack_phases_new: list[np.ndarray] = []
    ministack_temp_cohs: list[np.ndarray] = []
    compressed_slcs: list[np.ndarray] = []
    compressed_slcs_new: list[np.ndarray] = []

    for idx in range(n_ministacks):
        start_idx = idx * ministack_size
        end_idx = (idx + 1) * ministack_size

        cur_slc_data = full_slc_stack[start_idx:end_idx]

        if idx > 0:
            cur_pl_input = np.concatenate([compressed_slcs, cur_slc_data], axis=0)
            cur_pl_input_new = np.concatenate(
                [compressed_slcs_new, cur_slc_data], axis=0
            )
        else:
            cur_pl_input = cur_slc_data
            cur_pl_input_new = cur_slc_data

        pl_output = run_phase_linking(
            cur_pl_input,
            half_window=HalfWindow(*half_window),
            reference_idx=idx,  # the first real SLC
            first_real_slc_idx=idx,
            compute_crlb=False,
        )
        cur_comp_slc = compress(
            cur_pl_input,
            pl_output.cpx_phase,
            first_real_slc_idx=idx,
        )

        ministack_phases.append(pl_output.cpx_phase)
        ministack_temp_cohs.append(pl_output.temp_coh)
        compressed_slcs.append(cur_comp_slc)

        pl_output_new = run_phase_linking(
            cur_pl_input_new,
            half_window=HalfWindow(*half_window),
            reference_idx=max(0, idx - 1),  # the most recent compressed SLC
            first_real_slc_idx=idx,
            compute_crlb=False,
        )
        ministack_phases_new.append(pl_output_new.cpx_phase)
        cur_comp_slc_new = compress(
            cur_pl_input_new,
            pl_output_new.cpx_phase,
            first_real_slc_idx=idx,
        )
        compressed_slcs_new.append(cur_comp_slc_new)

        logger.info(f"Ministack {idx} complete")

    compressed_slc_stack = np.stack(compressed_slcs, axis=0)
    logger.info(f"Compressed SLC stack shape: {compressed_slc_stack.shape}")

    adjustment_pl_output = run_phase_linking(
        compressed_slc_stack,
        half_window=HalfWindow(*half_window),
        use_evd=False,
        reference_idx=0,
        first_real_slc_idx=0,
        compute_crlb=False,
    )
    adjustment_phases = adjustment_pl_output.cpx_phase
    out_phases = np.zeros_like(full_slc_stack, dtype="float32")
    for m_idx, cur_phase in enumerate(ministack_phases):
        logger.info(f"{m_idx = }, {cur_phase.shape = }")
        combined = cur_phase[m_idx:] * adjustment_phases[m_idx][None]
        out_phases[m_idx * ministack_size : (m_idx + 1) * ministack_size] = np.angle(
            combined
        )

    logger.info(f"Adjustment phases shape: {adjustment_phases.shape}")
    logger.info(
        "Datum adjustment complete. "
        f"Adjustments computed for {len(compressed_slcs)} ministacks."
    )

    out_phases_new = np.concatenate(
        [
            np.angle(ministack_phases_new[ii][ii:])
            for ii in range(len(ministack_phases_new))
        ]
    )

    # Get rmse values, trimming the edges away from the half window, taking every nth
    # pixel in space to avoid the correlated outputs
    trim_slices = (
        slice(None),
        slice(half_window[0], -half_window[0], half_window[0] // 2),
        slice(half_window[1], -half_window[1], half_window[1] // 2),
    )
    rmse_out_phases = rmse(out_phases[trim_slices])
    rmse_out_phases_new = rmse(out_phases_new[trim_slices])

    return crlb_std_devs, rmse_out_phases, rmse_out_phases_new


def plot_results(out_phases, out_phases_new, crlb_std_devs):
    import ultraplot as uplt

    fig1, ax1 = uplt.subplots()

    ax1.plot(rmse(out_phases_new), label="Near-real-time", lw=2)
    ax1.plot(rmse(out_phases), label="Datum Adj.", lw=2)
    ax1.plot(crlb_std_devs, label="CRLB", color="k")
    ax1.legend()
    return fig1


if __name__ == "__main__":
    import ultraplot as uplt

    # ms_sizes = [5, 10, 20, 25]
    # ms_sizes = [20, 10]
    ms_sizes = [20]
    rmses = [run(ministack_size=ms) for ms in ms_sizes]
    avg_da = np.mean(np.stack([r_da for (_, r_da, r_new) in rmses]), axis=0)
    avg_new = np.mean(np.stack([r_new for (_, r_da, r_new) in rmses]), axis=0)

    fig, ax = uplt.subplots(refwidth=4, refheight=3)
    ax.plot(rmses[0][0], label="CRLB", color="k", marker="s")
    # for ms, (_, r_da, r_new) in zip(ms_sizes, rmses):
    #     ax.plot(r_da, label=f"Datum Adj. (ms = {ms})")
    #     ax.plot(r_new, label=f"Dolphin. (ms = {ms})")

    ax.plot(avg_da, lw=4, marker="o", label="Datum Adj.")
    ax.plot(avg_new, lw=4, marker="d", label="Dolphin.")
    ax.legend()
    ax.format(ylabel="RMSE [rad]", xlabel="SLC index")
    fig.savefig("demo_sequential_datum_adjustment_avg.pdf")

    fig, ax = uplt.subplots(refwidth=4, refheight=2)
    ax.plot(rmses[0][0], label="CRLB", color="k", marker="s")
    crlb, r_da, r_new = rmses[0]
    ax.plot(r_da, marker="o", label="Datum Adj.")
    ax.plot(r_new, marker="d", label="Dolphin.")
    ax.legend()
    ax.format(ylabel="RMSE [rad]", xlabel="SLC index")
    fig.savefig("figures/figure_sequential_datum_adjustment.pdf")
