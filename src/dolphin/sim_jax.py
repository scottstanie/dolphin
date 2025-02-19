from __future__ import annotations

import logging
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .interpolation import get_circle_idxs

logger = logging.getLogger("dolphin")


@jax.jit
def phase_similarity(x1: jnp.ndarray, x2: jnp.ndarray) -> float:
    """Compute the similarity between two complex 1D vectors in JAX."""
    return jnp.real(x1 * jnp.conjugate(x2)).mean()


def median_similarity(
    ifg_stack: np.ndarray, search_radius: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """Compute the median similarity of each pixel and its neighbors."""
    # Convert to jax arrays
    ifg_stack_j = jnp.array(ifg_stack)
    mask_j = None if mask is None else jnp.array(mask, dtype=bool)

    # Vectorized call
    out = _create_loop_and_run_jax(
        ifg_stack_j, search_radius, mask_j, summary_func=jnp.nanmedian
    )
    return np.array(out, dtype=np.float32)


def max_similarity(
    ifg_stack: np.ndarray, search_radius: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """Compute the maximum similarity of each pixel and its neighbors."""
    ifg_stack_j = jnp.array(ifg_stack)
    mask_j = None if mask is None else jnp.array(mask, dtype=bool)

    out = _create_loop_and_run_jax(
        ifg_stack_j, search_radius, mask_j, summary_func=jnp.nanmax
    )
    return np.array(out, dtype=np.float32)


def _create_loop_and_run_jax(
    ifg_stack_j: jnp.ndarray,
    search_radius: int,
    mask_j: jnp.ndarray | None,
    summary_func: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Run a JAX-based loop for median/max similarity, returning 2D array."""
    n_ifg, rows, cols = ifg_stack_j.shape

    # Mark any nans/all zeros as invalid
    invalid_mask = jnp.sum(jnp.nan_to_num(ifg_stack_j), axis=0) == 0
    mask_j = ~invalid_mask if mask_j is None else mask_j & ~invalid_mask

    # Convert everything to "unit phasor" form:
    # if it is complex, we keep only angle; if real, treat it as phase
    is_complex = jnp.iscomplex(ifg_stack_j).any()
    ifg_unit = (
        jnp.exp(1j * jnp.angle(ifg_stack_j))
        if is_complex
        else jnp.exp(1j * ifg_stack_j)
    )

    # Circle idxs
    circle_idxs_np = get_circle_idxs(search_radius)
    circle_idxs_j = jnp.array(circle_idxs_np, dtype=jnp.int32)

    jnp.full((rows, cols), jnp.nan, dtype=jnp.float32)

    # We define a pixel-wise function that calculates the summary similarity
    @jax.jit
    def single_pixel_similarity(r0, c0):
        # If mask is False, no similarity
        cond = mask_j[r0, c0]

        def compute():
            x0 = ifg_unit[:, r0, c0]
            # Collect neighbor similarities
            # clip the neighbor coords
            nr = jnp.clip(r0 + circle_idxs_j[:, 0], 0, rows - 1)
            nc = jnp.clip(c0 + circle_idxs_j[:, 1], 0, cols - 1)

            # We skip the center itself
            # Also skip if the neighbor mask is off
            valid_neighbor = mask_j[nr, nc]
            # We'll store all similarities in an array, then mask out invalid.
            # shape (#neighbors,)
            sims = jnp.where(
                valid_neighbor,
                _phase_similarity_vmap(x0, ifg_unit[:, nr, nc]),
                jnp.nan,
            )
            return summary_func(sims)

        return jnp.where(cond, compute(), jnp.nan)

    # Vectorized over all r, c
    r_coords = jnp.arange(rows)
    c_coords = jnp.arange(cols)
    rr, cc = jnp.meshgrid(r_coords, c_coords, indexing="ij")
    coords = jnp.stack([rr, cc], axis=-1)

    @jax.jit
    def process_rc(rc):
        return single_pixel_similarity(rc[0], rc[1])

    # Now vmap over 2D
    return jax.vmap(jax.vmap(process_rc))(coords)


# We can define a vmapped version of phase_similarity for arrays
_phase_similarity_vmap = jax.vmap(
    lambda a, b: phase_similarity(a, b),  # a,b are both shape (n_ifg,)
    in_axes=(None, 1),  # we broadcast x0, and iterate over each neighbor's columns
    out_axes=0,
)
