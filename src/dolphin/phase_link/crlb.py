from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap
from jax.scipy.linalg import solve
from numpy.linalg import inv
from numpy.typing import ArrayLike


def compute_crlb(
    coherence_matrix: ArrayLike, num_looks: int, aps_variance: float = 0.01
) -> np.ndarray:
    r"""Compute the Cramer-Rao Lower Bound (CRLB) for phase linking estimation.

    Uses notation from [@Tebaldini2010MethodsPerformancesMultiPass], such that
    the Fisher information matrix, $X$, is computed as

    \begin{equation}
        X = \frac{2}{L} (\Gamma \circ \Gamma^{-1} - I)
    \end{equation}

    where $\Gamma$ is the complex coherence matrix, $L$ is the number of looks,
    and $I$ is the identity matrix.

    The CRLB is then computed as

    \begin{equation}
        \mathrm{CRLB} = \mathrm{inv}(\mathrm{\Theta}^T X \mathrm{\Theta})
    \end{equation}

    where $\mathrm{\Theta}$ is a matrix of partial derivatives, which, for direct
    phase estimation, is an identity matrix with on extra row of zeros.

    If the APS variance is non-zero, the CRLB is modified as

    \begin{equation}
        \mathrm{CRLB} = \mathrm{inv}(\mathrm{\Theta}^T (X + \mathrm{R}_\mathrm{APS}^{-1}) \mathrm{\Theta})
    \end{equation}

    where $\mathrm{R}_\mathrm{APS}^{-1}$ is the inverse of the APS covariance matrix,
    $\mathrm{R}_\mathrm{APS} = \alpha I

    See Equations (21) and (22) in [@Tebaldini2010MethodsPerformancesMultiPass].

    Parameters
    ----------
    coherence_matrix : ArrayLike
        Complex coherence matrix (N x N)
    num_looks : int
        Number of looks used in estimation
    aps_variance : float
        Variance of the atmospheric phase screen.
        If 0, no the portion of the fisher information matrix corresponding
        to the APS variance is skipped, and only phase decorrelation is considered.

    Returns
    -------
    np.ndarray
        Array (shape (N,)) of standard deviations (in radians) for the estimator
        variance lower bound at each date.

    """  # noqa: E501
    N = np.asarray(coherence_matrix).shape[0]

    # For direct phase estimation, Theta should be (N x (N-1))
    # This maps N-1 phase differences to N phases
    Theta = np.zeros((N, N - 1))
    # First row is 0 (using day 0 as reference)
    Theta[1:, :] = np.eye(N - 1)  # Last N-1 rows are identity

    # Compute X matrix as in equation (17)
    abs_coherence = np.abs(coherence_matrix)
    X = 2 * num_looks * (abs_coherence * inv(abs_coherence) - np.eye(N))

    if aps_variance == 0:
        # Compute CRLB portions in equation (21)
        fim = Theta.T @ X @ Theta  # Now should be (N-1 x N-1)
        inv_fim = inv(fim)
    else:
        # Add APS contribution
        R_aps_inv = np.eye(N) / aps_variance
        # Otherwise, use full hybrid version, equation (22)
        A = Theta.T @ X @ inv(X + R_aps_inv) @ X @ Theta
        fim = Theta.T @ X @ Theta - A
        inv_fim = inv(fim)

    return inv_fim


def compute_lower_bound_std(
    coherence_matrix: ArrayLike, num_looks: int, aps_variance: float = 0.01
) -> np.ndarray:
    """Compute the Cramer Rao lower bound on the phase linking estimator variance.

    Returns the result as a standard standard deviation (in radians) per epoch.

    Parameters
    ----------
    coherence_matrix : ArrayLike
        Complex coherence matrix (N x N)
    num_looks : int
        Number of looks used in estimation
    aps_variance : float
        Variance of the APS, in radians squared.
        If 0, The bound only considers the variance due to phase decorrelation, not
        atmospheric noise.
        Default is 0.01.

    Returns
    -------
    lower_bound_std : np.ndarray
        Lower bound on the standard deviation of the phase linking estimator.

    """
    crlb = compute_crlb(
        coherence_matrix=coherence_matrix,
        num_looks=num_looks,
        aps_variance=aps_variance,
    )

    estimator_stddev = np.sqrt(np.diag(crlb))
    return np.concatenate(([0], estimator_stddev))


def _theta_indices(n: int, ref: int) -> Array:
    return jnp.concatenate([jnp.arange(ref), jnp.arange(ref + 1, n)])


def _build_fisher_from_abs_gamma(
    abs_G: Array, abs_G_inv: Array, num_looks: float
) -> Array:
    """Create the Fisher Information Matrix from |coherence matrix|.

    Useful when |coherence matrix| is already computed and inverted.

    Parameters
    ----------
    abs_G : Array
        Absolute value of the coherence matrix
    abs_G_inv : Array
        Inverse of the absolute value of the coherence matrix
    num_looks : float
        Number of looks used in the coherence matrix

    Returns
    -------
    Array
        Fisher Information Matrix

    """
    eyeN = jnp.eye(abs_G.shape[-1], dtype=abs_G.dtype)
    eyeN = jnp.broadcast_to(eyeN, abs_G.shape)
    return 2.0 * num_looks * (abs_G * abs_G_inv - eyeN)


def _crlb_from_x(
    X: Array, reference_idx: int, aps_variance: float, fim_jitter: float
) -> Array:
    *batch, N, _ = X.shape
    idx = _theta_indices(N, reference_idx)

    eyeN = jnp.eye(N, dtype=X.dtype)
    eyeN1 = jnp.eye(N - 1, dtype=X.dtype)
    eyeN = jnp.broadcast_to(eyeN, X.shape)
    eyeN1 = jnp.broadcast_to(eyeN1, (*batch, N - 1, N - 1))

    # Θᵀ X Θ  by indexing
    F_base = X[..., idx[:, None], idx]

    if aps_variance > 0.0:
        R_inv = eyeN / aps_variance
        X_plus_R = X + R_inv + 0.0 * eyeN  # no implicit extra jitter here
        # (X + R⁻¹)⁻¹ (X Θ) via solve, where Θ selects columns 'idx'
        X_cols = X[..., :, idx]  # (..., N, N-1)
        AXTheta = solve(X_plus_R, X_cols, assume_a="pos")  # (..., N, N-1)
        A = (X @ AXTheta)[..., idx, :]  # (..., N-1, N-1)
        FIM = F_base - A
    else:
        FIM = F_base

    # Σ = inverse of FIM
    if fim_jitter != 0.0:
        FIM = FIM + fim_jitter * eyeN1
    Sigma = solve(FIM, eyeN1, assume_a="pos")
    sig = jnp.sqrt(jnp.diagonal(Sigma, axis1=-2, axis2=-1))
    return jnp.insert(sig, reference_idx, 0.0, axis=-1)


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def compute_crlb_jax(
    coherence_matrices: Array,
    num_looks: int,
    reference_idx: int,
    aps_variance: float = 0.0,
    gamma_jitter: float = 0.0,
    fim_jitter: float = 1e-6,
    mask_zero_blocks: bool = True,
    zero_tol: float = 1e-7,
) -> Array:
    """Compute CRLB for a batch of coherence matrices.

    Parameters
    ----------
    coherence_matrices : Array
        Coherence matrices, shape (..., N, N)
    num_looks : int
        Number of independent looks, `L`.
    reference_idx : int
        Reference epoch index (time index of 0 output)
    aps_variance : float
        Atmospheric phase screen variance.
        If 0, APS term is skipped.
    gamma_jitter : float
        Jitter added to regularize the inversion of |Γ|.
    fim_jitter : float
        Jitter added to regularize the inversion of the Fisher Information Matrix.
    mask_zero_blocks : bool
        Set output to nan where |Γ| is (near) zero.
        Default is True.
    zero_tol : float
        Tolerance for zero-blocks

    """
    *_batch, N, _ = coherence_matrices.shape
    eyeN = jnp.eye(N, dtype=coherence_matrices.dtype)
    eyeNb = jnp.broadcast_to(eyeN, coherence_matrices.shape)

    abs_G = jnp.abs(coherence_matrices)

    # Detect obviously singular blocks (your toy Γ=0 case)
    block_max = jnp.max(abs_G, axis=(-2, -1), keepdims=True)
    is_zero_block = block_max < zero_tol  # (..., 1, 1)

    # Keep the solve from crashing: replace zero-blocks by I *for the solve only*
    abs_G_safe = abs_G + gamma_jitter * eyeNb
    abs_G_safe = jnp.where(is_zero_block, eyeNb, abs_G_safe)

    abs_G_inv = solve(abs_G_safe, eyeNb, assume_a="pos")

    # Build X once and do the inverse-free CRLB from X
    X = _build_fisher_from_abs_gamma(abs_G, abs_G_inv, num_looks)
    sig = _crlb_from_x(X, reference_idx, aps_variance, fim_jitter)

    if mask_zero_blocks:
        # overwrite sigma on zero blocks to NaN to mimic NumPy error/NaN
        mask = jnp.squeeze(is_zero_block, axis=(-2, -1))
        nanv = jnp.full(sig.shape, jnp.nan, dtype=sig.dtype)
        sig = jnp.where(mask, nanv, sig)
    return sig


def penalization_weight(gamma_abs: Array, eps: float = 1e-10) -> Array:
    r"""Penalization weights from Zwieback & Meyer 2022 Improvement 2 (eq. 13).

    Returns a per-element weight matrix ``W`` such that the penalized
    sufficient statistic is ``C_tilde = C * W`` (Hadamard product). At high
    coherence ``W ≈ 1`` (no penalization); at low coherence ``W → 0``,
    suppressing the bias-prone entries.

    The published coefficients were fit at ``L = 100`` (R = 1200 replicates,
    P = 2). For ``L`` substantially larger than 100 the bias is smaller and
    these weights are slightly over-conservative; for ``L`` substantially
    smaller they may under-penalize.

    Parameters
    ----------
    gamma_abs : Array
        Coherence magnitudes ``|gamma_ij|`` in [0, 1], shape ``(..., N, N)``.
        Diagonal entries are forced to 1 (no autocorrelation penalty).
    eps : float
        Floor to avoid ``log(0)`` for fully decorrelated entries.

    Returns
    -------
    Array
        Weights in (0, 1], same shape as ``gamma_abs``, with the diagonal
        forced to 1.

    Notes
    -----
    The paper writes the fit as ``log(1 - W) = ...``; evaluating that form
    yields ``W → 0`` at high coherence, which contradicts the physical
    requirement that ``W = 1`` means "no penalization" (Section III-A,
    "If W_ij = 1 for all i, j, there is no penalization"). The form
    ``log W = -7 s1 - 0.6 s2`` reproduces Fig. 3(b): full penalization
    only for ``gamma <~ 0.03`` and weights ``≈ 1`` for ``gamma >~ 0.3``.
    A sign-convention sanity-check is included in the unit tests.

    References
    ----------
    Zwieback, S. and Meyer, F. J. (2022). Reliable InSAR Phase History
    Uncertainty Estimates. *IEEE Trans. Geosci. Remote Sens.* 60, 5222109,
    Eq. (13).

    """
    log_gamma = jnp.log(jnp.maximum(gamma_abs, eps))
    s1 = 1.0 / (1.0 + jnp.exp(15.02 * (log_gamma + 2.6)))
    s2 = 1.0 / (1.0 + jnp.exp(3.2 * (log_gamma + 1.8)))
    W = jnp.exp(-7.0 * s1 - 0.6 * s2)

    N = gamma_abs.shape[-1]
    diag_idx = jnp.arange(N)
    return W.at[..., diag_idx, diag_idx].set(1.0)


def _make_observed_fi_neg_loglik(N: int, reference_idx: int):
    r"""Build the Gaussian DS negative log-likelihood used for observed-FI Hessians.

    Returns a closure ``neg_loglik(beta, C, num_looks)`` returning a real scalar.
    The flat parameter vector ``beta`` packs:

      ``beta[:N-1]``  : phase parameters at the ``N-1`` non-reference epochs
                        (the reference phase is fixed to zero internally).
      ``beta[N-1:]``  : the ``N(N-1)/2`` upper-triangular off-diagonal entries
                        of the symmetric, real, unit-diagonal magnitude matrix G.

    The log-likelihood (dropping constants) is

        \\ell(\\beta) = -L \\log\\det \\Sigma - L \\, \\mathrm{tr}(\\Sigma^{-1} C),

    with ``Sigma = G \\circ exp(i theta) exp(i theta)^H`` (see Zwieback & Meyer
    2022 Eqs. (5)-(7)). We return ``-\\ell``.
    """
    n_theta = N - 1
    iu_row, iu_col = jnp.triu_indices(N, k=1)
    nonref = jnp.concatenate(
        [jnp.arange(reference_idx), jnp.arange(reference_idx + 1, N)]
    )

    def neg_loglik(beta: Array, C: Array, num_looks: float) -> Array:
        # Reconstruct full theta (length N) with theta[reference_idx] = 0
        theta_full = jnp.zeros(N, dtype=beta.dtype).at[nonref].set(beta[:n_theta])

        # Reconstruct symmetric, unit-diagonal G from upper-triangular off-diags
        G_off = (
            jnp.zeros((N, N), dtype=beta.dtype).at[iu_row, iu_col].set(beta[n_theta:])
        )
        G = G_off + G_off.T + jnp.eye(N, dtype=beta.dtype)

        # Sigma = G ∘ exp(iθ) exp(iθ)^H
        phasor = jnp.exp(1j * theta_full)
        Sigma = G.astype(C.dtype) * jnp.outer(phasor, jnp.conj(phasor))

        # -L logdet Σ - L tr(Σ⁻¹ C), then return the negative
        _, logdet = jnp.linalg.slogdet(Sigma)
        Sinv_C = jnp.linalg.solve(Sigma, C)
        return num_looks * (logdet + jnp.trace(Sinv_C)).real

    return neg_loglik


@partial(jit, static_argnums=(2, 3, 4))
def compute_observed_fi_crlb(
    coherence_matrices: Array,
    phase_estimates: Array,
    num_looks: float,
    reference_idx: int,
    fim_jitter: float = 1e-6,
) -> Array:
    r"""Observed-FI based phase uncertainty (Zwieback & Meyer 2022, Improvement 1).

    Computes the observed Fisher information matrix at the supplied estimates
    of phase and magnitudes, then forms the Schur-complement marginal
    covariance for the phase block:

    \\begin{equation}
        \\mathbf{K}^f_{\\theta\\theta} \\;=\\;
            \\big(\\mathbf{F}_{\\theta\\theta}
            - \\mathbf{F}_{\\theta G}\\,\\mathbf{F}_{GG}^{+}\\,
              \\mathbf{F}_{\\theta G}^{T}\\big)^{+},
    \\end{equation}

    accounting for the finite-sample uncertainty in the magnitude estimates G.

    For sufficiently large ``num_looks`` this converges to the expected-FI
    bound from :func:`compute_crlb_jax`; at small ``num_looks`` (or at low
    coherence) the partial expected-FI bound underestimates the actual error,
    and the observed-FI Schur correction is the leading-order finite-sample
    fix (Zwieback & Meyer 2022, Fig. 5).

    Parameters
    ----------
    coherence_matrices : Array
        Sample coherence matrices, shape ``(..., N, N)``. Hermitian, unit
        diagonal. The off-diagonal phases of these matrices are *not* used for
        the model phases (those are supplied in ``phase_estimates``); but the
        full complex matrix enters the likelihood as the sufficient statistic C.
    phase_estimates : Array
        Phase estimates θ̂, shape ``(..., N)``, in radians, e.g. from EMI. The
        entry at ``reference_idx`` is overridden to zero internally.
    num_looks : float
        Number of independent looks ``L``.
    reference_idx : int
        Reference epoch index. The corresponding phase is fixed at zero.
    fim_jitter : float
        Jitter added to the Schur-complemented FI before pseudoinversion to
        regularize against rank deficiency.

    Returns
    -------
    Array
        Per-epoch standard deviation in radians, shape ``(..., N)``, with
        ``sigma = 0`` at ``reference_idx``.

    Notes
    -----
    Hessians are computed via JAX automatic differentiation through the full
    Gaussian DS log-likelihood. Cost is dominated by the
    ``M`` by ``M`` pseudoinverse on the magnitude block, where
    ``M = N(N-1)/2``; for typical InSAR ``N`` (<= 60) this is tractable.

    Improvements 2 (penalized likelihood) and 3 (constrained G via the BIR1
    parametric model) of Zwieback & Meyer 2022 are not implemented here;
    they require modifying the point estimator and fitting a parametric
    structure to G respectively.

    References
    ----------
    Zwieback, S. and Meyer, F. J. (2022). Reliable InSAR Phase History
    Uncertainty Estimates. *IEEE Trans. Geosci. Remote Sens.* 60, 5222109.

    """
    *batch, N, _ = coherence_matrices.shape
    assert (
        phase_estimates.shape[-1] == N
    ), f"phase_estimates trailing dim {phase_estimates.shape[-1]} != N={N}"
    n_theta = N - 1

    neg_loglik = _make_observed_fi_neg_loglik(N, reference_idx)
    hess_fn = jax.hessian(neg_loglik, argnums=0)

    iu_row, iu_col = jnp.triu_indices(N, k=1)
    nonref = jnp.concatenate(
        [jnp.arange(reference_idx), jnp.arange(reference_idx + 1, N)]
    )

    def _single(C: Array, theta_est: Array) -> Array:
        # Magnitude estimate from the sample coherence (Zwieback Improvement 1
        # is evaluated at the joint MLE; we use the EMI/EVD phase estimate
        # together with |C| as the magnitude estimate, which matches the EMI
        # working point and is the cheapest sensible choice).
        G_est = jnp.abs(C)
        beta_init = jnp.concatenate([theta_est[nonref], G_est[iu_row, iu_col]])

        H = hess_fn(beta_init, C, num_looks)

        F_tt = H[:n_theta, :n_theta]
        F_tg = H[:n_theta, n_theta:]
        F_gg = H[n_theta:, n_theta:]

        # Schur complement of the θ block of the full observed FI
        F_corr = F_tt - F_tg @ jnp.linalg.pinv(F_gg) @ F_tg.T
        F_corr = F_corr + fim_jitter * jnp.eye(n_theta, dtype=F_corr.dtype)
        K_tt = jnp.linalg.pinv(F_corr)

        sigma_free = jnp.sqrt(jnp.maximum(jnp.diag(K_tt), 0.0))
        return jnp.insert(sigma_free, reference_idx, 0.0)

    # Vmap across leading batch dims by flattening then reshaping back
    flat_C = coherence_matrices.reshape((-1, N, N))
    flat_theta = phase_estimates.reshape((-1, N))
    sigmas_flat = vmap(_single)(flat_C, flat_theta)
    return sigmas_flat.reshape(*batch, N)


def _examples(N=10, gamma0=0.6, rho=0.8):
    """Make example covariance matrices used in Tebaldini, 2010."""
    idxs = np.abs(np.arange(N).reshape(-1, 1) - np.arange(N).reshape(1, -1))
    # {Γ}nm = ρ^|n−m|; ρ = 0.8  # noqa: RUF003
    C_ar1 = rho**idxs
    # {Γ}nm = γ0 + (1 - γ0) δ{n-m};  # noqa: RUF003
    C_const_gamma = (1 - gamma0) * np.eye(N) + gamma0 * np.ones((N, N))
    return C_ar1, C_const_gamma


def demo_from_slc_stack(  # noqa: D103
    slc_vrt_filename: str = "slc_stack.vrt",
    hw: tuple[int, int] = (5, 5),
    center_pixel: tuple[int, int] = (50, 50),
    aps_variance: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    from dolphin import io
    from dolphin.phase_link import covariance

    hwr, hwc = hw
    reader = io.VRTStack.from_vrt_file(slc_vrt_filename)
    r0, c0 = center_pixel
    samples = reader[:, r0 - hwr : r0 + hwr, c0 - hwc : c0 + hwc].reshape(
        len(reader), -1
    )
    C = covariance.coh_mat_single(samples)
    # num_looks = (2 * hwr + 1) * (2 * hwc + 1)
    num_looks = np.sqrt(hwr * hwc)
    return C, compute_lower_bound_std(C, num_looks, aps_variance=aps_variance)
