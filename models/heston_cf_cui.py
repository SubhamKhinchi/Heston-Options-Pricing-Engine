"""
Cui et al. (2016) Heston characteristic function - Eq. 18.

This representation is both numerically continuous (no branch-switching
discontinuities for long maturities) and analytically differentiable,
making it suitable for gradient-based calibration.

Reference: Cui, Y., del Baño Rollin, S., & Germano, G. (2016).
"Full and fast calibration of the Heston stochastic volatility model."
arXiv:1511.08718v2.
"""

import numpy as np


def _intermediates(u, kappa, theta, sigma, rho, T):
    """
    Compute shared intermediate quantities for the CF and its gradient, in a
    form that never overflows at long maturities / high quadrature nodes.

    The raw A1, A2 of Eqs. 15b/15c carry a factor cosh(dT/2) that blows up once
    Re(d)·T/2 approaches the double-precision ceiling (~709) — this is the SPX
    long-dated LEAPS failure: (u²+iu)·sinh(dT/2) → inf → A → inf → φ → NaN.
    But A1 and A2 only ever enter downstream as the ratio A = A1/A2 and as the
    ratios ∂A/A2, ∂D/A2, so the common cosh(dT/2) factor cancels identically.
    We therefore normalise it out up front: dividing A1, A2 by cosh(dT/2) turns
    sinh(dT/2)/cosh(dT/2) into tanh(dT/2), which is bounded (|tanh| ≤ 1 because
    the principal square root gives Re(d) ≥ 0 ⇒ Re(dT/2) ≥ 0). Nothing overflows,
    and every downstream quantity (A, D, and their ρ/σ derivatives) is unchanged.

    Returns
    -------
    xi, d, A1, A2, A, D, t
    where A1 = (u²+iu)·tanh(dT/2)   (= Eq. 15b A1 divided by cosh(dT/2))
          A2 = d + ξ·tanh(dT/2)      (= Eq. 15c A2 divided by cosh(dT/2))
          A  = A1 / A2               (Eq. 15a — the cosh factor cancels)
          t  = tanh(dT/2)            (the log-safe replacement for sinh/cosh)
          D  = log B (the stable log-B from Eq. 17b).
    """
    xi = kappa - sigma * rho * 1j * u                          # Eq. 11a
    d = np.sqrt(xi ** 2 + sigma ** 2 * (u ** 2 + 1j * u))     # Eq. 11b

    # tanh(dT/2) is bounded since Re(d) ≥ 0 ⇒ Re(dT/2) ≥ 0; this is the whole fix.
    t = np.tanh(d * T / 2)

    # cosh(dT/2)-normalised A1, A2 — the factor cancels in every downstream ratio.
    A1 = (u ** 2 + 1j * u) * t                                # Eq. 15b / cosh(dT/2)
    A2 = d + xi * t                                            # Eq. 15c / cosh(dT/2)
    A = A1 / A2                                                 # Eq. 15a

    # Stable log B = D, Eq. 17b — avoids discontinuities in log A2 for large T.
    # log B = log d + (κ-d)T/2 - log((d+ξ)/2 + (d-ξ)/2·e^{-dT})
    D = (np.log(d)
         + (kappa - d) * T / 2
         - np.log((d + xi) / 2 + (d - xi) / 2 * np.exp(-d * T)))  # Eq. 17b

    return xi, d, A1, A2, A, D, t


def heston_cf_cui(u, v0, kappa, theta, sigma, rho, S0, r, T, q=0.0):
    """
    Cui et al. CF, Eq. 18, with continuous carry q (the implied-forward yield).

    The risk-neutral log-drift is (r − q), i.e. the forward is S₀·e^{(r−q)T}, so
    the CF is consistent with the de-Americanized market quotes and the quad pricer.

    φ(θ; u, t) = exp{ iu(log S₀ + (r−q)t) − tκθρiu/σ − v₀A + (2κθ/σ²)D }
    """
    _, _, _, _, A, D, _ = _intermediates(u, kappa, theta, sigma, rho, T)

    exponent = (1j * u * (np.log(S0) + (r - q) * T)
                - T * kappa * theta * rho * 1j * u / sigma
                - v0 * A
                + 2.0 * kappa * theta / sigma ** 2 * D)

    return np.exp(exponent)


def heston_cf_and_gradient(u, v0, kappa, theta, sigma, rho, S0, r, T, q=0.0):
    """
    Compute φ(θ; u, T) and the gradient multiplier vector h(u) such that

        ∂φ/∂θⱼ = φ · hⱼ(u)

    as given by Theorem 1 / Eqs. 23-30 of Cui et al. (2016).

    Returns
    -------
    phi : complex array, shape (N,) for u of shape (N,)
    h   : complex array, shape (5, N)
          Row order matches project convention: [v0, kappa, theta, sigma, rho].
    """
    xi, d, A1, A2, A, D, t = _intermediates(
        u, kappa, theta, sigma, rho, T
    )

    # Carry q enters only the (r − q) log-drift; it is θ-independent, so the
    # gradient multipliers h below are unchanged.
    exponent = (1j * u * (np.log(S0) + (r - q) * T)
                - T * kappa * theta * rho * 1j * u / sigma
                - v0 * A
                + 2.0 * kappa * theta / sigma ** 2 * D)
    phi = np.exp(exponent)

    # ------------------------------------------------------------------ #
    # Derivatives w.r.t. ρ  (Eqs. 27)
    # ------------------------------------------------------------------ #
    dd_drho = -xi * sigma * 1j * u / d                                   # (27a)

    # cosh(dT/2)-normalised (÷cosh throughout): sinh→t=tanh, cosh→1. The factor
    # cancels against A2 in dA_drho/dD_drho below, exactly as in the un-normalised
    # form, so ∂A/∂ρ and ∂D/∂ρ are unchanged — but now nothing overflows.
    dA2_drho = (-sigma * 1j * u * (2.0 + T * xi) / (2.0 * d)
                * (xi + d * t))                                          # (27b) / cosh

    dA1_drho = (-1j * u * (u ** 2 + 1j * u) * T * xi * sigma
                / (2.0 * d))                                            # (27d) / cosh

    dA_drho = (dA1_drho - A * dA2_drho) / A2                             # (27e)

    # D = log B, so dD/dρ = dd/dρ / d − dA2/dρ / A2
    dD_drho = dd_drho / d - dA2_drho / A2

    # ------------------------------------------------------------------ #
    # Derivatives w.r.t. σ  (Eqs. 30)
    # ------------------------------------------------------------------ #
    dd_dsigma = (rho / sigma - 1.0 / xi) * dd_drho + sigma * u ** 2 / d  # (30a)

    dA1_dsigma = (u ** 2 + 1j * u) * T / 2.0 * dd_dsigma               # (30b) / cosh

    dA2_dsigma = (rho / sigma * dA2_drho
                  - (2.0 + T * xi) / (1j * u * T * xi) * dA1_drho
                  + sigma * T * A1 / 2.0)                                  # (30c)

    dA_dsigma = (dA1_dsigma - A * dA2_dsigma) / A2                       # (30d)

    dD_dsigma = dd_dsigma / d - dA2_dsigma / A2

    # ------------------------------------------------------------------ #
    # Gradient components h₁ … h₅  (Eqs. 23)
    # Paper parameter order: [v0, v̄=theta, ρ=rho, κ=kappa, σ=sigma]
    # Project parameter order: [v0, kappa, theta, sigma, rho]
    # ------------------------------------------------------------------ #

    # h₁ = −A                                             ∂/∂v₀  Eq. 23a
    h_v0 = -A

    # h₂ = (2κ/σ²)D − Tκρiu/σ                           ∂/∂θ   Eq. 23b
    h_theta = (2.0 * kappa / sigma ** 2 * D
               - T * kappa * rho * 1j * u / sigma)

    # h₃ = −v₀(∂A/∂ρ) + (2κθ/σ²)(∂D/∂ρ) − Tκθiu/σ     ∂/∂ρ   Eq. 23c
    h_rho = (-v0 * dA_drho
             + 2.0 * kappa * theta / sigma ** 2 * dD_drho
             - T * kappa * theta * 1j * u / sigma)

    # h₄ = v₀/(σiu)(∂A/∂ρ) + (2θ/σ²)D
    #       + (2κθ/σ²)(i/(σu)·∂D/∂ρ + T/2) − Tθρiu/σ   ∂/∂κ   Eq. 23d
    # [Uses dD/dκ = i/(σu)·dD/dρ + T/2,  since ∂ξ/∂κ=1, ∂ξ/∂ρ=−σiu]
    h_kappa = (v0 / (sigma * 1j * u) * dA_drho
               + 2.0 * theta / sigma ** 2 * D
               + 2.0 * kappa * theta / sigma ** 2
               * (1j / (sigma * u) * dD_drho + T / 2.0)
               - T * theta * rho * 1j * u / sigma)

    # h₅ = −v₀(∂A/∂σ) − (4κθ/σ³)D + (2κθ/σ²)(∂D/∂σ)
    #       + Tκθρiu/σ²                                   ∂/∂σ   Eq. 23e
    h_sigma = (-v0 * dA_dsigma
               - 4.0 * kappa * theta / sigma ** 3 * D
               + 2.0 * kappa * theta / sigma ** 2 * dD_dsigma
               + T * kappa * theta * rho * 1j * u / sigma ** 2)

    # Stack in project order: [v0, kappa, theta, sigma, rho]
    h = np.array([h_v0, h_kappa, h_theta, h_sigma, h_rho])  # shape (5, N)

    return phi, h
