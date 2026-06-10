"""
Residual vector and analytical Jacobian for Heston calibration.

Implements the nonlinear least-squares formulation of Cui et al. (2016):
    rᵢ(θ) = C_model(θ; Kᵢ, Tᵢ) − C*ᵢ
    f(θ)  = ½ ‖r(θ)‖²

The Jacobian Jᵢⱼ = ∂rᵢ/∂θⱼ is computed analytically for European options
via the GL-quadrature gradient (Eq. 22 of the paper).  American contracts
fall back to a finite-difference Jacobian automatically through scipy's
least_squares machinery (jac='2-point' path is never hit here; the fallback
is invoked by returning np.nan for those rows, triggering scipy's FD).
"""

import numpy as np

from pricing.european_gl import (
    heston_call_price_and_gradient,
    heston_put_price_and_gradient,
    heston_call_gl,
    heston_put_gl,
)
from pricing.american import (
    american_call_without_dividends,
    american_call_with_dividends,
    american_put_without_dividends,
    american_put_with_dividends,
)
from pricing.heston_pde_american import heston_pde_american
from calibration.implied_vol import implied_volatility

# Weight for the soft Feller-condition penalty appended to the residual vector.
# Scales the residual r_feller = FELLER_WEIGHT * max(0, σ²-2κθ).
#
# DEFAULT 0.0 (penalty OFF). The Feller condition 2κθ ≥ σ² keeps the
# continuous-time variance process from touching zero, but single-name equities
# (NVDA especially) routinely violate it — high vol-of-vol relative to mean
# reversion is the empirical norm. A nonzero weight pins the fit onto the
# boundary 2κθ = σ², which forces κ artificially high, σ toward its cap, and
# distorts ρ. A controlled recovery test (synthetic chain priced from known
# params with σ=1.5 > 2κθ) showed FELLER_WEIGHT=50 could NOT recover the truth
# (landed on Feller=0 with κ≈5.5), while FELLER_WEIGHT=0 recovered it to machine
# precision. Set to a small positive value (e.g. 0.1) only if you want a gentle
# tiebreaker toward feasibility without dominating the price residuals.
FELLER_WEIGHT = 0.0


# ------------------------------------------------------------------ #
# Low-level per-row helpers
# ------------------------------------------------------------------ #

def _model_price_and_grad(row, params, r, q, Ns, Nv, Nt, pricing_mode):
    """
    Compute (model_price, grad) for one contract row.

    grad has shape (5,) in [v0, kappa, theta, sigma, rho] order.
    For non-European paths, grad is None (caller handles FD fallback).
    """
    v0, kappa, theta, sigma, rho = params
    # Per-row rate and yield override global fallbacks when set by service layer.
    r = float(getattr(row, "r", r))
    q = float(getattr(row, "q", q))
    S0 = row.spot
    K = row.strike
    T = row.T
    opt_type = row.type
    exercise = row.ExerciseStyle

    if pricing_mode == "european_proxy" or exercise.lower() == "european":
        if opt_type == "call":
            price, grad = heston_call_price_and_gradient(
                S0, K, r, T, v0, kappa, theta, sigma, rho
            )
        else:
            price, grad = heston_put_price_and_gradient(
                S0, K, r, T, v0, kappa, theta, sigma, rho
            )
        return price, grad

    # American — no closed-form gradient; return price only
    if exercise.lower() == "american":
        # Call with no dividends: early exercise is never optimal → same as European
        if opt_type == "call" and abs(q) <= 1e-12:
            price = american_call_without_dividends(S0, K, r, T, v0, kappa, theta, sigma, rho)
        elif pricing_mode == "lsmc":
            # Use fewer paths during calibration to keep runtime tractable
            _M, _N = 50, 2000
            if opt_type == "call":
                price = american_call_with_dividends(
                    S0, K, r, T, v0, kappa, theta, sigma, rho, _M, _N, q
                )
            elif abs(q) <= 1e-12:
                price = american_put_without_dividends(
                    S0, K, r, T, v0, kappa, theta, sigma, rho, _M, _N
                )
            else:
                price = american_put_with_dividends(
                    S0, K, r, T, v0, kappa, theta, sigma, rho, _M, _N, q
                )
        else:
            # pde (default for American)
            price = heston_pde_american(
                S0, K, r, q, T, v0, kappa, theta, sigma, rho, opt_type, Ns, Nv, Nt
            )
        return price, None

    raise ValueError(f"Unknown exercise style: {exercise!r}")


# ------------------------------------------------------------------ #
# Public API consumed by calibrate_heston.py
# ------------------------------------------------------------------ #

def heston_residuals(params, r, q, options_df, Ns, Nv, Nt, pricing_mode):
    """
    Return the residual vector r(θ) ∈ ℝ^(n+1).
    rᵢ = model_priceᵢ − market_priceᵢ  for i = 1…n (price residuals)
    r_{n+1} = FELLER_WEIGHT * max(0, σ²−2κθ)  (soft Feller penalty)

    The Feller term is always present (zero when satisfied) so the vector
    size is constant across calls, as required by scipy's least_squares.
    Invalid parameters return a large constant vector of the same size.
    """
    v0, kappa, theta, sigma, rho = params
    n_valid = int((options_df["mid_price"] > 0).sum())
    if any(p <= 0 for p in [v0, kappa, theta, sigma]) or not (-1 < rho < 1):
        return np.full(n_valid + 1, 1e5)

    residuals = []
    for row in options_df.itertuples(index=False):
        market = row.mid_price
        if market <= 0:
            continue
        price, _ = _model_price_and_grad(row, params, r, q, Ns, Nv, Nt, pricing_mode)
        residuals.append(price - market)

    # Soft Feller penalty — zero when 2κθ ≥ σ², positive when violated
    feller_gap = max(0.0, sigma ** 2 - 2.0 * kappa * theta)
    residuals.append(FELLER_WEIGHT * feller_gap)

    return np.array(residuals) if residuals else np.array([1e5])


def heston_jacobian(params, r, q, options_df, Ns, Nv, Nt, pricing_mode):
    """
    Return the Jacobian matrix J ∈ ℝ^((n+1)×5).
    Rows 0…n−1 : ∂rᵢ/∂θⱼ = ∂C_model/∂θⱼ  (analytical for European contracts)
    Row  n      : gradient of the soft Feller penalty (analytical, always present)

    For European contracts the price gradient comes from Theorem 1 of Cui et al.
    For American contracts the row is np.nan so scipy falls back to finite differences.
    """
    v0, kappa, theta, sigma, rho = params
    rows = []
    for row in options_df.itertuples(index=False):
        if row.mid_price <= 0:
            continue
        _, grad = _model_price_and_grad(row, params, r, q, Ns, Nv, Nt, pricing_mode)
        rows.append(grad if grad is not None else np.full(5, np.nan))

    # Feller gradient: d/d[v0,κ,θ,σ,ρ] of FELLER_WEIGHT * max(0, σ²−2κθ)
    if sigma ** 2 > 2.0 * kappa * theta:
        feller_grad = FELLER_WEIGHT * np.array([0.0, -2.0 * theta, -2.0 * kappa, 2.0 * sigma, 0.0])
    else:
        feller_grad = np.zeros(5)
    rows.append(feller_grad)

    return np.array(rows)


# ------------------------------------------------------------------ #
# Legacy scalar-loss shim — no longer used by the main calibrator.
# Kept here for reference; uncomment if needed for standalone debugging.
# ------------------------------------------------------------------ #

# def heston_loss(params, r, q, options_df, Ns, Nv, Nt,
#                 objective="iv", pricing_mode="auto"):
#     """Scalar MSE loss — legacy, superseded by heston_residuals/heston_jacobian."""
#     v0, kappa, theta, sigma, rho = params
#     if any(p <= 0 for p in [v0, kappa, theta, sigma]) or not (-1 < rho < 1):
#         return 1e10
#
#     errors = []
#     for row in options_df.itertuples(index=False):
#         market = row.mid_price
#         if market <= 0:
#             continue
#         price, _ = _model_price_and_grad(row, params, r, q, Ns, Nv, Nt, pricing_mode)
#
#         if objective == "price":
#             scale = max(abs(market), 1.0)
#             errors.append(((price - market) / scale) ** 2)
#         else:
#             market_iv = getattr(row, "market_iv", np.nan)
#             if np.isnan(market_iv):
#                 market_iv = implied_volatility(
#                     market, row.spot, row.strike, r, row.T, row.type, q
#                 )
#             model_iv = implied_volatility(price, row.spot, row.strike, r, row.T, row.type, q)
#             if np.isnan(model_iv) or np.isnan(market_iv):
#                 continue
#             errors.append((model_iv - market_iv) ** 2)
#
#     return float(np.mean(errors)) if errors else 1e10
