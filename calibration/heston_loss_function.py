"""
Residual vector and analytical Jacobian for Heston calibration.

Implements the nonlinear least-squares formulation of Cui et al. (2016):
    rᵢ(θ) = C_model(θ; Kᵢ, Tᵢ) − C*ᵢ
    f(θ)  = ½ ‖r(θ)‖²

The Jacobian Jᵢⱼ = ∂rᵢ/∂θⱼ is computed analytically via the GL-quadrature
gradient (Eq. 22 of the paper).  This is a European-equivalent engine: quotes
are de-Americanized upstream and the calibration universe is forced to European,
so every contract is priced with the closed-form European pricer and always has
an analytic gradient.  (The American PDE/LSMC pricers were removed; see the
gitignored _graveyard.py.)
"""

import numpy as np
from scipy.stats import norm

from pricing.european_gl import (
    heston_call_price_and_gradient,
    heston_put_price_and_gradient,
)

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
# Residual weighting (price-space residuals scaled to equalise info)
# ------------------------------------------------------------------ #

def bs_vega(S, K, r, T, sigma, q=0.0):
    """Black-Scholes vega with carry q. Used to vega-weight price residuals."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def compute_residual_weights(
    options_df, weight_scheme, r, q, *, vega_floor=0.05, spread_floor=0.01
):
    """
    Per-contract residual weights, aligned to the valid rows (mid_price > 0) in
    itertuples order — the same rows heston_residuals/heston_jacobian iterate.

    weight_scheme:
      "none"       -> equal weights (plain price residuals)
      "vega"       -> 1/vega at market IV; approximates an IV-space objective so
                      near-intrinsic high-price contracts no longer dominate
      "inv_spread" -> 1/rel_spread; trust tight markets more

    Weights are normalised to mean 1 (keeps the loss scale comparable) and are
    constant in the parameters, so the analytic Jacobian stays valid (each row is
    simply scaled by its weight).
    """
    valid = options_df[options_df["mid_price"] > 0]
    n = len(valid)
    if n == 0:
        return np.ones(0)
    if weight_scheme in (None, "none"):
        return np.ones(n)

    if weight_scheme == "vega":
        w = np.empty(n)
        for i, row in enumerate(valid.itertuples(index=False)):
            rr = float(getattr(row, "r", r))
            qq = float(getattr(row, "q", q))
            iv = float(getattr(row, "market_iv", np.nan))
            if not np.isfinite(iv) or iv <= 0:
                iv = 0.3  # fallback when the market IV could not be inverted
            v = bs_vega(row.spot, row.strike, rr, row.T, iv, qq)
            w[i] = 1.0 / max(v, vega_floor)
    elif weight_scheme == "inv_spread":
        rs = valid.get("rel_spread")
        if rs is None:
            return np.ones(n)
        rs = rs.fillna(spread_floor).clip(lower=spread_floor).to_numpy()
        w = 1.0 / rs
    else:
        raise ValueError(f"unknown weight_scheme {weight_scheme!r}")

    mean = w.mean()
    return w / mean if mean > 0 else np.ones(n)


# ------------------------------------------------------------------ #
# Low-level per-row helpers
# ------------------------------------------------------------------ #

def _model_price_and_grad(row, params, r, q, Ns, Nv, Nt, pricing_mode):
    """
    Compute (model_price, grad) for one contract row via the European CF pricer.

    grad has shape (5,) in [v0, kappa, theta, sigma, rho] order and is always
    present (analytic) — the engine is European-equivalent, so every contract is
    priced as European.  Ns/Nv/Nt and pricing_mode are accepted for signature
    compatibility with the residual/Jacobian callers but are no longer used.
    """
    v0, kappa, theta, sigma, rho = params
    # Per-row rate and yield override global fallbacks when set by service layer.
    r = float(getattr(row, "r", r))
    q = float(getattr(row, "q", q))
    S0 = row.spot
    K = row.strike
    T = row.T

    if row.type == "call":
        return heston_call_price_and_gradient(S0, K, r, T, v0, kappa, theta, sigma, rho, q)
    return heston_put_price_and_gradient(S0, K, r, T, v0, kappa, theta, sigma, rho, q)


# ------------------------------------------------------------------ #
# Reporting metric: IV-space fit quality (does NOT affect calibration)
# ------------------------------------------------------------------ #

def iv_error_metrics(params, r, q, options_df, *, vega_floor=0.05):
    """
    Vega-linearised implied-vol fit error at ``params``, in IV (fraction) units.

    The raw calibration loss is a vega-weighted *price* sum-of-squares whose
    absolute value scales with the underlying's price level and contract count
    (so it reads ~1e5-1e6 for an index like ^SPX even for an excellent fit, and
    is not comparable across tickers). This converts each price residual to an
    approximate IV error via the first-order relationship the vega weighting
    already exploits,

        ΔIV_i ≈ (P_model,i − P_market,i) / vega_i        (BS vega at the market IV)

    and returns (iv_rmse, iv_mae) as fractions — multiply by 100 for vol points.
    These are dimensionless and directly interpretable ("we fit the surface to
    X vol points"), and comparable across underlyings.

    Reporting only: it re-prices at the fitted params and is independent of the
    optimiser's ``weight_scheme``, so it always reflects true IV-space quality.
    Vega is floored (as in the weights) to avoid blow-up on near-zero-vega wings.
    """
    errs = []
    for row in options_df.itertuples(index=False):
        market = row.mid_price
        if market <= 0:
            continue
        rr = float(getattr(row, "r", r))
        qq = float(getattr(row, "q", q))
        price, _ = _model_price_and_grad(row, params, rr, qq, 0, 0, 0, "european_proxy")
        iv = float(getattr(row, "market_iv", np.nan))
        if not np.isfinite(iv) or iv <= 0:
            iv = 0.3  # same fallback the vega weights use when market IV is missing
        v = bs_vega(row.spot, row.strike, rr, row.T, iv, qq)
        errs.append((price - market) / max(v, vega_floor))
    if not errs:
        return float("nan"), float("nan")
    e = np.asarray(errs)
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))


# ------------------------------------------------------------------ #
# Public API consumed by calibrate_heston.py
# ------------------------------------------------------------------ #

def heston_residuals(params, r, q, options_df, Ns, Nv, Nt, pricing_mode, weights=None):
    """
    Return the residual vector r(θ) ∈ ℝ^(n+1).
    rᵢ = wᵢ · (model_priceᵢ − market_priceᵢ)  for i = 1…n (weighted price residuals)
    r_{n+1} = FELLER_WEIGHT * max(0, σ²−2κθ)  (soft Feller penalty, unweighted)

    `weights` (length n, aligned to the valid rows in itertuples order; see
    compute_residual_weights) scales each price residual — e.g. 1/vega to make the
    fit behave like an IV-space objective. None ⇒ equal weights (plain prices).

    The Feller term is always present (zero when satisfied) so the vector
    size is constant across calls, as required by scipy's least_squares.
    Invalid parameters return a large constant vector of the same size.
    """
    v0, kappa, theta, sigma, rho = params
    n_valid = int((options_df["mid_price"] > 0).sum())
    if any(p <= 0 for p in [v0, kappa, theta, sigma]) or not (-1 < rho < 1):
        return np.full(n_valid + 1, 1e5)

    residuals = []
    wi = 0
    for row in options_df.itertuples(index=False):
        market = row.mid_price
        if market <= 0:
            continue
        price, _ = _model_price_and_grad(row, params, r, q, Ns, Nv, Nt, pricing_mode)
        w = 1.0 if weights is None else float(weights[wi])
        residuals.append(w * (price - market))
        wi += 1

    # Soft Feller penalty — zero when 2κθ ≥ σ², positive when violated
    feller_gap = max(0.0, sigma ** 2 - 2.0 * kappa * theta)
    residuals.append(FELLER_WEIGHT * feller_gap)

    return np.array(residuals) if residuals else np.array([1e5])


def heston_jacobian(params, r, q, options_df, Ns, Nv, Nt, pricing_mode, weights=None):
    """
    Return the Jacobian matrix J ∈ ℝ^((n+1)×5).
    Rows 0…n−1 : ∂rᵢ/∂θⱼ = wᵢ · ∂C_model/∂θⱼ  (analytical for European contracts)
    Row  n      : gradient of the soft Feller penalty (analytical, always present)

    `weights` scales each row by the same constant as its residual (residual
    weighting is parameter-independent, so the analytic gradient stays exact).
    For European contracts the price gradient comes from Theorem 1 of Cui et al.
    For American contracts the row is np.nan so scipy falls back to finite differences.
    """
    v0, kappa, theta, sigma, rho = params
    rows = []
    wi = 0
    for row in options_df.itertuples(index=False):
        if row.mid_price <= 0:
            continue
        _, grad = _model_price_and_grad(row, params, r, q, Ns, Nv, Nt, pricing_mode)
        w = 1.0 if weights is None else float(weights[wi])
        rows.append(w * grad if grad is not None else np.full(5, np.nan))
        wi += 1

    # Feller gradient: d/d[v0,κ,θ,σ,ρ] of FELLER_WEIGHT * max(0, σ²−2κθ)
    if sigma ** 2 > 2.0 * kappa * theta:
        feller_grad = FELLER_WEIGHT * np.array([0.0, -2.0 * theta, -2.0 * kappa, 2.0 * sigma, 0.0])
    else:
        feller_grad = np.zeros(5)
    rows.append(feller_grad)

    return np.array(rows)
