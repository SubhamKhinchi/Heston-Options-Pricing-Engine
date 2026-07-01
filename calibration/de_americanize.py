"""
De-Americanization of American option quotes for Heston calibration.

Heston's fast characteristic-function pricer only produces *European* prices,
so calibrating to raw American market quotes would force an American pricer
(PDE / LSMC) inside the optimizer loop — slow and, in this project, numerically
fragile.  The industry-standard alternative is to *de-Americanize* the quotes
first and calibrate to the European-equivalent surface.

Procedure (per quote), see e.g. In 't Hout / Eurex settlement methodology:

  1. Build a constant-volatility Black-Scholes **American** pricer (CRR binomial
     tree).  It assumes GBM dynamics but enforces the early-exercise constraint
     at every node, so it *does* account for early exercise.
  2. Invert it for the single volatility sigma* that reproduces the American
     market price:   AmerTree_BS(sigma*) = P_american.
     Because the tree handles the exercise premium explicitly, sigma* is a
     clean volatility — the premium does **not** leak into it.
  3. Re-price a European option at that same sigma* (closed-form BS):
     P_euro_equiv = EuroBS(sigma*).  This strips the early-exercise premium in a
     model-consistent way (same dynamics on both legs, so the crude-model error
     cancels in the difference).

sigma* is simultaneously the "de-Americanized implied vol": the European BS
implied vol of P_euro_equiv is sigma* by construction, so an IV-space objective
can target sigma* directly without forming P_euro_equiv.

The only residual approximation is that the *premium* is computed under
constant-vol GBM rather than under Heston — a small, smooth, second-order
correction.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from models.black_scholes import black_scholes_price
from calibration.implied_vol import implied_volatility


# Default tree resolution. De-Americanization is a one-off pre-step over a
# filtered set of contracts, and the premium is a smooth correction, so a few
# hundred steps is ample accuracy without hurting calibration responsiveness.
DEFAULT_STEPS = 256


def crr_american_price(S, K, r, q, T, sigma, option_type, steps=DEFAULT_STEPS):
    """
    American option price under Black-Scholes dynamics via a Cox-Ross-Rubinstein
    binomial tree with continuous dividend yield q.

    Constant volatility sigma; early exercise enforced at every node.  This is the
    BS-dynamics American pricer used only to back out the de-Americanized vol —
    it is *not* the Heston American pricer.
    """
    is_call = option_type.lower() == "call"
    intrinsic0 = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if T <= 0 or sigma <= 0:
        return intrinsic0

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    # With steps in the hundreds, dt is tiny and p is a valid probability; clip
    # only as a defensive guard against pathological inputs.
    p = min(max(p, 0.0), 1.0)

    # Terminal layer: asset prices and payoffs at maturity (steps+1 nodes).
    k = np.arange(steps + 1)
    ST = S * u ** (steps - k) * d ** k
    V = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)

    # Backward induction with the early-exercise constraint at each node.
    for step in range(steps - 1, -1, -1):
        kk = np.arange(step + 1)
        ST = S * u ** (step - kk) * d ** kk
        cont = disc * (p * V[: step + 1] + (1.0 - p) * V[1: step + 2])
        exercise = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
        V = np.maximum(cont, exercise)

    return float(V[0])


def american_implied_vol(price, S, K, r, q, T, option_type, steps=DEFAULT_STEPS):
    """
    Back out the constant BS volatility sigma* such that the CRR American tree
    reproduces the given American market price.  Returns np.nan if the price is
    not invertible (below intrinsic, above no-arbitrage cap, or no sign change).
    """
    try:
        S, K, T, price = float(S), float(K), float(T), float(price)
    except Exception:
        return np.nan

    if S <= 0 or K <= 0 or T <= 0 or price <= 0:
        return np.nan

    is_call = option_type.lower() == "call"
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    tol = 1e-10
    # An American price must sit between immediate intrinsic and the underlying
    # (call) / strike (put); outside that band it is not invertible.
    upper = S if is_call else K
    if price < intrinsic - tol or price > upper + tol:
        return np.nan

    def objective(sigma):
        return crr_american_price(S, K, r, q, T, sigma, option_type, steps) - price

    sigma_low, sigma_high, max_high = 1e-4, 5.0, 200.0
    try:
        f_low = objective(sigma_low)
        f_high = objective(sigma_high)
        if np.isnan(f_low) or np.isnan(f_high):
            return np.nan
        while f_low * f_high > 0 and sigma_high < max_high:
            sigma_high *= 2.0
            f_high = objective(sigma_high)
        if f_low * f_high > 0:
            return np.nan
        return float(brentq(objective, sigma_low, sigma_high, xtol=1e-8))
    except Exception:
        return np.nan


def de_americanize_price(market_price, S, K, r, q, T, option_type, steps=DEFAULT_STEPS):
    """
    Convert one American market price into a European-equivalent price.

    Returns (euro_equiv_price, sigma_star).  If the quote cannot be inverted the
    original price is returned unchanged with sigma_star = np.nan, so the caller
    can keep the contract rather than dropping it.
    """
    sigma_star = american_implied_vol(market_price, S, K, r, q, T, option_type, steps)
    if np.isnan(sigma_star):
        return float(market_price), np.nan
    euro_price = black_scholes_price(S, K, r, T, sigma_star, option_type, q)
    return float(euro_price), float(sigma_star)


def add_deamericanized_columns(df, r=0.0, q=0.0, steps=DEFAULT_STEPS):
    """
    Add `euro_mid` (European-equivalent mid price) and `deam_iv` (the de-Americanized
    implied vol sigma*) columns to an option-chain frame.

    American-style rows are de-Americanized via the binomial-tree procedure
    (de_americanize_price); European-style rows pass through unchanged
    (euro_mid = mid_price, deam_iv = European BS implied vol). Per-row `r`/`q` are
    used when present, else the scalar fallbacks; rows without a positive mid get NaNs.

    This is the single source of the "European-equivalent" market vol used across the
    project — both as the calibration target and as the market IV in the analytics /
    surface / mispricing layer — so the market side is always directly comparable to
    the European Heston model IV.
    """
    out = df.copy()

    def _row(row):
        try:
            mid = float(row.get("mid_price"))
        except (TypeError, ValueError):
            return (np.nan, np.nan)
        if not np.isfinite(mid) or mid <= 0:
            return (np.nan, np.nan)
        rr = float(row.get("r", r))
        qq = float(row.get("q", q))
        S, K, T = row.get("spot"), row.get("strike"), row.get("T")
        otype = str(row.get("type", "")).lower()
        is_american = str(row.get("ExerciseStyle", "american")).lower() == "american"
        if is_american:
            return de_americanize_price(mid, S, K, rr, qq, T, otype, steps)
        # European quote: the price already is European; deam_iv is its BS implied vol.
        return float(mid), implied_volatility(mid, S, K, rr, T, otype, qq)

    res = out.apply(lambda row: pd.Series(_row(row), index=["euro_mid", "deam_iv"]), axis=1)
    out["euro_mid"] = res["euro_mid"]
    out["deam_iv"] = res["deam_iv"]
    return out
