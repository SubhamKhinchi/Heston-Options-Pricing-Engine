"""
Data-driven initial guess and parameter bounds for Heston model calibration.

Uses the shape of the market implied-volatility surface to estimate
sensible starting values and tight search bounds for (v0, kappa, theta, sigma, rho).

Algorithm overview
------------------
1. v0    — ATM IV² at the shortest liquid maturity
2. theta — ATM IV² at the longest liquid maturity
3. sigma — vol-of-vol from quadratic smile curvature: σ ≈ sqrt(8 * c)
4. rho   — correlation from ATM smile slope: ρ ≈ 2 * b / sigma
5. kappa — mean reversion from term-structure slope of ATM variance
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibration.implied_vol import implied_volatility


# ── IV helper ─────────────────────────────────────────────────────────────────

def _compute_iv_column(df: pd.DataFrame, r: float, q: float) -> pd.Series:
    def _row_iv(row):
        return implied_volatility(
            heston_model_price=row["mid_price"],
            S=row["spot"],
            K=row["strike"],
            r=r,
            T=row["T"],
            option_type=row["type"],
            q=q,
        )
    return df.apply(_row_iv, axis=1)


# ── Smile fitting helpers ─────────────────────────────────────────────────────

def _select_liquid_maturities(df: pd.DataFrame, min_contracts: int = 5) -> list[float]:
    """Sorted list of maturities that have at least *min_contracts* valid IV rows."""
    counts = df.groupby("T").size()
    return sorted(counts[counts >= min_contracts].index.tolist())


def _fit_smile(k: np.ndarray, iv: np.ndarray, degree: int = 2):
    """
    Fit σ_imp ≈ a + b*k + c*k² by least squares.

    Returns coefficients [a, b, c] (lowest-degree first), or None on failure.
    """
    mask = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[mask], iv[mask]
    n = len(k)
    if n < 2:
        return None
    deg = min(degree, n - 1)
    try:
        poly = np.polyfit(k, iv, deg=deg)   # highest-degree first
        coeffs = poly[::-1]                  # → [a, b, c, ...]
        while len(coeffs) < 3:
            coeffs = np.append(coeffs, 0.0) # pad with zeros if deg < 2
        return coeffs
    except Exception:
        return None


def _atm_iv_from_slice(slice_df: pd.DataFrame, atm_threshold: float) -> float:
    """
    Estimate the ATM implied vol (k = 0) for one maturity slice.

    Prefers a polynomial fit evaluated at k = 0; falls back to the
    nearest-to-ATM contract if there are too few points.
    """
    sub = slice_df[
        (slice_df["log_moneyness"].abs() < atm_threshold)
        & slice_df["market_iv"].notna()
        & (slice_df["market_iv"] > 0.01)
    ]

    if sub.empty:
        idx = slice_df["log_moneyness"].abs().idxmin()
        return float(slice_df.loc[idx, "market_iv"]) if not pd.isna(slice_df.loc[idx, "market_iv"]) else np.nan

    k  = sub["log_moneyness"].values
    iv = sub["market_iv"].values

    coeffs = _fit_smile(k, iv, degree=min(2, len(sub) - 1))
    if coeffs is None:
        return float(np.nanmedian(iv))

    # Evaluate polynomial at k = 0: value is the intercept (coeffs[0])
    return float(np.clip(coeffs[0], 0.01, 5.0))


def _estimate_smile_shape(
    slice_df: pd.DataFrame, fit_threshold: float
) -> tuple[float, float, float]:
    """
    Fit σ_imp ≈ a + b*k + c*k² and return (atm_iv=a, slope=b, curvature=c).

    Uses a wider moneyness window than _atm_iv_from_slice so there are
    enough contracts on both sides of ATM for a reliable regression.
    """
    sub = slice_df[
        (slice_df["log_moneyness"].abs() < fit_threshold)
        & slice_df["market_iv"].notna()
        & (slice_df["market_iv"] > 0.01)
    ]

    if len(sub) < 3:
        atm_iv = _atm_iv_from_slice(slice_df, fit_threshold)
        return (atm_iv if not np.isnan(atm_iv) else 0.30), 0.0, 0.01

    k      = sub["log_moneyness"].values
    iv     = sub["market_iv"].values
    coeffs = _fit_smile(k, iv, degree=2)

    if coeffs is None:
        return float(np.nanmedian(iv)), 0.0, 0.01

    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return a, b, c


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_data_driven_bounds(
    df: pd.DataFrame,
    r: float = 0.05,
    q: float = 0.0,
    atm_threshold: float = 0.10,
    fit_threshold: float = 0.15,
    min_contracts: int = 5,
) -> dict:
    """
    Derive data-driven initial guess and search bounds for Heston calibration.

    Parameters
    ----------
    df : DataFrame
        Option chain after ensure_option_frame(). Must contain:
        mid_price, spot, strike, T, type.
        market_iv is computed internally if absent.
    r, q : float
        Risk-free rate and continuous dividend yield.
    atm_threshold : float
        |log(K/S)| cutoff for ATM IV estimation (default 10%).
    fit_threshold : float
        |log(K/S)| cutoff for the skew/curvature regression (default 15%).
    min_contracts : int
        Minimum valid-IV contracts per maturity to count as liquid.

    Returns
    -------
    dict with keys:
        'initial_guess' : [v0, kappa, theta, sigma, rho]
        'bounds'        : [(lo, hi), ...] for each parameter, same order
        'diagnostics'   : dict of intermediate derived quantities
    """
    df = df.copy()

    # Log-moneyness: k = log(K / S)
    df["log_moneyness"] = np.log(df["strike"] / df["spot"])

    # Ensure market_iv is present
    if "market_iv" not in df.columns or df["market_iv"].isna().all():
        df["market_iv"] = _compute_iv_column(df, r, q)
    else:
        missing = df["market_iv"].isna()
        if missing.any():
            df.loc[missing, "market_iv"] = _compute_iv_column(df[missing], r, q)

    # Working set: valid IV rows only
    iv_df = df[df["market_iv"].notna() & (df["market_iv"] > 0.01)].copy()

    liquid_T = _select_liquid_maturities(iv_df, min_contracts=min_contracts)

    if len(liquid_T) < 2:
        return _default_output("Fewer than 2 liquid maturities — static defaults used.")

    T_short = liquid_T[0]
    T_long  = liquid_T[-1]

    short_slice = iv_df[iv_df["T"] == T_short]
    long_slice  = iv_df[iv_df["T"] == T_long]

    # ── 1. v0: initial variance ───────────────────────────────────────────────
    # v0* = σ_ATM(T_short)²  using nearest-to-ATM strike at shortest maturity
    sigma_atm_short = _atm_iv_from_slice(short_slice, atm_threshold)
    if np.isnan(sigma_atm_short) or sigma_atm_short <= 0:
        sigma_atm_short = 0.20

    v0_star = sigma_atm_short ** 2
    v0_lo   = max(0.7  * v0_star, 1e-4)
    v0_hi   = min(1.3  * v0_star, 2.0)

    # ── 2. theta: long-run variance ───────────────────────────────────────────
    # θ* = σ_ATM(T_long)²  using ATM strike at longest maturity
    sigma_atm_long = _atm_iv_from_slice(long_slice, atm_threshold)
    if np.isnan(sigma_atm_long) or sigma_atm_long <= 0:
        sigma_atm_long = sigma_atm_short

    theta_star = sigma_atm_long ** 2
    theta_lo   = max(0.5  * theta_star, 1e-4)
    theta_hi   = min(2.0  * theta_star, 2.0)

    # ── 3 & 4. sigma (vol-of-vol) and rho (correlation) ──────────────────────
    # Fit σ_imp ≈ a + b*k + c*k²  at the short-maturity slice (richest in skew)
    _, slope_b, curv_c = _estimate_smile_shape(short_slice, fit_threshold)

    # sigma from curvature: Heston short-time approx → c ≈ σ² / 8
    c_safe     = max(curv_c, 1e-4)
    sigma_star = float(np.clip(np.sqrt(8.0 * c_safe), 0.10, 2.0))
    sigma_lo   = max(0.5 * sigma_star, 1e-4)
    sigma_hi   = min(2.0 * sigma_star, 2.0)

    # rho from skew: Heston short-time approx → b ≈ ρ * σ / 2
    # → ρ = 2 * b / σ
    rho_star = float(np.clip(2.0 * slope_b / sigma_star, -0.95, 0.95))
    rho_lo   = max(rho_star - 0.20, -0.999)
    rho_hi   = min(rho_star + 0.20,  0.999)

    # ── 5. kappa: mean reversion from ATM term structure ─────────────────────
    # slope_T = (σ_ATM(T_long) − σ_ATM(T_short)) / (T_long − T_short)
    #
    # Heston term structure for variance:
    #   E[v_T] ≈ θ + (v0 − θ) e^{−κT}
    # Differentiating and applying at T_short → 0:
    #   d(σ²)/dT ≈ −κ (v0 − θ)
    # Using d(σ²)/dT ≈ 2 σ_mid · slope_T:
    #   κ* = −2 σ_mid · slope_T / (v0* − θ*)
    slope_T   = (sigma_atm_long - sigma_atm_short) / (T_long - T_short)
    sigma_mid = (sigma_atm_short + sigma_atm_long) / 2.0
    dv        = v0_star - theta_star

    if abs(dv) < 5e-4:
        # Near-flat term structure: κ is weakly identified; use a neutral value
        kappa_star = 2.0
    else:
        kappa_star = float(np.clip(-2.0 * sigma_mid * slope_T / dv, 0.1, 10.0))

    kappa_lo = max(0.5, 0.5 * kappa_star)
    kappa_hi = min(3.0 * kappa_star, 10.0)

    # ── Assemble result ───────────────────────────────────────────────────────
    initial_guess = [v0_star, kappa_star, theta_star, sigma_star, rho_star]
    bounds = [
        (v0_lo,    v0_hi),
        (kappa_lo, kappa_hi),
        (theta_lo, theta_hi),
        (sigma_lo, sigma_hi),
        (rho_lo,   rho_hi),
    ]

    diagnostics = {
        "T_short":             T_short,
        "T_long":              T_long,
        "sigma_atm_short":     sigma_atm_short,
        "sigma_atm_long":      sigma_atm_long,
        "slope_T":             slope_T,
        "smile_slope_b":       slope_b,
        "smile_curvature_c":   curv_c,
        "n_liquid_maturities": len(liquid_T),
        "liquid_maturities":   liquid_T,
    }

    return {
        "initial_guess": initial_guess,
        "bounds":        bounds,
        "diagnostics":   diagnostics,
    }


def _default_output(reason: str = "") -> dict:
    """Fallback when data is insufficient to estimate bounds."""
    return {
        "initial_guess": [0.04, 2.0, 0.04, 0.5, -0.70],
        "bounds": [
            (1e-4, 2.0),
            (1e-4, 10.0),
            (1e-4, 2.0),
            (1e-4, 2.0),
            (-0.999, 0.999),
        ],
        "diagnostics": {"warning": reason or "Static defaults used."},
    }
