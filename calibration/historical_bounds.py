"""
Historical-data-based initial guess and bounds for Heston model calibration.

Estimates (v0, kappa, theta, sigma, rho) from the realized variance dynamics
of daily closing prices under the physical (real-world) measure P.

Physical vs risk-neutral
------------------------
These estimates live under P, not Q. The risk-neutral calibration targets Q.
Use these as sanity checks and starting bounds — signs and rough magnitudes
should agree across both measures; exact values will differ.

Parameter estimation logic
--------------------------
  v0    — most recent 3-month (63-day) rolling realized variance
  theta — long-run mean of the full rolling RV series
  kappa — AR(1) regression on the rolling RV series: v[t+1] = a + b*v[t] + ε
             kappa = (1 - b) / dt,  dt = 1/252
  sigma — vol-of-vol from normalized AR(1) residuals:
             ε[t] ≈ σ·√(v[t]·dt)·noise  →  σ = std(ε / √(v[t]·dt))
  rho   — Pearson correlation between daily log-returns and changes in
             rolling realized variance
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252   # annualization constant


# ── Data helpers ──────────────────────────────────────────────────────────────

def fetch_price_history(ticker: str, period: str = "2y") -> pd.Series:
    """Download adjusted daily closing prices from Yahoo Finance."""
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError(f"No price history returned for {ticker!r}.")
    return hist["Close"].dropna()


def _log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def _rolling_realized_variance(returns: pd.Series, window: int) -> pd.Series:
    """Annualized rolling realized variance over *window* trading days."""
    return (returns.rolling(window=window).var() * TRADING_DAYS).dropna()


# ── AR(1) regression ──────────────────────────────────────────────────────────

def _fit_ar1(rv: pd.Series) -> tuple[float, float, np.ndarray]:
    """
    OLS fit of v[t+1] = a + b·v[t] + ε on the realized variance series.

    Returns (a, b, residuals).
    """
    y = rv.iloc[1:].values
    x = rv.iloc[:-1].values
    X = np.column_stack([np.ones_like(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    residuals = y - (a + b * x)
    return a, b, residuals


# ── Main estimator ────────────────────────────────────────────────────────────

def estimate_heston_from_history(
    prices: pd.Series,
    v0_window: int = 63,
) -> dict:
    """
    Estimate Heston parameters from historical daily closing prices.

    Parameters
    ----------
    prices : pd.Series
        Daily adjusted closing prices indexed by date.
    v0_window : int
        Rolling window in trading days used for the realized variance series
        (default 63 ≈ 3 months).  This window drives v0, kappa, and sigma.

    Returns
    -------
    dict with keys:
        'initial_guess' : [v0, kappa, theta, sigma, rho]
        'bounds'        : [(lo, hi), ...] same order
        'diagnostics'   : dict of intermediate computed values
    """
    dt = 1.0 / TRADING_DAYS

    returns   = _log_returns(prices)
    rv_series = _rolling_realized_variance(returns, window=v0_window)

    if len(rv_series) < v0_window + 20:
        return _default_output("Insufficient history — need at least 2× v0_window observations.")

    # ── 1. v0: most recent 3-month realized variance ──────────────────────────
    # Uses the last value of the rolling RV window — the variance the market
    # has realized over the most recent v0_window trading days, annualized.
    v0_star = float(np.clip(rv_series.iloc[-1], 1e-4, 2.0))
    v0_lo   = max(0.7  * v0_star, 1e-4)
    v0_hi   = min(1.3  * v0_star, 2.0)

    # ── 2. theta: long-run mean of the rolling RV series ─────────────────────
    # Average realized variance over the full downloaded history — this is the
    # best available proxy for the unconditional (long-run) variance level.
    theta_star = float(np.clip(rv_series.mean(), 1e-4, 2.0))
    theta_lo   = max(0.5  * theta_star, 1e-4)
    theta_hi   = min(2.0  * theta_star, 2.0)

    # ── 3. kappa: mean reversion speed from AR(1) ────────────────────────────
    # Heston variance process discretized at daily frequency:
    #   v[t+dt] ≈ v[t] + kappa*(theta - v[t])*dt + noise
    #           = kappa*theta*dt + (1 - kappa*dt)*v[t] + noise
    #           = a + b*v[t] + noise          ← AR(1), b = 1 - kappa*dt
    # Therefore: kappa = (1 - b) / dt
    a_ar1, b_ar1, residuals = _fit_ar1(rv_series)

    # Clip b to [0, 1) to guarantee stationarity before inverting
    b_clipped  = np.clip(b_ar1, 0.0, 1.0 - 1e-6)
    kappa_star = float(np.clip((1.0 - b_clipped) / dt, 0.1, 15.0))
    kappa_lo   = max(0.5, 0.5 * kappa_star)
    kappa_hi   = min(3.0  * kappa_star, 15.0)

    # ── 4. sigma: vol-of-vol from normalized AR(1) residuals ─────────────────
    # The Heston diffusion term for variance is σ·√v·dW₂.
    # Over one day: ε[t] ≈ σ·√(v[t]·dt)·noise
    # So: σ = std(ε[t] / √(v[t]·dt))
    v_lag      = rv_series.iloc[:-1].values
    normalizer = np.sqrt(np.maximum(v_lag * dt, 1e-10))
    sigma_star = float(np.clip(np.std(residuals / normalizer), 0.05, 2.0))
    sigma_lo   = max(0.5  * sigma_star, 1e-4)
    sigma_hi   = min(2.0  * sigma_star, 2.0)

    # ── 5. rho: return–variance correlation ───────────────────────────────────
    # Heston: corr(dS/S, dv) = ρ·dt
    # Empirical proxy: corr(daily log-return, daily change in rolling RV)
    rv_changes = rv_series.diff().dropna()
    common_idx = returns.index.intersection(rv_changes.index)
    r_aligned  = returns.loc[common_idx].values
    dv_aligned = rv_changes.loc[common_idx].values

    if len(r_aligned) > 20:
        rho_star = float(np.clip(np.corrcoef(r_aligned, dv_aligned)[0, 1], -0.999, 0.999))
    else:
        rho_star = -0.70  # equity fallback

    rho_lo = max(rho_star - 0.20, -0.999)
    rho_hi = min(rho_star + 0.20,  0.999)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    diagnostics = {
        "n_price_obs":       len(prices),
        "n_return_obs":      len(returns),
        "history_start":     str(prices.index[0].date()),
        "history_end":       str(prices.index[-1].date()),
        "v0_window_days":    v0_window,
        "rv_latest":         float(rv_series.iloc[-1]),
        "rv_mean":           float(rv_series.mean()),
        "rv_std":            float(rv_series.std()),
        "rv_min":            float(rv_series.min()),
        "rv_max":            float(rv_series.max()),
        "ar1_b":             float(b_ar1),
        "ar1_implied_theta": float(a_ar1 / max(1.0 - b_ar1, 1e-8)),
        "feller_satisfied":  bool(2.0 * kappa_star * theta_star > sigma_star ** 2),
    }

    return {
        "initial_guess": [v0_star, kappa_star, theta_star, sigma_star, rho_star],
        "bounds": [
            (v0_lo,    v0_hi),
            (kappa_lo, kappa_hi),
            (theta_lo, theta_hi),
            (sigma_lo, sigma_hi),
            (rho_lo,   rho_hi),
        ],
        "diagnostics": diagnostics,
    }


def compute_historical_bounds(
    ticker: str,
    period: str = "2y",
    v0_window: int = 63,
) -> dict:
    """
    Fetch price history and estimate Heston bounds in one call.

    Parameters
    ----------
    ticker : str
        Equity ticker (e.g. "NVDA").
    period : str
        yfinance period string — how far back to download (default "2y").
    v0_window : int
        Rolling window in trading days for realized variance (default 63 ≈ 3 months).
    """
    prices = fetch_price_history(ticker, period=period)
    return estimate_heston_from_history(prices, v0_window=v0_window)


def _default_output(reason: str = "") -> dict:
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
