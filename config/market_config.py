"""
market_config.py — SOFR-based OIS discount curve for USD derivatives pricing.

Why SOFR/OIS?
  - Post-LIBOR (June 2023), SOFR is the standard risk-neutral discount rate
    for collateralised USD derivatives.
  - OIS (Overnight Index Swap) rates compound SOFR overnight fixings into term
    rates; they represent the market's expectation of SOFR over each horizon.
  - Using T-bills instead introduces a small credit/liquidity premium (~10-30 bps)
    not present in derivatives valuation.

Curve construction
  - Short end (<= 6M):  SOFR compound averages from FRED (backward-looking,
    published by NY Fed). Differ from CME Term SOFR by < 5 bps in normal markets.
  - Long end (>= 3M):   US Treasury On-The-Run yields from yfinance as anchors.
    OIS-Treasury spread is modelled as negligible (< 30 bps) for equity options.
  - Interpolation:      log-linear in discount factors
    (= linear interpolation of r·T, then divide by T).
  - Extrapolation:      flat at the nearest curve endpoint.

Data sources (tried in order)
  1. FRED public CSV API (no key required) for SOFR compound averages.
  2. yfinance for US Treasury On-The-Run yields (^IRX, ^FVX, ^TNX).
  3. Hard-coded fallback of 4.5% if all network calls fail.

Usage
-----
  from config.market_config import get_ois_curve, interpolate_rate, fetch_sofr_rate

  r_3m   = fetch_sofr_rate(T=0.25)          # single rate for most uses
  r_at_T = interpolate_rate(curve, T=1.5)   # rate at arbitrary maturity
  curve  = get_ois_curve()                  # full {T_years: rate} dict
"""

from __future__ import annotations

import io
import logging
import time
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FALLBACK_RATE: float = 0.045          # used when all network calls fail
_CACHE_TTL:     float = 3600.0         # seconds before a fresh fetch is attempted

# FRED public CSV endpoint — no API key required.
_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# SOFR compound-average series from NY Fed / FRED
# {maturity_years: FRED_series_id}
_SOFR_NODES: dict[float, str] = {
    1 / 365: "SOFR",            # overnight fixing
    30 / 365: "SOFR30DAYAVG",   # 30-day backward-looking compound average
    90 / 365: "SOFR90DAYAVG",   # 90-day
    180 / 365: "SOFR180DAYAVG", # 180-day
}

# US Treasury On-The-Run tickers (yfinance) — anchors for the long end
# {maturity_years: yfinance_ticker}
_TREASURY_NODES: dict[float, str] = {
    0.25: "^IRX",   # 13-week T-bill
    5.0:  "^FVX",   # 5-year note
    10.0: "^TNX",   # 10-year note
}

# Module-level cache: (timestamp, curve_dict) or None
_cache: tuple[float, dict[float, float]] | None = None


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _fetch_fred(series_id: str, timeout: int = 5) -> Optional[float]:
    """Return the latest value of a FRED series as a decimal (not percent)."""
    try:
        url = _FRED_URL.format(sid=series_id)
        req = urllib.request.Request(url, headers={"User-Agent": "HestonPricingEngine/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        # FRED uses "." for missing values
        val_col = df.columns[-1]
        df = df[df[val_col] != "."].copy()
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        latest = df[val_col].dropna().iloc[-1]
        return float(latest) / 100.0          # FRED quotes in percent
    except Exception as exc:
        logger.debug("FRED fetch failed for %s: %s", series_id, exc)
        return None


def _fetch_yf_yield(ticker: str) -> Optional[float]:
    """Return the latest yield from a yfinance treasury ticker as a decimal."""
    try:
        t = yf.Ticker(ticker)
        price = t.fast_info.get("lastPrice")
        if price is None or price == 0:
            hist = t.history(period="5d")
            if hist.empty:
                return None
            price = float(hist["Close"].iloc[-1])
        return float(price) / 100.0           # Yahoo quotes yields in percent
    except Exception as exc:
        logger.debug("yfinance fetch failed for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------

def fetch_ois_curve() -> dict[float, float]:
    """
    Build a SOFR/OIS term-structure dict {maturity_years: annualised_rate}.

    Makes live network calls to FRED and yfinance. Call ``get_ois_curve``
    for the cached version.
    """
    nodes: dict[float, float] = {}

    # Short end — SOFR compound averages from FRED
    for T, sid in _SOFR_NODES.items():
        rate = _fetch_fred(sid)
        if rate is not None:
            nodes[T] = rate
            logger.debug("SOFR node T=%.4f  r=%.4f  (%s)", T, rate, sid)

    # Long end — treasury yields from yfinance
    for T, ticker in _TREASURY_NODES.items():
        # Don't overwrite a SOFR node that already covers this maturity
        if not any(abs(existing - T) < 0.05 for existing in nodes):
            rate = _fetch_yf_yield(ticker)
            if rate is not None:
                nodes[T] = rate
                logger.debug("Treasury node T=%.2f  r=%.4f  (%s)", T, rate, ticker)

    if not nodes:
        logger.warning(
            "All market data fetches failed — using flat fallback rate %.4f", _FALLBACK_RATE
        )
        return {0.25: _FALLBACK_RATE, 5.0: _FALLBACK_RATE}

    return dict(sorted(nodes.items()))


def get_ois_curve(force_refresh: bool = False) -> dict[float, float]:
    """
    Return the SOFR/OIS curve, refreshing at most once per hour.

    Parameters
    ----------
    force_refresh : bool
        Bypass the cache and fetch fresh data immediately.
    """
    global _cache
    now = time.monotonic()

    if not force_refresh and _cache is not None:
        ts, curve = _cache
        if now - ts < _CACHE_TTL:
            return curve

    curve = fetch_ois_curve()
    _cache = (now, curve)
    return curve


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_rate(curve: dict[float, float], T: float) -> float:
    """
    Interpolate the OIS curve at maturity T (in years).

    Method: log-linear in discount factors, equivalent to linear interpolation
    of r·T (the continuously-compounded zero rate times maturity). This ensures
    no-arbitrage between adjacent curve nodes and a smooth forward rate curve.

    Flat extrapolation outside the curve endpoints.
    """
    if not curve:
        return _FALLBACK_RATE

    maturities = np.array(sorted(curve))
    rates = np.array([curve[m] for m in maturities])

    if T <= maturities[0]:
        return float(rates[0])
    if T >= maturities[-1]:
        return float(rates[-1])

    # Interpolate r·T linearly, then divide by T to recover r(T)
    rT = float(np.interp(T, maturities, rates * maturities))
    return rT / T


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def fetch_sofr_rate(T: float = 0.25) -> float:
    """
    Return the SOFR/OIS rate at maturity T (default: 3-month).

    Uses the module-level TTL cache — safe to call on every Streamlit rerender.
    """
    return interpolate_rate(get_ois_curve(), T)


def maturity_label(T: float) -> str:
    """Human-readable maturity label for sidebar display."""
    if T < 2 / 365:
        return "O/N"
    if T < 0.12:
        return f"{round(T * 365)}D"
    if T < 0.9:
        return f"{round(T * 12)}M"
    return f"{round(T)}Y"


def curve_summary(curve: dict[float, float]) -> str:
    """One-line string of key curve nodes for sidebar captions."""
    display_nodes = {T: r for T, r in curve.items() if T >= 1 / 12}
    return "  |  ".join(
        f"{maturity_label(T)}: {r * 100:.2f}%"
        for T, r in sorted(display_nodes.items())
    )
