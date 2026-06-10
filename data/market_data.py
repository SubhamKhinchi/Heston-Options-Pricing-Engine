from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from data.instrument_classifier import classify
from data.forward_curve import build_forward_curve


def _year_fraction(maturity: str) -> float:
    try:
        expiry = pd.to_datetime(maturity, errors="coerce")
        if pd.isna(expiry):
            return float("nan")
        return (expiry - pd.Timestamp.now()).total_seconds() / (365.0 * 24 * 3600)
    except Exception:
        return float("nan")


def _trailing_dividend_yield(tk: yf.Ticker, spot: float) -> float:
    """Robust trailing-12M continuous dividend yield computed from the actual
    cash dividend history — NOT yfinance's ``info['dividendYield']``, which is
    notoriously stale and unit-inconsistent (it handed back 47% for NVDA).

    Returns sum of cash dividends over the trailing 365 days divided by spot.
    Falls back to 0.0 on any failure (correct for non-dividend payers).
    """
    if not spot or spot <= 0:
        return 0.0
    try:
        divs = tk.dividends  # pandas Series indexed by ex-date
        if divs is None or len(divs) == 0:
            return 0.0
        cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
        ttm_cash = float(divs[divs.index >= cutoff].sum())
        q = ttm_cash / spot
        # Sanity clamp: a single-name continuous yield above ~15% is almost
        # certainly bad data; treat as zero rather than poison every price.
        return q if 0.0 <= q < 0.15 else 0.0
    except Exception:
        return 0.0


def _rate_at(T: float) -> float:
    """Maturity-matched SOFR/OIS rate, with a flat fallback if the curve is
    unavailable. Used only to convert an implied forward into an implied yield."""
    try:
        from config.market_config import get_ois_curve, interpolate_rate

        curve = get_ois_curve()
        return float(interpolate_rate(curve, T))
    except Exception:
        return 0.045


def get_all_options(ticker: str) -> pd.DataFrame:
    """Fetch raw option chain from Yahoo Finance. No filtering applied — returns every contract as-is.

    Dividend handling is no longer the broken yfinance ``dividendYield`` scalar.
    Instead, per expiry we recover an *implied forward* from put-call parity and
    express it as a per-expiry continuous yield; where the chain is too thin we
    fall back to the trailing-12M realised dividend yield. Exercise style and
    instrument type are stamped from the classifier (cash indices are European).
    """
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])

    info = classify(ticker)
    fallback_q = _trailing_dividend_yield(tk, spot)
    frames = []

    for maturity in tk.options:
        T = _year_fraction(maturity)
        if pd.isna(T) or T <= 0:
            continue

        chain = tk.option_chain(maturity)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        calls["type"] = "call"
        puts["type"] = "put"
        df = pd.concat([calls, puts], ignore_index=True)
        df["maturity"] = maturity
        df["spot"] = spot
        df["ticker"] = ticker
        df["instrument_type"] = info.instrument_type
        df["ExerciseStyle"] = info.exercise_style
        df["dividend_yield"] = fallback_q          # provisional; refined below
        df["dividend_source"] = "trailing"          # provisional
        df["forward"] = np.nan
        df["T"] = T

        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df = df.dropna(subset=["strike"]).copy()

        if {"bid", "ask"}.issubset(df.columns):
            df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
        elif "lastPrice" in df.columns:
            df["mid_price"] = df["lastPrice"]

        # When bid=0 and ask=0 (no live quote), fall back to lastPrice so the
        # contract isn't dropped by the mid-price filter outside market hours.
        if "lastPrice" in df.columns and "mid_price" in df.columns:
            no_quote = df["mid_price"].fillna(0) <= 0
            df.loc[no_quote, "mid_price"] = df.loc[no_quote, "lastPrice"]

        if {"bid", "ask", "mid_price"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                spread = (df["ask"] - df["bid"]) / df["mid_price"].replace(0, np.nan)
            # rel_spread is NaN when bid=ask=0 (no live quote) — spread filter
            # will skip these rather than treating them as infinitely wide.
            df["rel_spread"] = np.where(
                (df["bid"].fillna(0) == 0) & (df["ask"].fillna(0) == 0),
                np.nan,
                spread,
            )

        if {"strike", "spot"}.issubset(df.columns):
            df["moneyness"] = df["strike"] / df["spot"]

        df = df.replace([np.inf, -np.inf], np.nan)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # ── Refine dividends via implied forward (near-ATM put-call parity) ───────
    # Build a maturity-matched rate map, then recover the implied forward per
    # expiry from the near-ATM window (exercise-style-aware; see forward_curve).
    # The forward F(T) is the carried object. Where the fit passes every quality
    # gate we stamp the implied yield + forward; otherwise we keep the trailing
    # yield and make the forward consistent with it, so F(T) and q(T) never
    # disagree downstream.
    try:
        r_by_T = {float(T): _rate_at(float(T)) for T in out["T"].dropna().unique()}
        points = build_forward_curve(out, spot=spot, r_by_T=r_by_T)
        for p in points:
            mask = out["maturity"] == p.maturity
            if p.ok:
                out.loc[mask, "forward"] = p.forward
                out.loc[mask, "dividend_yield"] = p.implied_q
                out.loc[mask, "dividend_source"] = "implied_forward"
            else:
                # No reliable implied forward — keep the trailing-yield dividend
                # (already stamped above) and set F = S·e^{(r-q)T} to match it.
                r_exp = -np.log(p.discount) / p.T if p.T > 0 and p.discount > 0 else 0.0
                out.loc[mask, "forward"] = spot * np.exp((r_exp - fallback_q) * p.T)
    except Exception as exc:  # never let dividend refinement break the fetch
        print(f"  implied-forward refinement skipped for {ticker}: {exc}")

    return out


def get_multiple_tickers(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        print(f"pulling...{ticker}...")
        df = get_all_options(ticker)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
