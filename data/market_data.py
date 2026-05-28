from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def _year_fraction(maturity: str) -> float:
    try:
        expiry = pd.to_datetime(maturity, errors="coerce")
        if pd.isna(expiry):
            return float("nan")
        return (expiry - pd.Timestamp.now()).total_seconds() / (365.0 * 24 * 3600)
    except Exception:
        return float("nan")


def _fetch_dividend_yield(tk: yf.Ticker) -> float:
    """Return annualised continuous dividend yield from yfinance info, or 0.0 on failure."""
    try:
        info = tk.info
        raw = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0.0
        return float(raw)
    except Exception:
        return 0.0


def get_all_options(ticker: str) -> pd.DataFrame:
    """Fetch raw option chain from Yahoo Finance. No filtering applied — returns every contract as-is."""
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    div_yield = _fetch_dividend_yield(tk)
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
        df["dividend_yield"] = div_yield
        df["ExerciseStyle"] = "american"
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

    return pd.concat(frames, ignore_index=True)


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
