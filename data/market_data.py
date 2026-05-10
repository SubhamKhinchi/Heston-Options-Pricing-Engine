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


def get_all_options(ticker: str) -> pd.DataFrame:
    """Fetch raw option chain from Yahoo Finance. No filtering applied — returns every contract as-is."""
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
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
        df["ExerciseStyle"] = "american"
        df["T"] = T

        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df = df.dropna(subset=["strike"]).copy()

        if {"bid", "ask"}.issubset(df.columns):
            df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
        elif "lastPrice" in df.columns:
            df["mid_price"] = df["lastPrice"]

        if {"bid", "ask", "mid_price"}.issubset(df.columns):
            df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid_price"].replace(0, np.nan)

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
