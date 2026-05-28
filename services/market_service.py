from __future__ import annotations

from typing import Iterable

import pandas as pd

from analytics.schema import ensure_option_frame
from config.market_config import interpolate_rate
from data.option_filters import apply_filters


def parse_tickers(raw_tickers: str | Iterable[str]) -> list[str]:
    if isinstance(raw_tickers, str):
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    else:
        tickers = [str(t).strip().upper() for t in raw_tickers if str(t).strip()]
    return tickers or ["NVDA"]


def extract_dividend_yields(raw_df: pd.DataFrame) -> dict[str, float]:
    """Return {ticker: dividend_yield} extracted from the raw chain DataFrame."""
    if "dividend_yield" not in raw_df.columns or "ticker" not in raw_df.columns:
        return {}
    return (
        raw_df.dropna(subset=["dividend_yield"])
        .groupby("ticker")["dividend_yield"]
        .first()
        .to_dict()
    )


def load_live_chain(tickers: Iterable[str]) -> pd.DataFrame:
    """Fetch raw option chain from Yahoo Finance with no filtering applied.

    Mirrors the notebook: get_all_options() output is returned as-is so that
    filter_chain_with_stats (which calls ensure_option_frame internally) is the
    single place that normalises the schema.
    """
    from data.market_data import get_multiple_tickers

    return get_multiple_tickers(parse_tickers(tickers))


def filter_chain_with_stats(
    raw_df: pd.DataFrame,
    *,
    spread_limit: float = 0.05,
    r: float = 0.0,
    q: float = 0.0,
    rate_curve: dict[float, float] | None = None,
    tickers: list[str] | None = None,
    option_types: tuple[str, ...] | None = None,
    min_volume: int = 0,
    min_open_interest: int = 0,
    max_maturity: float | None = None,
    max_contracts: int | None = None,
    moneyness_lo: float = 0.8,
    moneyness_hi: float = 1.2,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Single entry point for all filtering. Returns (filtered_df, stats)."""
    df = ensure_option_frame(raw_df)
    # Stamp per-row q from dividend_yield before filtering so apply_filters
    # can use it for the arbitrage bound (multi-ticker: each ticker keeps its own yield).
    if "dividend_yield" in df.columns:
        df["q"] = df["dividend_yield"].fillna(0.0)
    elif "q" not in df.columns:
        df["q"] = q

    filtered_df, stats = apply_filters(
        df,
        spread_limit=spread_limit,
        r=r,
        q=q,
        rate_curve=rate_curve,
        tickers=tickers,
        option_types=option_types,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        max_maturity=max_maturity,
        max_contracts=max_contracts,
        moneyness_lo=moneyness_lo,
        moneyness_hi=moneyness_hi,
    )
    if "r" not in filtered_df.columns:
        filtered_df = filtered_df.copy()
        if rate_curve:
            filtered_df["r"] = filtered_df["T"].map(lambda T: interpolate_rate(rate_curve, T))
        else:
            filtered_df["r"] = r
    return filtered_df, stats
