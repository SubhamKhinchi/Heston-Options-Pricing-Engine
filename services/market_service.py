from __future__ import annotations

from typing import Iterable

import pandas as pd

from analytics.schema import ensure_option_frame
from data.option_filters import apply_filters


def parse_tickers(raw_tickers: str | Iterable[str]) -> list[str]:
    if isinstance(raw_tickers, str):
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    else:
        tickers = [str(t).strip().upper() for t in raw_tickers if str(t).strip()]
    return tickers or ["NVDA"]


def load_live_chain(tickers: Iterable[str]) -> pd.DataFrame:
    """Fetch raw option chain from Yahoo Finance with no filtering applied."""
    from data.market_data import get_multiple_tickers

    df = get_multiple_tickers(parse_tickers(tickers))
    return ensure_option_frame(df)


def filter_chain_with_stats(
    raw_df: pd.DataFrame,
    *,
    spread_limit: float = 0.05,
    r: float = 0.05,
    q: float = 0.0,
    tickers: list[str] | None = None,
    option_types: tuple[str, ...] | None = None,
    min_volume: int = 0,
    min_open_interest: int = 0,
    max_maturity: float | None = None,
    max_contracts: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Single entry point for all filtering. Returns (filtered_df, stats)."""
    df = ensure_option_frame(raw_df)
    return apply_filters(
        df,
        spread_limit=spread_limit,
        r=r,
        q=q,
        tickers=tickers,
        option_types=option_types,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        max_maturity=max_maturity,
        max_contracts=max_contracts,
    )
