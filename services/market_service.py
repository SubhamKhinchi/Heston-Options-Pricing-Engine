from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from analytics.schema import ensure_option_frame


DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "nvda_vol.xlsx"


def parse_tickers(raw_tickers: str | Iterable[str]) -> list[str]:
    if isinstance(raw_tickers, str):
        tickers = [ticker.strip().upper() for ticker in raw_tickers.split(",") if ticker.strip()]
    else:
        tickers = [str(ticker).strip().upper() for ticker in raw_tickers if str(ticker).strip()]
    return tickers or ["NVDA"]


def load_sample_chain(sample_path: str | Path = DEFAULT_SAMPLE_PATH) -> pd.DataFrame:
    path = Path(sample_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample chain not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported sample format: {path.suffix}")

    return ensure_option_frame(df)


def load_live_chain(
    tickers: Iterable[str],
    *,
    spread_limit: float,
    r: float,
    q: float,
) -> pd.DataFrame:
    from data.market_data import get_multiple_tickers

    df = get_multiple_tickers(parse_tickers(tickers), spread_limit=spread_limit, r=r, q=q)
    return ensure_option_frame(df)


def load_option_chain(
    *,
    source: str,
    tickers: Iterable[str] | str = ("NVDA",),
    spread_limit: float = 0.05,
    r: float = 0.05,
    q: float = 0.0,
    sample_path: str | Path = DEFAULT_SAMPLE_PATH,
) -> pd.DataFrame:
    if source == "sample":
        return load_sample_chain(sample_path=sample_path)
    if source == "live":
        return load_live_chain(tickers=tickers, spread_limit=spread_limit, r=r, q=q)
    raise ValueError("source must be 'sample' or 'live'")


def filter_option_chain(
    options_df: pd.DataFrame,
    *,
    tickers: Iterable[str] | None = None,
    option_types: Iterable[str] | None = None,
    min_volume: int = 0,
    min_open_interest: int = 0,
    max_maturity: float | None = None,
    keep_positive_time: bool = True,
    max_contracts: int | None = None,
) -> pd.DataFrame:
    df = ensure_option_frame(options_df)

    if keep_positive_time and "T" in df.columns:
        df = df[df["T"] > 0].copy()

    if tickers:
        df = df[df["ticker"].isin(parse_tickers(tickers))].copy()

    if option_types:
        normalized_types = {str(option_type).lower() for option_type in option_types}
        df = df[df["type"].isin(normalized_types)].copy()

    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= min_volume].copy()

    if "openInterest" in df.columns:
        df = df[df["openInterest"].fillna(0) >= min_open_interest].copy()

    if max_maturity is not None and "T" in df.columns:
        df = df[df["T"] <= max_maturity].copy()

    df = df.sort_values(["ticker", "T", "strike", "type"]).reset_index(drop=True)

    if max_contracts is not None and len(df) > max_contracts:
        df = df.head(max_contracts).copy()

    return df
