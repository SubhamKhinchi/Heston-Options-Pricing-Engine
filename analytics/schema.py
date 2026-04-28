from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

NUMERIC_OPTION_COLUMNS: tuple[str, ...] = (
    "spot",
    "strike",
    "bid",
    "ask",
    "lastPrice",
    "mid_price",
    "volume",
    "openInterest",
    "T",
)


def _year_fraction_from_maturity(series: pd.Series) -> pd.Series:
    expiry = pd.to_datetime(series, errors="coerce")
    now = pd.Timestamp.now()
    years = (expiry - now).dt.total_seconds() / (365.0 * 24 * 3600)
    return years


def ensure_option_frame(options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize option-chain columns into a consistent, analytics-friendly schema.
    """
    df = options_df.copy()

    if df.empty:
        return df
    print(df.columns)

    for column in NUMERIC_OPTION_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower()

    if "ticker" not in df.columns:
        df["ticker"] = "UNKNOWN"
    else:
        df["ticker"] = df["ticker"].astype(str).str.upper()

    if "ExerciseStyle" not in df.columns:
        df["ExerciseStyle"] = "american"
    else:
        df["ExerciseStyle"] = df["ExerciseStyle"].fillna("american").astype(str).str.lower()

    if "T" not in df.columns and "maturity" in df.columns:
        df["T"] = _year_fraction_from_maturity(df["maturity"])

    if "mid_price" not in df.columns:
        if {"bid", "ask"}.issubset(df.columns):
            df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
        elif "lastPrice" in df.columns:
            df["mid_price"] = df["lastPrice"]
        else:
            df["mid_price"] = np.nan
    else:
        if {"bid", "ask"}.issubset(df.columns):
            df["mid_price"] = df["mid_price"].fillna((df["bid"] + df["ask"]) / 2.0)
        elif "lastPrice" in df.columns:
            df["mid_price"] = df["mid_price"].fillna(df["lastPrice"])

    if {"bid", "ask", "mid_price"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid_price"]

    if {"spot", "strike"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["moneyness"] = df["strike"] / df["spot"]
            df["spot_over_strike"] = df["spot"] / df["strike"]
            df["atm_distance"] = np.abs(np.log(df["moneyness"]))

    if "contractSymbol" not in df.columns:
        ticker = df.get("ticker", pd.Series("UNKNOWN", index=df.index))
        option_type = df.get("type", pd.Series("option", index=df.index))
        maturity = df.get("maturity", pd.Series("unknown", index=df.index))
        strike = df.get("strike", pd.Series(np.nan, index=df.index))
        df["contractSymbol"] = (
            ticker.astype(str)
            + "_"
            + option_type.astype(str)
            + "_"
            + maturity.astype(str)
            + "_"
            + strike.round(4).astype(str)
        )

    df["contract_id"] = df["contractSymbol"].astype(str)

    df = df.replace([np.inf, -np.inf], np.nan)
    #print(df.columns)
    return df


def required_columns_present(options_df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return set(columns).issubset(options_df.columns)
