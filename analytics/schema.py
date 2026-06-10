from __future__ import annotations

from collections.abc import Iterable

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
    return (expiry - now).dt.total_seconds() / (365.0 * 24 * 3600)


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_OPTION_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_type_and_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower()
    if "ticker" not in df.columns:
        df["ticker"] = "UNKNOWN"
    else:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def _ensure_exercise_style(df: pd.DataFrame) -> pd.DataFrame:
    if "ExerciseStyle" not in df.columns:
        df["ExerciseStyle"] = "american"
    else:
        df["ExerciseStyle"] = df["ExerciseStyle"].fillna("american").astype(str).str.lower()
    return df


def _ensure_time_to_expiry(df: pd.DataFrame) -> pd.DataFrame:
    if "T" not in df.columns and "maturity" in df.columns:
        df["T"] = _year_fraction_from_maturity(df["maturity"])
    return df


def _ensure_mid_price(df: pd.DataFrame) -> pd.DataFrame:
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

    # When bid=ask=0 (no live quote), fall back to lastPrice so contracts
    # aren't dropped by the mid-price filter outside market hours.
    if "lastPrice" in df.columns and "mid_price" in df.columns:
        no_quote = df["mid_price"].fillna(0) <= 0
        df.loc[no_quote, "mid_price"] = df.loc[no_quote, "lastPrice"]

    return df


def _compute_rel_spread(df: pd.DataFrame) -> pd.DataFrame:
    if not {"bid", "ask", "mid_price"}.issubset(df.columns):
        return df
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = (df["ask"] - df["bid"]) / df["mid_price"].replace(0, np.nan)
    # NaN rel_spread = no live quote; spread filter skips these intentionally.
    df["rel_spread"] = np.where(
        (df["bid"].fillna(0) == 0) & (df["ask"].fillna(0) == 0),
        np.nan,
        spread,
    )
    return df


def _compute_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    if not {"spot", "strike"}.issubset(df.columns):
        return df
    with np.errstate(divide="ignore", invalid="ignore"):
        df["moneyness"] = df["strike"] / df["spot"]
        df["spot_over_strike"] = df["spot"] / df["strike"]
        # Forward-based moneyness K/F is the economically correct ATM measure.
        # Use it when an implied forward is available (stamped by market_data);
        # otherwise fall back to spot-based K/S. atm_distance drives ATM
        # selection downstream, so it should be forward-based when possible.
        if "forward" in df.columns and df["forward"].notna().any():
            fwd = df["forward"].where(df["forward"] > 0)
            df["forward_moneyness"] = df["strike"] / fwd
            df["forward_moneyness"] = df["forward_moneyness"].fillna(df["moneyness"])
            df["atm_distance"] = np.abs(np.log(df["forward_moneyness"]))
        else:
            df["atm_distance"] = np.abs(np.log(df["moneyness"]))
    return df


def _ensure_contract_id(df: pd.DataFrame) -> pd.DataFrame:
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
    return df


def ensure_option_frame(options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an option-chain DataFrame into a consistent, analytics-friendly schema.

    Applies in order: numeric coercion → type/ticker normalization → exercise style →
    time-to-expiry → mid price → relative spread → moneyness → contract ID.
    """
    df = options_df.copy()
    if df.empty:
        return df

    df = _coerce_numerics(df)
    df = _normalize_type_and_ticker(df)
    df = _ensure_exercise_style(df)
    df = _ensure_time_to_expiry(df)
    df = _ensure_mid_price(df)
    df = _compute_rel_spread(df)
    df = _compute_moneyness(df)
    df = _ensure_contract_id(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def required_columns_present(options_df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return set(columns).issubset(options_df.columns)
