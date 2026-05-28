from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from analytics.greeks import black_scholes_greeks
from analytics.schema import ensure_option_frame
from calibration.implied_vol import implied_volatility
from config.market_config import interpolate_rate
from services.pricing_service import HestonParameters, price_option_frame


GREEK_NAMES: tuple[str, ...] = ("delta", "gamma", "vega", "theta", "rho")


def _implied_vol_for_row(row: pd.Series, price_col: str, r: float, q: float) -> float:
    price = row.get(price_col)
    if pd.isna(price):
        return np.nan
    return implied_volatility(
        heston_model_price=price,
        S=row.get("spot"),
        K=row.get("strike"),
        r=float(row.get("r", r)),
        T=row.get("T"),
        option_type=row.get("type", ""),
        q=float(row.get("q", q)),
    )


def _greeks_from_iv(
    row: pd.Series,
    iv_col: str,
    prefix: str,
    r: float,
    q: float,
) -> pd.Series:
    greeks = black_scholes_greeks(
        S=row.get("spot"),
        K=row.get("strike"),
        r=float(row.get("r", r)),
        T=row.get("T"),
        sigma=row.get(iv_col),
        option_type=row.get("type", ""),
        q=float(row.get("q", q)),
    )
    return pd.Series({f"{prefix}_{name}": value for name, value in greeks.items()})


def compute_liquidity_score(options_df: pd.DataFrame) -> pd.Series:
    volume = options_df.get("volume", pd.Series(0.0, index=options_df.index)).fillna(0.0)
    open_interest = options_df.get("openInterest", pd.Series(0.0, index=options_df.index)).fillna(0.0)
    rel_spread = options_df.get("rel_spread", pd.Series(1.0, index=options_df.index)).fillna(1.0)
    mid_price = options_df.get("mid_price", pd.Series(0.0, index=options_df.index)).fillna(0.0)

    volume_component = np.tanh(volume / 100.0)
    oi_component = np.tanh(open_interest / 500.0)
    price_component = np.tanh(mid_price / 10.0)
    spread_component = 1.0 - np.clip(rel_spread, 0.0, 1.0)

    score = 100.0 * (
        0.35 * volume_component
        + 0.30 * oi_component
        + 0.20 * spread_component
        + 0.15 * price_component
    )
    return pd.Series(np.clip(score, 0.0, 100.0), index=options_df.index)


def _intrinsic_value(options_df: pd.DataFrame) -> pd.Series:
    calls = options_df["type"] == "call"
    return pd.Series(
        np.where(
            calls,
            np.maximum(options_df["spot"] - options_df["strike"], 0.0),
            np.maximum(options_df["strike"] - options_df["spot"], 0.0),
        ),
        index=options_df.index,
    )


def enrich_option_chain(
    options_df: pd.DataFrame,
    r: float = 0.0,
    q: float = 0.0,
    *,
    rate_curve: dict | None = None,
    heston_params: HestonParameters | Iterable[float] | None = None,
    compute_model_prices: bool = False,
    pricing_limit: int | None = None,
    Ns: int = 40,
    Nv: int = 20,
    Nt: int = 40,
) -> pd.DataFrame:
    """
    Add implied vols, greeks, liquidity metrics, and optional Heston model values.
    """
    df = ensure_option_frame(options_df)

    if df.empty:
        return df

    df = df.copy()
    if rate_curve and "r" not in df.columns:
        df["r"] = df["T"].map(lambda T: interpolate_rate(rate_curve, T))
    df["intrinsic_value"] = _intrinsic_value(df)
    df["time_value"] = df["mid_price"] - df["intrinsic_value"]
    df["liquidity_score"] = compute_liquidity_score(df)

    if "market_iv" not in df.columns:
        df["market_iv"] = df.apply(_implied_vol_for_row, axis=1, args=("mid_price", r, q))
    else:
        missing_market_iv = df["market_iv"].isna()
        if missing_market_iv.any():
            df.loc[missing_market_iv, "market_iv"] = df.loc[missing_market_iv].apply(
                _implied_vol_for_row,
                axis=1,
                args=("mid_price", r, q),
            )

    market_greeks = df.apply(_greeks_from_iv, axis=1, args=("market_iv", "market", r, q))
    df = pd.concat([df, market_greeks], axis=1)
    df["market_abs_delta"] = df["market_delta"].abs()

    if "calibrated_heston_price" in df.columns and "model_price" not in df.columns:
        df["model_price"] = df["calibrated_heston_price"]

    if compute_model_prices and heston_params is not None:
        df["model_price"] = price_option_frame(
            df,
            r=r,
            q=q,
            heston_params=heston_params,
            rate_curve=rate_curve,
            pricing_limit=pricing_limit,
            Ns=Ns,
            Nv=Nv,
            Nt=Nt,
        )

    if "model_price" in df.columns:
        df["model_iv"] = df.apply(_implied_vol_for_row, axis=1, args=("model_price", r, q))
        model_greeks = df.apply(_greeks_from_iv, axis=1, args=("model_iv", "model", r, q))
        df = pd.concat([df, model_greeks], axis=1)
        df["model_abs_delta"] = df["model_delta"].abs()
        df["price_error"] = df["model_price"] - df["mid_price"]
        df["iv_error"] = df["model_iv"] - df["market_iv"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["relative_price_error"] = df["price_error"] / df["mid_price"]
        df["abs_iv_error"] = df["iv_error"].abs()
        df["mispricing_score"] = df["abs_iv_error"] * (1.0 + df["liquidity_score"] / 100.0)
        df["mispricing_bias"] = np.where(
            df["iv_error"] > 0,
            "buy",
            np.where(df["iv_error"] < 0, "sell", "hold"),
        )
    else:
        for name in ("model_price", "model_iv", "price_error", "iv_error", "relative_price_error", "abs_iv_error", "mispricing_score"):
            if name not in df.columns:
                df[name] = np.nan
        for name in GREEK_NAMES:
            df[f"model_{name}"] = np.nan
        df["model_abs_delta"] = np.nan
        df["mispricing_bias"] = "hold"

    df = df.replace([np.inf, -np.inf], np.nan)
    return df
