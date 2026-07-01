"""
Mispricing screener: rank contracts by model-vs-market dislocation.

`rank_mispriced_contracts()` filters an enriched analytics table by IV-error
magnitude and liquidity and returns the top candidates (rich/cheap vs the
calibrated Heston model).

Upstream:   analytics/chain_metrics.py (needs model_iv/iv_error columns).
Downstream: app/pages/07_Mispricing_Screener.py.
"""

from __future__ import annotations

import pandas as pd


def rank_mispriced_contracts(
    analytics_df: pd.DataFrame,
    *,
    min_abs_iv_error: float = 0.02,
    top_n: int = 15,
) -> pd.DataFrame:
    df = analytics_df.copy()
    required_columns = {"model_iv", "market_iv", "iv_error", "mispricing_score"}
    if df.empty or not required_columns.issubset(df.columns):
        return df.head(0).copy()

    df = df.dropna(subset=["model_iv", "market_iv", "iv_error", "mispricing_score"]).copy()
    df = df[df["abs_iv_error"] >= min_abs_iv_error].copy()
    if df.empty:
        return df

    df["trade_action"] = df["mispricing_bias"].map(
        {
            "buy": "Buy option: model IV > market IV",
            "sell": "Sell option: model IV < market IV",
        }
    ).fillna("Hold")

    ordered_columns = [
        "contract_id",
        "ticker",
        "type",
        "maturity",
        "strike",
        "mid_price",
        "market_iv",
        "model_iv",
        "iv_error",
        "mispricing_score",
        "trade_action",
    ]
    return df.sort_values("mispricing_score", ascending=False).head(top_n)[ordered_columns]


def build_relative_value_strategies(
    analytics_df: pd.DataFrame,
    *,
    min_abs_iv_error: float = 0.02,
    top_n: int = 10,
) -> pd.DataFrame:
    df = analytics_df.copy()
    required_columns = {"model_iv", "market_iv", "iv_error", "mispricing_score", "maturity", "type", "strike"}
    if df.empty or not required_columns.issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(subset=["model_iv", "market_iv", "iv_error", "mispricing_score"]).copy()
    df = df[df["abs_iv_error"] >= min_abs_iv_error].copy()
    if df.empty:
        return df

    strategies: list[dict[str, object]] = []
    group_cols = ["ticker", "maturity", "type"]
    for group_key, group in df.groupby(group_cols):
        underpriced = group[group["iv_error"] > 0].sort_values("mispricing_score", ascending=False)
        overpriced = group[group["iv_error"] < 0].sort_values("mispricing_score", ascending=False)
        if underpriced.empty or overpriced.empty:
            continue

        long_leg = underpriced.iloc[0]
        short_leg = overpriced.iloc[0]
        if long_leg["contract_id"] == short_leg["contract_id"]:
            continue

        strategy_name = "Relative value pair"
        option_type = str(group_key[2])
        if option_type == "call" and float(long_leg["strike"]) < float(short_leg["strike"]):
            strategy_name = "Bull call spread proxy"
        elif option_type == "put" and float(long_leg["strike"]) > float(short_leg["strike"]):
            strategy_name = "Bear put spread proxy"

        strategies.append(
            {
                "strategy": strategy_name,
                "ticker": group_key[0],
                "maturity": group_key[1],
                "type": option_type,
                "long_contract": long_leg["contract_id"],
                "short_contract": short_leg["contract_id"],
                "long_iv_error": long_leg["iv_error"],
                "short_iv_error": short_leg["iv_error"],
                "combined_score": long_leg["mispricing_score"] + short_leg["mispricing_score"],
            }
        )

    if not strategies:
        return pd.DataFrame()

    return pd.DataFrame(strategies).sort_values("combined_score", ascending=False).head(top_n)
