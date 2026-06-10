from __future__ import annotations

import numpy as np
import pandas as pd


def _interpolate_rate(rate_curve: dict[float, float], T: float) -> float:
    if not rate_curve:
        return 0.0
    maturities = sorted(rate_curve)
    rates = [rate_curve[t] for t in maturities]
    rT = np.interp(T, maturities, [r * t for r, t in zip(rates, maturities)])
    return float(rT / T) if T > 0 else rates[0]


def _drop(df: pd.DataFrame, mask: pd.Series, reason: str, stats: dict[str, int]) -> pd.DataFrame:
    n = int(mask.sum())
    if n:
        stats[reason] = stats.get(reason, 0) + n
    return df[~mask].copy()


def apply_filters(
    df: pd.DataFrame,
    *,
    spread_limit: float = 0.05,
    r: float = 0.0,
    q: float = 0.0,
    rate_curve: dict[float, float] | None = None,
    min_mid_price: float = 1e-3,
    moneyness_lo: float = 0.1,
    moneyness_hi: float = 5.0,
    tickers: list[str] | None = None,
    option_types: tuple[str, ...] | None = None,
    min_volume: int = 0,
    min_open_interest: int = 0,
    max_maturity: float | None = None,
    keep_positive_time: bool = True,
    max_contracts: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply all option filters in one pass. Returns (filtered_df, stats) where stats maps each filter reason to contracts dropped."""
    stats: dict[str, int] = {}

    # 1. Expired contracts
    if keep_positive_time and "T" in df.columns:
        df = _drop(df, df["T"].fillna(0) <= 0, "Expired (T ≤ 0)", stats)

    # 2. Near-zero mid price
    if "mid_price" in df.columns:
        df = _drop(df, df["mid_price"].fillna(0) <= min_mid_price, f"Mid price ≤ {min_mid_price}", stats)

    # 3. Bid-ask spread too wide — only drop contracts with a KNOWN wide spread.
    #    NaN rel_spread means bid=ask=0 (no live quote, price from lastPrice);
    #    these are kept here and may be dropped later by volume/OI filters.
    if "rel_spread" in df.columns:
        known_wide = df["rel_spread"].notna() & (df["rel_spread"] >= spread_limit)
        df = _drop(df, known_wide, f"Rel. spread ≥ {spread_limit:.0%}", stats)

    # 4. Moneyness outside band (keeps near-ATM contracts only). Prefer the
    #    forward-based measure K/F when available — it is the correct ATM metric;
    #    K/S is used only as a coarse fallback before any forward is known.
    moneyness_col = "forward_moneyness" if "forward_moneyness" in df.columns else "moneyness"
    if moneyness_col in df.columns:
        m = df[moneyness_col]
        out_of_band = (m.fillna(0) < moneyness_lo) | (m.fillna(np.inf) > moneyness_hi)
        df = _drop(df, out_of_band, f"Moneyness outside [{moneyness_lo}, {moneyness_hi}]", stats)

    # 5. No-arbitrage lower bound: e^{-rT}·max(0, F-K) for calls, e^{-rT}·max(0, K-F)
    #    for puts. Uses the implied forward F directly when present (the carried
    #    object); else reconstructs it from the per-row yield as S·e^{(r-q)T}.
    if {"spot", "strike", "T", "type", "mid_price"}.issubset(df.columns):
        if rate_curve:
            r_vec = df["T"].map(lambda T: _interpolate_rate(rate_curve, T))
        else:
            r_vec = pd.Series(r, index=df.index)
        disc = np.exp(-r_vec * df["T"])
        if "forward" in df.columns and df["forward"].notna().any():
            forward_pv = df["forward"].fillna(df["spot"]) * disc      # e^{-rT}·F
        else:
            q_vec = df["q"] if "q" in df.columns else pd.Series(q, index=df.index)
            forward_pv = df["spot"] * np.exp(-q_vec * df["T"])         # = e^{-rT}·F
        disc_k = df["strike"] * disc
        lower = pd.Series(0.0, index=df.index)
        calls = df["type"] == "call"
        puts = df["type"] == "put"
        lower[calls] = np.maximum(0.0, forward_pv[calls] - disc_k[calls])
        lower[puts] = np.maximum(0.0, disc_k[puts] - forward_pv[puts])
        df = _drop(df, df["mid_price"] < lower - 1e-8, "Arbitrage violation", stats)

    # 6. Ticker selection
    if tickers and "ticker" in df.columns:
        df = _drop(df, ~df["ticker"].isin(tickers), "Ticker not in selection", stats)

    # 7. Option type selection
    if option_types and "type" in df.columns:
        normalized_types = {str(t).lower() for t in option_types}
        df = _drop(df, ~df["type"].isin(normalized_types), "Option type excluded", stats)

    # 8. Minimum volume
    if "volume" in df.columns and min_volume > 0:
        df = _drop(df, df["volume"].fillna(0) < min_volume, f"Volume < {min_volume}", stats)

    # 9. Minimum open interest
    if "openInterest" in df.columns and min_open_interest > 0:
        df = _drop(df, df["openInterest"].fillna(0) < min_open_interest, f"Open interest < {min_open_interest}", stats)

    # 10. Max time to maturity
    if max_maturity is not None and "T" in df.columns:
        df = _drop(df, df["T"] > max_maturity, f"Maturity > {max_maturity}y", stats)

    # 11. Hard contract cap (applied last, after sort)
    df = df.sort_values(["ticker", "T", "strike", "type"]).reset_index(drop=True)
    if max_contracts is not None and len(df) > max_contracts:
        stats[f"Truncated to max {max_contracts} contracts"] = len(df) - max_contracts
        df = df.head(max_contracts).copy()

    return df, stats
