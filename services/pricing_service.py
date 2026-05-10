from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from analytics.schema import ensure_option_frame
from pricing.european import heston_european_call_option, heston_european_put_option
from pricing.american import (
    american_call_with_dividends,
    american_call_without_dividends,
    american_put_with_dividends,
    american_put_without_dividends,
)
from pricing.heston_pde_american import heston_pde_american


@dataclass(frozen=True)
class HestonParameters:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    @classmethod
    def from_iterable(cls, params: Iterable[float]) -> "HestonParameters":
        v0, kappa, theta, sigma, rho = [float(value) for value in params]
        return cls(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.v0, self.kappa, self.theta, self.sigma, self.rho)


def coerce_heston_parameters(
    params: HestonParameters | Iterable[float],
) -> HestonParameters:
    if isinstance(params, HestonParameters):
        return params
    return HestonParameters.from_iterable(params)


def price_option_row(
    row: pd.Series,
    *,
    r: float,
    q: float,
    heston_params: HestonParameters | Iterable[float],
    Ns: int = 40,
    Nv: int = 20,
    Nt: int = 40,
    M: int = 100,
    N: int = 10000,
    american_method: str = "auto",
) -> float:
    params = coerce_heston_parameters(heston_params)
    option_type = str(row.get("type", "")).lower()
    exercise_style = str(row.get("ExerciseStyle", "american")).lower()
    S0 = float(row["spot"])
    K = float(row["strike"])
    T = float(row["T"])

    if T <= 0:
        return max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)

    if exercise_style == "european":
        if option_type == "call":
            return heston_european_call_option(S0, K, r, T, *params.as_tuple())
        if option_type == "put":
            return heston_european_put_option(S0, K, r, T, *params.as_tuple())
        raise ValueError("option_type must be 'call' or 'put'")

    if option_type == "call" and abs(q) <= 1e-12:
        return american_call_without_dividends(S0, K, r, T, *params.as_tuple())

    if american_method == "lsmc":
        if option_type == "call":
            return american_call_with_dividends(S0, K, r, T, *params.as_tuple(), M, N, q)
        if abs(q) <= 1e-12:
            return american_put_without_dividends(S0, K, r, T, *params.as_tuple(), M, N)
        return american_put_with_dividends(S0, K, r, T, *params.as_tuple(), M, N, q)

    return heston_pde_american(
        S0=S0,
        K=K,
        r=r,
        q=q,
        T=T,
        v0=params.v0,
        kappa=params.kappa,
        theta=params.theta,
        sigma=params.sigma,
        rho=params.rho,
        option_type=option_type,
        Ns=Ns,
        Nv=Nv,
        Nt=Nt,
    )


def prioritize_contracts(options_df: pd.DataFrame, max_contracts: int | None) -> pd.Index:
    if max_contracts is None or len(options_df) <= max_contracts:
        return options_df.index

    ranked = options_df.copy()
    ranked["volume_rank"] = ranked.get("volume", pd.Series(0.0, index=ranked.index)).fillna(0.0)
    ranked["spread_rank"] = ranked.get("rel_spread", pd.Series(1.0, index=ranked.index)).fillna(1.0)
    ranked["atm_rank"] = ranked.get("atm_distance", pd.Series(np.inf, index=ranked.index)).fillna(np.inf)
    ranked = ranked.sort_values(
        ["T", "atm_rank", "spread_rank", "volume_rank"],
        ascending=[True, True, True, False],
    )
    return ranked.head(max_contracts).index


def price_option_frame(
    options_df: pd.DataFrame,
    *,
    r: float,
    q: float,
    heston_params: HestonParameters | Iterable[float],
    pricing_limit: int | None = None,
    Ns: int = 40,
    Nv: int = 20,
    Nt: int = 40,
    M: int = 100,
    N: int = 10000,
    american_method: str = "auto",
) -> pd.Series:
    df = ensure_option_frame(options_df)
    prices = pd.Series(np.nan, index=df.index, dtype=float)

    if df.empty:
        return prices

    selected_index = prioritize_contracts(df[df["T"] > 0], pricing_limit)
    selected = df.loc[selected_index]

    for idx, row in selected.iterrows():
        try:
            prices.loc[idx] = price_option_row(
                row,
                r=r,
                q=q,
                heston_params=heston_params,
                Ns=Ns,
                Nv=Nv,
                Nt=Nt,
                M=M,
                N=N,
                american_method=american_method,
            )
        except Exception:
            prices.loc[idx] = np.nan

    expired = df.index.difference(selected_index)
    for idx in expired:
        row = df.loc[idx]
        if pd.notna(row.get("T")) and row.get("T") <= 0:
            option_type = str(row.get("type", "")).lower()
            prices.loc[idx] = (
                max(float(row["spot"]) - float(row["strike"]), 0.0)
                if option_type == "call"
                else max(float(row["strike"]) - float(row["spot"]), 0.0)
            )

    return prices

