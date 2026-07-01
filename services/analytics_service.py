"""
Analytics service: thin orchestration over the analytics engine.

`build_chain_analytics()` enriches a chain (market IV, greeks, liquidity, and
optionally model prices/mispricing). `calibrate_and_build_analytics()` chains a
calibration and an enrichment in one call.

Position in the pipeline: MarketService (+ optional Heston params) ->
[AnalyticsService -> analytics/chain_metrics] -> enriched table for the app pages.
"""

from __future__ import annotations

import pandas as pd

from analytics.chain_metrics import enrich_option_chain
from services.calibration_service import CalibrationResult, calibrate_option_chain
from services.pricing_service import HestonParameters


def build_chain_analytics(
    options_df: pd.DataFrame,
    *,
    r: float = 0.0,
    q: float = 0.0,
    rate_curve: dict | None = None,
    heston_params: HestonParameters | tuple[float, float, float, float, float] | None = None,
    compute_model_prices: bool = False,
    pricing_limit: int | None = None,
    Ns: int = 40,
    Nv: int = 20,
    Nt: int = 40,
) -> pd.DataFrame:
    return enrich_option_chain(
        options_df,
        r=r,
        q=q,
        rate_curve=rate_curve,
        heston_params=heston_params,
        compute_model_prices=compute_model_prices,
        pricing_limit=pricing_limit,
        Ns=Ns,
        Nv=Nv,
        Nt=Nt,
    )


def calibrate_and_build_analytics(
    options_df: pd.DataFrame,
    *,
    r: float = 0.0,
    q: float = 0.0,
    pricing_limit: int | None = None,
    Ns: int = 40,
    Nv: int = 20,
    Nt: int = 40,
    max_expiries: int = 6,
    contracts_per_expiry: int = 6,
) -> tuple[pd.DataFrame, CalibrationResult, pd.DataFrame]:
    calibration_result, calibration_df = calibrate_option_chain(
        options_df,
        r=r,
        q=q,
        Ns=Ns,
        Nv=Nv,
        Nt=Nt,
        max_expiries=max_expiries,
        contracts_per_expiry=contracts_per_expiry,
    )

    analytics_df = build_chain_analytics(
        options_df,
        r=r,
        q=q,
        heston_params=calibration_result.params,
        compute_model_prices=True,
        pricing_limit=pricing_limit,
        Ns=Ns,
        Nv=Nv,
        Nt=Nt,
    )
    return analytics_df, calibration_result, calibration_df
