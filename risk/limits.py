"""
Risk limits: thresholds and pass/warn/reject evaluation.

`RiskLimits` holds the configurable thresholds; `evaluate_limits()` compares a
strategy's net-greek exposure (risk/exposure) against them and returns a verdict
per metric. Consumed by risk/engine.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from risk.exposure import exposure_snapshot


@dataclass(frozen=True)
class RiskLimits:
    max_abs_delta: float = 1500.0
    max_abs_gamma: float = 250.0
    max_abs_vega: float = 4000.0
    max_premium_paid: float = 15000.0
    max_contracts: float = 20.0
    max_loss_on_grid: float = 20000.0


def _status_for(value: float, limit: float) -> str:
    abs_value = abs(value)
    if abs_value > limit:
        return "reject"
    if abs_value > 0.8 * limit:
        return "warn"
    return "pass"


def evaluate_limits(strategy_summary: dict[str, object], limits: RiskLimits) -> pd.DataFrame:
    exposure = exposure_snapshot(strategy_summary)
    rows = [
        ("net_delta", exposure["net_delta"], limits.max_abs_delta),
        ("net_gamma", exposure["net_gamma"], limits.max_abs_gamma),
        ("net_vega", exposure["net_vega"], limits.max_abs_vega),
        ("net_premium_paid", exposure["net_premium_paid"], limits.max_premium_paid),
        ("contract_count", exposure["contract_count"], limits.max_contracts),
        ("max_loss_on_grid", exposure["max_loss_on_grid"], limits.max_loss_on_grid),
    ]

    data = []
    for metric, value, limit in rows:
        data.append(
            {
                "metric": metric,
                "value": float(value),
                "limit": float(limit),
                "status": _status_for(float(value), float(limit)),
            }
        )
    return pd.DataFrame(data)

