"""
Risk engine: combines limit checks and scenario analysis for a strategy.

`evaluate_strategy_risk()` takes an aggregated strategy summary and returns the
limit verdicts (risk/limits) plus a spot/vol/time scenario table (risk/scenarios).

Upstream:   strategies/portfolio.py summaries.
Downstream: app/pages/4_Risk_Dashboard.py.
"""

from __future__ import annotations

from risk.limits import RiskLimits, evaluate_limits
from risk.scenarios import scenario_table


def evaluate_strategy_risk(
    strategy_summary: dict[str, object],
    *,
    spot: float,
    limits: RiskLimits,
) -> dict[str, object]:
    limits_df = evaluate_limits(strategy_summary, limits)
    scenarios_df = scenario_table(strategy_summary, spot=spot)
    overall_status = (
        "reject"
        if (limits_df["status"] == "reject").any()
        else "warn"
        if (limits_df["status"] == "warn").any()
        else "pass"
    )
    return {
        "overall_status": overall_status,
        "limits": limits_df,
        "scenarios": scenarios_df,
    }
