from __future__ import annotations

from itertools import product

import pandas as pd


def scenario_table(
    strategy_summary: dict[str, object],
    *,
    spot: float,
    spot_shocks: tuple[float, ...] = (-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15),
    iv_shocks: tuple[float, ...] = (-0.10, 0.0, 0.10),
    day_shifts: tuple[int, ...] = (0, 7, 30),
) -> pd.DataFrame:
    delta = float(strategy_summary.get("delta", 0.0))
    gamma = float(strategy_summary.get("gamma", 0.0))
    vega = float(strategy_summary.get("vega", 0.0))
    theta = float(strategy_summary.get("theta", 0.0))

    rows = []
    for spot_shock, iv_shock, days in product(spot_shocks, iv_shocks, day_shifts):
        d_s = spot * spot_shock
        pnl = delta * d_s + 0.5 * gamma * (d_s**2) + vega * iv_shock + theta * (days / 365.0)
        rows.append(
            {
                "spot_shock_pct": spot_shock,
                "iv_shock": iv_shock,
                "days_forward": days,
                "estimated_pnl": pnl,
            }
        )
    return pd.DataFrame(rows)

