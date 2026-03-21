from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from strategies.contracts import OptionLeg
from strategies.payoff import estimate_break_even_points, price_grid, strategy_payoff


def aggregate_greeks(legs: Iterable[OptionLeg]) -> dict[str, float]:
    totals = {name: 0.0 for name in ("delta", "gamma", "vega", "theta", "rho")}
    contract_count = 0
    premium_paid = 0.0

    for leg in legs:
        contract_count += abs(leg.quantity)
        premium_paid += leg.quantity * leg.premium * leg.multiplier
        for greek in totals:
            value = getattr(leg, greek)
            if not np.isnan(value):
                totals[greek] += leg.quantity * leg.multiplier * value

    totals["contract_count"] = float(contract_count)
    totals["net_premium_paid"] = float(premium_paid)
    totals["entry_cashflow"] = float(-premium_paid)
    return totals


def summarize_strategy(legs: list[OptionLeg], spot: float) -> dict[str, object]:
    prices = price_grid(spot)
    pnl = strategy_payoff(legs, prices)
    greeks = aggregate_greeks(legs)

    return {
        "price_grid": prices,
        "payoff": pnl,
        "break_evens": estimate_break_even_points(prices, pnl),
        "max_profit_on_grid": float(np.max(pnl)),
        "max_loss_on_grid": float(np.min(pnl)),
        **greeks,
    }

