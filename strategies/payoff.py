from __future__ import annotations

from typing import Iterable

import numpy as np

from strategies.contracts import OptionLeg


def price_grid(spot: float, lower: float = 0.5, upper: float = 1.5, points: int = 200) -> np.ndarray:
    return np.linspace(spot * lower, spot * upper, points)


def intrinsic_value(underlying_prices: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(underlying_prices - strike, 0.0)
    return np.maximum(strike - underlying_prices, 0.0)


def leg_payoff(leg: OptionLeg, underlying_prices: np.ndarray) -> np.ndarray:
    intrinsic = intrinsic_value(underlying_prices, leg.strike, leg.option_type)
    return leg.quantity * leg.multiplier * (intrinsic - leg.premium)


def strategy_payoff(legs: Iterable[OptionLeg], underlying_prices: np.ndarray) -> np.ndarray:
    total = np.zeros_like(underlying_prices, dtype=float)
    for leg in legs:
        total += leg_payoff(leg, underlying_prices)
    return total


def estimate_break_even_points(underlying_prices: np.ndarray, pnl: np.ndarray) -> list[float]:
    break_evens: list[float] = []
    for idx in range(1, len(underlying_prices)):
        left_pnl = pnl[idx - 1]
        right_pnl = pnl[idx]
        if left_pnl == 0:
            break_evens.append(float(underlying_prices[idx - 1]))
        elif left_pnl * right_pnl < 0:
            x1 = underlying_prices[idx - 1]
            x2 = underlying_prices[idx]
            y1 = left_pnl
            y2 = right_pnl
            x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
            break_evens.append(float(x_zero))
    return break_evens

