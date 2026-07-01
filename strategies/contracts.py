"""
OptionLeg: the atomic building block of a multi-leg strategy.

Frozen dataclass describing one option position (type, strike, quantity, premium,
greeks) with helpers for its payoff and greek contribution. Composed by
strategies/builders.py, strategies/payoff.py, and strategies/portfolio.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptionLeg:
    contract_id: str
    ticker: str
    option_type: str
    maturity: str
    strike: float
    premium: float
    quantity: int
    multiplier: int = 100
    delta: float = np.nan
    gamma: float = np.nan
    vega: float = np.nan
    theta: float = np.nan
    rho: float = np.nan
    implied_vol: float = np.nan

