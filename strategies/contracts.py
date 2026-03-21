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

