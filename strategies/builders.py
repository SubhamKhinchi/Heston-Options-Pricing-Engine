from __future__ import annotations

import pandas as pd

from strategies.contracts import OptionLeg


def build_leg_from_row(
    row: pd.Series,
    *,
    action: str,
    quantity: int,
    greek_prefix: str = "market",
) -> OptionLeg:
    signed_quantity = int(quantity)
    if action.lower() == "sell":
        signed_quantity *= -1

    return OptionLeg(
        contract_id=str(row.get("contract_id", row.get("contractSymbol", "unknown"))),
        ticker=str(row.get("ticker", "UNKNOWN")),
        option_type=str(row.get("type", "")).lower(),
        maturity=str(row.get("maturity", row.get("T", ""))),
        strike=float(row["strike"]),
        premium=float(row["mid_price"]),
        quantity=signed_quantity,
        delta=float(row.get(f"{greek_prefix}_delta", float("nan"))),
        gamma=float(row.get(f"{greek_prefix}_gamma", float("nan"))),
        vega=float(row.get(f"{greek_prefix}_vega", float("nan"))),
        theta=float(row.get(f"{greek_prefix}_theta", float("nan"))),
        rho=float(row.get(f"{greek_prefix}_rho", float("nan"))),
        implied_vol=float(row.get(f"{greek_prefix}_iv", row.get("market_iv", float("nan")))),
    )

