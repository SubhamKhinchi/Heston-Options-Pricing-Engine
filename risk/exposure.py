from __future__ import annotations


def exposure_snapshot(strategy_summary: dict[str, object]) -> dict[str, float]:
    return {
        "net_delta": float(strategy_summary.get("delta", 0.0)),
        "net_gamma": float(strategy_summary.get("gamma", 0.0)),
        "net_vega": float(strategy_summary.get("vega", 0.0)),
        "net_theta": float(strategy_summary.get("theta", 0.0)),
        "net_rho": float(strategy_summary.get("rho", 0.0)),
        "net_premium_paid": float(strategy_summary.get("net_premium_paid", 0.0)),
        "entry_cashflow": float(strategy_summary.get("entry_cashflow", 0.0)),
        "contract_count": float(strategy_summary.get("contract_count", 0.0)),
        "max_loss_on_grid": float(strategy_summary.get("max_loss_on_grid", 0.0)),
    }

