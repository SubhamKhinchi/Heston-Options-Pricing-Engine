"""
CLI: calibrate Heston parameters on a live option-chain snapshot.

Fetches the chain from Yahoo Finance (services/market_service.load_live_chain),
filters it (filter_chain_with_stats), and calibrates via the single
characteristic-function method (de-Americanize + Levenberg-Marquardt). Prints or
saves the fitted parameters as JSON.

Example:
    python pipelines/run_calibration.py --tickers NVDA --max-expiries 6 \\
        --contracts-per-expiry 6 --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.calibration_service import DEFAULT_INITIAL_GUESS, calibrate_option_chain
from services.market_service import filter_chain_with_stats, load_live_chain, parse_tickers
from services.pricing_service import HestonParameters


def _parse_initial_guess(raw_guess: str | None) -> HestonParameters:
    if not raw_guess:
        return DEFAULT_INITIAL_GUESS
    values = [float(value.strip()) for value in raw_guess.split(",")]
    if len(values) != 5:
        raise ValueError("Expected five comma-separated values for the initial guess.")
    return HestonParameters.from_iterable(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate Heston parameters on a live option-chain snapshot.")
    parser.add_argument("--tickers", default="NVDA")
    parser.add_argument("--spread-limit", type=float, default=0.05)
    parser.add_argument("--risk-free-rate", type=float, default=0.05)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--min-volume", type=int, default=1)
    parser.add_argument("--min-open-interest", type=int, default=0)
    parser.add_argument("--max-maturity", type=float, default=2.0)
    parser.add_argument("--max-contracts", type=int, default=400)
    parser.add_argument("--max-expiries", type=int, default=6)
    parser.add_argument("--contracts-per-expiry", type=int, default=6)
    parser.add_argument("--initial-guess", help="Optional v0,kappa,theta,sigma,rho tuple.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    tickers = parse_tickers(args.tickers)

    raw_df = load_live_chain(tickers)
    filtered_df, _stats = filter_chain_with_stats(
        raw_df,
        spread_limit=args.spread_limit,
        r=args.risk_free_rate,
        q=args.dividend_yield,
        tickers=tickers,
        min_volume=args.min_volume,
        min_open_interest=args.min_open_interest,
        max_maturity=args.max_maturity,
        max_contracts=args.max_contracts,
    )

    calibration_result, calibration_df = calibrate_option_chain(
        filtered_df,
        r=args.risk_free_rate,
        q=args.dividend_yield,
        initial_guess=_parse_initial_guess(args.initial_guess),
        max_expiries=args.max_expiries,
        contracts_per_expiry=args.contracts_per_expiry,
        american_method="european_proxy",
    )

    payload = calibration_result.as_dict()
    if "contract_id" in calibration_df.columns:
        payload["calibration_contracts"] = calibration_df["contract_id"].tolist()

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"Saved calibration result to {args.output}")
    else:
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
