from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.calibration_service import DEFAULT_INITIAL_GUESS, calibrate_option_chain
from services.market_service import DEFAULT_SAMPLE_PATH, filter_option_chain, load_option_chain, parse_tickers
from services.pricing_service import HestonParameters


def _parse_initial_guess(raw_guess: str | None) -> HestonParameters:
    if not raw_guess:
        return DEFAULT_INITIAL_GUESS
    values = [float(value.strip()) for value in raw_guess.split(",")]
    if len(values) != 5:
        raise ValueError("Expected five comma-separated values for the initial guess.")
    return HestonParameters.from_iterable(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate Heston parameters on an option-chain snapshot.")
    parser.add_argument("--source", choices=("sample", "live"), default="sample")
    parser.add_argument("--sample-path", default=str(DEFAULT_SAMPLE_PATH))
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
    parser.add_argument("--calibration-style", choices=("fast", "full"), default="fast")
    parser.add_argument("--initial-guess", help="Optional v0,kappa,theta,sigma,rho tuple.")
    parser.add_argument("--Ns", type=int, default=40)
    parser.add_argument("--Nv", type=int, default=20)
    parser.add_argument("--Nt", type=int, default=40)
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    raw_df = load_option_chain(
        source=args.source,
        sample_path=args.sample_path,
        tickers=parse_tickers(args.tickers),
        spread_limit=args.spread_limit,
        r=args.risk_free_rate,
        q=args.dividend_yield,
    )
    filtered_df = filter_option_chain(
        raw_df,
        tickers=parse_tickers(args.tickers),
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
        Ns=args.Ns,
        Nv=args.Nv,
        Nt=args.Nt,
        max_expiries=args.max_expiries,
        contracts_per_expiry=args.contracts_per_expiry,
        calibration_style=args.calibration_style,
    )

    payload = calibration_result.as_dict()
    payload["calibration_contracts"] = calibration_df["contract_id"].tolist()

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"Saved calibration result to {args.output}")
    else:
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
