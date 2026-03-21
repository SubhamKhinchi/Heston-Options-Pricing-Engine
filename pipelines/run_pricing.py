from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.analytics_service import build_chain_analytics
from services.market_service import DEFAULT_SAMPLE_PATH, filter_option_chain, load_option_chain, parse_tickers
from services.pricing_service import HestonParameters


def _parse_params(raw_params: str | None) -> HestonParameters | None:
    if not raw_params:
        return None
    values = [float(value.strip()) for value in raw_params.split(",")]
    if len(values) != 5:
        raise ValueError("Expected five comma-separated Heston parameters.")
    return HestonParameters.from_iterable(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an option-chain analytics table.")
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
    parser.add_argument("--option-types", default="call,put")
    parser.add_argument("--heston-params", help="Optional v0,kappa,theta,sigma,rho tuple.")
    parser.add_argument("--pricing-limit", type=int, default=150)
    parser.add_argument("--Ns", type=int, default=40)
    parser.add_argument("--Nv", type=int, default=20)
    parser.add_argument("--Nt", type=int, default=40)
    parser.add_argument("--output", help="Optional CSV output path.")
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
        option_types=[option_type.strip() for option_type in args.option_types.split(",") if option_type.strip()],
        min_volume=args.min_volume,
        min_open_interest=args.min_open_interest,
        max_maturity=args.max_maturity,
        max_contracts=args.max_contracts,
    )

    heston_params = _parse_params(args.heston_params)
    analytics_df = build_chain_analytics(
        filtered_df,
        r=args.risk_free_rate,
        q=args.dividend_yield,
        heston_params=heston_params,
        compute_model_prices=heston_params is not None,
        pricing_limit=args.pricing_limit,
        Ns=args.Ns,
        Nv=args.Nv,
        Nt=args.Nt,
    )

    if args.output:
        analytics_df.to_csv(args.output, index=False)
        print(f"Saved analytics table to {args.output}")
    else:
        columns = [
            column
            for column in (
                "contract_id",
                "ticker",
                "type",
                "maturity",
                "strike",
                "mid_price",
                "market_iv",
                "market_delta",
                "market_gamma",
                "market_vega",
                "model_iv",
                "iv_error",
            )
            if column in analytics_df.columns
        ]
        print(analytics_df[columns].head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
