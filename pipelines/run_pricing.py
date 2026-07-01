"""
CLI: build an enriched option-chain analytics table from live data.

Fetches the chain from Yahoo Finance (services/market_service.load_live_chain),
filters it, and runs the analytics enrichment (market IV, greeks, liquidity, and —
when --heston-params is given — model prices / IV error). Prints a preview or
writes the full table to CSV.

Example:
    python pipelines/run_pricing.py --tickers NVDA --max-contracts 100 \\
        --heston-params 0.04,2.0,0.04,0.5,-0.7 --output analytics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.analytics_service import build_chain_analytics
from services.market_service import filter_chain_with_stats, load_live_chain, parse_tickers
from services.pricing_service import HestonParameters


def _parse_params(raw_params: str | None) -> HestonParameters | None:
    if not raw_params:
        return None
    values = [float(value.strip()) for value in raw_params.split(",")]
    if len(values) != 5:
        raise ValueError("Expected five comma-separated Heston parameters.")
    return HestonParameters.from_iterable(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an option-chain analytics table from live data.")
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
    parser.add_argument("--output", help="Optional CSV output path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    tickers = parse_tickers(args.tickers)
    option_types = tuple(t.strip() for t in args.option_types.split(",") if t.strip())

    raw_df = load_live_chain(tickers)
    filtered_df, _stats = filter_chain_with_stats(
        raw_df,
        spread_limit=args.spread_limit,
        r=args.risk_free_rate,
        q=args.dividend_yield,
        tickers=tickers,
        option_types=option_types,
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
