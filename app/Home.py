from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import streamlit as st

from shared import configure_page, load_app_data, render_calibration_panel, render_chain_summary


configure_page("Options Analytics Home")
st.title("Options Analytics Platform")
st.caption(
    "Use the sidebar to refresh live option data, calibrate Heston once, store the parameters, and update pricing, volatility, and mispricing views."
)

try:
    config, raw_df, filtered_df, analytics_df, calibration_meta, calibration_df = load_app_data("home")
except Exception as exc:
    st.error(f"Failed to load analytics data: {exc}")
    st.stop()

render_chain_summary(raw_df, filtered_df, analytics_df)
render_calibration_panel(calibration_meta, calibration_df)

# ── Two-panel data explorer ──────────────────────────────────────────────────
tab_raw, tab_filtered = st.tabs(["Raw Contracts", "Filtered Contracts"])

_RAW_COLS = [
    "ticker", "type", "maturity", "strike", "spot",
    "bid", "ask", "mid_price", "volume", "openInterest", "rel_spread", "T",
]

with tab_raw:
    st.metric("Total raw contracts", f"{len(raw_df):,}")
    available = [c for c in _RAW_COLS if c in raw_df.columns]
    st.dataframe(raw_df[available], use_container_width=True, hide_index=True)

with tab_filtered:
    n_raw = len(raw_df)
    n_filtered = len(filtered_df)
    n_dropped = n_raw - n_filtered

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Raw contracts", f"{n_raw:,}")
    col_b.metric("After filtering", f"{n_filtered:,}")
    col_c.metric("Total dropped", f"{n_dropped:,}")

    st.markdown("#### Filter breakdown")
    filter_stats: dict[str, int] = st.session_state.get("_filter_stats", {})

    if filter_stats:
        breakdown = pd.DataFrame(
            [
                {
                    "Filter reason": reason,
                    "Contracts dropped": count,
                    "% of raw": f"{count / n_raw * 100:.1f}%" if n_raw else "—",
                }
                for reason, count in filter_stats.items()
            ]
        )
        st.dataframe(breakdown, use_container_width=True, hide_index=True)
    else:
        st.info("No contracts were filtered out — all raw contracts passed every filter.")

    st.markdown("#### Filtered contracts")
    _FILTERED_COLS = [
        "contract_id", "ticker", "type", "maturity", "strike", "spot",
        "mid_price", "market_iv", "model_iv",
        "market_delta", "market_gamma", "market_vega",
        "mispricing_bias", "mispricing_score", "liquidity_score",
    ]
    available_filtered = [c for c in _FILTERED_COLS if c in analytics_df.columns]
    st.dataframe(analytics_df[available_filtered], use_container_width=True, hide_index=True)

st.markdown(
    """
Use the pages on the left to inspect the option chain, volatility surfaces,
construct simple strategies, inspect model-vs-market mispricing, and check basic risk limits.
"""
)
