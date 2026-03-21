from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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

st.subheader("Current Dataset Preview")
preview_columns = [
    "contract_id",
    "ticker",
    "type",
    "maturity",
    "strike",
    "spot",
    "mid_price",
    "market_iv",
    "model_iv",
    "market_delta",
    "market_gamma",
    "market_vega",
    "mispricing_bias",
    "mispricing_score",
    "liquidity_score",
]
available_columns = [column for column in preview_columns if column in analytics_df.columns]
st.dataframe(analytics_df[available_columns], use_container_width=True, hide_index=True)

st.markdown(
    """
Use the pages on the left to inspect the option chain, volatility surfaces,
construct simple strategies, inspect model-vs-market mispricing, and check basic risk limits.
"""
)
