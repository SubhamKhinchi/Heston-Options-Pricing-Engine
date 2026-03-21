from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st

from shared import configure_page, load_app_data, render_calibration_panel, render_chain_summary
from strategies.screener import build_relative_value_strategies, rank_mispriced_contracts


configure_page("Mispricing Screener")
st.title("Mispricing Screener")
st.caption("Compare market IV against model IV and surface trade candidates from the current option chain.")

try:
    config, raw_df, filtered_df, analytics_df, calibration_meta, calibration_df = load_app_data("mispricing")
except Exception as exc:
    st.error(f"Failed to load screener data: {exc}")
    st.stop()

render_chain_summary(raw_df, filtered_df, analytics_df)
render_calibration_panel(calibration_meta, calibration_df)

if "model_iv" not in analytics_df.columns or not analytics_df["model_iv"].notna().any():
    st.info("Run Heston calibration or provide manual parameters to generate model IV and mispricing signals.")
    st.stop()

min_abs_iv_error = st.slider("Minimum |IV error|", min_value=0.0, max_value=0.25, value=0.02, step=0.005)
top_n = st.slider("Top contracts", min_value=5, max_value=50, value=15, step=5)

ranked = rank_mispriced_contracts(analytics_df, min_abs_iv_error=min_abs_iv_error, top_n=top_n)
st.subheader("Ranked Single-Leg Opportunities")
if ranked.empty:
    st.info("No contracts met the current mispricing threshold.")
else:
    st.dataframe(ranked, use_container_width=True, hide_index=True)

strategies_df = build_relative_value_strategies(analytics_df, min_abs_iv_error=min_abs_iv_error, top_n=10)
st.subheader("Suggested Relative Value Strategies")
if strategies_df.empty:
    st.info("No relative value pairs met the current threshold.")
else:
    st.dataframe(strategies_df, use_container_width=True, hide_index=True)
