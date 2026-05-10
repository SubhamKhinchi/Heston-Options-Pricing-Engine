from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st

from shared import configure_page, load_app_data, render_surface_chart


configure_page("Vol Surface Explorer")
st.title("Volatility Surface Explorer")
st.caption("Standalone compatibility view backed by the new analytics and app service layer.")

try:
    config, raw_df, filtered_df, analytics_df, calibration_meta, calibration_df = load_app_data("legacy_surface")
except Exception as exc:
    st.error(f"Failed to load surface data: {exc}")
    st.stop()

surface_columns = [
    column
    for column in ("market_iv", "model_iv", "iv_error", "market_delta", "market_gamma", "market_vega")
    if column in analytics_df.columns and analytics_df[column].notna().any()
]

x_col = st.selectbox("X axis", options=["moneyness", "strike", "market_abs_delta"], index=0)
z_col = st.selectbox("Surface metric", options=surface_columns, index=0)

try:
    render_surface_chart(
        analytics_df,
        x_col=x_col,
        y_col="T",
        z_col=z_col,
        title=f"{z_col} surface",
    )
except Exception as exc:
    st.warning(f"Unable to build the selected surface: {exc}")
