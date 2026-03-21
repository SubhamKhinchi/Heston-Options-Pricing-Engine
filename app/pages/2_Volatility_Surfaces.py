from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st

from shared import configure_page, load_app_data, render_surface_chart


configure_page("Volatility Surfaces")
st.title("Volatility And Greek Surfaces")

try:
    config, raw_df, filtered_df, analytics_df, calibration_meta, calibration_df = load_app_data("surfaces")
except Exception as exc:
    st.error(f"Failed to load surface data: {exc}")
    st.stop()

surface_columns = [
    "market_iv",
    "model_iv",
    "iv_error",
    "mispricing_score",
    "relative_price_error",
    "market_delta",
    "market_gamma",
    "market_vega",
    "market_theta",
    "model_delta",
    "model_gamma",
    "model_vega",
]
surface_columns = [column for column in surface_columns if column in analytics_df.columns and analytics_df[column].notna().any()]

x_axis = st.selectbox(
    "X axis",
    options=["moneyness", "strike", "market_abs_delta", "model_abs_delta"],
    index=0,
)
y_axis = st.selectbox("Y axis", options=["T"], index=0)
z_axis = st.selectbox("Surface metric", options=surface_columns, index=0)

try:
    render_surface_chart(
        analytics_df,
        x_col=x_axis,
        y_col=y_axis,
        z_col=z_axis,
        title=f"{z_axis} surface",
    )
except Exception as exc:
    st.warning(f"Unable to build the selected surface: {exc}")

scatter_columns = [column for column in [x_axis, y_axis, z_axis, "contract_id", "type", "maturity"] if column in analytics_df.columns]
st.subheader("Underlying Surface Points")
st.dataframe(
    analytics_df[scatter_columns].dropna(subset=[x_axis, y_axis, z_axis]),
    use_container_width=True,
    hide_index=True,
)
