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


configure_page("Option Chain")
st.title("Option Chain")

try:
    config, raw_df, filtered_df, analytics_df, calibration_meta, calibration_df = load_app_data("chain")
except Exception as exc:
    st.error(f"Failed to load option chain: {exc}")
    st.stop()

render_chain_summary(raw_df, filtered_df, analytics_df)
render_calibration_panel(calibration_meta, calibration_df)

default_columns = [
    "contract_id",
    "ticker",
    "type",
    "maturity",
    "strike",
    "spot",
    "mid_price",
    "market_iv",
    "market_delta",
    "market_gamma",
    "market_vega",
    "market_theta",
    "market_rho",
    "liquidity_score",
    "model_price",
    "model_iv",
    "iv_error",
    "relative_price_error",
    "mispricing_bias",
    "mispricing_score",
]
available_columns = [column for column in default_columns if column in analytics_df.columns]
selected_columns = st.multiselect(
    "Visible columns",
    options=list(analytics_df.columns),
    default=available_columns,
)

st.dataframe(analytics_df[selected_columns], use_container_width=True, hide_index=True)

if "mispricing_score" in analytics_df.columns and analytics_df["mispricing_score"].notna().any():
    st.subheader("Top Mispriced Contracts")
    top_columns = [
        column
        for column in [
            "contract_id",
            "ticker",
            "type",
            "maturity",
            "strike",
            "mid_price",
            "market_iv",
            "model_iv",
            "iv_error",
            "mispricing_bias",
            "mispricing_score",
        ]
        if column in analytics_df.columns
    ]
    top_df = analytics_df.dropna(subset=["mispricing_score"]).sort_values("mispricing_score", ascending=False).head(20)
    st.dataframe(top_df[top_columns], use_container_width=True, hide_index=True)

csv_bytes = analytics_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download analytics CSV",
    data=csv_bytes,
    file_name="option_chain_analytics.csv",
    mime="text/csv",
)
