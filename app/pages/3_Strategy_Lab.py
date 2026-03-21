from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plotly.graph_objects as go
import streamlit as st

from shared import configure_page, load_app_data
from strategies.builders import build_leg_from_row
from strategies.portfolio import summarize_strategy
from strategies.screener import build_relative_value_strategies, rank_mispriced_contracts


configure_page("Strategy Lab")
st.title("Strategy Lab")

try:
    config, raw_df, filtered_df, analytics_df, calibration_meta, calibration_df = load_app_data("strategy")
except Exception as exc:
    st.error(f"Failed to load strategy data: {exc}")
    st.stop()

if analytics_df.empty:
    st.warning("No contracts available for strategy construction.")
    st.stop()

maturities = sorted(analytics_df["maturity"].dropna().astype(str).unique().tolist())
selected_maturity = st.selectbox("Strategy expiry", options=maturities, index=0)
strategy_df = analytics_df[analytics_df["maturity"].astype(str) == selected_maturity].copy()

st.caption("The current strategy lab assumes all legs share one expiry so payoff is well-defined at expiration.")

option_labels = {
    row["contract_id"]: f"{row['contract_id']} | {row['type']} | K={row['strike']} | mid={row['mid_price']:.2f} | delta={row['market_delta']:.3f}"
    for _, row in strategy_df.iterrows()
}

num_legs = st.number_input("Number of legs", min_value=1, max_value=4, value=2, step=1)
legs = []
for idx in range(int(num_legs)):
    col1, col2, col3 = st.columns([5, 2, 2])
    contract_id = col1.selectbox(
        f"Leg {idx + 1} contract",
        options=list(option_labels.keys()),
        format_func=lambda value: option_labels[value],
        key=f"strategy_contract_{idx}",
    )
    action = col2.selectbox(f"Leg {idx + 1} side", options=["Buy", "Sell"], key=f"strategy_action_{idx}")
    quantity = col3.number_input(f"Leg {idx + 1} qty", min_value=1, value=1, step=1, key=f"strategy_qty_{idx}")

    row = strategy_df.loc[strategy_df["contract_id"] == contract_id].iloc[0]
    legs.append(build_leg_from_row(row, action=action, quantity=int(quantity)))

spot = float(strategy_df["spot"].median())
summary = summarize_strategy(legs, spot=spot)
st.session_state["latest_strategy_summary"] = summary
st.session_state["latest_strategy_spot"] = spot

col1, col2, col3, col4 = st.columns(4)
col1.metric("Entry cashflow", f"{summary['entry_cashflow']:.2f}")
col2.metric("Net delta", f"{summary['delta']:.2f}")
col3.metric("Net vega", f"{summary['vega']:.2f}")
col4.metric("Grid max loss", f"{summary['max_loss_on_grid']:.2f}")

col5, col6, col7 = st.columns(3)
col5.metric("Net gamma", f"{summary['gamma']:.2f}")
col6.metric("Net theta", f"{summary['theta']:.2f}")
col7.metric("Break-evens", ", ".join(f"{value:.2f}" for value in summary["break_evens"]) or "None found")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=summary["price_grid"],
        y=summary["payoff"],
        mode="lines",
        name="Expiry PnL",
    )
)
fig.add_hline(y=0.0, line_dash="dash")
fig.add_vline(x=spot, line_dash="dot")
fig.update_layout(
    title="Strategy payoff at expiry",
    xaxis_title="Underlying price",
    yaxis_title="PnL",
    height=600,
)
st.plotly_chart(fig, use_container_width=True)

if "model_iv" in analytics_df.columns and analytics_df["model_iv"].notna().any():
    st.subheader("Mispricing Candidates For This Expiry")
    expiry_candidates = rank_mispriced_contracts(strategy_df, min_abs_iv_error=0.01, top_n=10)
    if expiry_candidates.empty:
        st.info("No strong mispricing candidates on the selected expiry.")
    else:
        st.dataframe(expiry_candidates, use_container_width=True, hide_index=True)

    pair_candidates = build_relative_value_strategies(strategy_df, min_abs_iv_error=0.01, top_n=5)
    if not pair_candidates.empty:
        st.subheader("Relative Value Strategy Ideas")
        st.dataframe(pair_candidates, use_container_width=True, hide_index=True)
