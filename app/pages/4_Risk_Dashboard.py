"""
Risk Dashboard.

Evaluates the latest strategy (from Strategy Lab) against risk limits and a
spot/vol/time scenario table (risk/engine). Upstream: Strategy Lab summary in
session state. The only page that still imports app/shared.configure_page.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st

from risk.engine import evaluate_strategy_risk
from risk.limits import RiskLimits
from shared import configure_page


configure_page("Risk Dashboard")
st.title("Risk Dashboard")

summary = st.session_state.get("latest_strategy_summary")
spot = st.session_state.get("latest_strategy_spot")

if not summary or spot is None:
    st.info("Build a strategy in the Strategy Lab first. The latest strategy summary will appear here.")
    st.stop()

st.sidebar.header("Risk Limits")
limits = RiskLimits(
    max_abs_delta=st.sidebar.number_input("Max |delta|", min_value=100.0, value=1500.0, step=100.0),
    max_abs_gamma=st.sidebar.number_input("Max |gamma|", min_value=10.0, value=250.0, step=10.0),
    max_abs_vega=st.sidebar.number_input("Max |vega|", min_value=100.0, value=4000.0, step=100.0),
    max_premium_paid=st.sidebar.number_input("Max premium paid", min_value=100.0, value=15000.0, step=100.0),
    max_contracts=st.sidebar.number_input("Max contracts", min_value=1.0, value=20.0, step=1.0),
    max_loss_on_grid=st.sidebar.number_input("Max grid loss", min_value=100.0, value=20000.0, step=100.0),
)

report = evaluate_strategy_risk(summary, spot=float(spot), limits=limits)

st.metric("Overall status", report["overall_status"])
st.subheader("Limit Checks")
st.dataframe(report["limits"], use_container_width=True, hide_index=True)

st.subheader("Scenario PnL")
st.dataframe(report["scenarios"], use_container_width=True, hide_index=True)
