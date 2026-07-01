"""
Strategy Lab.

Builds multi-leg option strategies from the analytics chain (strategies/*),
charts their payoff/P&L and net greeks, and stashes the strategy summary for the
Risk Dashboard. Upstream: the analytics chain; Downstream: Risk Dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from strategies.builders import build_leg_from_row
from strategies.contracts import OptionLeg
from strategies.payoff import strategy_payoff, estimate_break_even_points, price_grid

st.set_page_config(page_title="Strategy Lab", layout="wide")
st.title("Step 6 — Strategy Lab")
st.caption(
    "Build, inspect, and analyse multi-leg options strategies using live market data. "
    "Visualise P&L profiles, net Greeks, breakevens, and strategy metrics."
)

ss = st.session_state

# ── Data source ───────────────────────────────────────────────────────────────
if "analytics_df" in ss:
    chain: pd.DataFrame = ss["analytics_df"].copy()
    greek_prefix = "market"
    has_model_price = "model_price" in chain.columns and chain["model_price"].notna().any()
elif "filtered_df" in ss and not ss["filtered_df"].empty:
    chain = ss["filtered_df"].copy()
    greek_prefix = "market"
    has_model_price = False
    st.info("Using filtered chain (no model prices). Run Step 4 to add Heston pricing.")
else:
    st.warning("No data available. Complete at least Step 2 — Filter Options first.")
    st.page_link("pages/02_Filter_Options.py", label="← Go to Filter Options", icon="🔍")
    st.stop()

# ── Ticker / expiry pickers ───────────────────────────────────────────────────
hdr1, hdr2, hdr3 = st.columns(3)

with hdr1:
    if "ticker" in chain.columns and chain["ticker"].nunique() > 1:
        sel_ticker = st.selectbox("Ticker", options=sorted(chain["ticker"].unique()), key="sl_ticker")
        chain = chain[chain["ticker"] == sel_ticker].copy()
    else:
        sel_ticker = chain["ticker"].iloc[0] if "ticker" in chain.columns else "—"

spot = float(chain["spot"].iloc[0]) if "spot" in chain.columns and not chain.empty else 100.0

with hdr2:
    maturities = sorted(chain["maturity"].unique()) if "maturity" in chain.columns else []
    sel_mat = st.selectbox("Primary expiry", options=maturities, index=0, key="sl_mat")

with hdr3:
    st.metric("Spot price", f"${spot:.2f}")

# ── Strategy template selector ────────────────────────────────────────────────
st.divider()

STRATEGIES = {
    "Long Call": "Bullish: profit above breakeven, capped downside at premium paid.",
    "Long Put": "Bearish: profit below breakeven, capped downside at premium paid.",
    "Covered Call": "Yield enhancement: long 100 shares + short OTM call. Capped upside.",
    "Protective Put": "Downside hedge: long 100 shares + long put. Floor on losses.",
    "Bull Call Spread": "Bullish, defined risk: long lower strike call + short higher strike call.",
    "Bear Put Spread": "Bearish, defined risk: long higher strike put + short lower strike put.",
    "Long Straddle": "Volatility long: profit from large moves either direction (ATM call + put).",
    "Long Strangle": "Cheaper vol long: OTM call + OTM put — wider breakevens than straddle.",
    "Short Strangle": "Volatility short: sell OTM call + OTM put. Profit if spot stays range-bound.",
    "Iron Condor": "Range-bound premium seller: short OTM strangle + long wider OTM strangle hedge.",
    "Butterfly": "Low-vol play: long 2 ATM + short 2 OTM wings. Max profit at ATM at expiry.",
    "Calendar Spread": "Long back-month / short front-month at same strike.",
    "Custom": "Add your own legs manually from the chain below.",
}

strat_col, desc_col = st.columns([1, 2])
with strat_col:
    strategy = st.selectbox(
        "Strategy template",
        options=list(STRATEGIES.keys()),
        index=0,
        key="sl_strategy",
    )
with desc_col:
    st.info(STRATEGIES[strategy])

# ── Auto-select legs ──────────────────────────────────────────────────────────
mat_chain = chain[chain["maturity"] == sel_mat].copy() if "maturity" in chain.columns else chain.copy()
calls = mat_chain[mat_chain["type"] == "call"].sort_values("strike") if "type" in mat_chain.columns else pd.DataFrame()
puts  = mat_chain[mat_chain["type"] == "put"].sort_values("strike")  if "type" in mat_chain.columns else pd.DataFrame()


def _nearest_strike(df: pd.DataFrame, target: float) -> pd.Series | None:
    if df.empty:
        return None
    idx = (df["strike"] - target).abs().idxmin()
    return df.loc[idx]


def _otm_call(pct: float = 1.05) -> pd.Series | None:
    return _nearest_strike(calls, spot * pct)


def _otm_put(pct: float = 0.95) -> pd.Series | None:
    return _nearest_strike(puts, spot * pct)


def _atm_call() -> pd.Series | None:
    return _nearest_strike(calls, spot)


def _atm_put() -> pd.Series | None:
    return _nearest_strike(puts, spot)


# Build automatic legs for each strategy
AutoLeg = tuple[pd.Series | None, str, int]  # (row, action, qty)

auto_legs: list[AutoLeg] = []
needs_second_expiry = strategy == "Calendar Spread"

if strategy == "Long Call":
    auto_legs = [(_atm_call(), "buy", 1)]
elif strategy == "Long Put":
    auto_legs = [(_atm_put(), "buy", 1)]
elif strategy == "Covered Call":
    auto_legs = [(_otm_call(1.03), "sell", 1)]  # stock leg shown separately
elif strategy == "Protective Put":
    auto_legs = [(_otm_put(0.97), "buy", 1)]    # stock leg shown separately
elif strategy == "Bull Call Spread":
    auto_legs = [(_atm_call(), "buy", 1), (_otm_call(1.05), "sell", 1)]
elif strategy == "Bear Put Spread":
    auto_legs = [(_atm_put(), "buy", 1), (_otm_put(0.95), "sell", 1)]
elif strategy == "Long Straddle":
    auto_legs = [(_atm_call(), "buy", 1), (_atm_put(), "buy", 1)]
elif strategy == "Long Strangle":
    auto_legs = [(_otm_call(1.05), "buy", 1), (_otm_put(0.95), "buy", 1)]
elif strategy == "Short Strangle":
    auto_legs = [(_otm_call(1.05), "sell", 1), (_otm_put(0.95), "sell", 1)]
elif strategy == "Iron Condor":
    auto_legs = [
        (_otm_put(0.90),  "sell", 1),
        (_otm_put(0.85),  "buy",  1),
        (_otm_call(1.10), "sell", 1),
        (_otm_call(1.15), "buy",  1),
    ]
elif strategy == "Butterfly":
    auto_legs = [
        (_atm_call(),    "buy",  1),
        (_otm_call(1.05), "sell", 2),
        (_otm_call(1.10), "buy",  1),
    ]
elif strategy == "Calendar Spread":
    auto_legs = [(_atm_call(), "sell", 1)]  # back-month added below
elif strategy == "Custom":
    auto_legs = []

# ── Leg builder UI ────────────────────────────────────────────────────────────
st.subheader("Legs")

# Stock component notice for covered/protective
if strategy in ("Covered Call", "Protective Put"):
    st.caption(f"Stock position: Long 100 shares of **{sel_ticker}** at ${spot:.2f} assumed.")

# Calendar — second expiry
if needs_second_expiry and len(maturities) > 1:
    back_mat = st.selectbox(
        "Back-month expiry (long leg)",
        options=[m for m in maturities if m != sel_mat],
        index=0,
        key="sl_back_mat",
    )
    back_calls = chain[(chain["maturity"] == back_mat) & (chain["type"] == "call")]
    back_atm = _nearest_strike(back_calls, spot)
    if back_atm is not None:
        auto_legs.append((back_atm, "buy", 1))

# Render editable legs table
leg_rows = []
for i, (row, default_action, default_qty) in enumerate(auto_legs):
    if row is None:
        continue
    lc1, lc2, lc3, lc4, lc5, lc6 = st.columns([1, 1, 1, 1, 1, 1])
    opt_type = str(row.get("type", "call")).upper()
    strike_val = float(row["strike"])
    premium_val = float(row["mid_price"])

    with lc1:
        st.markdown(f"**Leg {i+1}**")
        st.caption(f"{opt_type}  K={strike_val:.0f}")
    with lc2:
        action = st.selectbox(
            "Action", options=["buy", "sell"], index=0 if default_action == "buy" else 1,
            key=f"sl_action_{i}",
        )
    with lc3:
        qty = st.number_input(
            "Qty (contracts)", min_value=1, max_value=100,
            value=default_qty, step=1, key=f"sl_qty_{i}",
        )
    with lc4:
        premium = st.number_input(
            "Premium ($)", min_value=0.001, value=round(premium_val, 3),
            step=0.01, format="%.3f", key=f"sl_prem_{i}",
        )
    with lc5:
        st.caption("Strike")
        st.write(f"${strike_val:.2f}")
    with lc6:
        st.caption("Market IV")
        iv_val = row.get("market_iv", float("nan"))
        st.write(f"{iv_val*100:.1f}%" if pd.notna(iv_val) else "—")

    row_copy = row.copy()
    row_copy["mid_price"] = premium
    leg_rows.append((row_copy, action, int(qty)))

# Custom strategy: let user pick legs from chain
if strategy == "Custom":
    st.markdown("**Add a custom leg from the option chain:**")
    chain_display = mat_chain[
        [c for c in ["type", "strike", "mid_price", "market_iv",
                      "moneyness", "market_delta", "openInterest"] if c in mat_chain.columns]
    ].dropna(subset=["mid_price"]).sort_values(["type", "strike"])

    selected = st.data_editor(
        chain_display.assign(_include=False),
        column_config={
            "_include": st.column_config.CheckboxColumn("Add?", default=False),
        },
        use_container_width=True,
        num_rows="fixed",
        key="sl_custom_picker",
    )
    custom_rows = selected[selected["_include"]].drop(columns=["_include"])
    if not custom_rows.empty:
        cc1, cc2 = st.columns(2)
        with cc1:
            custom_action = st.selectbox("Action for selected", ["buy", "sell"], key="sl_custom_action")
        with cc2:
            custom_qty = st.number_input("Qty", min_value=1, max_value=100, value=1, key="sl_custom_qty")

        for _, row in custom_rows.iterrows():
            leg_rows.append((row, custom_action, int(custom_qty)))

# ── Build OptionLeg objects ────────────────────────────────────────────────────
if not leg_rows:
    st.warning("No legs defined. Select a strategy template or add custom legs.")
    st.stop()

legs: list[OptionLeg] = []
for row, action, qty in leg_rows:
    try:
        leg = build_leg_from_row(row, action=action, quantity=qty, greek_prefix=greek_prefix)
        legs.append(leg)
    except Exception as e:
        st.warning(f"Skipping leg (error: {e})")

if not legs:
    st.warning("Could not build any valid legs.")
    st.stop()

# ── Stock component (for covered/protective) ──────────────────────────────────
stock_pnl = np.zeros(1)  # placeholder — replaced below with grid values

# ── P&L profile ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("P&L Profile at Expiration")

spot_range_pct = st.slider(
    "Spot range (% of current spot)",
    min_value=10, max_value=80, value=35, step=5, key="sl_range",
)
lower = 1 - spot_range_pct / 100
upper = 1 + spot_range_pct / 100
spots = price_grid(spot, lower=lower, upper=upper, points=300)

options_pnl = strategy_payoff(legs, spots) * 100  # per-contract: ×100 shares

# Add stock P&L for covered call / protective put
total_pnl = options_pnl.copy()
if strategy == "Covered Call":
    stock_pnl = (spots - spot) * 100
    total_pnl = options_pnl + stock_pnl
elif strategy == "Protective Put":
    stock_pnl = (spots - spot) * 100
    total_pnl = options_pnl + stock_pnl

breakevens = estimate_break_even_points(spots, total_pnl)

fig_pnl = go.Figure()

# Options-only trace (show separately for covered/protective)
if strategy in ("Covered Call", "Protective Put"):
    fig_pnl.add_trace(go.Scatter(
        x=spots, y=stock_pnl,
        mode="lines", name="Stock P&L",
        line=dict(color="#aaaaaa", dash="dot", width=1),
    ))
    fig_pnl.add_trace(go.Scatter(
        x=spots, y=options_pnl,
        mode="lines", name="Options P&L",
        line=dict(color="#4C78A8", dash="dash", width=1),
    ))

# Total / combined P&L
fig_pnl.add_trace(go.Scatter(
    x=spots, y=total_pnl,
    mode="lines",
    name="Total P&L",
    line=dict(color="#54A24B" if strategy not in ("Short Strangle",) else "#E45756", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(84,162,75,0.08)" if total_pnl.max() > 0 else "rgba(228,87,86,0.08)",
    hovertemplate="Spot: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>",
))

# Zero line
fig_pnl.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)

# Current spot
fig_pnl.add_vline(x=spot, line_dash="dash", line_color="orange",
                  annotation_text=f"Spot ${spot:.2f}", annotation_position="top")

# Breakeven markers
for be in breakevens:
    fig_pnl.add_vline(x=be, line_dash="dot", line_color="#9467bd", line_width=1,
                      annotation_text=f"BE ${be:.2f}", annotation_position="bottom")

fig_pnl.update_layout(
    xaxis_title="Underlying price at expiry ($)",
    yaxis_title="P&L per strategy unit ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.01),
    height=400,
    hovermode="x unified",
    margin=dict(t=20),
)
st.plotly_chart(fig_pnl, use_container_width=True)

# ── Strategy metrics ──────────────────────────────────────────────────────────
st.subheader("Strategy Metrics")

net_debit = sum(
    leg.premium * leg.quantity * leg.multiplier
    for leg in legs
)
if strategy in ("Covered Call", "Protective Put"):
    net_debit += 0  # stock cost tracked separately

max_profit = float(total_pnl.max())
max_loss = float(total_pnl.min())
be_str = ", ".join(f"${b:.2f}" for b in breakevens) if breakevens else "None in range"

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric(
    "Net debit / credit",
    f"{'−' if net_debit > 0 else '+'}${abs(net_debit):.2f}",
    help="Positive = net debit (cost to enter); negative = net credit (premium received).",
)
mc2.metric("Max profit", f"${max_profit:.2f}" if max_profit < 1e6 else "Unlimited")
mc3.metric("Max loss", f"${abs(max_loss):.2f}" if max_loss > -1e6 else "Unlimited")
mc4.metric("Breakeven(s)", be_str)
mc5.metric("Legs", len(legs))

# ── Net Greeks ────────────────────────────────────────────────────────────────
st.subheader("Net Greeks (per strategy unit)")

net_delta = sum(leg.delta * leg.quantity for leg in legs if pd.notna(leg.delta))
net_gamma = sum(leg.gamma * leg.quantity for leg in legs if pd.notna(leg.gamma))
net_vega  = sum(leg.vega  * leg.quantity for leg in legs if pd.notna(leg.vega))
net_theta = sum(leg.theta * leg.quantity for leg in legs if pd.notna(leg.theta))

# Add delta for stock legs
if strategy == "Covered Call":
    net_delta += 100  # long 100 shares
elif strategy == "Protective Put":
    net_delta += 100

gc1, gc2, gc3, gc4 = st.columns(4)
gc1.metric("Net Δ (delta)", f"{net_delta:.4f}" if net_delta != 0 else "n/a",
           help="Sensitivity to $1 spot move.")
gc2.metric("Net Γ (gamma)", f"{net_gamma:.6f}" if net_gamma != 0 else "n/a",
           help="Rate of change of delta.")
gc3.metric("Net ν (vega)",  f"{net_vega:.4f}" if net_vega != 0 else "n/a",
           help="Sensitivity to 1% IV change.")
gc4.metric("Net θ (theta)", f"{net_theta:.4f}" if net_theta != 0 else "n/a",
           help="Daily time decay (P&L per calendar day).")

# ── Legs detail table ─────────────────────────────────────────────────────────
st.subheader("Legs detail")
leg_table_rows = []
for leg in legs:
    leg_table_rows.append({
        "Type": leg.option_type.upper(),
        "Strike": f"${leg.strike:.2f}",
        "Maturity": leg.maturity,
        "Action": "BUY" if leg.quantity > 0 else "SELL",
        "Qty (contracts)": abs(leg.quantity),
        "Premium ($)": f"${leg.premium:.3f}",
        "Cost ($)": f"${leg.premium * leg.quantity * leg.multiplier:.2f}",
        "Market IV": f"{leg.implied_vol*100:.2f}%" if pd.notna(leg.implied_vol) else "—",
        "Delta": f"{leg.delta:.4f}" if pd.notna(leg.delta) else "—",
        "Gamma": f"{leg.gamma:.6f}" if pd.notna(leg.gamma) else "—",
        "Vega": f"{leg.vega:.4f}" if pd.notna(leg.vega) else "—",
    })
st.dataframe(pd.DataFrame(leg_table_rows), use_container_width=True, hide_index=True)

# ── Sensitivity analysis ──────────────────────────────────────────────────────
st.subheader("Sensitivity — P&L vs IV shift")
st.caption(
    "Approximate effect of a parallel IV shift on strategy value "
    "(uses net vega × ΔIV; linear approximation)."
)

iv_shifts = np.linspace(-0.20, 0.20, 41)  # −20% to +20% in vol pts
vega_pnl = net_vega * iv_shifts * 100      # ×100 per contract

fig_vega = go.Figure(go.Scatter(
    x=iv_shifts * 100, y=vega_pnl,
    mode="lines+markers",
    line=dict(color="#4C78A8", width=2),
    marker=dict(size=4),
    hovertemplate="ΔIV: %{x:.1f}%<br>P&L impact: $%{y:.2f}<extra></extra>",
))
fig_vega.add_vline(x=0, line_dash="dot", line_color="gray")
fig_vega.add_hline(y=0, line_dash="dot", line_color="gray")
fig_vega.update_layout(
    xaxis_title="IV shift (vol pts %)",
    yaxis_title="Approx P&L impact ($)",
    height=300,
    margin=dict(t=10),
)
st.plotly_chart(fig_vega, use_container_width=True)

# ── Mispricing view (if model data available) ─────────────────────────────────
if has_model_price:
    st.subheader("Model vs Market Pricing")
    mp_rows = []
    for leg in legs:
        row_match = chain[
            (chain["type"] == leg.option_type) &
            (chain["strike"] == leg.strike) &
            (chain["maturity"] == leg.maturity)
        ]
        if row_match.empty:
            continue
        row_m = row_match.iloc[0]
        model_p = row_m.get("model_price", float("nan"))
        market_p = row_m.get("mid_price", float("nan"))
        mp_rows.append({
            "Type": leg.option_type.upper(),
            "Strike": f"${leg.strike:.2f}",
            "Market Price ($)": f"${market_p:.3f}" if pd.notna(market_p) else "—",
            "Model Price ($)": f"${model_p:.3f}" if pd.notna(model_p) else "—",
            "Edge ($)": f"${(model_p - market_p):.3f}" if pd.notna(model_p) and pd.notna(market_p) else "—",
            "Market IV": f"{row_m.get('market_iv', float('nan'))*100:.2f}%" if pd.notna(row_m.get("market_iv")) else "—",
            "Model IV": f"{row_m.get('model_iv', float('nan'))*100:.2f}%" if pd.notna(row_m.get("model_iv")) else "—",
        })
    if mp_rows:
        st.dataframe(pd.DataFrame(mp_rows), use_container_width=True, hide_index=True)

# ── Navigation ────────────────────────────────────────────────────────────────
st.divider()
col_back, col_fwd = st.columns([1, 1])
with col_back:
    st.page_link("pages/05_Volatility_Surface.py", label="← Back to Volatility Surface", icon="📈")
with col_fwd:
    st.page_link("pages/07_Mispricing_Screener.py", label="Next: Mispricing Screener →", icon="🔍")
