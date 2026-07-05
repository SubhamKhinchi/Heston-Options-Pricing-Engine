"""
Step 2 — Filter Options.

Applies liquidity / no-arbitrage filters (services/market_service.filter_chain_with_stats)
to the raw chain from Step 1 and stashes the filtered chain in session state.
Upstream: Step 1 (Load Market Data). Downstream: Step 3 (Calibrate Heston).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.market_service import filter_chain_with_stats, parse_tickers

st.set_page_config(page_title="Filter Options", layout="wide")
st.title("Filter Options")
st.caption("Apply liquidity and model-quality filters to the raw option chain.")

ss = st.session_state

# ── Prerequisite check ────────────────────────────────────────────────────────
if "raw_df" not in ss:
    st.warning("No data loaded yet. Go to **Load Market Data** first and pull an option chain.")
    st.page_link("pages/01_Load_Market_Data.py", label="← Go to Load Market Data", icon="📥")
    st.stop()

raw_df: pd.DataFrame = ss["raw_df"]
params: dict = ss.get("fetch_params", {})
div_yields: dict = ss.get("_div_yields", {})
rate_curve: dict = ss.get("rate_curve", {})
r = ss.get("r_scalar", params.get("r", 0.045))

if not rate_curve:
    st.warning(
        f"⚠️ SOFR/OIS rates unavailable — using {r*100:.2f}% flat rate. "
        "Go to Load Market Data and refresh rates to reload the curve."
    )

# Per-ticker status line
tickers_list = parse_tickers(params.get("tickers", "NVDA"))
for tkr in tickers_list:
    tkr_df = raw_df[raw_df["ticker"] == tkr] if "ticker" in raw_df.columns else raw_df
    spot = tkr_df["spot"].iloc[0] if not tkr_df.empty and "spot" in tkr_df.columns else None
    q_val = div_yields.get(tkr, params.get("q", 0.0))
    st.caption(
        f"**{tkr}**: spot {'${:.2f}'.format(spot) if spot else 'n/a'}  |  "
        f"r = {r*100:.3f}%  |  q = {q_val*100:.3f}%  |  "
        f"{len(tkr_df):,} raw contracts"
    )

# ── Auto-detect max maturity from the data ────────────────────────────────────
data_max_T = float(raw_df["T"].max()) if "T" in raw_df.columns else 2.0
data_max_T = round(data_max_T + 0.05, 2)  # slight buffer so no contract is clipped

prev = ss.get("filter_params", {})

# ── Filter inputs ─────────────────────────────────────────────────────────────
st.subheader("Filter parameters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Liquidity filters**")
    spread_limit = st.number_input(
        "Max relative bid-ask spread  (ask−bid)/mid",
        min_value=0.01, max_value=1.0,
        value=prev.get("spread_limit", 0.05),
        step=0.01, format="%.2f",
        help="Contracts with a known spread wider than this are dropped. "
             "Contracts with no live quote (bid=ask=0) are kept — they use lastPrice.",
    )
    abs_spread_floor = st.number_input(
        "Absolute spread rescue floor ($)",
        min_value=0.0, max_value=5.0,
        value=prev.get("abs_spread_floor", 0.10),
        step=0.05, format="%.2f",
        help="Contracts that fail the relative-spread gate are kept anyway if their "
             "absolute bid-ask spread (ask−bid) is at or below this floor and they have "
             "a live bid — at low premiums the relative spread measures tick size, not "
             "illiquidity. Default $0.10 ≈ 2 ticks; set to 0 to disable the rescue.",
    )
    min_volume = st.number_input(
        "Min daily volume",
        min_value=0, value=prev.get("min_volume", 0), step=10,
        help="Keep at 0 outside market hours — yfinance volume is 0 for stale quotes.",
    )
    min_open_interest = st.number_input(
        "Min open interest",
        min_value=0, value=prev.get("min_open_interest", 0), step=100,
        help="Keep at 0 outside market hours — yfinance returns OI=0 for stale quotes.",
    )
    option_types = st.multiselect(
        "Option types",
        options=["call", "put"],
        default=prev.get("option_types", ["call", "put"]),
    )

with col2:
    st.markdown("**Strike / maturity filters**")
    moneyness_lo = st.number_input(
        "Moneyness min  (strike / spot)",
        min_value=0.01, max_value=1.0,
        value=prev.get("moneyness_lo", 0.8),
        step=0.05, format="%.2f",
        help="0.8 = strikes down to 80% of spot (standard near-ATM band).",
    )
    moneyness_hi = st.number_input(
        "Moneyness max  (strike / spot)",
        min_value=1.0, max_value=20.0,
        value=prev.get("moneyness_hi", 1.2),
        step=0.05, format="%.2f",
        help="1.2 = strikes up to 120% of spot (standard near-ATM band).",
    )
    _prev_min_mat = prev.get("min_maturity", 7.0 / 365.0)
    _min_days_default = int(round(_prev_min_mat * 365)) if _prev_min_mat else 0
    min_maturity_days = st.number_input(
        "Min maturity (days)",
        min_value=0, max_value=90,
        value=_min_days_default,
        step=1,
        help="Drop contracts expiring within this many days — near-expiry options are "
             "microstructure-noisy and not reliably priceable by the Fourier model "
             "(default 7 = 1 week).",
    )
    max_maturity = st.number_input(
        "Max maturity (years)",
        min_value=0.01, max_value=float(max(data_max_T, 10.0)),
        value=prev.get("max_maturity", data_max_T),
        step=0.1, format="%.2f",
        help=f"Auto-set to the longest expiry in your data ({data_max_T:.2f}y). Reduce to restrict.",
    )

filter_clicked = st.button("Apply Filters", type="primary")

# ── Filter ────────────────────────────────────────────────────────────────────
if filter_clicked:
    filter_params = dict(
        spread_limit=spread_limit,
        abs_spread_floor=float(abs_spread_floor),
        r=r,
        q=params.get("q", 0.0),
        rate_curve=rate_curve,
        min_volume=int(min_volume),
        min_open_interest=int(min_open_interest),
        moneyness_lo=moneyness_lo,
        moneyness_hi=moneyness_hi,
        min_maturity=(int(min_maturity_days) / 365.0 if int(min_maturity_days) > 0 else None),
        max_maturity=max_maturity,
        option_types=tuple(option_types),
    )

    with st.spinner("Filtering …"):
        filtered_df, filter_stats = filter_chain_with_stats(raw_df, **filter_params)

    ss["filtered_df"] = filtered_df
    ss["filter_stats"] = filter_stats
    ss["filter_params"] = filter_params

    # Invalidate downstream calibration
    ss.pop("calibration", None)

    if filtered_df.empty:
        st.error(
            "All contracts were filtered out. Try relaxing the filters — "
            "especially Min open interest (set to 0 outside market hours) "
            "and Moneyness band (try 0.5–2.0 to include more strikes)."
        )
    else:
        st.success(
            f"{len(filtered_df):,} contracts passed all filters and were normalized to "
            "European-equivalent prices (de-Americanized)."
        )
    st.rerun()

# ── Display results ───────────────────────────────────────────────────────────
if "filtered_df" not in ss:
    st.info("Set your filter parameters above and click **Apply Filters**.")
    st.stop()

filtered_df: pd.DataFrame = ss["filtered_df"]
filter_stats: dict = ss.get("filter_stats", {})
n_raw = len(raw_df)
n_filt = len(filtered_df)

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw contracts", f"{n_raw:,}")
c2.metric("After filtering", f"{n_filt:,}", delta=f"-{n_raw - n_filt:,} dropped")
c3.metric("Expiries", filtered_df["maturity"].nunique() if not filtered_df.empty else 0)
c4.metric(
    "Calls / Puts",
    f"{(filtered_df['type']=='call').sum():,} / {(filtered_df['type']=='put').sum():,}"
    if not filtered_df.empty else "—",
)

# ── Filter insights ───────────────────────────────────────────────────────────
_TYPE_COLORS = {"call": "#2196F3", "put": "#EF5350"}


def _dropped_by_reason_fig(stats: dict) -> go.Figure | None:
    items = [(k, v) for k, v in stats.items() if v]
    if not items:
        return None
    items.sort(key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items],
        orientation="h", marker_color="#EF5350",
        text=[f"{v:,}" for _, v in items], textposition="outside",
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Contracts dropped", yaxis_title="")
    return fig


def _kept_per_expiry_fig(df: pd.DataFrame) -> go.Figure | None:
    if df.empty or "maturity" not in df.columns or "type" not in df.columns:
        return None
    g = df.groupby(["maturity", "type"]).size().reset_index(name="n")
    fig = px.bar(
        g, x="maturity", y="n", color="type", barmode="group",
        color_discrete_map=_TYPE_COLORS,
        labels={"maturity": "Expiry", "n": "Contracts", "type": ""},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      legend_title_text="", xaxis_tickangle=-45)
    return fig


st.subheader("Filter insights")
ins1, ins2 = st.columns(2)
with ins1:
    st.caption("Contracts removed by each filter")
    fig_drop = _dropped_by_reason_fig(filter_stats)
    if fig_drop is not None:
        st.plotly_chart(fig_drop, use_container_width=True)
    else:
        st.info("No contracts were dropped — every raw contract passed.")
with ins2:
    st.caption("Surviving contracts per expiry (calls vs puts)")
    fig_keep = _kept_per_expiry_fig(filtered_df)
    if fig_keep is not None:
        st.plotly_chart(fig_keep, use_container_width=True)
    else:
        st.info("No contracts survived the filters.")

# Filter breakdown
st.subheader("Filter breakdown")
if filter_stats:
    breakdown = pd.DataFrame([
        {
            "Filter": reason,
            "Contracts dropped": count,
            "% of raw": f"{count / n_raw * 100:.1f}%" if n_raw else "—",
        }
        for reason, count in filter_stats.items()
    ])
    st.dataframe(breakdown, use_container_width=True, hide_index=True)
else:
    st.info("No contracts were filtered out — all raw contracts passed every filter.")

# Filtered contracts table
if not filtered_df.empty:
    st.subheader("Filtered contracts")
    st.caption(
        "`euro_mid` is the **European-equivalent** mid (American early-exercise premium "
        "stripped via a binomial tree); `deam_iv` is its implied vol σ*. These are the "
        "single source of the European market side used by calibration and analytics — "
        "always applied, so the model (European Heston) and market are compared like-for-like."
    )
    _COLS = ["ticker", "type", "maturity", "strike", "spot",
             "mid_price", "euro_mid", "deam_iv", "rel_spread", "volume",
             "openInterest", "moneyness", "T"]
    show_cols = [c for c in _COLS if c in filtered_df.columns]
    st.dataframe(filtered_df[show_cols], use_container_width=True, hide_index=True)

# ── Navigation ────────────────────────────────────────────────────────────────
st.divider()
col_back, col_fwd = st.columns([1, 1])
with col_back:
    st.page_link("pages/01_Load_Market_Data.py", label="← Back to Load Market Data", icon="📥")
with col_fwd:
    st.page_link("pages/03_Calibrate_Heston.py", label="Next: Calibrate Heston →", icon="⚙️")
