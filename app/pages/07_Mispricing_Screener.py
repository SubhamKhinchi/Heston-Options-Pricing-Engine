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
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Mispricing Screener", layout="wide")
st.title("Step 7 — Mispricing Screener")
st.caption(
    "Six-lens analysis: ranked opportunities, put-call parity violations, "
    "systematic bias, volatility risk premium, strategy-level edge, and Greeks filter."
)

ss = st.session_state

# ── Data source ───────────────────────────────────────────────────────────────
if "analytics_df" in ss:
    df_full: pd.DataFrame = ss["analytics_df"].copy()
    has_model = "model_iv" in df_full.columns and df_full["model_iv"].notna().any()
    has_model_price = "model_price" in df_full.columns and df_full["model_price"].notna().any()
elif "filtered_df" in ss and not ss["filtered_df"].empty:
    df_full = ss["filtered_df"].copy()
    has_model = False
    has_model_price = False
else:
    st.warning("No data available. Complete at least Step 2 — Filter Options first.")
    st.page_link("pages/02_Filter_Options.py", label="← Go to Filter Options", icon="🔍")
    st.stop()

if not has_model:
    st.warning(
        "Model IV not available — most screener lenses require Heston pricing. "
        "Run **Step 4 — Price Contracts** first."
    )
    st.page_link("pages/04_Price_Contracts.py", label="← Go to Price Contracts", icon="💰")
    st.stop()

# Ensure derived columns
df_full["abs_iv_error"] = df_full["iv_error"].abs() if "iv_error" in df_full.columns else np.nan
if "liquidity_score" not in df_full.columns:
    df_full["liquidity_score"] = 1.0

# ── Global controls ───────────────────────────────────────────────────────────
gc1, gc2, gc3 = st.columns(3)

with gc1:
    if "ticker" in df_full.columns and df_full["ticker"].nunique() > 1:
        sel_ticker = st.selectbox("Ticker", options=sorted(df_full["ticker"].unique()), key="ms_ticker")
        df = df_full[df_full["ticker"] == sel_ticker].copy()
    else:
        sel_ticker = df_full["ticker"].iloc[0] if "ticker" in df_full.columns else "—"
        df = df_full.copy()

with gc2:
    min_iv_err = st.slider(
        "Min |IV error| threshold (vol pts)",
        min_value=0.0, max_value=0.30, value=0.01, step=0.005,
        format="%.3f", key="ms_min_iv",
    )

with gc3:
    min_liq = st.slider(
        "Min liquidity score",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        key="ms_min_liq",
    )

spot = float(df["spot"].iloc[0]) if "spot" in df.columns and not df.empty else 100.0
r = ss.get("fetch_params", {}).get("r", 0.05)
q = ss.get("fetch_params", {}).get("q", 0.0)
if "ticker" in df_full.columns:
    q = ss.get("_div_yields", {}).get(sel_ticker, q)

st.caption(
    f"**{sel_ticker}**  spot ${spot:.2f}  |  r={r*100:.2f}%  q={q*100:.3f}%  |  "
    f"{len(df):,} contracts  |  "
    f"{int(df['abs_iv_error'].notna().sum()):,} with IV error"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏆 Ranked",
    "⚖ Put-Call Parity",
    "🗺 Bias Heatmap",
    "📉 Vol Risk Premium",
    "🎯 Strategy Edge",
    "🔬 Greeks Filter",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — RANKED OPPORTUNITIES
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Ranked Mispricing Opportunities")
    st.caption(
        "Composite score = |IV error| × liquidity score. "
        "Green = model IV > market IV (underpriced → buy). "
        "Red = model IV < market IV (overpriced → sell)."
    )

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        dir_filter = st.selectbox("Direction", ["All", "Buy (underpriced)", "Sell (overpriced)"], key="ms_dir")
    with fc2:
        type_filter = st.selectbox("Option type", ["All", "Calls only", "Puts only"], key="ms_type")
    with fc3:
        top_n = st.number_input("Show top N", min_value=5, max_value=200, value=30, step=5, key="ms_topn")
    with fc4:
        sort_by = st.selectbox("Sort by", ["Composite score", "|IV error|", "Price error ($)", "Liquidity score"], key="ms_sort")

    df_ranked = df.dropna(subset=["iv_error", "market_iv", "model_iv"]).copy()
    df_ranked["abs_iv_error"] = df_ranked["iv_error"].abs()
    df_ranked = df_ranked[df_ranked["abs_iv_error"] >= min_iv_err]
    df_ranked = df_ranked[df_ranked["liquidity_score"] >= min_liq]

    if "mispricing_bias" not in df_ranked.columns:
        df_ranked["mispricing_bias"] = np.where(df_ranked["iv_error"] > 0, "buy", "sell")

    if dir_filter == "Buy (underpriced)":
        df_ranked = df_ranked[df_ranked["mispricing_bias"] == "buy"]
    elif dir_filter == "Sell (overpriced)":
        df_ranked = df_ranked[df_ranked["mispricing_bias"] == "sell"]

    if type_filter == "Calls only":
        df_ranked = df_ranked[df_ranked["type"] == "call"]
    elif type_filter == "Puts only":
        df_ranked = df_ranked[df_ranked["type"] == "put"]

    # Composite score
    liq_max = df_ranked["liquidity_score"].max()
    df_ranked["composite_score"] = (
        df_ranked["abs_iv_error"] *
        (df_ranked["liquidity_score"] / liq_max if liq_max > 0 else 1.0)
    )

    sort_col_map = {
        "Composite score": "composite_score",
        "|IV error|": "abs_iv_error",
        "Price error ($)": "price_error" if "price_error" in df_ranked.columns else "abs_iv_error",
        "Liquidity score": "liquidity_score",
    }
    df_ranked = df_ranked.sort_values(sort_col_map[sort_by], ascending=False).head(int(top_n))

    display_cols = [c for c in [
        "type", "maturity", "strike", "moneyness", "mid_price", "model_price",
        "price_error", "market_iv", "model_iv", "iv_error",
        "liquidity_score", "composite_score", "mispricing_bias",
    ] if c in df_ranked.columns]

    def _color_rows(row):
        bias = row.get("mispricing_bias", "")
        if bias == "buy":
            return ["background-color: #1a3a1a"] * len(row)
        if bias == "sell":
            return ["background-color: #3a1a1a"] * len(row)
        return [""] * len(row)

    if df_ranked.empty:
        st.info("No contracts meet the current thresholds.")
    else:
        st.dataframe(
            df_ranked[display_cols].style.apply(_color_rows, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Contracts shown", len(df_ranked))
        m2.metric("Buy signals", int((df_ranked["mispricing_bias"] == "buy").sum()))
        m3.metric("Sell signals", int((df_ranked["mispricing_bias"] == "sell").sum()))
        avg_err = df_ranked["abs_iv_error"].mean()
        m4.metric("Avg |IV error|", f"{avg_err*100:.2f} vol pts")

        csv = df_ranked[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download opportunities CSV", data=csv,
                           file_name="mispricing_opportunities.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PUT-CALL PARITY
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Put-Call Parity Violation Detector")
    st.caption(
        "Parity: C − P = S·e^(−qT) − K·e^(−rT).  "
        "Violation = deviation exceeds the combined bid-ask half-spread — a potential arbitrage."
    )

    if "type" not in df.columns or "strike" not in df.columns:
        st.info("Need call and put data with strike information.")
    else:
        _pcp_cols = [c for c in ["maturity", "strike", "T", "mid_price", "bid", "ask", "model_price"] if c in df.columns]
        calls_pcp = df[df["type"] == "call"][_pcp_cols].copy()
        puts_pcp  = df[df["type"] == "put" ][_pcp_cols].copy()

        merged = calls_pcp.merge(
            puts_pcp,
            on=["maturity", "strike", "T"],
            suffixes=("_call", "_put"),
        )

        if merged.empty:
            st.info("No matching call-put pairs found for the same (maturity, strike).")
        else:
            merged["C_minus_P"] = merged["mid_price_call"] - merged["mid_price_put"]
            merged["parity_rhs"] = spot * np.exp(-q * merged["T"]) - merged["strike"] * np.exp(-r * merged["T"])
            merged["deviation"] = merged["C_minus_P"] - merged["parity_rhs"]
            merged["abs_deviation"] = merged["deviation"].abs()
            if "ask_call" in merged.columns and "bid_call" in merged.columns:
                merged["half_spread_call"] = ((merged["ask_call"] - merged["bid_call"]) / 2).clip(lower=0)
                merged["half_spread_put"]  = ((merged["ask_put"]  - merged["bid_put"])  / 2).clip(lower=0)
                merged["spread_threshold"] = merged["half_spread_call"] + merged["half_spread_put"]
            else:
                merged["spread_threshold"] = 0.05  # fallback: 5-cent threshold
            merged["tradeable"] = merged["abs_deviation"] > merged["spread_threshold"]
            merged["violation"] = merged["abs_deviation"] > 0.01  # basic flag

            pcp_thresh = st.slider("Flag violations above ($)", min_value=0.01, max_value=5.0,
                                   value=0.05, step=0.01, key="ms_pcp_thresh")
            merged["flagged"] = merged["abs_deviation"] > pcp_thresh

            # Chart
            fig_pcp = go.Figure()
            for mat in sorted(merged["maturity"].unique()):
                sub = merged[merged["maturity"] == mat].sort_values("strike")
                fig_pcp.add_trace(go.Scatter(
                    x=sub["strike"], y=sub["deviation"],
                    mode="markers+lines",
                    name=mat,
                    marker=dict(
                        color=["#d62728" if v else "#2ca02c" for v in sub["tradeable"]],
                        size=8,
                    ),
                    hovertemplate="K=%{x:.0f}<br>Deviation: $%{y:.4f}<extra>" + mat + "</extra>",
                ))
            fig_pcp.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_pcp.add_hline(y=pcp_thresh, line_dash="dash", line_color="orange",
                              annotation_text=f"Threshold ${pcp_thresh:.2f}")
            fig_pcp.add_hline(y=-pcp_thresh, line_dash="dash", line_color="orange")
            fig_pcp.update_layout(
                xaxis_title="Strike ($)", yaxis_title="C − P − Parity RHS ($)",
                height=400, legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_pcp, use_container_width=True)

            # Table — sort on full df first, then select display columns
            show_cols = [c for c in [
                "maturity", "strike", "T", "mid_price_call", "mid_price_put",
                "C_minus_P", "parity_rhs", "deviation", "abs_deviation",
                "spread_threshold", "tradeable",
            ] if c in merged.columns]
            disp = merged.sort_values("abs_deviation", ascending=False)[show_cols]

            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Pairs checked", len(merged))
            pc2.metric(f"Violations > ${pcp_thresh:.2f}", int(merged["flagged"].sum()))
            pc3.metric("Tradeable (> spread)", int(merged["tradeable"].sum()))

            with st.expander("All pairs detail", expanded=False):
                st.dataframe(disp, use_container_width=True, hide_index=True)

            flagged = disp[disp["tradeable"] == True].copy() if "tradeable" in disp.columns else pd.DataFrame()
            if not flagged.empty:
                st.subheader("Tradeable violations (deviation > bid-ask spread)")
                st.dataframe(flagged, use_container_width=True, hide_index=True)
            else:
                st.success("No tradeable put-call parity violations found.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SYSTEMATIC BIAS HEATMAP
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Systematic Model Bias Heatmap")
    st.caption(
        "Average IV error (market − model) per moneyness × maturity cell. "
        "Blue = model overprices (sell signal region). Red = model underprices (buy signal region)."
    )

    df_hm = df.dropna(subset=["iv_error", "moneyness", "maturity"]).copy()

    if df_hm.empty:
        st.info("No IV error data available.")
    else:
        hm_col1, hm_col2 = st.columns(2)
        with hm_col1:
            n_bins = st.slider("Moneyness bins", min_value=5, max_value=20, value=10, key="ms_bins")
        with hm_col2:
            hm_metric = st.selectbox(
                "Metric",
                options=["IV error (vol pts %)", "Price error ($)", "Abs IV error"],
                key="ms_hm_metric",
            )

        df_hm["m_bin"] = pd.cut(df_hm["moneyness"], bins=n_bins, precision=2)
        df_hm["m_bin_str"] = df_hm["m_bin"].astype(str)

        if hm_metric == "IV error (vol pts %)":
            val_col, scale = "iv_error", 100.0
            colorscale, zmid = "RdBu_r", 0
        elif hm_metric == "Price error ($)":
            val_col = "price_error" if "price_error" in df_hm.columns else "iv_error"
            scale, colorscale, zmid = 1.0, "RdBu_r", 0
        else:
            val_col, scale = "abs_iv_error", 100.0
            colorscale, zmid = "Reds", None

        pivot = (
            df_hm.pivot_table(index="m_bin_str", columns="maturity",
                               values=val_col, aggfunc="mean") * scale
        )

        if not pivot.empty:
            hm_kwargs = dict(zmid=zmid) if zmid is not None else {}
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale=colorscale,
                colorbar=dict(title=hm_metric),
                hovertemplate="Maturity: %{x}<br>Moneyness: %{y}<br>Value: %{z:.3f}<extra></extra>",
                **hm_kwargs,
            ))
            fig_hm.update_layout(
                xaxis_title="Maturity",
                yaxis_title="Moneyness bin",
                height=420,
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            # Count and avg per cell
            st.subheader("Bias summary by option type")
            bias_cols = st.columns(2)
            for i, opt_t in enumerate(["call", "put"]):
                sub_t = df_hm[df_hm["type"] == opt_t] if "type" in df_hm.columns else df_hm
                if sub_t.empty:
                    continue
                with bias_cols[i]:
                    avg_bias = sub_t["iv_error"].mean() * 100
                    max_under = sub_t["iv_error"].max() * 100
                    max_over  = sub_t["iv_error"].min() * 100
                    st.markdown(f"**{opt_t.capitalize()}s**")
                    st.metric("Avg IV error", f"{avg_bias:+.2f} vol pts")
                    st.metric("Max underpricing", f"{max_under:+.2f} vol pts")
                    st.metric("Max overpricing",  f"{max_over:+.2f} vol pts")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — VOLATILITY RISK PREMIUM
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Volatility Risk Premium (VRP)")
    st.caption(
        "VRP = ATM implied vol − realized vol.  "
        "Positive VRP → market prices more uncertainty than was realised → premium selling opportunity."
    )

    vrp_col1, vrp_col2 = st.columns(2)
    with vrp_col1:
        rv_window = st.selectbox("Realized vol window (trading days)",
                                 options=[10, 21, 30, 42, 63],
                                 index=1, key="ms_rv_window")
    with vrp_col2:
        rv_period = st.selectbox("Historical lookback",
                                 options=["30d", "60d", "90d", "180d"],
                                 index=1, key="ms_rv_period")

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_rv(ticker: str, period: str, window: int) -> tuple[float, pd.Series]:
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period=period)
            if hist.empty or len(hist) < window:
                return float("nan"), pd.Series(dtype=float)
            log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            rv_series = log_ret.rolling(window).std() * np.sqrt(252)
            return float(rv_series.iloc[-1]), rv_series
        except Exception:
            return float("nan"), pd.Series(dtype=float)

    with st.spinner(f"Fetching {rv_window}-day realized vol for {sel_ticker}…"):
        rv_current, rv_series = _fetch_rv(sel_ticker, rv_period, rv_window)

    # ATM IV per expiry
    atm_rows = []
    for mat in sorted(df["maturity"].unique() if "maturity" in df.columns else []):
        mdf = df[df["maturity"] == mat]
        T_val = mdf["T"].iloc[0]
        near_atm = mdf[mdf["moneyness"].between(0.97, 1.03)] if "moneyness" in mdf.columns else mdf
        if near_atm.empty:
            near_atm = mdf.loc[(mdf["moneyness"] - 1.0).abs().nsmallest(2).index] if "moneyness" in mdf.columns else mdf
        atm_iv = near_atm["market_iv"].mean() if "market_iv" in near_atm.columns else np.nan
        atm_rows.append({"maturity": mat, "T": T_val, "atm_iv": atm_iv,
                          "rv": rv_current, "vrp": atm_iv - rv_current})

    vrp_df = pd.DataFrame(atm_rows).dropna(subset=["atm_iv"])

    if pd.isna(rv_current):
        st.warning(f"Could not fetch historical data for {sel_ticker}. Check ticker or internet connection.")
    else:
        v1, v2, v3 = st.columns(3)
        v1.metric(f"{rv_window}-day Realized Vol", f"{rv_current*100:.2f}%")
        v2.metric("Nearest ATM IV",
                  f"{vrp_df['atm_iv'].iloc[0]*100:.2f}%" if not vrp_df.empty else "n/a")
        v3.metric("VRP (nearest expiry)",
                  f"{vrp_df['vrp'].iloc[0]*100:+.2f} vol pts" if not vrp_df.empty else "n/a")

        vrp_l, vrp_r = st.columns(2)

        with vrp_l:
            # VRP by expiry
            fig_vrp = go.Figure()
            if not vrp_df.empty:
                fig_vrp.add_trace(go.Bar(
                    x=vrp_df["maturity"], y=vrp_df["vrp"] * 100,
                    marker_color=["#2ca02c" if v > 0 else "#d62728" for v in vrp_df["vrp"]],
                    hovertemplate="Expiry: %{x}<br>VRP: %{y:.2f} vol pts<extra></extra>",
                    name="VRP",
                ))
                fig_vrp.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_vrp.update_layout(
                    title="VRP = ATM IV − Realized Vol by Expiry",
                    xaxis_title="Expiry", yaxis_title="VRP (vol pts %)",
                    height=360,
                )
            st.plotly_chart(fig_vrp, use_container_width=True)

        with vrp_r:
            # Rolling realized vol history
            if not rv_series.empty:
                fig_rv = go.Figure()
                fig_rv.add_trace(go.Scatter(
                    x=rv_series.index, y=rv_series.values * 100,
                    mode="lines", name=f"{rv_window}d RV",
                    line=dict(color="#4C78A8", width=2),
                ))
                if not vrp_df.empty:
                    for _, row in vrp_df.iterrows():
                        fig_rv.add_hline(
                            y=row["atm_iv"] * 100,
                            line_dash="dot", line_color="#F58518",
                            annotation_text=f"ATM IV {row['maturity']}",
                            annotation_position="right",
                        )
                fig_rv.update_layout(
                    title=f"{rv_window}-day Rolling Realized Vol",
                    xaxis_title="Date", yaxis_title="Annualised Vol (%)",
                    height=360,
                )
                st.plotly_chart(fig_rv, use_container_width=True)

        # VRP table + recommendation
        if not vrp_df.empty:
            st.subheader("VRP Table & Signal")
            vrp_disp = vrp_df.copy()
            vrp_disp["ATM IV"] = (vrp_disp["atm_iv"] * 100).round(3).astype(str) + "%"
            vrp_disp["Realized Vol"] = f"{rv_current*100:.2f}%"
            vrp_disp["VRP"] = (vrp_disp["vrp"] * 100).apply(lambda x: f"{x:+.2f}%")
            vrp_disp["Signal"] = vrp_disp["vrp"].apply(
                lambda v: "Sell premium (IV rich)" if v > 0.02
                else ("Buy vol (IV cheap)" if v < -0.02 else "Neutral")
            )
            st.dataframe(vrp_disp[["maturity", "T", "ATM IV", "Realized Vol", "VRP", "Signal"]],
                         use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — STRATEGY-LEVEL EDGE
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Strategy-Level Edge Screener")
    st.caption(
        "For each expiry, finds common strategy setups and computes the edge: "
        "model value − market cost.  Positive edge = model says strategy is cheap → buy."
    )

    if not has_model_price:
        st.info("Model prices not available. Run Step 4 — Price Contracts to enable strategy edge.")
    elif "type" not in df.columns:
        st.info("Need call and put type column.")
    else:
        strat_rows = []

        for mat in sorted(df["maturity"].unique() if "maturity" in df.columns else []):
            mdf = df[df["maturity"] == mat].copy()
            T_val = mdf["T"].iloc[0] if not mdf.empty else 0
            calls_s = mdf[mdf["type"] == "call"].sort_values("strike")
            puts_s  = mdf[mdf["type"] == "put"].sort_values("strike")

            def _nearest(frame, target):
                if frame.empty:
                    return None
                return frame.loc[(frame["strike"] - target).abs().idxmin()]

            atm_c = _nearest(calls_s, spot)
            atm_p = _nearest(puts_s,  spot)
            otm_c = _nearest(calls_s[calls_s["strike"] > spot * 1.04], spot * 1.05)
            otm_p = _nearest(puts_s[puts_s["strike"]   < spot * 0.96], spot * 0.95)
            wide_c = _nearest(calls_s[calls_s["strike"] > spot * 1.09], spot * 1.10)
            wide_p = _nearest(puts_s[puts_s["strike"]   < spot * 0.91], spot * 0.90)

            def _edge(legs):
                """legs: list of (row, qty) where qty > 0 = long, < 0 = short"""
                market_cost = sum(row["mid_price"]   * qty for row, qty in legs if row is not None)
                model_val   = sum(row["model_price"] * qty for row, qty in legs if row is not None and pd.notna(row.get("model_price")))
                return model_val - market_cost, market_cost, model_val

            def _leg_str(legs):
                parts = []
                for row, qty in legs:
                    if row is None:
                        continue
                    direction = "L" if qty > 0 else "S"
                    parts.append(f"{direction} {row['type'][0].upper()} K={row['strike']:.0f}")
                return " + ".join(parts)

            strategies_here = []

            # Straddle (ATM call + ATM put)
            if atm_c is not None and atm_p is not None:
                legs = [(atm_c, 1), (atm_p, 1)]
                edge, cost, mval = _edge(legs)
                strategies_here.append({
                    "Strategy": "Long Straddle",
                    "Maturity": mat, "T": round(T_val, 3),
                    "Legs": _leg_str(legs),
                    "Market Cost ($)": round(cost * 100, 2),
                    "Model Value ($)": round(mval * 100, 2),
                    "Edge ($)": round(edge * 100, 2),
                    "Edge (%)": round(edge / abs(cost) * 100, 2) if cost != 0 else 0,
                    "Signal": "Buy" if edge > 0 else "Sell",
                })

            # Strangle (OTM call + OTM put)
            if otm_c is not None and otm_p is not None:
                legs = [(otm_c, 1), (otm_p, 1)]
                edge, cost, mval = _edge(legs)
                strategies_here.append({
                    "Strategy": "Long Strangle",
                    "Maturity": mat, "T": round(T_val, 3),
                    "Legs": _leg_str(legs),
                    "Market Cost ($)": round(cost * 100, 2),
                    "Model Value ($)": round(mval * 100, 2),
                    "Edge ($)": round(edge * 100, 2),
                    "Edge (%)": round(edge / abs(cost) * 100, 2) if cost != 0 else 0,
                    "Signal": "Buy" if edge > 0 else "Sell",
                })

            # Bull Call Spread (ATM long + OTM short)
            if atm_c is not None and otm_c is not None and atm_c["strike"] < otm_c["strike"]:
                legs = [(atm_c, 1), (otm_c, -1)]
                edge, cost, mval = _edge(legs)
                strategies_here.append({
                    "Strategy": "Bull Call Spread",
                    "Maturity": mat, "T": round(T_val, 3),
                    "Legs": _leg_str(legs),
                    "Market Cost ($)": round(cost * 100, 2),
                    "Model Value ($)": round(mval * 100, 2),
                    "Edge ($)": round(edge * 100, 2),
                    "Edge (%)": round(edge / abs(cost) * 100, 2) if cost != 0 else 0,
                    "Signal": "Buy" if edge > 0 else "Sell",
                })

            # Bear Put Spread (ATM long + OTM short)
            if atm_p is not None and otm_p is not None and atm_p["strike"] > otm_p["strike"]:
                legs = [(atm_p, 1), (otm_p, -1)]
                edge, cost, mval = _edge(legs)
                strategies_here.append({
                    "Strategy": "Bear Put Spread",
                    "Maturity": mat, "T": round(T_val, 3),
                    "Legs": _leg_str(legs),
                    "Market Cost ($)": round(cost * 100, 2),
                    "Model Value ($)": round(mval * 100, 2),
                    "Edge ($)": round(edge * 100, 2),
                    "Edge (%)": round(edge / abs(cost) * 100, 2) if cost != 0 else 0,
                    "Signal": "Buy" if edge > 0 else "Sell",
                })

            # Iron Condor (short strangle + long wider strangle hedge)
            if all(x is not None for x in [otm_c, otm_p, wide_c, wide_p]):
                legs = [(otm_p, -1), (wide_p, 1), (otm_c, -1), (wide_c, 1)]
                edge, cost, mval = _edge(legs)
                strategies_here.append({
                    "Strategy": "Iron Condor",
                    "Maturity": mat, "T": round(T_val, 3),
                    "Legs": _leg_str(legs),
                    "Market Cost ($)": round(cost * 100, 2),
                    "Model Value ($)": round(mval * 100, 2),
                    "Edge ($)": round(edge * 100, 2),
                    "Edge (%)": round(edge / abs(cost) * 100, 2) if abs(cost) > 0.001 else 0,
                    "Signal": "Buy" if edge > 0 else "Sell",
                })

            strat_rows.extend(strategies_here)

        if not strat_rows:
            st.info("No strategy combinations found. Ensure model prices are available.")
        else:
            strat_df = pd.DataFrame(strat_rows).sort_values("Edge ($)", ascending=False)

            # Filter controls
            se1, se2 = st.columns(2)
            with se1:
                strat_filter = st.multiselect(
                    "Strategy types",
                    options=sorted(strat_df["Strategy"].unique()),
                    default=sorted(strat_df["Strategy"].unique()),
                    key="ms_strat_filter",
                )
            with se2:
                min_edge = st.number_input("Min |Edge| ($)", min_value=0.0, value=0.0,
                                           step=1.0, key="ms_min_edge")

            strat_df_show = strat_df[
                strat_df["Strategy"].isin(strat_filter) &
                (strat_df["Edge ($)"].abs() >= min_edge)
            ]

            # Chart
            fig_se = go.Figure(go.Bar(
                x=strat_df_show["Strategy"] + " | " + strat_df_show["Maturity"],
                y=strat_df_show["Edge ($)"],
                marker_color=["#2ca02c" if v > 0 else "#d62728" for v in strat_df_show["Edge ($)"]],
                hovertemplate="%{x}<br>Edge: $%{y:.2f}<extra></extra>",
            ))
            fig_se.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_se.update_layout(
                xaxis_title="Strategy | Expiry",
                yaxis_title="Edge per contract ($)",
                xaxis_tickangle=-35,
                height=380,
            )
            st.plotly_chart(fig_se, use_container_width=True)

            st.dataframe(strat_df_show, use_container_width=True, hide_index=True)

            csv_s = strat_df_show.to_csv(index=False).encode("utf-8")
            st.download_button("Download strategy edge CSV", data=csv_s,
                               file_name="strategy_edge.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — GREEKS FILTER
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Greeks-Based Filter")
    st.caption(
        "Screen contracts by Greek exposure ranges. "
        "Delta discrepancy = market delta (from market IV) vs model delta (from Heston) — "
        "large gaps signal distributional disagreement."
    )

    gf1, gf2 = st.columns(2)

    with gf1:
        st.markdown("**Delta filter**")
        delta_lo, delta_hi = st.slider(
            "Delta range", min_value=-1.0, max_value=1.0,
            value=(-1.0, 1.0), step=0.05, key="ms_delta_range",
        )
        st.markdown("**Vega filter**")
        vega_lo = st.number_input("Min vega", min_value=0.0, value=0.0,
                                   step=0.01, key="ms_vega_lo")

    with gf2:
        st.markdown("**IV filter**")
        iv_lo, iv_hi = st.slider(
            "Market IV range", min_value=0.0, max_value=3.0,
            value=(0.0, 3.0), step=0.05, key="ms_iv_range",
        )
        st.markdown("**Delta discrepancy**")
        show_disc = st.checkbox("Flag delta discrepancy", value=True, key="ms_disc")
        disc_thresh = st.number_input("Discrepancy threshold", min_value=0.01,
                                      value=0.05, step=0.01, key="ms_disc_thresh")

    df_gf = df.copy()

    # Delta filter
    delta_col = "market_delta" if "market_delta" in df_gf.columns else None
    if delta_col:
        df_gf = df_gf[df_gf[delta_col].between(delta_lo, delta_hi)]

    # Vega filter
    vega_col = "market_vega" if "market_vega" in df_gf.columns else None
    if vega_col:
        df_gf = df_gf[df_gf[vega_col] >= vega_lo]

    # IV filter
    if "market_iv" in df_gf.columns:
        df_gf = df_gf[df_gf["market_iv"].between(iv_lo, iv_hi)]

    # Delta discrepancy
    if show_disc and "market_delta" in df_gf.columns and "model_delta" in df_gf.columns:
        df_gf["delta_discrepancy"] = (df_gf["market_delta"] - df_gf["model_delta"]).abs()
        df_gf["delta_flag"] = df_gf["delta_discrepancy"] > disc_thresh
    else:
        df_gf["delta_flag"] = False

    gf_display = [c for c in [
        "type", "maturity", "strike", "moneyness", "mid_price",
        "market_iv", "model_iv", "iv_error",
        "market_delta", "model_delta", "delta_discrepancy", "delta_flag",
        "market_gamma", "market_vega", "liquidity_score",
    ] if c in df_gf.columns]

    gm1, gm2, gm3 = st.columns(3)
    gm1.metric("Contracts after filter", len(df_gf))
    if "delta_flag" in df_gf.columns:
        gm2.metric("Delta discrepancies", int(df_gf["delta_flag"].sum()))
    if "delta_discrepancy" in df_gf.columns:
        gm3.metric("Avg delta discrepancy", f"{df_gf['delta_discrepancy'].mean():.4f}")

    if df_gf.empty:
        st.info("No contracts match the current Greek filter.")
    else:
        st.dataframe(df_gf[gf_display], use_container_width=True, hide_index=True)

        # Delta discrepancy scatter
        if "market_delta" in df_gf.columns and "model_delta" in df_gf.columns:
            st.subheader("Market Delta vs Model Delta")
            fig_dd = go.Figure()
            for opt_t, sym, clr in [("call", "circle", "#4C78A8"), ("put", "square", "#E45756")]:
                sub_dd = df_gf[df_gf["type"] == opt_t] if "type" in df_gf.columns else df_gf
                if sub_dd.empty:
                    continue
                fig_dd.add_trace(go.Scatter(
                    x=sub_dd["model_delta"],
                    y=sub_dd["market_delta"],
                    mode="markers",
                    name=opt_t,
                    marker=dict(symbol=sym, color=clr, size=7, opacity=0.7),
                    hovertemplate=(
                        "K=%{customdata[0]:.0f} | Mat=%{customdata[1]}<br>"
                        "Model Δ=%{x:.3f}  Market Δ=%{y:.3f}<extra></extra>"
                    ),
                    customdata=list(zip(sub_dd["strike"], sub_dd["maturity"]))
                    if "strike" in sub_dd.columns else None,
                ))

            d_min = min(df_gf["model_delta"].min(), df_gf["market_delta"].min())
            d_max = max(df_gf["model_delta"].max(), df_gf["market_delta"].max())
            fig_dd.add_shape(type="line", x0=d_min, x1=d_max, y0=d_min, y1=d_max,
                             line=dict(dash="dot", color="gray"))
            fig_dd.update_layout(
                xaxis_title="Model Delta (Heston)",
                yaxis_title="Market Delta (from market IV)",
                height=380,
                title="45° line = perfect agreement",
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        csv_g = df_gf[gf_display].to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered contracts CSV", data=csv_g,
                           file_name="greeks_filtered.csv", mime="text/csv")

# ── Navigation ────────────────────────────────────────────────────────────────
st.divider()
col_back, col_fwd = st.columns([1, 1])
with col_back:
    st.page_link("pages/06_Strategy_Lab.py", label="← Back to Strategy Lab", icon="⚗️")
with col_fwd:
    st.page_link("pages/04_Price_Contracts.py", label="Re-price contracts →", icon="💰")
