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
import streamlit as st

from services.analytics_service import build_chain_analytics
from services.market_service import parse_tickers
from services.pricing_service import HestonParameters, price_option_frame

st.set_page_config(page_title="Price Contracts", layout="wide")
st.title("Step 4 — Price Contracts")
st.caption(
    "Apply calibrated Heston parameters to price every filtered contract. "
    "Adds model price, model IV, BS Greeks, and mispricing metrics to the chain."
)

ss = st.session_state

_METHOD_LABELS = {
    "european_proxy": "European Proxy",
    "pde":            "PDE Solver",
    "lsmc":           "LSMC Simulation",
}


def _completed_methods(cal: dict) -> dict[str, str]:
    return {
        code: _METHOD_LABELS[code]
        for code in _METHOD_LABELS
        if code in cal and "meta" in cal[code]
    }


# ── Prerequisite checks ───────────────────────────────────────────────────────
if "raw_df" not in ss:
    st.warning("No data loaded. Go to **Fetch Data** first.")
    st.page_link("pages/01_Fetch_Data.py", label="← Go to Fetch Data", icon="📥")
    st.stop()

if "filtered_df" not in ss or ss["filtered_df"].empty:
    st.warning("No filtered data. Go to **Filter Options** first.")
    st.page_link("pages/02_Filter_Options.py", label="← Go to Filter Options", icon="🔍")
    st.stop()

cal: dict = ss.get("calibration", {})
done = _completed_methods(cal)

if not done:
    st.warning("No calibration results yet. Go to **Calibrate Heston** first.")
    st.page_link("pages/03_Calibrate_Heston.py", label="← Go to Calibrate Heston", icon="⚙️")
    st.stop()

filtered_df: pd.DataFrame = ss["filtered_df"]
params: dict = ss.get("fetch_params", {})
div_yields: dict = ss.get("_div_yields", {})
rate_curve: dict = ss.get("rate_curve", {})
r = ss.get("r_scalar", params.get("r", 0.045))
q = params.get("q", 0.0)

if not rate_curve:
    st.warning(
        f"⚠️ SOFR/OIS rates unavailable — using {r*100:.2f}% flat rate. "
        "Go to Fetch Data and refresh rates to reload the curve."
    )

# Per-ticker caption
for tkr in parse_tickers(params.get("tickers", "NVDA")):
    tkr_df = filtered_df[filtered_df["ticker"] == tkr] if "ticker" in filtered_df.columns else filtered_df
    spot = tkr_df["spot"].iloc[0] if not tkr_df.empty and "spot" in tkr_df.columns else None
    q_val = div_yields.get(tkr, q)
    st.caption(
        f"**{tkr}**: spot {'${:.2f}'.format(spot) if spot else 'n/a'}  |  "
        f"r = {r*100:.3f}%  |  q = {q_val*100:.3f}%  |  "
        f"{len(tkr_df):,} filtered contracts"
    )

# ── Inputs ────────────────────────────────────────────────────────────────────
st.subheader("Pricing inputs")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Calibration parameters**")
    cal_code = st.selectbox(
        "Use parameters from",
        options=list(done.keys()),
        format_func=lambda c: done[c],
        help="Selects which calibration run's Heston parameters are used for pricing.",
    )
    selected_meta = cal[cal_code]["meta"]
    hp = HestonParameters(
        v0=selected_meta["v0"],
        kappa=selected_meta["kappa"],
        theta=selected_meta["theta"],
        sigma=selected_meta["sigma"],
        rho=selected_meta["rho"],
    )

    # Show selected params as a compact table
    feller = 2 * hp.kappa * hp.theta - hp.sigma ** 2
    p_df = pd.DataFrame({
        "Param": ["v₀", "κ", "θ̄", "σ", "ρ", "Feller"],
        "Value": [
            f"{hp.v0:.6f}",
            f"{hp.kappa:.4f}",
            f"{hp.theta:.6f}",
            f"{hp.sigma:.4f}",
            f"{hp.rho:.4f}",
            f"{feller:+.4f} ({'✓' if feller > 0 else '✗'})",
        ],
    })
    st.dataframe(p_df, hide_index=True, use_container_width=True)

with col_right:
    st.markdown("**Pricing method for American options**")
    pricing_method = st.selectbox(
        "American-option pricing method",
        options=["european_proxy", "pde", "lsmc"],
        format_func=lambda c: _METHOD_LABELS[c],
        index=0,
        help=(
            "European Proxy: fastest — treats American as European. "
            "PDE: accurate finite-difference grid. "
            "LSMC: Monte Carlo, slowest."
        ),
        key="pricing_method_select",
    )

    pricing_limit = st.number_input(
        "Pricing limit (max contracts)",
        min_value=10, max_value=5000,
        value=min(500, len(filtered_df)),
        step=50,
        help=(
            "Contracts are sorted by ATM distance + spread + volume; "
            "the most liquid are priced first. "
            "Set to total filtered count to price everything."
        ),
    )

    # Method-specific inputs
    if pricing_method == "pde":
        st.markdown("**PDE grid settings**")
        g1, g2, g3 = st.columns(3)
        Ns = g1.number_input("Ns (stock steps)", min_value=5, max_value=200, value=40, step=5, key="p_Ns")
        Nv = g2.number_input("Nv (var steps)",   min_value=5, max_value=200, value=20, step=5, key="p_Nv")
        Nt = g3.number_input("Nt (time steps)",  min_value=5, max_value=200, value=40, step=5, key="p_Nt")
        M, N_lsmc = 100, 10000  # unused for PDE

    elif pricing_method == "lsmc":
        st.markdown("**LSMC settings**")
        l1, l2 = st.columns(2)
        M      = l1.number_input("Time steps (M)",   min_value=10, max_value=500,   value=50,   step=10, key="p_M")
        N_lsmc = l2.number_input("Paths (N)",        min_value=100, max_value=50000, value=1000, step=100, key="p_N")
        Ns, Nv, Nt = 40, 20, 40  # unused for LSMC

    else:  # european_proxy
        Ns, Nv, Nt, M, N_lsmc = 40, 20, 40, 100, 10000

price_clicked = st.button(
    f"Price Contracts  ({len(filtered_df):,} in chain, limit {int(pricing_limit)})",
    type="primary",
)

# ── Price ─────────────────────────────────────────────────────────────────────
if price_clicked:
    progress = st.progress(0, text="Starting pricing …")

    with st.spinner(
        f"Pricing with {_METHOD_LABELS[pricing_method]} — "
        f"up to {int(pricing_limit)} contracts …"
    ):
        try:
            # Step 1: compute model prices via price_option_frame (controls american_method)
            progress.progress(10, text="Computing Heston model prices …")
            model_prices: pd.Series = price_option_frame(
                filtered_df,
                r=r,
                q=q,
                rate_curve=rate_curve,
                heston_params=hp,
                pricing_limit=int(pricing_limit),
                Ns=int(Ns),
                Nv=int(Nv),
                Nt=int(Nt),
                M=int(M),
                N=int(N_lsmc),
                american_method=pricing_method,
            )
            progress.progress(60, text="Computing market IV, Greeks, mispricing …")

            # Step 2: inject model_price into df, then enrich (market IV, Greeks, errors)
            df_with_prices = filtered_df.copy()
            df_with_prices["model_price"] = model_prices

            analytics_df = build_chain_analytics(
                df_with_prices,
                r=r,
                q=q,
                rate_curve=rate_curve,
                compute_model_prices=False,  # already in df
            )
            progress.progress(100, text="Done.")

            ss["analytics_df"] = analytics_df
            ss["pricing_params"] = {
                "cal_method": cal_code,
                "pricing_method": pricing_method,
                "pricing_limit": int(pricing_limit),
            }

            n_priced = int(model_prices.notna().sum())
            st.success(
                f"Priced {n_priced:,} of {len(filtered_df):,} contracts "
                f"using **{done[cal_code]}** params + **{_METHOD_LABELS[pricing_method]}** pricing."
            )
        except Exception as exc:
            st.error(f"Pricing failed: {exc}")
            progress.empty()
            st.stop()

    st.rerun()

# ── Display results ───────────────────────────────────────────────────────────
if "analytics_df" not in ss:
    st.info(
        "Configure inputs above and click **Price Contracts** to run pricing. "
        "This may take a while for large contract sets or slow pricing methods."
    )
    st.stop()

analytics_df: pd.DataFrame = ss["analytics_df"]
pp: dict = ss.get("pricing_params", {})

st.caption(
    f"Priced with: **{done.get(pp.get('cal_method', cal_code), '?')} params**  |  "
    f"Method: **{_METHOD_LABELS.get(pp.get('pricing_method', pricing_method), '?')}**  |  "
    f"Limit: {pp.get('pricing_limit', '?')}"
)

# ── Summary metrics ───────────────────────────────────────────────────────────
n_total   = len(analytics_df)
n_priced  = int(analytics_df["model_price"].notna().sum()) if "model_price" in analytics_df.columns else 0
n_nan     = n_total - n_priced

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total contracts", f"{n_total:,}")
m2.metric("Model priced", f"{n_priced:,}")
m3.metric("Not priced (NaN)", f"{n_nan:,}")

if "price_error" in analytics_df.columns:
    avg_abs_err = analytics_df["price_error"].abs().mean()
    m4.metric("Avg |price error|", f"${avg_abs_err:.4f}" if pd.notna(avg_abs_err) else "n/a")
if "iv_error" in analytics_df.columns:
    avg_iv_err = analytics_df["iv_error"].abs().mean()
    m5.metric("Avg |IV error|", f"{avg_iv_err*100:.3f}%" if pd.notna(avg_iv_err) else "n/a")

# ── Enriched contract table ───────────────────────────────────────────────────
st.subheader("Priced option chain")

_DISPLAY_COLS = [
    "ticker", "type", "maturity", "strike", "spot",
    "mid_price", "model_price", "price_error", "relative_price_error",
    "market_iv", "model_iv", "iv_error",
    "market_delta", "market_gamma", "market_vega",
    "model_delta",  "model_gamma",  "model_vega",
    "liquidity_score", "mispricing_score", "mispricing_bias",
    "moneyness", "T",
]
show_cols = [c for c in _DISPLAY_COLS if c in analytics_df.columns]

# Column selector
selected_cols = st.multiselect(
    "Visible columns",
    options=show_cols,
    default=show_cols,
    key="pricing_col_selector",
)

# Highlight rows with large pricing errors
def _highlight_error(row: pd.Series):
    err = row.get("price_error", 0)
    if pd.isna(err):
        return ["background-color: #3a3a3a"] * len(row)
    if abs(err) > 1.0:
        return ["background-color: #4a1a1a"] * len(row)
    return [""] * len(row)

if selected_cols:
    st.dataframe(
        analytics_df[selected_cols],
        use_container_width=True,
        hide_index=True,
    )

# ── Download ──────────────────────────────────────────────────────────────────
st.divider()
csv = analytics_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download pricing results (CSV)",
    data=csv,
    file_name="heston_pricing_results.csv",
    mime="text/csv",
)

# ── Navigation ────────────────────────────────────────────────────────────────
st.divider()
col_back, col_fwd = st.columns([1, 1])
with col_back:
    st.page_link("pages/03_Calibrate_Heston.py", label="← Back to Calibrate Heston", icon="⚙️")
with col_fwd:
    if "analytics_df" in ss:
        st.page_link("pages/05_Volatility_Surface.py", label="Next: Volatility Surface →", icon="📈")
