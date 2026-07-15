"""
Streamlit landing page for the Heston options analytics app.

Introduces the step-by-step pipeline and links to the page flow:
Load Market Data -> Filter Options -> Calibrate Heston -> Price Contracts ->
Volatility Surface.
Each page under app/pages/ is self-contained. Run with `streamlit run app/Home.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.surfaces import build_surface_grid, select_otm_smile

st.set_page_config(page_title="Options Analytics Home", layout="wide")

st.title("Options Analytics Platform")
st.caption(
    "A step-by-step pipeline for pricing, calibrating, and analysing options "
    "under the Heston stochastic-volatility model."
)

ss = st.session_state

_METHOD_LABELS = {
    "european_proxy": "Characteristic-Function",
}


def _completed_methods(cal: dict) -> dict[str, str]:
    """Return {code: label} for every method that has finished calibrating."""
    return {
        code: _METHOD_LABELS[code]
        for code in _METHOD_LABELS
        if code in cal and "meta" in cal[code]
    }


def _render_cal_params(meta: dict) -> None:
    feller = 2 * meta["kappa"] * meta["theta"] - meta["sigma"] ** 2
    color = "green" if feller > 0 else "red"
    params_df = pd.DataFrame({
        "Param": ["v₀", "κ", "θ̄", "σ", "ρ"],
        "Value": [
            f"{meta['v0']:.4f}",
            f"{meta['kappa']:.4f}",
            f"{meta['theta']:.4f}",
            f"{meta['sigma']:.4f}",
            f"{meta['rho']:.4f}",
        ],
    })
    st.dataframe(params_df, hide_index=True, use_container_width=True)
    _iv = meta.get("iv_rmse")
    _fit = (f"IV-RMSE: {_iv * 100:.2f} vol pts" if pd.notna(_iv)
            else f"Loss: {meta['loss']:.3e}")
    st.markdown(
        f"Feller: :{color}[{feller:+.4f} "
        f"({'✓' if feller > 0 else '✗'})]  "
        f"| {_fit}"
    )


# ── Pipeline status ───────────────────────────────────────────────────────────
st.subheader("Pipeline status")
step_col1, step_col2, step_col3 = st.columns(3)

# Fetch
with step_col1:
    with st.container(border=True):
        st.markdown("**Load Market Data**")
        if "raw_df" in ss:
            p = ss.get("fetch_params", {})
            st.success("Data loaded")
            st.metric("Raw contracts", f"{len(ss['raw_df']):,}")
            st.caption(
                f"Tickers: {p.get('tickers', '?')}  |  "
                f"r={p.get('r', 0)*100:.2f}%  q={p.get('q', 0)*100:.3f}%"
            )
        else:
            st.warning("Not loaded yet")
            st.page_link("pages/01_Load_Market_Data.py", label="Go to Load Market Data →")

# Filter
with step_col2:
    with st.container(border=True):
        st.markdown("**Filter Options**")
        if "filtered_df" in ss:
            fdf: pd.DataFrame = ss["filtered_df"]
            n_raw = len(ss.get("raw_df", pd.DataFrame()))
            n_filt = len(fdf)
            st.success("Filters applied")
            st.metric("Filtered contracts", f"{n_filt:,}",
                      delta=f"-{n_raw - n_filt:,} dropped")
            if not fdf.empty:
                st.caption(
                    f"Expiries: {fdf['maturity'].nunique()}  |  "
                    f"Calls: {(fdf['type']=='call').sum()}  "
                    f"Puts: {(fdf['type']=='put').sum()}"
                )
        else:
            st.warning("Not filtered yet")
            st.page_link("pages/02_Filter_Options.py", label="Go to Filter Options →")

# Calibrate
with step_col3:
    with st.container(border=True):
        st.markdown("**Calibrate Heston**")
        cal: dict = ss.get("calibration", {})
        done = _completed_methods(cal)

        if done:
            st.success(f"Calibrated: {', '.join(done.values())}")

            # Method selector — only shows methods that are actually done
            selected_code = st.selectbox(
                "View results for",
                options=list(done.keys()),
                format_func=lambda c: done[c],
                key="home_cal_selector",
            )
            meta = cal[selected_code]["meta"]
            _render_cal_params(meta)
        else:
            st.warning("Not calibrated yet")
            st.page_link("pages/03_Calibrate_Heston.py", label="Go to Calibrate →")

# ── Latest results tabs ───────────────────────────────────────────────────────
st.divider()

has_raw      = "raw_df" in ss
has_filtered = "filtered_df" in ss and not ss["filtered_df"].empty
has_cal      = bool(_completed_methods(ss.get("calibration", {})))
has_priced   = "analytics_df" in ss and not ss["analytics_df"].empty

if not has_raw:
    st.info(
        "**Start here:** go to **Load Market Data** in the sidebar, "
        "then work through Filter → Calibrate → Price."
    )
else:
    tab_raw, tab_filtered, tab_cal, tab_pricing, tab_surface = st.tabs(
        ["Raw Data", "Filtered Data", "Calibration", "Pricing", "Vol Surface"]
    )

    with tab_raw:
        raw_df: pd.DataFrame = ss["raw_df"]
        st.metric("Total raw contracts", f"{len(raw_df):,}")
        _RAW_COLS = ["ticker", "type", "maturity", "strike", "spot",
                     "bid", "ask", "mid_price", "volume", "openInterest", "rel_spread", "T"]
        show = [c for c in _RAW_COLS if c in raw_df.columns]
        st.dataframe(raw_df[show], use_container_width=True, hide_index=True)

    with tab_filtered:
        if has_filtered:
            fdf = ss["filtered_df"]
            stats = ss.get("filter_stats", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("After filtering", f"{len(fdf):,}")
            c2.metric("Expiries", fdf["maturity"].nunique())
            c3.metric("Dropped", f"{len(ss['raw_df']) - len(fdf):,}")
            if stats:
                breakdown = pd.DataFrame([
                    {"Filter": k, "Dropped": v,
                     "% of raw": f"{v / len(ss['raw_df']) * 100:.1f}%"}
                    for k, v in stats.items()
                ])
                st.dataframe(breakdown, use_container_width=True, hide_index=True)
            _FCOLS = ["ticker", "type", "maturity", "strike", "mid_price",
                      "rel_spread", "volume", "openInterest", "moneyness", "T"]
            show_f = [c for c in _FCOLS if c in fdf.columns]
            st.dataframe(fdf[show_f], use_container_width=True, hide_index=True)
        else:
            st.info("No filtered data yet. Go to **Filter Options**.")

    with tab_cal:
        cal = ss.get("calibration", {})
        done = _completed_methods(cal)

        if not done:
            st.info("No calibration yet. Go to **Calibrate Heston**.")
        else:
            # Method selector
            selected_code = st.selectbox(
                "Select calibration method to inspect",
                options=list(done.keys()),
                format_func=lambda c: done[c],
                key="home_tab_cal_selector",
            )
            meta = cal[selected_code]["meta"]
            cal_df = cal[selected_code].get("df")

            # Full parameter table
            feller = 2 * meta["kappa"] * meta["theta"] - meta["sigma"] ** 2
            color = "green" if feller > 0 else "red"
            full_df = pd.DataFrame({
                "Parameter": ["v₀  (initial variance)", "κ  (mean-reversion speed)",
                              "θ̄  (long-run variance)", "σ  (vol of vol)",
                              "ρ  (spot-vol correlation)"],
                "Value": [
                    f"{meta['v0']:.6f}  →  init vol {meta['v0']**0.5*100:.2f}%",
                    f"{meta['kappa']:.4f}",
                    f"{meta['theta']:.6f}  →  long-run vol {meta['theta']**0.5*100:.2f}%",
                    f"{meta['sigma']:.4f}",
                    f"{meta['rho']:.4f}",
                ],
            })
            st.dataframe(full_df, use_container_width=True, hide_index=True)

            m1, m2, m3 = st.columns(3)
            _iv = meta.get("iv_rmse")
            if pd.notna(_iv):
                m1.metric("IV-RMSE", f"{_iv * 100:.2f} vpts")
            else:
                m1.metric("Loss", f"{meta['loss']:.4e}")
            m2.metric("Contracts used", int(meta["contract_count"]))
            m3.metric("Runtime", f"{meta['runtime_seconds']:.1f}s")

            st.markdown(
                f"**Feller condition** 2κθ−σ²: :{color}[{feller:+.4f}  "
                f"({'satisfied ✓' if feller > 0 else 'violated ✗'})]"
            )

            # Show calibration universe for the selected method
            if cal_df is not None and not cal_df.empty:
                with st.expander("Calibration universe", expanded=False):
                    show_c = [c for c in ["maturity", "type", "strike", "T",
                                           "moneyness", "mid_price", "market_iv"]
                              if c in cal_df.columns]
                    st.dataframe(cal_df[show_c].sort_values(["maturity", "strike"]),
                                 use_container_width=True, hide_index=True)

            # If multiple methods done, show comparison
            if len(done) > 1:
                st.subheader("All calibrated methods")
                rows = []
                for code, label in done.items():
                    m = cal[code]["meta"]
                    f = 2 * m["kappa"] * m["theta"] - m["sigma"] ** 2
                    rows.append({
                        "Method": label,
                        "v₀": round(m["v0"], 6),
                        "κ": round(m["kappa"], 4),
                        "θ̄": round(m["theta"], 6),
                        "σ": round(m["sigma"], 4),
                        "ρ": round(m["rho"], 4),
                        "Feller": round(f, 4),
                        "IV-RMSE (vpts)": (f"{m['iv_rmse']*100:.2f}"
                                           if pd.notna(m.get("iv_rmse"))
                                           else f"loss {m['loss']:.2e}"),
                        "Runtime (s)": round(m["runtime_seconds"], 1),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_pricing:
        if not has_priced:
            st.info("No pricing results yet. Go to **Price Contracts**.")
        else:
            adf: pd.DataFrame = ss["analytics_df"]

            n_total  = len(adf)
            n_priced = int(adf["model_price"].notna().sum()) if "model_price" in adf.columns else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total contracts", f"{n_total:,}")
            m2.metric("Model priced", f"{n_priced:,}")
            if "price_error" in adf.columns:
                avg_abs_err = adf["price_error"].abs().mean()
                m3.metric("Avg |price error|",
                          f"${avg_abs_err:.4f}" if pd.notna(avg_abs_err) else "n/a")
            if "iv_error" in adf.columns:
                avg_iv_err = adf["iv_error"].abs().mean()
                m4.metric("Avg |IV error|",
                          f"{avg_iv_err*100:.3f}%" if pd.notna(avg_iv_err) else "n/a")

            _PCOLS = ["ticker", "type", "maturity", "strike", "spot",
                      "mid_price", "model_price", "price_error",
                      "market_iv", "model_iv", "iv_error",
                      "mispricing_score", "mispricing_bias", "moneyness", "T"]
            show_p = [c for c in _PCOLS if c in adf.columns]
            st.dataframe(adf[show_p], use_container_width=True, hide_index=True)

    with tab_surface:
        if not has_priced or "market_iv" not in ss["analytics_df"].columns \
                or ss["analytics_df"]["market_iv"].isna().all():
            st.info(
                "No vol surface yet — market/model IVs are computed during pricing. "
                "Go to **Price Contracts**."
            )
        else:
            sdf: pd.DataFrame = ss["analytics_df"].copy()

            ctrl_t, ctrl_b = st.columns(2)
            with ctrl_t:
                if "ticker" in sdf.columns and sdf["ticker"].nunique() > 1:
                    surf_ticker = st.selectbox(
                        "Ticker", options=sorted(sdf["ticker"].unique()),
                        key="home_surface_ticker",
                    )
                    sdf = sdf[sdf["ticker"] == surf_ticker]
            with ctrl_b:
                # Same basis selector as the Volatility Surface page; applied to BOTH
                # panels so market and model surfaces are built from the same slice.
                smile_basis = st.selectbox(
                    "Smile basis (both surfaces)",
                    options=["OTM only (clean)", "Calls only", "Puts only", "All quotes (raw)"],
                    index=0,
                    key="home_surface_basis",
                    help="OTM only keeps the out-of-the-money leg per strike (one IV per "
                         "strike, no ITM noise or call/put double-count). 'All quotes' is "
                         "the raw both-legs view including ITM.",
                )
            if smile_basis == "OTM only (clean)":
                sdf = select_otm_smile(sdf)
            elif smile_basis == "Calls only":
                sdf = sdf[sdf["type"] == "call"].copy()
            elif smile_basis == "Puts only":
                sdf = sdf[sdf["type"] == "put"].copy()

            # Forward-moneyness K/F is the ATM-correct axis; fall back to spot K/S.
            if "forward_moneyness" in sdf.columns and sdf["forward_moneyness"].notna().any():
                x_col, x_label = "forward_moneyness", "Forward-moneyness K/F"
            else:
                x_col, x_label = "moneyness", "Moneyness K/S"

            def _home_surface_fig(df_src: pd.DataFrame, iv_col: str, title: str,
                                  colorscale: str,
                                  z_range: tuple[float, float] | None = None) -> go.Figure | None:
                data = df_src[[x_col, "T", iv_col]].dropna()
                if len(data) < 8:
                    return None
                try:
                    grid = build_surface_grid(data, x_col=x_col, y_col="T",
                                              z_col=iv_col, x_points=60, y_points=40)
                except Exception:
                    return None
                fig = go.Figure(go.Surface(
                    x=grid.x_grid, y=grid.y_grid, z=grid.z_grid * 100,
                    colorscale=colorscale,
                    cmin=z_range[0] if z_range else None,
                    cmax=z_range[1] if z_range else None,
                    colorbar=dict(title="IV (%)"),
                    hovertemplate=(f"{x_label}: %{{x:.3f}}<br>"
                                   "T: %{y:.3f}y<br>IV: %{z:.2f}%<extra></extra>"),
                ))
                fig.update_layout(
                    title=title,
                    scene=dict(xaxis_title=x_label, yaxis_title="Maturity (yrs)",
                               zaxis_title="IV (%)",
                               zaxis=dict(range=list(z_range)) if z_range else dict(),
                               camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8))),
                    height=500,
                    margin=dict(l=0, r=0, t=50, b=0),
                )
                return fig

            has_model_iv = "model_iv" in sdf.columns and sdf["model_iv"].notna().any()
            st.caption("Drag to rotate · scroll to zoom · double-click to reset view")
            surf_cols = st.columns(2 if has_model_iv else 1)

            # Shared z/color range so the two panels are visually comparable
            # (independent autoscaling exaggerates the flatter surface).
            _z_cols = ["market_iv"] + (["model_iv"] if has_model_iv else [])
            _z_vals = pd.concat([sdf[c] for c in _z_cols if c in sdf.columns]).dropna() * 100
            if len(_z_vals):
                _pad = max(0.03 * (float(_z_vals.max()) - float(_z_vals.min())), 0.5)
                _shared_z = (float(_z_vals.min()) - _pad, float(_z_vals.max()) + _pad)
            else:
                _shared_z = None

            with surf_cols[0]:
                fig_mkt = _home_surface_fig(sdf, "market_iv", "Market IV Surface", "Viridis",
                                            z_range=_shared_z)
                if fig_mkt:
                    st.plotly_chart(fig_mkt, use_container_width=True)
                else:
                    st.warning("Not enough market IV points to build a surface (need ≥ 8).")

            if has_model_iv:
                with surf_cols[1]:
                    fig_mdl = _home_surface_fig(sdf.dropna(subset=["model_iv"]), "model_iv",
                                                "Heston Model IV Surface", "Plasma",
                                                z_range=_shared_z)
                    if fig_mdl:
                        st.plotly_chart(fig_mdl, use_container_width=True)
                    else:
                        st.warning("Not enough model IV points to build a surface (need ≥ 8).")

            st.page_link("pages/05_Volatility_Surface.py",
                         label="Full surface analysis (smile, term structure, arb checks) →",
                         icon="📈")
