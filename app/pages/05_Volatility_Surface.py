"""
Step 5 — Volatility Surface.

Builds and contrasts the market IV surface against the calibrated Heston model IV
surface (analytics/surfaces.build_surface_grid). The model-vs-market gap is the
basis for the mispricing view. Upstream: Steps 3-4 (calibration + analytics).
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
import plotly.express as px
import streamlit as st

from analytics.surfaces import build_surface_grid, select_otm_smile
from calibration.implied_vol import implied_volatility
from pricing.european_gl import heston_call_gl, heston_put_gl

st.set_page_config(page_title="Volatility Surface", layout="wide")
st.title("Volatility Surface Analysis")
st.caption(
    "3D IV surface, smile/skew, term structure, forward variance, risk metrics, "
    "Greeks surfaces, and no-arbitrage checks."
)

ss = st.session_state

# ── Data source resolution ────────────────────────────────────────────────────
if "analytics_df" in ss:
    df_full: pd.DataFrame = ss["analytics_df"].copy()
    has_model = "model_iv" in df_full.columns and df_full["model_iv"].notna().any()
    has_greeks = "market_delta" in df_full.columns and df_full["market_delta"].notna().any()
    data_label = "Priced chain"
elif "filtered_df" in ss and not ss["filtered_df"].empty:
    df_full = ss["filtered_df"].copy()
    has_model = False
    has_greeks = False
    data_label = "Filtered chain (no model)"
    st.info(
        "Showing market data only. Run **Price Contracts** to add model IV, "
        "Heston Greeks, and mispricing data."
    )
else:
    st.warning("No data available. Complete at least **Filter Options** first.")
    st.page_link("pages/02_Filter_Options.py", label="← Go to Filter Options", icon="🔍")
    st.stop()

if "market_iv" not in df_full.columns or df_full["market_iv"].isna().all():
    st.error(
        "No implied volatility data found. "
        "Market IV is computed during pricing — run **Price Contracts** first."
    )
    st.page_link("pages/04_Price_Contracts.py", label="← Go to Price Contracts", icon="💰")
    st.stop()

# ── Global controls ───────────────────────────────────────────────────────────
# Option lists were trimmed (Calls/Puts-only basis and spot-moneyness axes removed);
# clear stale widget state from an older session so the selectboxes don't error.
if ss.get("vs_smile_basis") not in (None, "OTM only (clean)", "All quotes (raw)"):
    del ss["vs_smile_basis"]
if ss.get("vs_x_axis") not in (None, "forward_moneyness", "log_forward_moneyness", "strike"):
    del ss["vs_x_axis"]

ctrl1, ctrl2, ctrl3 = st.columns(3)

with ctrl1:
    if "ticker" in df_full.columns and df_full["ticker"].nunique() > 1:
        sel_ticker = st.selectbox(
            "Ticker", options=sorted(df_full["ticker"].unique()), key="vs_ticker"
        )
        df = df_full[df_full["ticker"] == sel_ticker].copy()
    else:
        sel_ticker = df_full["ticker"].iloc[0] if "ticker" in df_full.columns else "—"
        df = df_full.copy()

with ctrl2:
    smile_basis = st.selectbox(
        "Market smile basis",
        options=["OTM only (clean)", "All quotes (raw)"],
        index=0,
        key="vs_smile_basis",
        help="OTM only keeps the liquid out-of-the-money leg per strike (one IV per "
             "strike, no ITM noise or call/put double-count) — the working view. "
             "'All quotes' is the raw both-legs view including ITM: use it to spot "
             "bad quotes, not to analyse the smile.",
    )

with ctrl3:
    x_axis_mode = st.selectbox(
        "X-axis (smile / surface)",
        options=["forward_moneyness", "log_forward_moneyness", "strike"],
        format_func=lambda x: {
            "forward_moneyness": "Forward-moneyness  K/F",
            "log_forward_moneyness": "Log forward-moneyness  ln(K/F)",
            "strike": "Strike price",
        }[x],
        index=0,
        key="vs_x_axis",
    )

# Forward-moneyness K/F is the economically-correct ATM measure (K = F is ATM) and
# matches the basis used by calibration / de-Americanization. Fall back to spot K/S
# only when no implied forward was stamped. `m_col` is the moneyness basis used by all
# the ATM / skew / risk-reversal / arbitrage logic below, independent of the x-axis the
# user picks for the smile/surface plots. Computed on the full (both-leg) frame *before*
# the basis selection, so the OTM-leg split below can use K/F.
has_fwd = "forward_moneyness" in df.columns and df["forward_moneyness"].notna().any()
if has_fwd:
    df["forward_moneyness"] = (
        df["forward_moneyness"].where(df["forward_moneyness"] > 0).fillna(df.get("moneyness"))
    )
    m_col = "forward_moneyness"
    m_label = "K/F"
else:
    df["forward_moneyness"] = df.get("moneyness")
    m_col = "moneyness"
    m_label = "K/S"
if "moneyness" in df.columns:
    df["log_moneyness"] = np.log(df["moneyness"].clip(lower=1e-4))
df["log_forward_moneyness"] = np.log(df["forward_moneyness"].clip(lower=1e-4))

# Market smile basis — which contracts represent the smile at each strike.
#  - "OTM only" keeps the liquid OTM leg per strike (one IV/strike, no ITM noise, no
#    call/put double-count) — the clean, recommended representation (see select_otm_smile).
#  - "All quotes (raw)" keeps both legs incl. ITM — a data-QC view.
if smile_basis == "OTM only (clean)":
    df = select_otm_smile(df)

x_col = x_axis_mode if x_axis_mode in df.columns else m_col
df_iv = df.dropna(subset=["market_iv", "T", x_col]).copy()

n_expiries = df_iv["maturity"].nunique() if "maturity" in df_iv.columns else 0
st.caption(
    f"**{sel_ticker}** — {data_label}  |  "
    f"basis: **{smile_basis}**  |  "
    f"{len(df_iv):,} contracts with market IV  |  "
    f"{n_expiries} expiries  |  "
    f"{'Model IV available ✓' if has_model else 'Market IV only'}"
)

if len(df_iv) < 4:
    st.error("Not enough data points to build surfaces. Relax filters or run the pricing step.")
    st.stop()

# ── Tab layout ────────────────────────────────────────────────────────────────
_tabs = ["📈 IV Surface", "📐 Smile & Skew", "📅 Term Structure", "📊 Risk Metrics"]
if has_model:
    _tabs.append("🔄 Market vs Model")
if has_greeks:
    _tabs.append("🏛 Greeks")
_tabs.append("⚖ Arb Checks")

tabs = st.tabs(_tabs)
tidx = {name: i for i, name in enumerate(_tabs)}

x_label = {
    "forward_moneyness": "Forward-moneyness K/F",
    "log_forward_moneyness": "ln(K/F)",
    "moneyness": "Moneyness K/S",
    "log_moneyness": "ln(K/S)",
    "strike": "Strike",
}.get(x_axis_mode, x_axis_mode)

# ATM reference on the x-axis: K/F=1 or K/S=1 (linear), ln=0 (log); no fixed ATM for strike.
_ATM_X = {
    "forward_moneyness": 1.0,
    "moneyness": 1.0,
    "log_forward_moneyness": 0.0,
    "log_moneyness": 0.0,
    "strike": None,
}


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — IV SURFACE  (interactive Plotly 3D — drag to rotate)
# ════════════════════════════════════════════════════════════════════════════
def _surface_fig(df_src: pd.DataFrame, iv_col: str, title: str,
                 colorscale: str = "Viridis",
                 z_range: tuple[float, float] | None = None) -> go.Figure | None:
    data = df_src[[x_col, "T", iv_col]].dropna()
    if len(data) < 8:
        return None
    try:
        grid = build_surface_grid(data, x_col=x_col, y_col="T", z_col=iv_col,
                                  x_points=60, y_points=40)
    except Exception:
        return None

    fig = go.Figure(go.Surface(
        x=grid.x_grid,
        y=grid.y_grid,
        z=grid.z_grid * 100,
        colorscale=colorscale,
        # Shared color limits across panels so the same IV maps to the same color.
        cmin=z_range[0] if z_range else None,
        cmax=z_range[1] if z_range else None,
        colorbar=dict(title="IV (%)"),
        hovertemplate=(
            f"{x_label}: %{{x:.3f}}<br>"
            "T: %{y:.3f}y<br>"
            "IV: %{z:.2f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title="Maturity (yrs)",
            zaxis_title="IV (%)",
            # Shared z-axis so smile/wing amplitudes are visually comparable.
            zaxis=dict(range=list(z_range)) if z_range else dict(),
            camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8)),
        ),
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


with tabs[tidx["📈 IV Surface"]]:
    st.subheader("Implied Volatility Surface")
    st.caption("Drag to rotate · scroll to zoom · double-click to reset view")

    n_cols = 2 if has_model else 1
    surf_cols = st.columns(n_cols)

    # Shared z-axis + color range across market and model panels. Independent
    # auto-scaling visually shrinks whichever surface spans more (the market's
    # put wing) and stretches the flatter one (the model smile), making the two
    # look more alike than they are — the comparison is the entire point here.
    _z_cols = ["market_iv"] + (["model_iv"] if has_model else [])
    _z_vals = pd.concat([df_iv[c] for c in _z_cols if c in df_iv.columns]).dropna() * 100
    if len(_z_vals):
        _pad = max(0.03 * (float(_z_vals.max()) - float(_z_vals.min())), 0.5)
        shared_z = (float(_z_vals.min()) - _pad, float(_z_vals.max()) + _pad)
    else:
        shared_z = None

    with surf_cols[0]:
        fig = _surface_fig(df_iv, "market_iv", "Market IV Surface",
                           colorscale="Viridis", z_range=shared_z)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough market IV points to build surface (need ≥ 8).")

    if has_model and n_cols > 1:
        with surf_cols[1]:
            fig = _surface_fig(
                df_iv.dropna(subset=["model_iv"]), "model_iv",
                "Heston Model IV Surface", colorscale="Plasma", z_range=shared_z,
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough model IV points.")

    with st.expander("Raw surface data", expanded=False):
        show = [c for c in [x_col, "T", "maturity", "strike", "type",
                             "market_iv", "model_iv"] if c in df_iv.columns]
        st.dataframe(
            df_iv[show].sort_values(["T", x_col]),
            use_container_width=True, hide_index=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SMILE & SKEW
# ════════════════════════════════════════════════════════════════════════════
with tabs[tidx["📐 Smile & Skew"]]:
    st.subheader("IV Smile & Skew per Expiry")

    # Calibrated Heston parameters — used to draw a dense model smile curve
    # (desk convention: market quotes are discrete points, the model is a smooth line).
    _cal_meta = None
    if isinstance(ss.get("calibration"), dict):
        _cal_meta = (ss["calibration"].get("european_proxy") or {}).get("meta")
    _heston_p = None
    if _cal_meta and all(k in _cal_meta for k in ("v0", "kappa", "theta", "sigma", "rho")):
        _heston_p = tuple(float(_cal_meta[k]) for k in ("v0", "kappa", "theta", "sigma", "rho"))

    def _model_smile_curve(mdf: pd.DataFrame):
        """Dense Heston IV curve across the expiry's strike range (OTM leg per point).

        Pricing a 60-strike grid through the GL pricer costs milliseconds and shows the
        model's smile between/beyond the quoted strikes instead of a jagged polyline
        through whatever strikes happen to trade. Returns (x, iv) in the current x-axis
        convention, or None when no calibration is in session.
        """
        if _heston_p is None or mdf.empty:
            return None
        row = mdf.iloc[0]
        S, T_val = float(row["spot"]), float(row["T"])
        F = float(row["forward"]) if "forward" in mdf.columns and pd.notna(row["forward"]) else S
        rr_ = float(mdf["r"].median()) if "r" in mdf.columns else 0.045
        qq_ = float(mdf["q"].median()) if "q" in mdf.columns else 0.0
        v0_, ka_, th_, sg_, rh_ = _heston_p
        xs, ivs = [], []
        for K in np.linspace(float(mdf["strike"].min()), float(mdf["strike"].max()), 60):
            is_call = K > F
            try:
                px = (heston_call_gl if is_call else heston_put_gl)(
                    S, K, rr_, T_val, v0_, ka_, th_, sg_, rh_, qq_)
                iv = implied_volatility(heston_model_price=px, S=S, K=K, r=rr_, T=T_val,
                                        option_type="call" if is_call else "put", q=qq_)
            except Exception:
                iv = np.nan
            if iv is not None and np.isfinite(iv) and iv > 0:
                if x_axis_mode == "strike":
                    xs.append(float(K))
                elif x_axis_mode == "log_forward_moneyness":
                    xs.append(float(np.log(K / F)))
                else:
                    xs.append(float(K / F))
                ivs.append(float(iv))
        return (xs, ivs) if len(ivs) >= 5 else None

    maturities = sorted(df_iv["maturity"].unique()) if "maturity" in df_iv.columns else []
    if not maturities:
        st.info("No maturity data available.")
    else:
        sm_left, sm_right = st.columns([1, 3])
        with sm_left:
            sel_mats = st.multiselect(
                "Expiries",
                options=maturities,
                default=maturities[:min(6, len(maturities))],
                key="vs_smile_mats",
            )
            show_model_smile = has_model and st.checkbox(
                "Overlay Heston smile", value=True, key="vs_smile_model"
            )

        with sm_right:
            if not sel_mats:
                st.info("Select at least one expiry.")
            else:
                palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
                fig_sm = go.Figure()

                for i, mat in enumerate(sel_mats):
                    mdf = df_iv[df_iv["maturity"] == mat].sort_values(x_col)
                    clr = palette[i % len(palette)]
                    T_val = mdf["T"].iloc[0] if not mdf.empty else 0

                    # Market quotes as points — connecting them just draws quote noise.
                    fig_sm.add_trace(go.Scatter(
                        x=mdf[x_col],
                        y=mdf["market_iv"] * 100,
                        mode="markers",
                        name=f"{mat} (T={T_val:.2f}y) market",
                        marker=dict(color=clr, size=6),
                        hovertemplate=f"{mat}<br>{x_label}: %{{x:.3f}}<br>Market IV: %{{y:.2f}}%<extra></extra>",
                    ))

                    if show_model_smile:
                        curve = _model_smile_curve(mdf)
                        if curve is not None:
                            fig_sm.add_trace(go.Scatter(
                                x=curve[0], y=np.asarray(curve[1]) * 100,
                                mode="lines",
                                name=f"{mat} Heston",
                                line=dict(color=clr, width=2),
                                hovertemplate=f"{mat}<br>{x_label}: %{{x:.3f}}<br>Heston IV: %{{y:.2f}}%<extra></extra>",
                            ))
                        elif has_model:
                            # No calibration in session — fall back to model IV at quoted strikes.
                            mdf_m = mdf.dropna(subset=["model_iv"])
                            if not mdf_m.empty:
                                fig_sm.add_trace(go.Scatter(
                                    x=mdf_m[x_col], y=mdf_m["model_iv"] * 100,
                                    mode="lines",
                                    name=f"{mat} Heston (at quotes)",
                                    line=dict(color=clr, dash="dash"),
                                    hovertemplate=f"{mat}<br>{x_label}: %{{x:.3f}}<br>Heston IV: %{{y:.2f}}%<extra></extra>",
                                ))

                atm_x = _ATM_X.get(x_axis_mode)
                if atm_x is not None:
                    fig_sm.add_vline(
                        x=atm_x,
                        line=dict(dash="dot", color="gray", width=1),
                        annotation_text="ATM",
                    )

                fig_sm.update_layout(
                    xaxis_title=x_label,
                    yaxis_title="Implied Volatility (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=430,
                    hovermode="x unified",
                )
                st.plotly_chart(fig_sm, use_container_width=True)

        # ── ATM / 25Δ risk reversal / 25Δ butterfly — market vs model ────────────
        st.subheader("Smile Metrics — ATM / 25Δ Risk Reversal / 25Δ Butterfly")
        st.caption(
            "Desk-standard smile decomposition in **delta** buckets — 25Δ marks the same "
            "distributional position at every maturity, which fixed moneyness does not. "
            "RR = 25Δ put IV − 25Δ call IV (equity convention: > 0 = put skew, ↔ ρ). "
            "Fly = ½(25Δ put + 25Δ call) − ATM (smile curvature, ↔ vol-of-vol σ). "
            "Model values use the model's own IV and delta, so the ΔRR/ΔFly columns are "
            "the per-expiry read on how much skew/curvature Heston concedes to the market."
        )

        if "market_delta" not in df_iv.columns or df_iv["market_delta"].isna().all():
            st.info("Delta buckets need Greeks — run **Price Contracts** first.")
        else:
            _mdl_dcol = "model_delta" if "model_delta" in df_iv.columns else "market_delta"

            def _iv_at_delta(sub: pd.DataFrame, iv_col: str, d_col: str,
                             target: float = 0.25) -> float:
                """IV linearly interpolated at |delta| = target; NaN when not bracketed."""
                s = sub.dropna(subset=[iv_col, d_col]).copy()
                if len(s) < 2:
                    return float("nan")
                s["_ad"] = s[d_col].abs()
                s = s[(s["_ad"] > 0.02) & (s["_ad"] < 0.98)].sort_values("_ad").drop_duplicates("_ad")
                if len(s) < 2 or not (s["_ad"].iloc[0] <= target <= s["_ad"].iloc[-1]):
                    return float("nan")
                return float(np.interp(target, s["_ad"].values, s[iv_col].values))

            rows25 = []
            for mat in maturities:
                mdf = df_iv[df_iv["maturity"] == mat]
                T_val = float(mdf["T"].iloc[0])
                puts = mdf[mdf["type"] == "put"] if "type" in mdf.columns else mdf.iloc[0:0]
                calls = mdf[mdf["type"] == "call"] if "type" in mdf.columns else mdf.iloc[0:0]

                # ATM from the bracketing pair around the forward (same construction as
                # the κ₀ estimator): nearest call at K ≥ F + nearest put at K ≤ F, averaged.
                brackets = []
                for side in (calls[calls[m_col] >= 1.0], puts[puts[m_col] <= 1.0]):
                    side = side.dropna(subset=["market_iv"])
                    if len(side):
                        brackets.append(side.loc[(side[m_col] - 1.0).abs().idxmin()])
                atm_mkt = float(np.mean([b["market_iv"] for b in brackets])) if brackets else float("nan")
                atm_mdl = (float(np.nanmean([b.get("model_iv", np.nan) for b in brackets]))
                           if brackets and has_model else float("nan"))

                p_mkt = _iv_at_delta(puts, "market_iv", "market_delta")
                c_mkt = _iv_at_delta(calls, "market_iv", "market_delta")
                p_mdl = _iv_at_delta(puts, "model_iv", _mdl_dcol) if has_model else float("nan")
                c_mdl = _iv_at_delta(calls, "model_iv", _mdl_dcol) if has_model else float("nan")

                rows25.append({
                    "Expiry": mat, "T": T_val,
                    "atm_mkt": atm_mkt, "atm_mdl": atm_mdl,
                    "rr_mkt": p_mkt - c_mkt, "rr_mdl": p_mdl - c_mdl,
                    "fly_mkt": 0.5 * (p_mkt + c_mkt) - atm_mkt,
                    "fly_mdl": 0.5 * (p_mdl + c_mdl) - atm_mdl,
                })

            m25 = pd.DataFrame(rows25).sort_values("T")

            sk1, sk2 = st.columns(2)
            with sk1:
                fig_rr = go.Figure()
                fig_rr.add_trace(go.Scatter(
                    x=m25["T"], y=m25["rr_mkt"] * 100, mode="markers+lines",
                    name="Market RR", line=dict(color="#4C78A8"), marker=dict(size=7),
                    hovertemplate="T=%{x:.3f}y<br>RR=%{y:.2f} vol pts<extra>Market</extra>",
                ))
                if has_model and m25["rr_mdl"].notna().any():
                    fig_rr.add_trace(go.Scatter(
                        x=m25["T"], y=m25["rr_mdl"] * 100, mode="markers+lines",
                        name="Heston RR", line=dict(color="#E45756", dash="dash"),
                        marker=dict(size=6),
                        hovertemplate="T=%{x:.3f}y<br>RR=%{y:.2f} vol pts<extra>Heston</extra>",
                    ))
                # Empirical benchmark: equity skew decays roughly like 1/√T. Anchored to
                # the first liquid market point past ~5 weeks.
                ref = m25.dropna(subset=["rr_mkt"])
                ref = ref[ref["T"] >= 0.1]
                if len(ref) >= 2:
                    T0, rr0 = float(ref["T"].iloc[0]), float(ref["rr_mkt"].iloc[0])
                    Tg_ = np.linspace(max(float(m25["T"].min()), 0.02), float(m25["T"].max()), 100)
                    fig_rr.add_trace(go.Scatter(
                        x=Tg_, y=rr0 * np.sqrt(T0 / Tg_) * 100, mode="lines",
                        name="∝ 1/√T reference", line=dict(color="gray", dash="dot", width=1),
                        hoverinfo="skip",
                    ))
                fig_rr.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_rr.update_layout(
                    title="25Δ Risk Reversal Term Structure  (skew decay)",
                    xaxis_title="Maturity (years)", yaxis_title="RR (vol pts)",
                    height=380, legend=dict(orientation="h", yanchor="bottom", y=1.0),
                )
                st.plotly_chart(fig_rr, use_container_width=True)

            with sk2:
                fig_fl = go.Figure()
                fig_fl.add_trace(go.Scatter(
                    x=m25["T"], y=m25["fly_mkt"] * 100, mode="markers+lines",
                    name="Market fly", line=dict(color="#4C78A8"), marker=dict(size=7),
                    hovertemplate="T=%{x:.3f}y<br>Fly=%{y:.2f} vol pts<extra>Market</extra>",
                ))
                if has_model and m25["fly_mdl"].notna().any():
                    fig_fl.add_trace(go.Scatter(
                        x=m25["T"], y=m25["fly_mdl"] * 100, mode="markers+lines",
                        name="Heston fly", line=dict(color="#E45756", dash="dash"),
                        marker=dict(size=6),
                        hovertemplate="T=%{x:.3f}y<br>Fly=%{y:.2f} vol pts<extra>Heston</extra>",
                    ))
                fig_fl.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_fl.update_layout(
                    title="25Δ Butterfly Term Structure  (smile curvature)",
                    xaxis_title="Maturity (years)", yaxis_title="Fly (vol pts)",
                    height=380, legend=dict(orientation="h", yanchor="bottom", y=1.0),
                )
                st.plotly_chart(fig_fl, use_container_width=True)

            disp = m25.copy()
            disp["ΔRR (mkt−mdl)"] = (m25["rr_mkt"] - m25["rr_mdl"]).apply(
                lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
            disp["ΔFly (mkt−mdl)"] = (m25["fly_mkt"] - m25["fly_mdl"]).apply(
                lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
            for c_ in ["atm_mkt", "atm_mdl", "rr_mkt", "rr_mdl", "fly_mkt", "fly_mdl"]:
                disp[c_] = disp[c_].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
            disp["T"] = disp["T"].round(3)
            disp = disp.rename(columns={
                "atm_mkt": "ATM mkt", "atm_mdl": "ATM mdl",
                "rr_mkt": "25Δ RR mkt", "rr_mdl": "25Δ RR mdl",
                "fly_mkt": "25Δ Fly mkt", "fly_mdl": "25Δ Fly mdl",
            })
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.caption(
                "— rows: the chain's strikes never reach 25Δ at that expiry. The broad "
                "filter keeps K/F ∈ [0.8, 1.2], which at long maturities spans well under "
                "one standard deviation — e.g. at T≈2.5y and ~42% vol, even K/F=1.2 still "
                "carries Δ≈0.5. RR/fly are therefore only measurable at the short end "
                "unless the filter band is widened (or made delta-aware, the desk "
                "convention — e.g. keep |Δ| ∈ [0.05, 0.95])."
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — TERM STRUCTURE
# ════════════════════════════════════════════════════════════════════════════
with tabs[tidx["📅 Term Structure"]]:
    st.subheader("ATM Implied Volatility Term Structure")

    ts_rows = []
    for mat in sorted(df_iv["maturity"].unique() if "maturity" in df_iv.columns else []):
        mdf = df_iv[df_iv["maturity"] == mat]
        T_val = mdf["T"].iloc[0]
        near_atm = mdf[mdf[m_col].between(0.97, 1.03)]
        if near_atm.empty:
            near_atm = mdf.loc[(mdf[m_col] - 1.0).abs().nsmallest(3).index]
        for opt_t in ["call", "put"]:
            sub = near_atm[near_atm["type"] == opt_t] if "type" in near_atm.columns else near_atm
            if sub.empty:
                continue
            atm_iv = sub["market_iv"].mean()
            model_atm = sub["model_iv"].mean() if has_model and "model_iv" in sub.columns else np.nan
            ts_rows.append({
                "maturity": mat, "T": T_val, "type": opt_t,
                "atm_mkt_iv": atm_iv,
                "atm_mdl_iv": model_atm,
                "total_var": atm_iv ** 2 * T_val,
            })

    if not ts_rows:
        st.info("No ATM term structure data available.")
    else:
        ts_df = pd.DataFrame(ts_rows).sort_values("T")
        ts_left, ts_right = st.columns(2)

        with ts_left:
            fig_ts = go.Figure()
            style_map = {
                "call": ("solid", "circle", "#4C78A8"),
                "put": ("dash", "square", "#E45756"),
            }
            for opt_t, (dash, sym, clr) in style_map.items():
                sub = ts_df[ts_df["type"] == opt_t]
                if sub.empty:
                    continue
                fig_ts.add_trace(go.Scatter(
                    x=sub["T"], y=sub["atm_mkt_iv"] * 100,
                    mode="markers+lines",
                    name=f"Market {opt_t}",
                    line=dict(dash=dash, color=clr),
                    marker=dict(symbol=sym, size=8, color=clr),
                    hovertemplate=f"T=%{{x:.3f}}y<br>ATM IV=%{{y:.2f}}%<extra>Market {opt_t}</extra>",
                ))
                if has_model and sub["atm_mdl_iv"].notna().any():
                    fig_ts.add_trace(go.Scatter(
                        x=sub["T"], y=sub["atm_mdl_iv"] * 100,
                        mode="markers+lines",
                        name=f"Heston {opt_t}",
                        line=dict(dash="dot", color=clr, width=1.5),
                        marker=dict(symbol=sym, size=5, color=clr),
                        hovertemplate=f"T=%{{x:.3f}}y<br>Heston IV=%{{y:.2f}}%<extra>Heston {opt_t}</extra>",
                    ))

            fig_ts.update_layout(
                title="ATM IV Term Structure",
                xaxis_title="Maturity (years)",
                yaxis_title="ATM IV (%)",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.0),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        with ts_right:
            # Forward variance: Fvar(T1,T2) = (IV²·T2 − IV²·T1) / (T2 − T1)
            ts_call = ts_df[ts_df["type"] == "call"].sort_values("T").reset_index(drop=True)
            if len(ts_call) >= 2:
                fwd_rows = []
                for i in range(len(ts_call) - 1):
                    T1, T2 = ts_call.loc[i, "T"], ts_call.loc[i + 1, "T"]
                    var1, var2 = ts_call.loc[i, "total_var"], ts_call.loc[i + 1, "total_var"]
                    dT = T2 - T1
                    if dT > 1e-6:
                        fvar = (var2 - var1) / dT
                        fwd_rows.append({
                            "T_start": round(T1, 3),
                            "T_end": round(T2, 3),
                            "T_mid": round((T1 + T2) / 2, 3),
                            "fwd_vol_pct": np.sqrt(max(fvar, 0)) * 100,
                        })

                if fwd_rows:
                    fwd_df = pd.DataFrame(fwd_rows)
                    fig_fv = go.Figure(go.Bar(
                        x=fwd_df["T_mid"],
                        y=fwd_df["fwd_vol_pct"],
                        marker_color="#54A24B",
                        customdata=list(zip(fwd_df["T_start"], fwd_df["T_end"])),
                        hovertemplate="T=[%{customdata[0]:.2f}, %{customdata[1]:.2f}]<br>Fwd Vol: %{y:.2f}%<extra></extra>",
                    ))
                    fig_fv.update_layout(
                        title="Forward Variance (Implied)",
                        xaxis_title="Midpoint Maturity (yrs)",
                        yaxis_title="Forward Vol (%)",
                        height=380,
                    )
                    st.plotly_chart(fig_fv, use_container_width=True)
            else:
                st.info("Need ≥ 2 expiries to compute forward variance.")

        # Term structure table
        disp_ts = ts_df.copy()
        disp_ts["atm_mkt_iv"] = (disp_ts["atm_mkt_iv"] * 100).round(3).astype(str) + "%"
        disp_ts["atm_mdl_iv"] = disp_ts["atm_mdl_iv"].apply(
            lambda x: f"{x*100:.3f}%" if pd.notna(x) else "—"
        )
        disp_ts["total_var"] = disp_ts["total_var"].round(6)
        st.dataframe(
            disp_ts.rename(columns={"atm_mkt_iv": "Market ATM IV", "atm_mdl_iv": "Heston ATM IV"}),
            use_container_width=True, hide_index=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK METRICS
# ════════════════════════════════════════════════════════════════════════════
with tabs[tidx["📊 Risk Metrics"]]:
    st.subheader("Smile Risk Metrics by Expiry")
    st.caption(
        "Risk reversal ≈ OTM put IV − OTM call IV (same |moneyness offset|). "
        "Butterfly ≈ ½(OTM put IV + OTM call IV) − ATM IV."
    )

    # Use full dataset (both calls and puts) for RR/butterfly regardless of type filter.
    # Bucket in the same forward-moneyness basis as the rest of the page.
    df_rr_base = df_full.copy()
    if "ticker" in df_rr_base.columns:
        df_rr_base = df_rr_base[df_rr_base["ticker"] == sel_ticker]
    if m_col == "forward_moneyness" and "forward_moneyness" in df_rr_base.columns:
        df_rr_base["forward_moneyness"] = (
            df_rr_base["forward_moneyness"].where(df_rr_base["forward_moneyness"] > 0)
            .fillna(df_rr_base.get("moneyness"))
        )
    elif "forward_moneyness" not in df_rr_base.columns:
        df_rr_base["forward_moneyness"] = df_rr_base.get("moneyness")
    df_rr_base = df_rr_base.dropna(subset=["market_iv", m_col, "T"])

    rr_rows = []
    for mat in sorted(df_rr_base["maturity"].unique() if "maturity" in df_rr_base.columns else []):
        mdf = df_rr_base[df_rr_base["maturity"] == mat]
        T_val = mdf["T"].iloc[0]

        near_atm = mdf[mdf[m_col].between(0.97, 1.03)]
        if near_atm.empty:
            near_atm = mdf.loc[(mdf[m_col] - 1.0).abs().nsmallest(2).index]
        atm_iv = near_atm["market_iv"].mean()

        puts = mdf[(mdf.get("type", pd.Series(["call"] * len(mdf))) == "put") &
                   mdf[m_col].between(0.85, 0.95)] if "type" in mdf.columns else pd.DataFrame()
        calls = mdf[(mdf.get("type", pd.Series(["call"] * len(mdf))) == "call") &
                    mdf[m_col].between(1.05, 1.15)] if "type" in mdf.columns else pd.DataFrame()

        put_iv = puts["market_iv"].mean() if not puts.empty else np.nan
        call_iv = calls["market_iv"].mean() if not calls.empty else np.nan
        rr = put_iv - call_iv if pd.notna(put_iv) and pd.notna(call_iv) else np.nan
        fly = (put_iv + call_iv) / 2 - atm_iv if pd.notna(put_iv) and pd.notna(call_iv) and pd.notna(atm_iv) else np.nan

        rr_rows.append({
            "Expiry": mat, "T": round(T_val, 3),
            "ATM IV": atm_iv, "OTM Put IV (≈90%)": put_iv, "OTM Call IV (≈110%)": call_iv,
            "Risk Reversal": rr, "Butterfly": fly,
            "Total Var (IV²·T)": atm_iv ** 2 * T_val if pd.notna(atm_iv) else np.nan,
        })

    if not rr_rows:
        st.info("No expiry data available.")
    else:
        rr_df = pd.DataFrame(rr_rows).sort_values("T")

        rm1, rm2 = st.columns(2)

        with rm1:
            valid_rr = rr_df.dropna(subset=["Risk Reversal"])
            if not valid_rr.empty:
                fig_rr = go.Figure(go.Bar(
                    x=valid_rr["Expiry"],
                    y=valid_rr["Risk Reversal"] * 100,
                    marker_color=["#d62728" if v < 0 else "#2ca02c" for v in valid_rr["Risk Reversal"]],
                    hovertemplate="Expiry: %{x}<br>RR: %{y:.2f} vol pts<extra></extra>",
                ))
                fig_rr.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_rr.update_layout(
                    title="Risk Reversal by Expiry  (90% Put − 110% Call)",
                    xaxis_title="Expiry", yaxis_title="Vol pts (%)",
                    height=350,
                )
                st.plotly_chart(fig_rr, use_container_width=True)
            else:
                st.info(
                    "Risk reversal requires puts in [0.85, 0.95] moneyness "
                    "and calls in [1.05, 1.15] moneyness."
                )

        with rm2:
            valid_fly = rr_df.dropna(subset=["Butterfly"])
            if not valid_fly.empty:
                fig_fly = go.Figure(go.Bar(
                    x=valid_fly["Expiry"],
                    y=valid_fly["Butterfly"] * 100,
                    marker_color="#9467bd",
                    hovertemplate="Expiry: %{x}<br>Butterfly: %{y:.2f} vol pts<extra></extra>",
                ))
                fig_fly.update_layout(
                    title="Butterfly Spread by Expiry  (Smile Curvature)",
                    xaxis_title="Expiry", yaxis_title="Vol pts (%)",
                    height=350,
                )
                st.plotly_chart(fig_fly, use_container_width=True)
            else:
                st.info("Butterfly requires puts and calls at ≈90% and ≈110% moneyness.")

        # Summary table
        st.subheader("Risk Metrics Table")
        disp_rr = rr_df.copy()
        for col in ["ATM IV", "OTM Put IV (≈90%)", "OTM Call IV (≈110%)",
                    "Risk Reversal", "Butterfly"]:
            disp_rr[col] = disp_rr[col].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
            )
        disp_rr["Total Var (IV²·T)"] = disp_rr["Total Var (IV²·T)"].apply(
            lambda x: f"{x:.5f}" if pd.notna(x) else "—"
        )
        st.dataframe(disp_rr, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — MARKET vs MODEL
# ════════════════════════════════════════════════════════════════════════════
if has_model:
    with tabs[tidx["🔄 Market vs Model"]]:
        st.subheader("Market IV vs Heston Model IV")

        df_cmp = df_iv.dropna(subset=["market_iv", "model_iv"]).copy()

        if len(df_cmp) < 4:
            st.info("Not enough contracts with both market and model IV.")
        else:
            mv1, mv2 = st.columns(2)

            with mv1:
                # Scatter: market vs model IV
                iv_min = min(df_cmp["market_iv"].min(), df_cmp["model_iv"].min()) * 100
                iv_max = max(df_cmp["market_iv"].max(), df_cmp["model_iv"].max()) * 100

                fig_scat = go.Figure()
                for opt_t, sym, clr in [("call", "circle", "#4C78A8"), ("put", "square", "#E45756")]:
                    sub = df_cmp[df_cmp["type"] == opt_t] if "type" in df_cmp.columns else df_cmp
                    if sub.empty:
                        continue
                    hover = (
                        [f"mat={r['maturity']} K={r['strike']:.0f} {m_label}={r[m_col]:.3f}"
                         for _, r in sub.iterrows()]
                        if "maturity" in sub.columns else None
                    )
                    fig_scat.add_trace(go.Scatter(
                        x=sub["model_iv"] * 100,
                        y=sub["market_iv"] * 100,
                        mode="markers",
                        name=opt_t,
                        marker=dict(symbol=sym, color=clr, size=6, opacity=0.7),
                        text=hover,
                        hovertemplate="Heston IV: %{x:.2f}%<br>Market IV: %{y:.2f}%<br>%{text}<extra></extra>",
                    ))

                fig_scat.add_shape(
                    type="line",
                    x0=iv_min, x1=iv_max, y0=iv_min, y1=iv_max,
                    line=dict(dash="dot", color="gray"),
                )
                fig_scat.update_layout(
                    title="Market IV vs Heston Model IV (45° = perfect fit)",
                    xaxis_title="Heston Model IV (%)",
                    yaxis_title="Market IV (%)",
                    height=400,
                )
                st.plotly_chart(fig_scat, use_container_width=True)

            with mv2:
                if "iv_error" in df_cmp.columns:
                    iv_err = df_cmp["iv_error"].dropna() * 100
                    fig_hist = go.Figure(go.Histogram(
                        x=iv_err,
                        nbinsx=40,
                        marker_color="#4C78A8",
                        hovertemplate="IV Error: %{x:.2f}%<br>Count: %{y}<extra></extra>",
                    ))
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="red",
                                       annotation_text="0")
                    fig_hist.add_vline(x=float(iv_err.mean()), line_dash="dot",
                                       line_color="orange",
                                       annotation_text=f"Mean {iv_err.mean():.2f}%")
                    fig_hist.update_layout(
                        title="IV Error Distribution  (Market − Heston)",
                        xaxis_title="IV Error (vol pts %)",
                        yaxis_title="Count",
                        height=400,
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

            # IV error heatmap: forward-moneyness × maturity
            if "iv_error" in df_cmp.columns and "maturity" in df_cmp.columns:
                st.subheader(f"IV Error Heatmap  (Market − Heston, vol pts %, {m_label})")
                hdf = df_cmp.dropna(subset=["iv_error", m_col]).copy()
                hdf["m_bin"] = pd.cut(hdf[m_col], bins=10, precision=2).astype(str)
                pivot = (
                    hdf.pivot_table(
                        index="m_bin", columns="maturity",
                        values="iv_error", aggfunc="mean"
                    ) * 100
                )
                if not pivot.empty:
                    fig_hm = go.Figure(go.Heatmap(
                        z=pivot.values,
                        x=[str(c) for c in pivot.columns],
                        y=[str(r) for r in pivot.index],
                        colorscale="RdBu_r",
                        zmid=0,
                        colorbar=dict(title="IV Error (%)"),
                        hovertemplate="Maturity: %{x}<br>Moneyness: %{y}<br>Error: %{z:.2f}%<extra></extra>",
                    ))
                    fig_hm.update_layout(
                        xaxis_title="Maturity",
                        yaxis_title=f"Moneyness Bin ({m_label})",
                        height=400,
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)

            # Fit quality stats
            st.subheader("Fit Quality Statistics")
            if "iv_error" in df_cmp.columns:
                iv_e = df_cmp["iv_error"].dropna() * 100
                pe_e = df_cmp["price_error"].dropna() if "price_error" in df_cmp.columns else pd.Series(dtype=float)
                ss_res = ((df_cmp["market_iv"] - df_cmp["model_iv"]) ** 2).sum()
                ss_tot = ((df_cmp["market_iv"] - df_cmp["market_iv"].mean()) ** 2).sum()
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                stats = {
                    "Metric": [
                        "IV Error — Mean", "IV Error — RMSE", "IV Error — Max |·|",
                        "Price Error — Mean", "Price Error — RMSE", "Price Error — Max |·|",
                        "R² (market vs Heston IV)", "Contracts compared",
                    ],
                    "Value": [
                        f"{iv_e.mean():.4f} vol pts",
                        f"{np.sqrt((iv_e ** 2).mean()):.4f} vol pts",
                        f"{iv_e.abs().max():.4f} vol pts",
                        f"${pe_e.mean():.4f}" if len(pe_e) else "n/a",
                        f"${np.sqrt((pe_e ** 2).mean()):.4f}" if len(pe_e) else "n/a",
                        f"${pe_e.abs().max():.4f}" if len(pe_e) else "n/a",
                        f"{r2:.4f}" if pd.notna(r2) else "n/a",
                        f"{len(df_cmp):,}",
                    ],
                }
                st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — GREEKS
# ════════════════════════════════════════════════════════════════════════════
if has_greeks:
    with tabs[tidx["🏛 Greeks"]]:
        st.subheader("Greek Surfaces")

        greek_opts = [c for c in [
            "market_delta", "market_gamma", "market_vega",
            "model_delta", "model_gamma", "model_vega",
        ] if c in df.columns and df[c].notna().any()]

        if not greek_opts:
            st.info("No Greek columns available.")
        else:
            g1, g2 = st.columns(2)
            with g1:
                greek_z = st.selectbox("Greek", options=greek_opts, key="vs_greek_z")
            with g2:
                greek_x = st.selectbox(
                    "X-axis",
                    options=["forward_moneyness", "moneyness", "strike"],
                    format_func=lambda x: {"forward_moneyness": "Forward-moneyness K/F",
                                           "moneyness": "Moneyness K/S", "strike": "Strike"}.get(x, x),
                    key="vs_greek_x",
                )

            df_g = df.dropna(subset=[greek_z, greek_x, "T"]).copy()

            if len(df_g) >= 8:
                try:
                    grid_g = build_surface_grid(
                        df_g, x_col=greek_x, y_col="T", z_col=greek_z,
                        x_points=50, y_points=30,
                    )
                    fig_g = go.Figure(go.Surface(
                        x=grid_g.x_grid,
                        y=grid_g.y_grid,
                        z=grid_g.z_grid,
                        colorscale="RdYlGn",
                        colorbar=dict(title=greek_z),
                        hovertemplate=(
                            f"{greek_x}: %{{x:.3f}}<br>"
                            "T: %{y:.3f}y<br>"
                            f"{greek_z}: %{{z:.4f}}<extra></extra>"
                        ),
                    ))
                    fig_g.update_layout(
                        scene=dict(
                            xaxis_title=greek_x.capitalize(),
                            yaxis_title="Maturity (yrs)",
                            zaxis_title=greek_z,
                            camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8)),
                        ),
                        height=500,
                        margin=dict(l=0, r=0, t=20, b=0),
                    )
                    st.plotly_chart(fig_g, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not build Greek surface: {e}")
            else:
                st.warning("Not enough data points for a Greek surface (need ≥ 8).")

            # Cross-section: Greek vs x for a selected expiry
            st.subheader(f"{greek_z} — Cross-section by Expiry")
            cs1, cs2 = st.columns([1, 3])
            mats_g = sorted(df_g["maturity"].unique()) if "maturity" in df_g.columns else []
            with cs1:
                sel_mat_g = st.selectbox("Expiry", options=mats_g, key="vs_greek_mat")
            with cs2:
                cs_df = df_g[df_g["maturity"] == sel_mat_g].sort_values(greek_x)
                fig_cs = go.Figure(go.Scatter(
                    x=cs_df[greek_x], y=cs_df[greek_z],
                    mode="markers+lines",
                    marker=dict(size=7, color="#4C78A8"),
                    line=dict(color="#4C78A8"),
                    hovertemplate=f"{greek_x}: %{{x:.3f}}<br>{greek_z}: %{{y:.4f}}<extra></extra>",
                ))
                fig_cs.update_layout(
                    xaxis_title=greek_x.capitalize(), yaxis_title=greek_z,
                    height=320, margin=dict(t=20),
                )
                st.plotly_chart(fig_cs, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — ARB CHECKS
# ════════════════════════════════════════════════════════════════════════════
with tabs[tidx["⚖ Arb Checks"]]:
    st.subheader("No-Arbitrage Checks")
    st.caption(
        "Calendar spread: total variance IV²·T must be non-decreasing in T for the same moneyness. "
        "Butterfly: IV must be convex in strike (positive probability density implied by call prices)."
    )

    arb_issues: list[str] = []

    # ── Calendar spread ───────────────────────────────────────────────────
    st.markdown("#### Calendar Spread  (Total Variance non-decreasing in T)")
    cal_rows = []
    for mb in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]:
        near = df_iv[df_iv[m_col].between(mb - 0.025, mb + 0.025)]
        if "maturity" not in near.columns or near.empty:
            continue
        ts_sub = (
            near.groupby("maturity", sort=False)
            .apply(lambda g: pd.Series({
                "T": g["T"].iloc[0],
                "total_var": (g["market_iv"] ** 2 * g["T"]).mean(),
            }))
            .sort_values("T")
            .reset_index()
        )
        prev_var = None
        for _, row in ts_sub.iterrows():
            ok = prev_var is None or row["total_var"] >= prev_var - 1e-5
            cal_rows.append({
                "Moneyness ≈": f"{mb:.0%}",
                "Expiry": row["maturity"],
                "T": round(row["T"], 3),
                "Total Var (IV²·T)": round(row["total_var"], 6),
                "Calendar OK?": "✓" if ok else "✗ Violated",
            })
            if not ok:
                arb_issues.append(f"Calendar arb at moneyness≈{mb:.0%}, expiry {row['maturity']}")
            prev_var = row["total_var"]

    if cal_rows:
        cdf = pd.DataFrame(cal_rows)
        st.dataframe(cdf, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data to check calendar spread arbitrage.")

    # ── Butterfly / convexity ─────────────────────────────────────────────
    st.markdown("#### Butterfly / Convexity  (IV convex in strike per expiry)")
    bfly_rows = []
    mats_arb = sorted(df_iv["maturity"].unique()) if "maturity" in df_iv.columns else []
    for mat in mats_arb:
        mdf = df_iv[df_iv["maturity"] == mat]
        if "strike" not in mdf.columns or len(mdf) < 3:
            continue
        mdf_s = mdf.sort_values("strike")
        ivs = mdf_s["market_iv"].values
        strikes = mdf_s["strike"].values
        for i in range(1, len(mdf_s) - 1):
            dK1 = strikes[i] - strikes[i - 1]
            dK2 = strikes[i + 1] - strikes[i]
            if dK1 <= 0 or dK2 <= 0:
                continue
            d2iv = (ivs[i + 1] - ivs[i]) / dK2 - (ivs[i] - ivs[i - 1]) / dK1
            if d2iv < -0.001:
                bfly_rows.append({
                    "Expiry": mat,
                    "Strike": strikes[i],
                    f"Moneyness {m_label}": round(mdf_s.iloc[i][m_col], 3) if m_col in mdf_s.columns else "—",
                    "d²IV/dK² (×1000)": round(d2iv * 1000, 4),
                    "Butterfly OK?": "✗ Violated",
                })
                arb_issues.append(f"Butterfly arb at {mat}, K={strikes[i]:.1f}")

    if bfly_rows:
        st.dataframe(pd.DataFrame(bfly_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No butterfly violations detected (IV is convex in strike across all expiries).")

    # ── Summary ───────────────────────────────────────────────────────────
    st.divider()
    if arb_issues:
        st.warning(f"**{len(arb_issues)} potential arbitrage issue(s) detected.**")
        for issue in arb_issues[:25]:
            st.markdown(f"- {issue}")
        if len(arb_issues) > 25:
            st.caption(f"… and {len(arb_issues) - 25} more.")
    else:
        st.success(
            "All arbitrage checks passed. "
            "The IV surface appears internally consistent (calendar and convexity)."
        )


# ── Navigation ────────────────────────────────────────────────────────────────
st.divider()
col_back, col_fwd = st.columns([1, 1])
with col_back:
    st.page_link("pages/04_Price_Contracts.py", label="← Back to Price Contracts", icon="💰")
with col_fwd:
    st.page_link("pages/06_Strategy_Lab.py", label="Next: Strategy Lab →", icon="⚗️")
