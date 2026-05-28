from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Options Analytics Home", layout="wide")

st.title("Options Analytics Platform")
st.caption(
    "A step-by-step pipeline for pricing, calibrating, and analysing options "
    "under the Heston stochastic-volatility model."
)

ss = st.session_state

_METHOD_LABELS = {
    "european_proxy": "European Proxy",
    "pde":            "PDE Solver",
    "lsmc":           "LSMC Simulation",
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
    st.markdown(
        f"Feller: :{color}[{feller:+.4f} "
        f"({'✓' if feller > 0 else '✗'})]  "
        f"| Loss: {meta['loss']:.3e}"
    )


# ── Pipeline status ───────────────────────────────────────────────────────────
st.subheader("Pipeline status")
step_col1, step_col2, step_col3 = st.columns(3)

# Step 1 — Fetch
with step_col1:
    with st.container(border=True):
        st.markdown("**Step 1 — Fetch Data**")
        if "raw_df" in ss:
            p = ss.get("fetch_params", {})
            st.success("Data loaded")
            st.metric("Raw contracts", f"{len(ss['raw_df']):,}")
            st.caption(
                f"Tickers: {p.get('tickers', '?')}  |  "
                f"r={p.get('r', 0)*100:.2f}%  q={p.get('q', 0)*100:.3f}%"
            )
        else:
            st.warning("Not fetched yet")
            st.page_link("pages/01_Fetch_Data.py", label="Go to Fetch Data →")

# Step 2 — Filter
with step_col2:
    with st.container(border=True):
        st.markdown("**Step 2 — Filter Options**")
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

# Step 3 — Calibrate
with step_col3:
    with st.container(border=True):
        st.markdown("**Step 3 — Calibrate Heston**")
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

if not has_raw:
    st.info(
        "**Start here:** go to **Fetch Data** in the sidebar, "
        "then work through Filter → Calibrate."
    )
else:
    tab_raw, tab_filtered, tab_cal = st.tabs(["Raw Data", "Filtered Data", "Calibration"])

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
                        "Loss": f"{m['loss']:.4e}",
                        "Runtime (s)": round(m["runtime_seconds"], 1),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
