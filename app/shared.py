from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.surfaces import build_surface_grid
from config.market_config import (
    get_ois_curve,
    interpolate_rate,
    curve_summary,
    fetch_sofr_rate,
)
from services.analytics_service import build_chain_analytics
from services.calibration_service import (
    calibration_scope_id,
    calibrate_option_chain,
    load_triple_calibration,
    save_triple_calibration,
)
from services.market_service import (
    extract_dividend_yields,
    filter_chain_with_stats,
    load_live_chain,
    parse_tickers,
)
from services.pricing_service import HestonParameters


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_ois_curve() -> dict[float, float]:
    """Fetch the SOFR/OIS curve once per hour across all Streamlit sessions."""
    return get_ois_curve(force_refresh=True)


DEFAULT_PARAMS = HestonParameters(
    0.24403566968414625,
    2.000169494947594,
    0.1087414325971798,
    0.5085892157308337,
    -0.7084200062062825,
)

# Session-state key helpers  (three independent calibration slots)
_CAL_META_A  = "cal::{sid}::meta_slot_a"
_CAL_DF_A    = "cal::{sid}::df_slot_a"
_CAL_META_B  = "cal::{sid}::meta_slot_b"
_CAL_DF_B    = "cal::{sid}::df_slot_b"
_CAL_META_C  = "cal::{sid}::meta_slot_c"
_CAL_DF_C    = "cal::{sid}::df_slot_c"
_CAL_ACTIVE  = "cal::{sid}::active"   # "a" | "b" | "c"

# Mapping from UI label → internal method code passed to calibrate_option_chain
_METHOD_CODE: dict[str, str] = {
    "European Proxy": "european_proxy",
    "PDE Solver":     "pde",
    "LSMC Simulation": "lsmc",
}
_METHOD_OPTIONS = list(_METHOD_CODE.keys())


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")


def parse_heston_params(raw_params: str) -> HestonParameters:
    values = [float(value.strip()) for value in raw_params.split(",")]
    if len(values) != 5:
        raise ValueError("Expected five comma-separated Heston parameters.")
    return HestonParameters.from_iterable(values)


def _default_params_text() -> str:
    return ",".join(str(value) for value in DEFAULT_PARAMS.as_tuple())


def clear_data_caches() -> None:
    cached_load_chain.clear()
    cached_filter_chain.clear()
    cached_build_analytics.clear()
    cached_calibrate_triple.clear()


def model_params_from_meta(calibration_meta: dict[str, float]) -> HestonParameters:
    return HestonParameters(
        calibration_meta["v0"],
        calibration_meta["kappa"],
        calibration_meta["theta"],
        calibration_meta["sigma"],
        calibration_meta["rho"],
    )


def render_sidebar_controls(
    page_key: str,
    fetched_div_yields: dict[str, float] | None = None,
) -> dict[str, object]:
    st.sidebar.header("Market Data")
    tickers_text = st.sidebar.text_input(
        "Tickers",
        value="NVDA",
        key=f"{page_key}_tickers",
        help="Comma-separated ticker list for live mode.",
    )

    # ── SOFR/OIS rate curve (auto-fetched, read-only) ────────────────────
    if st.sidebar.button("↻ Refresh rates", key=f"{page_key}_rates_refresh",
                         use_container_width=False):
        _cached_ois_curve.clear()
        st.rerun()

    _sofr_fallback = False
    try:
        _rate_curve = _cached_ois_curve()
        r = interpolate_rate(_rate_curve, 0.25)
        st.sidebar.caption(f"**Risk-free (SOFR/OIS)** — {curve_summary(_rate_curve)}")
    except Exception:
        _rate_curve = {}
        r = 0.045
        _sofr_fallback = True
        st.sidebar.warning(
            f"⚠️ SOFR/OIS rates unavailable — interest rate switched to "
            f"3M SOFR fallback ({r*100:.2f}%)"
        )

    # ── Dividend yield (auto-fetched per ticker from yfinance, read-only) ─
    selected_tickers = parse_tickers(tickers_text)
    if fetched_div_yields:
        q_per_ticker = {t: fetched_div_yields.get(t, 0.0) for t in selected_tickers}
        q = float(sum(q_per_ticker.values()) / max(len(q_per_ticker), 1))
        yield_parts = [f"{t}: {v * 100:.3f}%" for t, v in q_per_ticker.items()]
        st.sidebar.caption("**Dividend yield** (yfinance) — " + "  |  ".join(yield_parts))
        zero_yield_tickers = [t for t, v in q_per_ticker.items() if v == 0.0]
        if zero_yield_tickers:
            st.sidebar.info(
                f"ℹ️ Dividend yield for {', '.join(zero_yield_tickers)} is 0.0% "
                f"(yfinance returned 0 — using 0.0% fallback)"
            )
    else:
        q = 0.0
        st.sidebar.info(
            "ℹ️ Dividend yield not yet fetched — using 0.0% fallback. "
            "Refresh the options chain to load live yields."
        )
    spread_limit = st.sidebar.number_input(
        "Spread limit",
        min_value=0.01,
        max_value=1.0,
        value=0.05,
        step=0.01,
        key=f"{page_key}_spread_limit",
    )
    refresh_requested = st.sidebar.button(
        "Refresh options chain", key=f"{page_key}_refresh", use_container_width=True
    )

    st.sidebar.header("View Filters")
    option_types = st.sidebar.multiselect(
        "Option types",
        options=("call", "put"),
        default=("call", "put"),
        key=f"{page_key}_option_types",
    )
    min_volume = st.sidebar.number_input(
        "Min volume", min_value=0, value=100, step=10, key=f"{page_key}_min_volume"
    )
    min_open_interest = st.sidebar.number_input(
        "Min open interest", min_value=0, value=1000, step=100, key=f"{page_key}_min_open_interest"
    )
    max_maturity = st.sidebar.number_input(
        "Max maturity (years)",
        min_value=0.1, max_value=5.0, value=2.0, step=0.1,
        key=f"{page_key}_max_maturity",
    )
    max_contracts = st.sidebar.number_input(
        "Max contracts in view",
        min_value=25, max_value=5000, value=2000, step=25,
        key=f"{page_key}_max_contracts",
    )
    moneyness_lo = st.sidebar.number_input(
        "Moneyness min (strike/spot)",
        min_value=0.01, max_value=1.0, value=0.1, step=0.05,
        format="%.2f",
        key=f"{page_key}_moneyness_lo",
    )
    moneyness_hi = st.sidebar.number_input(
        "Moneyness max (strike/spot)",
        min_value=1.0, max_value=20.0, value=5.0, step=0.25,
        format="%.2f",
        key=f"{page_key}_moneyness_hi",
    )

    st.sidebar.header("Model Layer")
    model_mode_label = st.sidebar.selectbox(
        "Model mode",
        (
            "Use stored / calibrated Heston params",
            "Use existing/precomputed model prices",
            "Use manual Heston params",
            "Market metrics only",
        ),
        index=0,
        key=f"{page_key}_model_mode",
    )

    params_text = _default_params_text()
    if model_mode_label == "Use manual Heston params":
        params_text = st.sidebar.text_input(
            "Heston params",
            value=_default_params_text(),
            key=f"{page_key}_params_text",
            help="Format: v0,kappa,theta,sigma,rho",
        )

    pricing_limit = st.sidebar.number_input(
        "Model pricing limit",
        min_value=10,
        max_value=5000,
        value=500,
        step=50,
        key=f"{page_key}_pricing_limit",
        help="Maximum contracts to reprice. Set to 5000 to price all.",
    )
    Ns = st.sidebar.number_input(
        "PDE stock steps", min_value=10, max_value=100, value=30, step=5, key=f"{page_key}_Ns"
    )
    Nv = st.sidebar.number_input(
        "PDE variance steps", min_value=10, max_value=100, value=15, step=5, key=f"{page_key}_Nv"
    )
    Nt = st.sidebar.number_input(
        "PDE time steps", min_value=10, max_value=100, value=30, step=5, key=f"{page_key}_Nt"
    )

    st.sidebar.header("Calibration")
    st.sidebar.caption(
        "Each slot runs the LM calibrator (Cui et al. 2016) with a different "
        "American-option pricing method. PDE and LSMC slots are slow — expect "
        "several minutes per run."
    )
    method_a = st.sidebar.selectbox(
        "Slot A — method",
        _METHOD_OPTIONS,
        index=0,
        key=f"{page_key}_method_a",
    )
    method_b = st.sidebar.selectbox(
        "Slot B — method",
        _METHOD_OPTIONS,
        index=1,
        key=f"{page_key}_method_b",
    )
    method_c = st.sidebar.selectbox(
        "Slot C — method",
        _METHOD_OPTIONS,
        index=2,
        key=f"{page_key}_method_c",
    )
    calibrate_requested = st.sidebar.button(
        "Calibrate All Slots", key=f"{page_key}_calibrate", use_container_width=True
    )

    return {
        "tickers_text": tickers_text,
        "r": float(r),              # 3M SOFR — representative scalar for logging/display
        "rate_curve": _rate_curve,  # full curve for maturity-matched pricing
        "q": float(q),
        "spread_limit": float(spread_limit),
        "refresh_requested": refresh_requested,
        "option_types": tuple(option_types),
        "min_volume": int(min_volume),
        "min_open_interest": int(min_open_interest),
        "max_maturity": float(max_maturity),
        "max_contracts": int(max_contracts),
        "model_mode": model_mode_label,
        "params_text": params_text,
        "pricing_limit": int(pricing_limit),
        "Ns": int(Ns),
        "Nv": int(Nv),
        "Nt": int(Nt),
        "calibrate_requested": calibrate_requested,
        "method_a": method_a,
        "method_b": method_b,
        "method_c": method_c,
        "moneyness_lo": float(moneyness_lo),
        "moneyness_hi": float(moneyness_hi),
    }


# ── Cached data loaders ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load_chain(tickers_text: str) -> pd.DataFrame:
    return load_live_chain(tickers=parse_tickers(tickers_text))


@st.cache_data(show_spinner=False)
def cached_filter_chain(
    raw_df: pd.DataFrame,
    tickers_text: str,
    spread_limit: float,
    r: float,
    q: float,
    option_types: tuple[str, ...],
    min_volume: int,
    min_open_interest: int,
    max_maturity: float,
    max_contracts: int,
    moneyness_lo: float = 0.5,
    moneyness_hi: float = 2.0,
    rate_curve: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    return filter_chain_with_stats(
        raw_df,
        spread_limit=spread_limit,
        r=r,
        q=q,
        rate_curve=rate_curve,
        tickers=parse_tickers(tickers_text),
        option_types=option_types,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        max_maturity=max_maturity,
        max_contracts=max_contracts,
        moneyness_lo=moneyness_lo,
        moneyness_hi=moneyness_hi,
    )


@st.cache_data(show_spinner=False)
def cached_build_analytics(
    filtered_df: pd.DataFrame,
    model_mode: str,
    params_text: str,
    r: float,
    q: float,
    pricing_limit: int,
    Ns: int,
    Nv: int,
    Nt: int,
    rate_curve: dict | None = None,
) -> pd.DataFrame:
    if model_mode in ("Market metrics only", "Use existing/precomputed model prices"):
        return build_chain_analytics(filtered_df, r=r, q=q, rate_curve=rate_curve)

    params = parse_heston_params(params_text)
    return build_chain_analytics(
        filtered_df,
        r=r,
        q=q,
        rate_curve=rate_curve,
        heston_params=params,
        compute_model_prices=True,
        pricing_limit=pricing_limit,
        Ns=Ns,
        Nv=Nv,
        Nt=Nt,
    )


@st.cache_data(show_spinner=False)
def cached_calibrate_triple(
    filtered_df: pd.DataFrame,
    r: float,
    q: float,
    Ns: int,
    Nv: int,
    Nt: int,
    method_a_code: str,
    method_b_code: str,
    method_c_code: str,
    rate_curve: dict | None = None,
) -> tuple[dict, pd.DataFrame, dict, pd.DataFrame, dict, pd.DataFrame]:
    """
    Run three independent LM calibrations using maturity-matched SOFR/OIS rates.
    """
    results = []
    for code in (method_a_code, method_b_code, method_c_code):
        label = next(k for k, v in _METHOD_CODE.items() if v == code)
        result, df = calibrate_option_chain(
            filtered_df,
            r=r, q=q, Ns=Ns, Nv=Nv, Nt=Nt,
            max_expiries=None,
            contracts_per_expiry=None,
            american_method=code,
            rate_curve=rate_curve,
        )
        meta = result.as_dict()
        meta["method_label"] = label
        results.append((meta, df))
    (ma, da), (mb, db), (mc, dc) = results
    return ma, da, mb, db, mc, dc


# ── Calibration state helpers ──────────────────────────────────────────────────

def _skey(template: str, scope_id: str) -> str:
    return template.format(sid=scope_id)


def _load_calibration_state(scope_id: str) -> tuple[
    dict | None, pd.DataFrame | None,
    dict | None, pd.DataFrame | None,
    dict | None, pd.DataFrame | None,
]:
    meta_a = st.session_state.get(_skey(_CAL_META_A, scope_id))
    df_a   = st.session_state.get(_skey(_CAL_DF_A,   scope_id))
    meta_b = st.session_state.get(_skey(_CAL_META_B, scope_id))
    df_b   = st.session_state.get(_skey(_CAL_DF_B,   scope_id))
    meta_c = st.session_state.get(_skey(_CAL_META_C, scope_id))
    df_c   = st.session_state.get(_skey(_CAL_DF_C,   scope_id))

    if meta_a is None and meta_b is None and meta_c is None:
        meta_a, meta_b, meta_c = load_triple_calibration(scope_id)
        for key, val in [
            (_CAL_META_A, meta_a), (_CAL_META_B, meta_b), (_CAL_META_C, meta_c),
        ]:
            if val:
                st.session_state[_skey(key, scope_id)] = val

    return meta_a, df_a, meta_b, df_b, meta_c, df_c


def _handle_triple_calibration_action(
    config: dict,
    filtered_df: pd.DataFrame,
    scope_id: str,
) -> None:
    if not config["calibrate_requested"]:
        return

    code_a = _METHOD_CODE[config["method_a"]]
    code_b = _METHOD_CODE[config["method_b"]]
    code_c = _METHOD_CODE[config["method_c"]]

    with st.spinner(
        f"Calibrating — Slot A: {config['method_a']} | "
        f"Slot B: {config['method_b']} | "
        f"Slot C: {config['method_c']}  …"
    ):
        meta_a, df_a, meta_b, df_b, meta_c, df_c = cached_calibrate_triple(
            filtered_df,
            r=float(config["r"]),
            q=float(config["q"]),
            Ns=int(config["Ns"]),
            Nv=int(config["Nv"]),
            Nt=int(config["Nt"]),
            method_a_code=code_a,
            method_b_code=code_b,
            method_c_code=code_c,
            rate_curve=config.get("rate_curve"),
        )

    for key, val, df in [
        (_CAL_META_A, meta_a, df_a),
        (_CAL_META_B, meta_b, df_b),
        (_CAL_META_C, meta_c, df_c),
    ]:
        st.session_state[_skey(key, scope_id)] = val
    st.session_state[_skey(_CAL_DF_A, scope_id)] = df_a
    st.session_state[_skey(_CAL_DF_B, scope_id)] = df_b
    st.session_state[_skey(_CAL_DF_C, scope_id)] = df_c

    save_triple_calibration(scope_id, meta_a, meta_b, meta_c)
    parts = " | ".join(
        f"{m['method_label']} loss={m['loss']:.4e} ({m['runtime_seconds']:.1f}s)"
        for m in (meta_a, meta_b, meta_c)
    )
    st.success(f"Calibration complete — {parts}")
    st.rerun()


def _render_method_selector(scope_id: str, page_key: str) -> str:
    """Render active-slot radio in sidebar; return 'a', 'b', or 'c'."""
    active_key = _skey(_CAL_ACTIVE, scope_id)
    current = st.session_state.get(active_key, "a")
    idx = {"a": 0, "b": 1, "c": 2}.get(current, 0)

    selected = st.sidebar.radio(
        "Active slot for pricing",
        options=["Slot A", "Slot B", "Slot C"],
        index=idx,
        key=f"{page_key}_method_radio",
    )
    active = {"Slot A": "a", "Slot B": "b", "Slot C": "c"}[selected]
    st.session_state[active_key] = active
    return active


# ── Render helpers ─────────────────────────────────────────────────────────────

def _feller_str(meta: dict) -> str:
    feller = 2 * meta["kappa"] * meta["theta"] - meta["sigma"] ** 2
    status = "✓ satisfied" if feller > 0 else "✗ violated"
    return f"{feller:+.4f}  ({status})"


def _render_cal_slot(col, meta: dict | None, df: pd.DataFrame | None,
                     slot_label: str, active_slot: str, slot_key: str) -> None:
    """Render one calibration slot column."""
    with col:
        badge = "  ← **active**" if slot_key == active_slot else ""
        method_name = meta.get("method_label", slot_label) if meta else slot_label
        st.markdown(f"**{slot_label}: {method_name}**{badge}")

        if meta is None:
            st.info("Not yet calibrated.")
            return

        params_df = pd.DataFrame({
            "Parameter": ["v₀", "κ", "θ̄", "σ", "ρ"],
            "Value": [
                f"{meta['v0']:.6f}  (vol {meta['v0']**0.5*100:.1f}%)",
                f"{meta['kappa']:.4f}",
                f"{meta['theta']:.6f}  (vol {meta['theta']**0.5*100:.1f}%)",
                f"{meta['sigma']:.4f}",
                f"{meta['rho']:.4f}",
            ],
        })
        st.dataframe(params_df, hide_index=True, use_container_width=True)

        feller = 2 * meta["kappa"] * meta["theta"] - meta["sigma"] ** 2
        feller_color = "green" if feller > 0 else "red"
        st.markdown(
            f"**Feller 2κθ−σ²** = "
            f":{feller_color}[{feller:+.4f}  "
            f"({'satisfied ✓' if feller > 0 else 'violated ✗'})]"
        )
        st.caption(
            f"Loss: {meta['loss']:.4e}  |  "
            f"Contracts: {int(meta['contract_count'])}  |  "
            f"Runtime: {meta['runtime_seconds']:.1f}s"
        )
        if df is not None and not df.empty:
            with st.expander("Calibration universe", expanded=False):
                show_cols = [c for c in ["maturity", "type", "strike", "T",
                                         "moneyness", "mid_price", "market_iv"]
                             if c in df.columns]
                st.dataframe(df[show_cols].sort_values(["maturity", "strike"]),
                             hide_index=True, use_container_width=True)


def render_triple_calibration_panel(
    meta_a: dict | None, df_a: pd.DataFrame | None,
    meta_b: dict | None, df_b: pd.DataFrame | None,
    meta_c: dict | None, df_c: pd.DataFrame | None,
    active_slot: str,
) -> None:
    """Three-column display of all calibration slots with active slot highlighted."""
    if not any([meta_a, meta_b, meta_c]):
        return

    with st.expander("Calibration Results — All Slots", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        _render_cal_slot(col_a, meta_a, df_a, "Slot A", active_slot, "a")
        _render_cal_slot(col_b, meta_b, df_b, "Slot B", active_slot, "b")
        _render_cal_slot(col_c, meta_c, df_c, "Slot C", active_slot, "c")


def render_calibration_panel(
    calibration_meta: dict[str, float] | None,
    calibration_df: pd.DataFrame | None,
) -> None:
    """Single-method panel — kept for backward compatibility with existing pages."""
    if not calibration_meta:
        return
    with st.expander("Active Calibration Parameters", expanded=False):
        st.json(calibration_meta)
        if calibration_df is not None and not calibration_df.empty:
            st.caption("Calibration universe")
            cols = [
                c for c in ["contract_id", "type", "maturity", "strike", "mid_price", "market_iv"]
                if c in calibration_df.columns
            ]
            st.dataframe(calibration_df[cols], use_container_width=True, hide_index=True)


def render_chain_summary(
    raw_df: pd.DataFrame, filtered_df: pd.DataFrame, analytics_df: pd.DataFrame
) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw contracts", f"{len(raw_df):,}")
    col2.metric("Filtered contracts", f"{len(filtered_df):,}")
    col3.metric("Expiries", analytics_df["maturity"].nunique() if "maturity" in analytics_df.columns else 0)
    col4.metric("Tickers", analytics_df["ticker"].nunique() if "ticker" in analytics_df.columns else 0)

    col5, col6, col7, col8 = st.columns(4)
    col5.metric(
        "Market IV points",
        int(analytics_df["market_iv"].notna().sum()) if "market_iv" in analytics_df.columns else 0,
    )
    col6.metric(
        "Model IV points",
        int(analytics_df["model_iv"].notna().sum()) if "model_iv" in analytics_df.columns else 0,
    )
    col7.metric(
        "Median market delta",
        f"{analytics_df['market_delta'].median():.3f}" if "market_delta" in analytics_df.columns else "n/a",
    )
    top_score = analytics_df["mispricing_score"].max() if "mispricing_score" in analytics_df.columns else None
    col8.metric("Top mispricing score", f"{top_score:.4f}" if pd.notna(top_score) else "n/a")


# ── Main data loader (called from every page) ──────────────────────────────────

def load_app_data(
    page_key: str,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float] | None,
    pd.DataFrame | None,
]:
    fetched_div_yields: dict[str, float] = st.session_state.get("_div_yields", {})
    config = render_sidebar_controls(page_key, fetched_div_yields=fetched_div_yields)

    if config["refresh_requested"]:
        clear_data_caches()

    raw_df = cached_load_chain(tickers_text=str(config["tickers_text"]))

    # Store per-ticker dividend yields for the next rerun
    new_yields = extract_dividend_yields(raw_df)
    if new_yields:
        st.session_state["_div_yields"] = new_yields
        fetched_div_yields = new_yields
    filtered_df, filter_stats = cached_filter_chain(
        raw_df,
        tickers_text=str(config["tickers_text"]),
        spread_limit=float(config["spread_limit"]),
        r=float(config["r"]),
        q=float(config["q"]),
        option_types=tuple(config["option_types"]),
        min_volume=int(config["min_volume"]),
        min_open_interest=int(config["min_open_interest"]),
        max_maturity=float(config["max_maturity"]),
        max_contracts=int(config["max_contracts"]),
        moneyness_lo=float(config["moneyness_lo"]),
        moneyness_hi=float(config["moneyness_hi"]),
        rate_curve=config.get("rate_curve"),
    )
    st.session_state["_filter_stats"] = filter_stats

    # Use a style-agnostic scope (both methods always run together)
    scope_id = calibration_scope_id(
        source="live",
        tickers_text=str(config["tickers_text"]),
        r=float(config["r"]),
        q=float(config["q"]),
        calibration_style="dual",
    )

    _handle_triple_calibration_action(config, filtered_df, scope_id)

    # Load calibration state (session → disk fallback)
    meta_a, df_a, meta_b, df_b, meta_c, df_c = _load_calibration_state(scope_id)

    # Show slot selector only when at least one slot has been calibrated
    active_slot = "a"
    if any([meta_a, meta_b, meta_c]):
        active_slot = _render_method_selector(scope_id, page_key)

    # Active calibration params = whichever slot is selected
    _slot_meta = {"a": meta_a, "b": meta_b, "c": meta_c}
    _slot_df   = {"a": df_a,   "b": df_b,   "c": df_c}
    calibration_meta = _slot_meta[active_slot]
    calibration_df   = _slot_df[active_slot]

    # Build analytics
    model_mode = str(config["model_mode"])

    _rate_curve = config.get("rate_curve")
    if model_mode == "Use stored / calibrated Heston params" and calibration_meta:
        params = model_params_from_meta(calibration_meta)
        analytics_df = build_chain_analytics(
            filtered_df,
            r=float(config["r"]),
            q=float(config["q"]),
            rate_curve=_rate_curve,
            heston_params=params,
            compute_model_prices=True,
            pricing_limit=int(config["pricing_limit"]),
            Ns=int(config["Ns"]),
            Nv=int(config["Nv"]),
            Nt=int(config["Nt"]),
        )
    elif model_mode == "Use stored / calibrated Heston params":
        analytics_df = build_chain_analytics(
            filtered_df, r=float(config["r"]), q=float(config["q"]),
            rate_curve=_rate_curve,
        )
    else:
        analytics_df = cached_build_analytics(
            filtered_df,
            model_mode=model_mode,
            params_text=str(config["params_text"]),
            r=float(config["r"]),
            q=float(config["q"]),
            pricing_limit=int(config["pricing_limit"]),
            Ns=int(config["Ns"]),
            Nv=int(config["Nv"]),
            Nt=int(config["Nt"]),
            rate_curve=_rate_curve,
        )
        if model_mode == "Use manual Heston params":
            params = parse_heston_params(str(config["params_text"]))
            calibration_meta = {
                "mode": "manual",
                "v0": params.v0,
                "kappa": params.kappa,
                "theta": params.theta,
                "sigma": params.sigma,
                "rho": params.rho,
            }

    # Attach triple panel state to session so pages can call render_triple_calibration_panel
    st.session_state["_triple_cal"] = {
        "meta_a": meta_a, "df_a": df_a,
        "meta_b": meta_b, "df_b": df_b,
        "meta_c": meta_c, "df_c": df_c,
        "active": active_slot,
    }

    return config, raw_df, filtered_df, analytics_df, calibration_meta, calibration_df


def render_surface_chart(
    analytics_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    title: str,
) -> None:
    grid = build_surface_grid(analytics_df, x_col=x_col, y_col=y_col, z_col=z_col)
    fig = go.Figure(
        data=[
            go.Surface(
                x=grid.x_grid,
                y=grid.y_grid,
                z=grid.z_grid,
                colorscale="Viridis",
            )
        ]
    )
    fig.update_layout(
        title=f"{title} ({grid.point_count} points)",
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        height=750,
    )
    st.plotly_chart(fig, use_container_width=True)
