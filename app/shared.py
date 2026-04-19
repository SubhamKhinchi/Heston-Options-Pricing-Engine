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
from services.analytics_service import build_chain_analytics
from services.calibration_service import (
    calibration_scope_id,
    calibrate_option_chain,
    load_saved_calibration,
    save_calibration_result,
)
from services.market_service import filter_chain_with_stats, load_live_chain, parse_tickers
from services.pricing_service import HestonParameters


DEFAULT_PARAMS = HestonParameters(
    0.24403566968414625,
    2.000169494947594,
    0.1087414325971798,
    0.5085892157308337,
    -0.7084200062062825,
)


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")


def parse_heston_params(raw_params: str) -> HestonParameters:
    values = [float(value.strip()) for value in raw_params.split(",")]
    if len(values) != 5:
        raise ValueError("Expected five comma-separated Heston parameters.")
    return HestonParameters.from_iterable(values)


def _default_params_text() -> str:
    return ",".join(str(value) for value in DEFAULT_PARAMS.as_tuple())


def _session_key(scope_id: str, suffix: str) -> str:
    return f"calibration::{scope_id}::{suffix}"


def clear_data_caches() -> None:
    cached_load_chain.clear()
    cached_filter_chain.clear()
    cached_build_analytics.clear()
    cached_calibrate_chain.clear()


def model_params_from_meta(calibration_meta: dict[str, float]) -> HestonParameters:
    return HestonParameters(
        calibration_meta["v0"],
        calibration_meta["kappa"],
        calibration_meta["theta"],
        calibration_meta["sigma"],
        calibration_meta["rho"],
    )


def render_sidebar_controls(page_key: str) -> dict[str, object]:
    st.sidebar.header("Market Data")
    tickers_text = st.sidebar.text_input(
        "Tickers",
        value="NVDA",
        key=f"{page_key}_tickers",
        help="Comma-separated ticker list for live mode.",
    )
    r = st.sidebar.number_input(
        "Risk-free rate",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.005,
        format="%.4f",
        key=f"{page_key}_r",
    )
    q = st.sidebar.number_input(
        "Dividend yield",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.005,
        format="%.4f",
        key=f"{page_key}_q",
    )
    spread_limit = st.sidebar.number_input(
        "Spread limit",
        min_value=0.01,
        max_value=1.0,
        value=0.05,
        step=0.01,
        key=f"{page_key}_spread_limit",
    )
    refresh_requested = st.sidebar.button("Refresh options chain", key=f"{page_key}_refresh", use_container_width=True)

    st.sidebar.header("View Filters")
    option_types = st.sidebar.multiselect(
        "Option types",
        options=("call", "put"),
        default=("call", "put"),
        key=f"{page_key}_option_types",
    )
    min_volume = st.sidebar.number_input(
        "Min volume",
        min_value=0,
        value=1,
        step=1,
        key=f"{page_key}_min_volume",
    )
    min_open_interest = st.sidebar.number_input(
        "Min open interest",
        min_value=0,
        value=0,
        step=1,
        key=f"{page_key}_min_open_interest",
    )
    max_maturity = st.sidebar.number_input(
        "Max maturity (years)",
        min_value=0.1,
        max_value=5.0,
        value=2.0,
        step=0.1,
        key=f"{page_key}_max_maturity",
    )
    max_contracts = st.sidebar.number_input(
        "Max contracts in view",
        min_value=25,
        max_value=5000,
        value=2000,
        step=25,
        key=f"{page_key}_max_contracts",
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

    calibration_style_label = st.sidebar.selectbox(
        "Calibration style",
        ("Fast proxy calibration (Recommended)", "Full IV calibration"),
        index=0,
        key=f"{page_key}_calibration_style",
        help="Fast proxy calibration restricts the universe and uses European proxy pricing with price error.",
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
        max_value=2000,
        value=150,
        step=10,
        key=f"{page_key}_pricing_limit",
        help="Only the highest-priority contracts are repriced when model values are computed.",
    )
    Ns = st.sidebar.number_input("PDE stock steps", min_value=10, max_value=100, value=30, step=5, key=f"{page_key}_Ns")
    Nv = st.sidebar.number_input("PDE variance steps", min_value=10, max_value=100, value=15, step=5, key=f"{page_key}_Nv")
    Nt = st.sidebar.number_input("PDE time steps", min_value=10, max_value=100, value=30, step=5, key=f"{page_key}_Nt")
    max_expiries = st.sidebar.number_input(
        "Calibration expiries",
        min_value=1,
        max_value=12,
        value=4,
        step=1,
        key=f"{page_key}_max_expiries",
    )
    contracts_per_expiry = st.sidebar.number_input(
        "Contracts per expiry",
        min_value=2,
        max_value=20,
        value=4,
        step=1,
        key=f"{page_key}_contracts_per_expiry",
    )
    calibrate_requested = st.sidebar.button("Calibrate Heston", key=f"{page_key}_calibrate", use_container_width=True)

    calibration_style = "fast" if calibration_style_label.startswith("Fast") else "full"

    return {
        "tickers_text": tickers_text,
        "r": float(r),
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
        "max_expiries": int(max_expiries),
        "contracts_per_expiry": int(contracts_per_expiry),
        "calibration_style": calibration_style,
        "calibrate_requested": calibrate_requested,
    }


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
) -> tuple[pd.DataFrame, dict[str, int]]:
    return filter_chain_with_stats(
        raw_df,
        spread_limit=spread_limit,
        r=r,
        q=q,
        tickers=parse_tickers(tickers_text),
        option_types=option_types,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        max_maturity=max_maturity,
        max_contracts=max_contracts,
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
) -> pd.DataFrame:
    if model_mode == "Market metrics only":
        return build_chain_analytics(filtered_df, r=r, q=q)

    if model_mode == "Use existing/precomputed model prices":
        return build_chain_analytics(filtered_df, r=r, q=q)

    params = parse_heston_params(params_text)
    return build_chain_analytics(
        filtered_df,
        r=r,
        q=q,
        heston_params=params,
        compute_model_prices=True,
        pricing_limit=pricing_limit,
        Ns=Ns,
        Nv=Nv,
        Nt=Nt,
    )


@st.cache_data(show_spinner=False)
def cached_calibrate_chain(
    filtered_df: pd.DataFrame,
    r: float,
    q: float,
    Ns: int,
    Nv: int,
    Nt: int,
    max_expiries: int,
    contracts_per_expiry: int,
    calibration_style: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    result, calibration_df = calibrate_option_chain(
        filtered_df,
        r=r,
        q=q,
        Ns=Ns,
        Nv=Nv,
        Nt=Nt,
        max_expiries=max_expiries,
        contracts_per_expiry=contracts_per_expiry,
        calibration_style=calibration_style,
    )
    return result.as_dict(), calibration_df


def _handle_calibration_action(
    config: dict[str, object],
    filtered_df: pd.DataFrame,
    scope_id: str,
) -> None:
    if not config["calibrate_requested"]:
        return

    with st.spinner("Running Heston calibration..."):
        calibration_meta, calibration_df = cached_calibrate_chain(
            filtered_df,
            r=float(config["r"]),
            q=float(config["q"]),
            Ns=int(config["Ns"]),
            Nv=int(config["Nv"]),
            Nt=int(config["Nt"]),
            max_expiries=int(config["max_expiries"]),
            contracts_per_expiry=int(config["contracts_per_expiry"]),
            calibration_style=str(config["calibration_style"]),
        )

    st.session_state[_session_key(scope_id, "meta")] = calibration_meta
    st.session_state[_session_key(scope_id, "df")] = calibration_df
    save_calibration_result(scope_id, calibration_meta)
    st.success(
        "Calibration stored. "
        f"Loss={calibration_meta['loss']:.6f}, runtime={calibration_meta['runtime_seconds']:.2f}s"
    )
    st.rerun()


def _stored_calibration_for_scope(scope_id: str) -> tuple[dict[str, float] | None, pd.DataFrame | None]:
    meta_key = _session_key(scope_id, "meta")
    df_key = _session_key(scope_id, "df")

    calibration_meta = st.session_state.get(meta_key)
    calibration_df = st.session_state.get(df_key)
    if calibration_meta:
        return calibration_meta, calibration_df

    saved_meta = load_saved_calibration(scope_id)
    if saved_meta:
        st.session_state[meta_key] = saved_meta
        return saved_meta, None

    return None, None


def load_app_data(
    page_key: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, pd.DataFrame | None]:
    config = render_sidebar_controls(page_key)

    if config["refresh_requested"]:
        clear_data_caches()

    raw_df = cached_load_chain(tickers_text=str(config["tickers_text"]))
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
    )
    st.session_state["_filter_stats"] = filter_stats

    scope_id = calibration_scope_id(
        source="live",
        tickers_text=str(config["tickers_text"]),
        r=float(config["r"]),
        q=float(config["q"]),
        calibration_style=str(config["calibration_style"]),
    )
    _handle_calibration_action(config, filtered_df, scope_id)

    model_mode = str(config["model_mode"])
    calibration_meta: dict[str, float] | None = None
    calibration_df: pd.DataFrame | None = None

    if model_mode == "Use stored / calibrated Heston params":
        calibration_meta, calibration_df = _stored_calibration_for_scope(scope_id)
        if calibration_meta:
            params = model_params_from_meta(calibration_meta)
            analytics_df = build_chain_analytics(
                filtered_df,
                r=float(config["r"]),
                q=float(config["q"]),
                heston_params=params,
                compute_model_prices=True,
                pricing_limit=int(config["pricing_limit"]),
                Ns=int(config["Ns"]),
                Nv=int(config["Nv"]),
                Nt=int(config["Nt"]),
            )
        else:
            analytics_df = build_chain_analytics(filtered_df, r=float(config["r"]), q=float(config["q"]))
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

    return config, raw_df, filtered_df, analytics_df, calibration_meta, calibration_df


def render_chain_summary(raw_df: pd.DataFrame, filtered_df: pd.DataFrame, analytics_df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw contracts", f"{len(raw_df):,}")
    col2.metric("Filtered contracts", f"{len(filtered_df):,}")
    col3.metric("Expiries", analytics_df["maturity"].nunique() if "maturity" in analytics_df.columns else 0)
    col4.metric("Tickers", analytics_df["ticker"].nunique() if "ticker" in analytics_df.columns else 0)

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Market IV points", int(analytics_df["market_iv"].notna().sum()) if "market_iv" in analytics_df.columns else 0)
    col6.metric("Model IV points", int(analytics_df["model_iv"].notna().sum()) if "model_iv" in analytics_df.columns else 0)
    col7.metric("Median market delta", f"{analytics_df['market_delta'].median():.3f}" if "market_delta" in analytics_df.columns else "n/a")
    top_score = analytics_df["mispricing_score"].max() if "mispricing_score" in analytics_df.columns else None
    col8.metric("Top mispricing score", f"{top_score:.4f}" if pd.notna(top_score) else "n/a")


def render_calibration_panel(calibration_meta: dict[str, float] | None, calibration_df: pd.DataFrame | None) -> None:
    if not calibration_meta:
        return

    st.subheader("Calibration Metadata")
    st.json(calibration_meta)
    if calibration_df is not None and not calibration_df.empty:
        st.caption("Calibration universe")
        columns = [column for column in ["contract_id", "type", "maturity", "strike", "mid_price", "market_iv"] if column in calibration_df.columns]
        st.dataframe(calibration_df[columns], use_container_width=True, hide_index=True)


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
