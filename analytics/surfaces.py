"""
Interpolated volatility / greek surfaces.

`build_surface_grid()` interpolates a scattered analytics column (e.g. market_iv,
model_iv, a greek) over a (moneyness or strike) × maturity mesh and returns a
`SurfaceGrid`. Used to compare the market IV surface against the model IV surface.

Upstream:   enriched chains from analytics/chain_metrics.py.
Downstream: app/pages/05_Volatility_Surface.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def select_otm_smile(df: pd.DataFrame, *, atm_band: float = 0.0) -> pd.DataFrame:
    """Keep the out-of-the-money leg per strike — the clean one-IV-per-strike smile.

    Relative to the implied forward F (via `forward_moneyness` = K/F, falling back to
    spot `moneyness` = K/S), keep the OTM leg at each strike:
        K ≤ F  -> the put   (its ITM call mirror is dropped)
        K > F  -> the call  (its ITM put mirror is dropped)
    A call and put at the same (K, T) carry the same implied vol, so this removes the
    redundant double-count and the noisier ITM inversion, leaving the liquid OTM quote.

    `atm_band` widens the at-the-money zone (|ln(K/F)| ≤ band) where the marginally-OTM
    leg is kept; 0 splits exactly at F. This is strict OTM: ITM quotes are always dropped,
    including a lone ITM quote at a strike whose OTM mirror didn't survive filtering — an
    ITM-only quote is exactly the noisy inversion the clean smile is meant to exclude.
    """
    if df.empty or "type" not in df.columns:
        return df

    fm = df["forward_moneyness"] if "forward_moneyness" in df.columns else df.get("moneyness")
    if fm is None:
        return df
    log_fm = np.log(fm.where(fm > 0))
    is_put = df["type"].eq("put")
    is_call = df["type"].eq("call")

    # OTM leg per strike (with a symmetric ATM band kept on the marginally-OTM side).
    otm = (
        ((log_fm < -atm_band) & is_put)
        | ((log_fm > atm_band) & is_call)
        | ((log_fm.abs() <= atm_band) & (((log_fm <= 0) & is_put) | ((log_fm > 0) & is_call)))
    ).fillna(False)

    return df[otm].copy()


@dataclass(frozen=True)
class SurfaceGrid:
    x_grid: np.ndarray
    y_grid: np.ndarray
    z_grid: np.ndarray
    x_col: str
    y_col: str
    z_col: str
    point_count: int


def build_surface_grid(
    analytics_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    x_points: int = 50,
    y_points: int = 50,
    min_points: int = 8,
) -> SurfaceGrid:
    data = analytics_df[[x_col, y_col, z_col]].dropna()
    if len(data) < min_points:
        raise ValueError(f"Need at least {min_points} valid points to build a surface.")

    if data[x_col].nunique() < 2 or data[y_col].nunique() < 2:
        raise ValueError("Need at least two unique coordinates on both axes.")

    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()
    z = data[z_col].to_numpy()

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), x_points)
    y_grid = np.linspace(float(np.min(y)), float(np.max(y)), y_points)
    xx, yy = np.meshgrid(x_grid, y_grid)

    method = "cubic" if len(data) >= 16 else "linear"
    zz = griddata((x, y), z, (xx, yy), method=method)
    zz_nearest = griddata((x, y), z, (xx, yy), method="nearest")
    zz = np.where(np.isnan(zz), zz_nearest, zz)

    return SurfaceGrid(
        x_grid=xx,
        y_grid=yy,
        z_grid=zz,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        point_count=len(data),
    )

