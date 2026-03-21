from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


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

