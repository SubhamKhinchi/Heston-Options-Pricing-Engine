import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

st.title("Heston Volatility Surface Analyzer")

# -----------------------
# Sidebar Controls
# -----------------------
surface_type = st.sidebar.selectbox(
    "Select Surface",
    ["Market IV", "Model IV", "Error Surface"]
)

# -----------------------
# Load your precomputed dataframe
# -----------------------
df = options_df_nvda_vol.copy()

df = df.dropna(subset=["market_iv", "model_iv", "moneyness", "T"])

# -----------------------
# Select Z
# -----------------------
X = df["moneyness"].values
Y = df["T"].values

if surface_type == "Market IV":
    Z = df["market_iv"].values
elif surface_type == "Model IV":
    Z = df["model_iv"].values
else:
    Z = (df["model_iv"] - df["market_iv"]).values

# -----------------------
# Grid interpolation
# -----------------------
x_grid = np.linspace(X.min(), X.max(), 50)
y_grid = np.linspace(Y.min(), Y.max(), 50)

X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

Z_grid = griddata((X, Y), Z, (X_grid, Y_grid), method="cubic")

# Fill NaNs
Z_grid_nearest = griddata((X, Y), Z, (X_grid, Y_grid), method="nearest")
Z_grid = np.where(np.isnan(Z_grid), Z_grid_nearest, Z_grid)

# -----------------------
# Plot
# -----------------------
fig = go.Figure(
    data=[go.Surface(
        x=X_grid,
        y=Y_grid,
        z=Z_grid,
        colorscale="Jet"
    )]
)

fig.update_layout(
    title=surface_type,
    scene=dict(
        xaxis_title="Moneyness (S/K)",
        yaxis_title="Time to Maturity",
        zaxis_title="Volatility"
    ),
    height=800
)

st.plotly_chart(fig)