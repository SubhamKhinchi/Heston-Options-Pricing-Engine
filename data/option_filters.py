import pandas as pd


def liquid_options(df: pd.DataFrame, spread_limit=0.05):
    # Removing completely illiquid options
    df_volume = df[df.volume > 0].copy()

    # Calculating mid price
    mid = (df_volume.ask + df_volume.bid) / 2

    # Removing wide bid-ask spreads (relative to mid price)
    df_final = df_volume[(df_volume.ask - df_volume.bid) / mid < spread_limit]

    return df_final