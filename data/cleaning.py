
def arbitrage_filter(df):
    calls = df['type'] == 'call'
    puts = df['type'] == 'put'

    df['lower_bound'] = 0.0

    df.loc[calls, 'lower_bound'] = np.maximum(
        0, df['forward_spot'] - df['disc_strike']
    )

    df.loc[puts, 'lower_bound'] = np.maximum(
        0, df['disc_strike'] - df['forward_spot']
    )

    # Remove invalid prices
    df = df[df['mid_price'] >= df['lower_bound'] - 1e-8]

    return df
