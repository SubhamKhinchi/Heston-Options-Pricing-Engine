import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import pandas as pd
from data.option_filters import liquid_options

def year_fraction(maturity: str) -> float:
    try:
        expiry = pd.to_datetime(maturity, errors='coerce')
        if pd.isna(expiry):
            return float('nan')
        now = pd.Timestamp.now()
        return (expiry - now).total_seconds() / (365.0 * 24 * 3600)
    except Exception:
        return float('nan')

def arbitrage_filter(df, r, q):
    """
    Enforce no-arbitrage bounds for European options
    """
    df = df.copy()

    df['forward_spot'] = df['spot'] * np.exp(-q * df['T'])
    df['disc_strike'] = df['strike'] * np.exp(-r * df['T'])

    calls = df['type'] == 'call'
    puts = df['type'] == 'put'

    df['lower_bound'] = 0.0

    df.loc[calls, 'lower_bound'] = np.maximum(
        0, df['forward_spot'] - df['disc_strike']
    )

    df.loc[puts, 'lower_bound'] = np.maximum(
        0, df['disc_strike'] - df['forward_spot']
    )

    # Remove arbitrage-violating prices
    df = df[df['mid_price'] >= df['lower_bound'] - 1e-8]

    return df


def get_all_options(ticker, spread_limit, r, q):
    tk = yf.Ticker(ticker)
    spread = spread_limit
    spot = tk.history(period='1d')['Close'].iloc[-1]
    maturities = tk.options

    all_data=[]

    for maturity in maturities:
        # compute precise time-to-maturity
        T = year_fraction(maturity)

        if pd.isna(T) or T<= 0 :
            continue

        chain = tk.option_chain(maturity)

        calls = liquid_options(chain.calls, spread)
        puts = liquid_options(chain.puts, spread)

        calls['type'] = 'call'
        puts['type'] = 'put'

        df = pd.concat([calls, puts], ignore_index=True)
        # attach metadata
        df["maturity"] = maturity
        df["spot"] = spot
        df["ticker"] = ticker
        df['ExerciseStyle'] = 'american'

        # Clean strike
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        df = df.dropna(subset=['strike']).copy()

        # Mid price
        if 'bid' in df.columns and 'ask' in df.columns:
            df['mid_price'] = (df['bid'] + df['ask']) / 2.0
        else:
            df['mid_price'] = df.get('lastPrice', pd.NA)

        df = df.dropna(subset=['mid_price']).copy()

         # Time to maturity (per row)
        df['T'] = T
        
        # Remove invalid prices
        df = df[df['mid_price'] > 1e-3]

         # Spread filter (extra safety)
        df['rel_spread'] = (df['ask'] - df['bid']) / df['mid_price']
        df = df[df['rel_spread'] < 0.3]

        # Moneyness filter (CRITICAL)
        df['moneyness'] = df['strike'] / df['spot']
        df = df[
            (df['moneyness'] > 0.8) &
            (df['moneyness'] < 1.2)
        ]

         # Arbitrage filter (MOST IMPORTANT)
        df = arbitrage_filter(df, r, q)

        # Final cleanup
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['mid_price', 'T', 'strike'])

        all_data.append(df)

        if len(all_data) == 0:
            return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)
    

def get_multiple_tickers(tickers, spread_limit, r, q):
    all_tickers = []

    for ticker in tickers:
        print(f'pulling...{ticker}...')
        df = get_all_options(ticker, spread_limit, r, q)

        if not df.empty:
            all_tickers.append(df)
    
    if len(all_tickers)==0:
        return pd.DataFrame()
    
    return pd.concat(all_tickers, ignore_index=True)


