import yfinance as yf
import pandas as pd
from datetime import datetime
from data.option_filters import liquid_options

def year_fraction(maturity: str) -> float:
    today = datetime.today()
    expiry = datetime.strptime(maturity, "%Y-%m-%d")
    return (expiry - today).days / 365.0

def get_all_options(ticker, spread_limit):
    tk = yf.Ticker(ticker)
    spread = spread_limit
    spot = tk.history(period='1d')['Close'].iloc[-1]
    maturities = tk.options

    all_data=[]

    for maturity in maturities:
        T= year_fraction(maturity)

        chain = tk.option_chain(maturity)

        calls = liquid_options(chain.calls, spread)
        puts = liquid_options(chain.puts, spread)

        calls['type'] = 'call'
        puts['type'] = 'put'

        df = pd.concat([calls, puts])
        df["maturity"] = maturity
        df["T"] = T
        df["spot"] = spot
        df["ticker"] = ticker
        df['ExerciseStyle'] = 'american'

        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)
    

def get_multiple_tickers(tickers, spread_limit):
    all_tickers = []

    for ticker in tickers:
        print(f'pulling...{ticker}...')
        df = get_all_options(ticker, spread_limit)
        all_tickers.append(df)
    return pd.concat(all_tickers, ignore_index=True)


