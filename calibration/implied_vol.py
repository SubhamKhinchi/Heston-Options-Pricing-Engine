import numpy as np
from scipy.optimize import brentq
from models.black_scholes import black_scholes_price

"""
Since there is no algebraic way to rearrange the Black-Scholes equation to solve for sigma, 
we use a numerical root-finding algorithm (like Newton-Raphson or Brent's method) to find the volatility 
that makes the Black-Scholes price equal to our Heston model price. Here we are using Brent's method.
"""

def implied_volatility(heston_model_price, S, K, T, r, option_type, q):

    if heston_model_price <= 0 or T <= 0:
        return np.nan
    
    def objective(sigma):
        return black_scholes_price(
            S, K, T, r, sigma, option_type, q
        ) - heston_model_price

    try:
        iv = brentq(objective, 1e-6, 5.0)
        return iv
    except ValueError:
        return np.nan