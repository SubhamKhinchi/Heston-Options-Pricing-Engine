import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_price(S, K, r, T, sigma, option_type, q):
    """
    European Black-Scholes price with continuous dividend yield.
    """
    if T <= 0:
        if option_type.lower() == "call":
            return max(S - K, 0.0)
        elif option_type.lower() == "put":
            return max(K - S, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type.lower() == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    elif option_type.lower() == "put":
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

    else:
        raise ValueError("option_type must be 'call' or 'put'")