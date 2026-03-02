from models.heston_european import heston_european_call_price
from models.heston_european import heston_european_put_price

def heston_european_call_option(S0, K, r, T, v0, kappa, theta, sigma, rho):
    """
    Wrapper for European option pricing under Heston.

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike
    r : float
        Risk-free rate
    T : float
        Maturity
    kappa, theta, sigma, rho, v0 : Heston parameters

    Returns
    -------
    price : float
    """
    params = (v0, kappa, theta, sigma, rho)

    european_call_price = heston_european_call_price(
        S0=S0,
        K=K,
        r=r,
        T=T,
        params=params
    )

    return european_call_price

def heston_european_put_option(S0, K, r, T, v0, kappa, theta, sigma, rho):
    
    params = (kappa, theta, sigma, rho, v0)

    european_put_price = heston_european_put_price(
        S0=S0,
        K=K,
        r=r,
        T=T,
        params=params
    )

    return european_put_price