from models.heston_european import heston_european_call_price
from models.heston_european import heston_european_put_price

def heston_european_call_option(S0, K, r, T, v0, kappa, theta, sigma, rho, q=0.0):
    params = (v0, kappa, theta, sigma, rho)
    return heston_european_call_price(S0=S0, K=K, r=r, T=T, params=params, q=q)


def heston_european_put_option(S0, K, r, T, v0, kappa, theta, sigma, rho, q=0.0):
    params = (v0, kappa, theta, sigma, rho)
    return heston_european_put_price(S0=S0, K=K, r=r, T=T, params=params, q=q)