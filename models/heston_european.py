import numpy as np
from scipy.integrate import quad
from models.Heston_cf import heston_cf



def P_integral(params, S0, K, r, T, j, q=0.0):
    integrand = lambda u: np.real(np.exp(-1j*u*np.log(K)) * heston_cf(u, params, S0, r, T, j, q=q) / (1j*u))
    return 0.5 + (1/np.pi) * quad(integrand, 0, 100)[0]


def heston_european_call_price(S0, K, r, T, params, q=0.0):
    P1 = P_integral(params, S0, K, r, T, 1, q=q)
    P2 = P_integral(params, S0, K, r, T, 2, q=q)
    return S0*np.exp(-q*T)*P1 - K*np.exp(-r*T)*P2

def heston_european_put_price(S0, K, r, T, params, q=0.0):
    P1 = P_integral(params, S0, K, r, T, 1, q=q)
    P2 = P_integral(params, S0, K, r, T, 2, q=q)
    return K*np.exp(-r*T)*(1-P2) - S0*np.exp(-q*T)*(1-P1)