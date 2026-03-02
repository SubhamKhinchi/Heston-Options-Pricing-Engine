import numpy as np
from scipy.integrate import quad


def heston_cf(u, params, S0, r, T, j):

    v0, kappa, theta, sigma, rho = params
    x = np.log(S0)

    if j == 1:
        b = kappa - rho*sigma
        u_bar = 0.5
    else:
        b = kappa
        u_bar = -0.5

    d = np.sqrt((rho*sigma*1j*u - b)**2 - sigma**2*(2*u_bar*1j*u - u**2))
    g = (b - rho*sigma*1j*u + d) / (b - rho*sigma*1j*u - d)

    C = r*1j*u*T + (kappa*theta/sigma**2)*(
        (b - rho*sigma*1j*u + d)*T
        - 2*np.log((1 - g*np.exp(d*T)) / (1 - g))
    )

    D = (b - rho*sigma*1j*u + d)/sigma**2 * (
        (1 - np.exp(d*T)) / (1 - g*np.exp(d*T))
    )

    return np.exp(C + D*v0 + 1j*u*x)