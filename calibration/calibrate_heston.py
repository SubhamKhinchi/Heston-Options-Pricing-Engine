import numpy as np
from scipy.optimize import minimize
from calibration.heston_loss_function import heston_loss

def calibrate_heston(r, q, option_df, Ns, Nv, Nt, initial_guess, bounds=None):
    if bounds is None:
        bounds=[(1e-4, 2), #v0
                (1e-4, 10), #kappa
                (1e-4, 2), #theta
                (1e-4, 2), #sigma
                (-0.999, 0.999)] #rho
    
    result = minimize(
        heston_loss, 
        x0=initial_guess, 
        args=(r, q, option_df, Ns, Nv, Nt),
        bounds = bounds,
        method='L-BFGS-B'
    )
    
    return result.x, result.fun