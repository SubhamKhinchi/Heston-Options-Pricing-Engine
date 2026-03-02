# ---------------------------
# 2️⃣ LSMC(Longstaff–Schwartz regression) American Option Pricing 
# ---------------------------
import numpy as np

#Applicable for american put (with or w/o dividends)
def american_put_lsmc_vec(S, K, r, dt):
    N, M = S.shape
    M -= 1
    #Terminal payoff
    cashflow = np.maximum(K - S[:,-1], 0)

    #backward induction
    for t in range(M-1, 0, -1):
        X = S[:,t]
        #identify in-the-money paths
        itm = X < K
        if np.sum(itm) == 0:
            #only ITM paths matter for exercise decisions
            cashflow *= np.exp(-r*dt)
            continue
        #discount continuation value
        Y = cashflow * np.exp(-r*dt)
        #regression step (in the future, include v_t also for regression)
        coeffs = np.polyfit(X[itm], Y[itm], 2)
        #computing continuation value
        C = np.polyval(coeffs, X[itm])
        #exercise decision
        #exercise if : Immediate payoff > Continuation value
        exercise = (K - X[itm]) > C

        #update cashflow
        cashflow[itm] = np.where(exercise, K - X[itm], Y[itm])
        cashflow[~itm] *= np.exp(-r*dt)

        #discount to time 0
    return np.mean(cashflow * np.exp(-r*dt))

#Applicable for american call (with or w/o dividends)
def american_call_lsmc_vec(S, K, r, dt):
    N, M = S.shape
    M -= 1
    #Terminal payoff
    cashflow = np.maximum(S[:,-1] - K, 0)

    #backward induction
    for t in range(M-1, 0, -1):
        X = S[:,t]
        #identify in-the-money paths (counting number of paths in the money)
        itm = X > K
        if np.sum(itm) == 0: #if all paths are out-of-the-money--> no early exercise possible
            #we just discount the existing cashflows one time step back
            #only ITM paths matter for exercise decisions
            cashflow *= np.exp(-r*dt)
            continue # continue --> skip the regression and exercise logic fof this time step
        
        Y = cashflow * np.exp(-r*dt)
        coeffs = np.polyfit(X[itm], Y[itm], 2)
        C = np.polyval(coeffs, X[itm])
        exercise = (X[itm] - K) > C
        cashflow[itm] = np.where(exercise, X[itm] - K, Y[itm])
        cashflow[~itm] *= np.exp(-r*dt)
    return np.mean(cashflow * np.exp(-r*dt))