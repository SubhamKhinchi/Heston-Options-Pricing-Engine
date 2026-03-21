import numpy as np
from pricing.american import american_call_without_dividends
from pricing.european import heston_european_call_option, heston_european_put_option
from pricing.heston_pde_american import heston_pde_american
from calibration.implied_vol import implied_volatility


def heston_loss(params, r, q, options_df, Ns, Nv, Nt):
    print("params type:", type(params), "value:", params)
    v0, kappa, theta, sigma, rho = params

    #variance should be positive
    if any (p<0 for p in [v0, kappa, theta, sigma]) or not (-1 < rho< 1):
        return 1e10
    
    errors = []

    for _, row in options_df.iterrows():
        S0= row['spot']
        K = row['strike']
        T = row['T']
        r = r
        market_price = row['mid_price'] #or mid price
        option_type = row['type']
        exercise_style = row['ExerciseStyle']
        if market_price <=0 :
            continue
        market_iv = implied_volatility(market_price, S0, K, r, T, option_type, q)
        
        if exercise_style.lower() == 'european':
                if row['type'] == 'call':
                    model_price = heston_european_call_option(S0, K, r,T, v0, kappa, theta, sigma, rho)
                elif row['type']=='put':
                    model_price = heston_european_put_option(S0, K, r, T, v0, kappa, theta, sigma, rho)
                else:
                    raise ValueError("European option_type must be 'call' or 'put'")
            
        elif exercise_style.lower()=='american':
            if row['type'] == 'call' and q ==0:
                model_price = american_call_without_dividends(S0, K, r, T, v0, kappa, theta, sigma, rho) #it's using heston CF under the hood
            elif row['type']=='call' and q != 0:
                model_price = heston_pde_american(S0, K, r, q, T, v0, kappa, theta, sigma, rho, option_type, Ns, Nv, Nt)
            elif row['type']== 'put' and q==0:
                model_price = heston_pde_american(S0, K, r, q, T, v0, kappa, theta, sigma, rho, option_type, Ns, Nv, Nt)
            elif row['type'] == 'put' and q!=0:
                model_price = heston_pde_american(S0, K, r, q, T, v0, kappa, theta, sigma, rho, option_type, Ns, Nv, Nt)
            else:
                raise ValueError("American option_type must be 'call' or 'put'")
        else: 
            raise ValueError("Exercise style is not european or american")
        
        model_iv = implied_volatility(model_price, S0, K, r, T, option_type, q)

        if np.isnan(model_iv) or np.isnan(market_iv):
            continue
            #ValueError('model_iv or market_iv is NaN')
        errors.append((model_iv - market_iv)**2)
    return np.mean(errors)