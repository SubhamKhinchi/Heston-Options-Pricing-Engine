import numpy as np
from pricing.american import american_call_without_dividends
from pricing.european import heston_european_call_option, heston_european_put_option
from pricing.heston_pde_american import heston_pde_american
from calibration.implied_vol import implied_volatility


def _model_price_from_row(row, params, r, q, Ns, Nv, Nt, pricing_mode):
    v0, kappa, theta, sigma, rho = params
    S0 = row.spot
    K = row.strike
    T = row.T
    option_type = row.type
    exercise_style = row.ExerciseStyle

    if pricing_mode == "european_proxy":
        if option_type == "call":
            return heston_european_call_option(S0, K, r, T, v0, kappa, theta, sigma, rho)
        if option_type == "put":
            return heston_european_put_option(S0, K, r, T, v0, kappa, theta, sigma, rho)
        raise ValueError("European option_type must be 'call' or 'put'")

    if exercise_style.lower() == "european":
        if option_type == "call":
            return heston_european_call_option(S0, K, r, T, v0, kappa, theta, sigma, rho)
        if option_type == "put":
            return heston_european_put_option(S0, K, r, T, v0, kappa, theta, sigma, rho)
        raise ValueError("European option_type must be 'call' or 'put'")

    if exercise_style.lower() == "american":
        if option_type == "call" and q == 0:
            return american_call_without_dividends(S0, K, r, T, v0, kappa, theta, sigma, rho)
        return heston_pde_american(S0, K, r, q, T, v0, kappa, theta, sigma, rho, option_type, Ns, Nv, Nt)

    raise ValueError("Exercise style is not european or american")


def heston_loss(
    params,
    r,
    q,
    options_df,
    Ns,
    Nv,
    Nt,
    objective="iv",
    pricing_mode="auto",
):
    v0, kappa, theta, sigma, rho = params

    #variance should be positive
    if any (p<0 for p in [v0, kappa, theta, sigma]) or not (-1 < rho< 1):
        return 1e10
    
    errors = []

    for row in options_df.itertuples(index=False):
        S0 = row.spot
        K = row.strike
        T = row.T
        market_price = row.mid_price
        option_type = row.type
        if market_price <=0 :
            continue

        model_price = _model_price_from_row(
            row=row,
            params=params,
            r=r,
            q=q,
            Ns=Ns,
            Nv=Nv,
            Nt=Nt,
            pricing_mode=pricing_mode,
        )

        if objective == "price":
            scale = max(abs(market_price), 1.0)
            errors.append(((model_price - market_price) / scale) ** 2)
            continue

        market_iv = getattr(row, "market_iv", np.nan)
        if np.isnan(market_iv):
            market_iv = implied_volatility(market_price, S0, K, r, T, option_type, q)

        model_iv = implied_volatility(model_price, S0, K, r, T, option_type, q)

        if np.isnan(model_iv) or np.isnan(market_iv):
            continue
        errors.append((model_iv - market_iv)**2)
    if not errors:
        return 1e10

    return float(np.mean(errors))
