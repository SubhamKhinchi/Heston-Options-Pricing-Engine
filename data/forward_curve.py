"""Implied forward curve from put-call parity.

For each expiry we recover the implied forward F(T) directly from the option
chain, rather than trusting a vendor dividend yield. The forward — not a
dividend yield — is the object option prices actually depend on, and it is
independent of the option's exercise style.

European put-call parity:
    C - P = e^{-rT} (F - K)

So across the common call/put strikes of one expiry, regressing (C - P) on K
gives a line whose slope is -e^{-rT} and whose intercept is e^{-rT} F. Hence:

    F = intercept / discount,   discount = -slope = e^{-rT}

Two facts make this robust and exercise-style-aware:

1. **Near-ATM window.** The equality form of parity is exact only for European
   options. For American options the early-exercise premium breaks it — but
   that premium is concentrated in *deep-ITM* strikes (especially ITM puts).
   Near the money it is negligible, so we fit only on strikes close to the
   money. Cash-settled index options (SPX/NDX) are European, so parity is exact
   there regardless. We select the window with *coarse* spot-based moneyness
   (K/S): the forward sits within ~2% of spot for these maturities, so a K/S
   band reliably brackets the strikes near the money — no forward needed yet,
   which breaks the chicken-and-egg with the downstream forward-based filter.

2. **Sanity clamp.** A single-name forward cannot sit miles from S·e^{rT}.
   Fits implying an absurd carry (|q| above a generous cap) are rejected and
   fall back to a no-dividend forward, so a noisy expiry never poisons pricing.

The implied dividend yield q_implied(T) = r(T) - ln(F/S)/T is exposed only as a
*diagnostic*; the forward F(T) is the carried object. Because q = r-ln(F/S)/T
is amplified by 1/T at short maturities, q is unstable for weeklies even when F
is clean — which is exactly why we carry F, not q.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForwardPoint:
    maturity: str
    T: float
    forward: float
    discount: float        # e^{-rT} implied by the regression
    implied_q: float       # continuous dividend yield consistent with F (diagnostic)
    n_pairs: int           # call/put pairs used in the near-ATM fit
    r2: float
    ok: bool               # passed every quality gate (else forward is the no-div fallback)


def _select_atm_window(
    merged: pd.DataFrame, spot: float, *, atm_band: float, min_pairs: int
) -> pd.DataFrame:
    """Keep call/put pairs near the money, widening the band until enough pairs.

    Uses coarse spot-based moneyness |K/S - 1|; the forward is within ~2% of
    spot, so this reliably brackets the money without needing F first.
    """
    rel = (merged["strike"] / spot - 1.0).abs()
    band = atm_band
    while True:
        win = merged[rel <= band]
        if len(win) >= min_pairs or band >= 0.5:
            return win
        band += 0.05


def _fit_one_expiry(
    sub: pd.DataFrame,
    spot: float,
    r: float,
    *,
    min_pairs: int,
    min_r2: float,
    atm_band: float,
    max_abs_q: float,
) -> ForwardPoint | None:
    """Fit the implied forward for a single expiry from its near-ATM call/put pairs."""
    maturity = sub["maturity"].iloc[0]
    T = float(sub["T"].iloc[0])
    if T <= 0:
        return None

    calls = sub[sub["type"] == "call"][["strike", "mid_price"]]
    puts = sub[sub["type"] == "put"][["strike", "mid_price"]]
    merged = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
    merged = merged[(merged["mid_price_c"] > 0) & (merged["mid_price_p"] > 0)]

    if merged.empty:
        return _fallback(maturity, T, spot, r, 0)

    # Restrict to a near-ATM window: minimises American early-exercise bias and
    # excludes the widest-spread deep strikes.
    merged = _select_atm_window(merged, spot, atm_band=atm_band, min_pairs=min_pairs)

    n = len(merged)
    if n < min_pairs:
        return _fallback(maturity, T, spot, r, n)

    K = merged["strike"].to_numpy(dtype=float)
    cmp_ = (merged["mid_price_c"] - merged["mid_price_p"]).to_numpy(dtype=float)

    # Linear fit C - P = a + b K  ->  slope b = -e^{-rT}, intercept a = e^{-rT} F
    slope, intercept = np.polyfit(K, cmp_, 1)
    if slope >= 0:  # economically impossible; chain too noisy
        return _fallback(maturity, T, spot, r, n)

    discount = -slope
    forward = intercept / discount
    if forward <= 0:
        return _fallback(maturity, T, spot, r, n)

    # R^2 of the fit
    pred = slope * K + intercept
    ss_res = float(np.sum((cmp_ - pred) ** 2))
    ss_tot = float(np.sum((cmp_ - cmp_.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    implied_q = r - np.log(forward / spot) / T

    # Quality gates: enough near-ATM pairs, a clean linear fit, AND a forward
    # that implies an economically plausible carry. A blown-up regression
    # (e.g. F = 5x spot from deep-ITM American-put contamination) implies a
    # wild |q| and is rejected here, falling back to the no-dividend forward.
    ok = (n >= min_pairs) and (r2 >= min_r2) and (abs(implied_q) <= max_abs_q)
    if not ok:
        return _fallback(maturity, T, spot, r, n)

    return ForwardPoint(maturity, T, float(forward), float(discount),
                        float(implied_q), n, float(r2), True)


def _fallback(maturity: str, T: float, spot: float, r: float, n: int) -> ForwardPoint:
    """No reliable forward — return a no-dividend forward S·e^{rT}, flagged ok=False.

    The forward is still economically sane (never the garbage regressed value),
    so it is safe to carry downstream; the caller treats ok=False as a signal to
    source the dividend from the trailing-yield fallback instead.
    """
    forward = spot * np.exp(r * T)
    return ForwardPoint(maturity, T, float(forward), float(np.exp(-r * T)),
                        0.0, n, 0.0, False)


def build_forward_curve(
    df: pd.DataFrame,
    *,
    spot: float,
    r_by_T: dict[float, float] | float = 0.0,
    min_pairs: int = 4,
    min_r2: float = 0.99,
    atm_band: float = 0.10,
    max_abs_q: float = 0.25,
) -> list[ForwardPoint]:
    """Build an implied-forward curve over all expiries in ``df``.

    ``df`` needs columns: type, strike, mid_price, maturity, T.
    ``r_by_T`` may be a scalar or a dict {T: r}; per-expiry rate is matched by
    nearest T when a dict is supplied.

    ``atm_band``   — half-width of the near-ATM K/S window for the parity fit.
    ``max_abs_q``  — reject fits whose implied continuous yield exceeds this
                     (sanity bound against blown-up regressions; not a target).
    """
    if df.empty:
        return []

    points: list[ForwardPoint] = []
    for _maturity, sub in df.groupby("maturity"):
        T = float(sub["T"].iloc[0])
        if isinstance(r_by_T, dict) and r_by_T:
            keys = np.array(sorted(r_by_T))
            r = float(r_by_T[keys[int(np.argmin(np.abs(keys - T)))]])
        else:
            r = float(r_by_T) if not isinstance(r_by_T, dict) else 0.0
        pt = _fit_one_expiry(
            sub, spot, r,
            min_pairs=min_pairs, min_r2=min_r2,
            atm_band=atm_band, max_abs_q=max_abs_q,
        )
        if pt is not None:
            points.append(pt)

    points.sort(key=lambda p: p.T)
    return points


def forward_curve_frame(points: list[ForwardPoint]) -> pd.DataFrame:
    """Tabular view of a forward curve (for display / debugging)."""
    return pd.DataFrame([{
        "maturity": p.maturity,
        "T": round(p.T, 4),
        "forward": round(p.forward, 4),
        "implied_q": round(p.implied_q, 5),
        "n_pairs": p.n_pairs,
        "r2": round(p.r2, 4),
        "ok": p.ok,
    } for p in points])
