"""Instrument classification for dividend treatment and exercise style.

The bucket a ticker falls into determines two things the pricing engine needs:

1. **Exercise style** — cash-settled index options (SPX, NDX, DJX, RUT) are
   European; ETF options (SPY, QQQ, DIA, IWM) and single stocks are American.
2. **Dividend method** — single stocks and ETFs pay *discrete* cash dividends;
   cash indices are modelled with a *continuous* dividend yield.

The primary forward-curve method (implied forward from put-call parity) is
type-agnostic, so this classifier only decides the *fallback* dividend method
and the exercise style.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Hard overrides ────────────────────────────────────────────────────────────
# Never fully trust a vendor field for something this load-bearing. The handful
# of tickers we actually care about are pinned here.
_INDEX_OVERRIDES = {
    "SPX", "^SPX", "^GSPC", "XSP",          # S&P 500 index
    "NDX", "^NDX", "XND",                    # Nasdaq-100 index
    "DJX", "^DJI",                           # Dow Jones index
    "RUT", "^RUT",                           # Russell 2000 index
    "VIX", "^VIX",                           # Volatility index
}
_ETF_OVERRIDES = {
    "SPY", "QQQ", "DIA", "IWM", "VOO", "IVV",
    "EEM", "EFA", "GLD", "SLV", "TLT", "HYG", "XLF", "XLE", "XLK",
}

INSTRUMENT_TYPES = ("EQUITY", "ETF", "INDEX")


@dataclass(frozen=True)
class InstrumentInfo:
    ticker: str
    instrument_type: str   # EQUITY | ETF | INDEX
    exercise_style: str    # american | european
    dividend_method: str   # discrete | continuous
    source: str            # how the classification was decided


def _yf_quote_type(ticker: str) -> str | None:
    """Best-effort yfinance quoteType lookup. Returns None on any failure."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        qt = info.get("quoteType")
        return str(qt).upper() if qt else None
    except Exception:
        return None


def classify(ticker: str, *, use_network: bool = True) -> InstrumentInfo:
    """Classify a ticker into EQUITY / ETF / INDEX.

    Resolution order (most trustworthy first):
      1. Hardcoded override maps
      2. ``^`` prefix convention  -> INDEX
      3. yfinance ``quoteType``   -> EQUITY / ETF / INDEX
      4. default                  -> EQUITY (safest single-name treatment)
    """
    t = ticker.strip().upper()

    if t in _INDEX_OVERRIDES:
        return _build(t, "INDEX", "override")
    if t in _ETF_OVERRIDES:
        return _build(t, "ETF", "override")
    if t.startswith("^"):
        return _build(t, "INDEX", "caret_prefix")

    if use_network:
        qt = _yf_quote_type(t)
        if qt == "INDEX":
            return _build(t, "INDEX", "yfinance_quoteType")
        if qt == "ETF":
            return _build(t, "ETF", "yfinance_quoteType")
        if qt in ("EQUITY", "MUTUALFUND"):
            return _build(t, "EQUITY", "yfinance_quoteType")

    return _build(t, "EQUITY", "default")


def _build(ticker: str, instrument_type: str, source: str) -> InstrumentInfo:
    if instrument_type == "INDEX":
        exercise, div_method = "european", "continuous"
    else:  # EQUITY, ETF
        exercise, div_method = "american", "discrete"
    return InstrumentInfo(
        ticker=ticker,
        instrument_type=instrument_type,
        exercise_style=exercise,
        dividend_method=div_method,
        source=source,
    )
