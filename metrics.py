import numpy as np
from data_fetcher import fetch_info, fetch_history
from analyzer import compute_volatility, compute_technical_indicators, compute_trend

"""
metrics.py: Supplemental metrics for stock analysis
Includes:
  - Style classification (Value vs Growth)
  - Risk scoring (composite of volatility, beta, leverage)
  - Short-term and long-term technical signals
"""

def classify_style(fundamentals: dict) -> str:
    """
    Simple wrapper that classifies a stock style as 'Value', 'Growth', or 'Blend'.
    """
    style, _ = classify_style_explained(fundamentals)
    return style

def classify_style_explained(fundamentals: dict) -> tuple[str, str]:
    pe = fundamentals.get('pe_ratio', np.inf)
    pb = fundamentals.get('pb_ratio', np.inf)
    growth = fundamentals.get('earnings_growth_pct', 0.0)

    if pe < 15 and pb < 1.5:
        return "Value", f"Low P/E ({pe:.2f}) and P/B ({pb:.2f}) indicate undervaluation"
    if growth > 0.2:
        return "Growth", f"High earnings growth ({growth:.0%}) suggests aggressive expansion"
    return "Blend", "Moderate valuation and growth metrics"


def compute_risk_score(ticker: str,
                       days: int = 252,
                       hist_period: str = '1y') -> float:
    """
    Compute a composite risk score for a stock:
      - Annualized volatility (40% weight)
      - Beta (30% weight)
      - Debt-to-equity ratio (30% weight)
    Returns a normalized score in [0,1], higher means higher risk.
    """
    # Fetch fundamental info and history
    info = fetch_info(ticker)
    hist = fetch_history(ticker, period=hist_period)

    # Volatility
    vol = compute_volatility(hist).get('annual_volatility', 0)
    # Beta
    beta = info.get('beta', 1.0)
    # Leverage
    de = info.get('debtToEquity', 0.0)

    # Normalize components
    vol_norm = min(vol / 1.0, 1.0)          # assume 100% vol is max
    beta_norm = min(beta / 2.0, 1.0)       # assume beta 2 is high
    de_norm = min(de / 2.0, 1.0)           # assume D/E of 2 is high

    # Weighted sum
    score = 0.4 * vol_norm + 0.3 * beta_norm + 0.3 * de_norm
    return float(np.clip(score, 0.0, 1.0))


def short_term_signal(ticker: str,
                      hist_period: str = '1mo') -> str:
    """
    Generate a short-term technical signal based on 5-day vs 20-day SMA and RSI.
    Returns 'Bullish' if SMA5 > SMA20 and RSI < 70, else 'Bearish'.
    """
    hist = fetch_history(ticker, period=hist_period)
    tech = compute_technical_indicators(hist)
    # Compute SMA5
    sma5 = hist['Close'].rolling(window=5).mean().iloc[-1]
    sma20 = tech['SMA20'].iloc[-1]
    rsi = tech['RSI14'].iloc[-1]
    if sma5 > sma20 and rsi < 70:
        return 'Bullish'
    return 'Bearish'


def long_term_signal(ticker: str,
                     hist_period: str = '1y') -> str:
    """
    Generate a long-term trend signal via annualized trend on log prices.
    Returns 'Bullish' if trend > 0, else 'Bearish'.
    """
    hist = fetch_history(ticker, period=hist_period)
    trend_data = compute_trend(hist)
    trend = trend_data.get('annual_trend', 0.0)
    if trend > 0:
        return 'Bullish'
    return 'Bearish'
