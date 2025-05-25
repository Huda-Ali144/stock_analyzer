import pandas as pd
import numpy as np
from data_fetcher import fetch_info, fetch_history


def compute_liquidity_ratios(info: dict) -> dict:
    """
    Liquidity measures for debt coverage, cash flow, and working-capital health.
    """
    cur_assets = info.get("currentAssets", np.nan)
    cur_liabilities = info.get("currentLiabilities", np.nan)
    inventory = info.get("inventory", 0)
    cash = info.get("cash", 0)
    op_cf = info.get("operatingCashflow", np.nan)

    return {
        "working_capital": cur_assets - cur_liabilities,
        "current_ratio": cur_assets / cur_liabilities,
        "quick_ratio": (cur_assets - inventory) / cur_liabilities,
        "cash_ratio": cash / cur_liabilities,
        "operating_cf_ratio": op_cf / cur_liabilities,
    }


def compute_solvency_ratios(info: dict) -> dict:
    """
    Long-term leverage and capital structure stability.
    """
    total_debt = info.get("totalDebt", np.nan)
    total_liabilities = info.get("totalLiabilities", np.nan)
    equity = info.get("shareholdersEquity", np.nan)
    total_assets = info.get("totalAssets", np.nan)
    ebit = info.get("ebit", np.nan)
    interest_expense = info.get("interestExpense", np.nan)

    return {
        "debt_ratio": total_liabilities / total_assets,
        "debt_to_equity": total_debt / equity,
        "debt_to_assets": total_debt / total_assets,
        "equity_ratio": equity / total_assets,
        "interest_coverage": ebit / interest_expense,
    }


def compute_profitability_ratios(info: dict) -> dict:
    """
    Earnings efficiency and returns on capital.
    """
    revenue = info.get("totalRevenue", np.nan)
    gross_profit = info.get("grossProfit", np.nan)
    operating_income = info.get("operatingIncome", np.nan)
    net_income = info.get("netIncome", np.nan)
    assets = info.get("totalAssets", np.nan)
    equity = info.get("shareholdersEquity", np.nan)

    return {
        "gross_margin": gross_profit / revenue,
        "operating_margin": operating_income / revenue,
        "net_margin": net_income / revenue,
        "return_on_assets": net_income / assets,
        "return_on_equity": net_income / equity,
        "eps": net_income / info.get("sharesOutstanding", np.nan),
    }


def compute_efficiency_ratios(info: dict) -> dict:
    """
    Asset and working-capital turnover metrics plus operating cycle.
    """
    revenue = info.get("totalRevenue", np.nan)
    cost_of_revenue = info.get("costOfRevenue", np.nan)
    inventory = info.get("inventory", np.nan)
    receivables = info.get("accountsReceivable", np.nan)
    payables = info.get("accountsPayable", np.nan)

    receivables_turnover = revenue / receivables if receivables else np.nan
    inventory_turnover = cost_of_revenue / inventory if inventory else np.nan

    days_sales_rec = 365 / receivables_turnover if receivables_turnover else np.nan
    days_sales_inv = 365 / inventory_turnover if inventory_turnover else np.nan
    operating_cycle = days_sales_rec + days_sales_inv if not np.isnan(days_sales_rec + days_sales_inv) else np.nan

    return {
        "asset_turnover": revenue / info.get("totalAssets", np.nan),
        "inventory_turnover": inventory_turnover,
        "receivables_turnover": receivables_turnover,
        "payables_turnover": cost_of_revenue / payables,
        "days_sales_receivables": days_sales_rec,
        "days_sales_inventory": days_sales_inv,
        "operating_cycle": operating_cycle,
    }


def compute_valuation_ratios(info: dict) -> dict:
    """
    Market multiples, per-share metrics, and capitalized cash flows.
    """
    price = info.get("currentPrice", np.nan)
    eps_trail = info.get("epsTrailingTwelveMonths", np.nan)
    book_value = info.get("bookValue", np.nan)
    shares = info.get("sharesOutstanding", np.nan)
    revenue = info.get("totalRevenue", np.nan)
    sales_per_share = info.get("revenuePerShare", np.nan)
    market_cap = info.get("marketCap", np.nan)
    debt = info.get("totalDebt", 0)
    cash = info.get("cash", 0)
    ebitda = info.get("ebitda", np.nan)

    enterprise_value = market_cap + debt - cash
    dividend_yield = info.get("dividendYield", 0)
    payout_ratio = (info.get("dividendRate", 0) * shares) / info.get("netIncome", np.nan)
    free_cash_flow = info.get("freeCashflow", np.nan)

    return {
        "earnings_per_share": eps_trail,
        "book_value_per_share": book_value / shares,
        "pe_ratio": price / eps_trail,
        "pb_ratio": price / book_value,
        "ps_ratio": price / sales_per_share,
        "ev_ebitda": enterprise_value / ebitda,
        "ev_sales": enterprise_value / revenue,
        "dividend_yield": dividend_yield,
        "payout_ratio": payout_ratio,
        "free_cash_flow_yield": free_cash_flow / enterprise_value,
    }


def compute_growth_rates(info: dict) -> dict:
    """
    Year-over-year percentage changes.
    """
    return {
        "revenue_growth_pct": info.get("revenueGrowth", np.nan),
        "earnings_growth_pct": info.get("netIncomeGrowth", np.nan),
        "dividend_growth_pct": info.get("dividendGrowth", np.nan),
    }


def compute_cashflow_ratios(info: dict) -> dict:
    """
    Cash-flow margins and free-cash generation.
    """
    op_cf = info.get("operatingCashflow", np.nan)
    capex = info.get("capitalExpenditures", 0)
    revenue = info.get("totalRevenue", np.nan)
    free_cf = op_cf + capex

    return {
        "operating_cf_margin": op_cf / revenue,
        "free_cf": free_cf,
        "free_cf_margin": free_cf / revenue,
    }


def compute_technical_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Common technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands.
    """
    df = hist.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['SMA20'] + 2 * bb_std
    df['BB_lower'] = df['SMA20'] - 2 * bb_std

    return df


def compute_volatility(hist: pd.DataFrame) -> dict:
    """
    Annualized volatility from daily returns.
    """
    returns = hist['Close'].pct_change().dropna()
    annual_vol = returns.std() * np.sqrt(252)
    return {"annual_volatility": annual_vol}


def compute_trend(hist: pd.DataFrame) -> dict:
    """
    Annualized trend via regression on log prices.
    """
    df = hist.dropna(subset=['Close'])
    y = np.log(df['Close'].values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    annualized_return = np.exp(slope * 252) - 1
    return {"annual_trend": annualized_return}

def analyze_fundamentals(ticker: str) -> dict:
    info = fetch_info(ticker)
    if not info:
        return {}

    return {
        "liquidity": {
            "ratios": compute_liquidity_ratios(info),
            "description": "Measures ability to meet short-term obligations"
        },
        "solvency": {
            "ratios": compute_solvency_ratios(info),
            "description": "Indicates long-term debt handling capability"
        },
        "profitability": {
            "ratios": compute_profitability_ratios(info),
            "description": "Shows how efficiently the company earns profits"
        },
        "efficiency": {
            "ratios": compute_efficiency_ratios(info),
            "description": "Reveals how well assets are utilized"
        },
        "valuation": {
            "ratios": compute_valuation_ratios(info),
            "description": "Compares market value to fundamentals"
        },
        "growth": {
            "ratios": compute_growth_rates(info),
            "description": "Year-over-year revenue, earnings, and dividend growth"
        },
        "cashflow": {
            "ratios": compute_cashflow_ratios(info),
            "description": "Measures actual cash performance"
        },
    }


def analyze_technicals(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch price history and attach technical indicators.
    """
    hist = fetch_history(ticker, period=period)
    if hist is None or hist.empty:
        return pd.DataFrame()
    return compute_technical_indicators(hist)


def analyze_stock(ticker: str) -> dict:
    """
    Run full-spectrum analysis: fundamentals grouped by KPI, technicals, volatility, and trend.
    """
    result = {}
    result['fundamentals'] = analyze_fundamentals(ticker)

    hist = fetch_history(ticker, period="1y")
    if hist is not None and not hist.empty:
        tech = compute_technical_indicators(hist)
        result['technicals'] = tech.iloc[-1].to_dict()
        result['volatility'] = compute_volatility(hist)
        result['trend'] = compute_trend(hist)

    return result


if __name__ == "__main__":
    tick = input("Ticker: ").strip().upper()
    analysis = analyze_stock(tick)
    from pprint import pprint
    pprint(analysis)
def summarize_kpis(fundamentals: dict, model) -> str:
    prompt = "Provide a detailed investment summary based on these KPIs:\n"
    for category, data in fundamentals.items():
        prompt += f"\n🧩 {category.upper()} ({data['description']}):\n"
        for ratio, value in data["ratios"].items():
            prompt += f"  - {ratio.replace('_', ' ').title()}: {value:.2f}\n"
    prompt += "\nSummarize this in plain English for an investor."

    return model.generate_content(prompt).text

    