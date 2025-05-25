import sys
import numpy as np
import pandas as pd
from data_fetcher import fetch_info, fetch_history
from analyzer import analyze_stock
from metrics import classify_style, compute_risk_score, short_term_signal, long_term_signal
from sentiment import analyze_news_sentiment
from utils import ask_user_profile
from price_predictor import RandomForestPredictor

# Gemini
import google.generativeai as genai
genai.configure(api_key="AIzaSyD6D3-kZ-WRM9EOmaBL3TovtBG22pSeI84")
model = genai.GenerativeModel("gemini-pro")

# Evaluation helpers
def evaluate_ratio(value: float, thresholds=(1.0, 1.0)) -> str:
    if value is None or np.isnan(value):
        return "N/A"
    low, high = thresholds
    if value > high:
        return "Good"
    if low <= value <= high:
        return "Neutral"
    return "Poor"

def evaluate_category(ratios: dict[str, float], thresholds_map: dict[str, tuple[float, float]]) -> str:
    labels = [evaluate_ratio(val, thresholds_map.get(name, (0.0, 1.0))) for name, val in ratios.items()]
    goods = labels.count("Good")
    poors = labels.count("Poor")
    if goods >= len(labels) / 2:
        return "Good"
    if poors >= len(labels) / 2:
        return "Poor"
    return "Neutral"

THRESHOLDS = {
    "current_ratio": (1.0, 1.5),
    "quick_ratio": (1.0, 1.5),
    "cash_ratio": (0.5, 1.0),
    "debt_to_equity": (0.0, 1.0),
    "interest_coverage": (1.0, 2.0),
    "net_margin": (0.0, 0.1),
    "return_on_assets": (0.0, 0.05),
    "asset_turnover": (0.2, 0.5),
    "pe_ratio": (10.0, 20.0),
}

def print_company_info(info: dict, ticker: str):
    df = pd.DataFrame({
        'Metric': ['Price', 'P/E Ratio', 'Dividend Yield', 'Beta', 'Debt/Equity'],
        'Value': [
            f"${info.get('currentPrice', np.nan):.2f}",
            f"{info.get('trailingPE', np.nan):.2f}",
            f"{info.get('dividendYield', 0)*100:.2f}%",
            f"{info.get('beta', np.nan):.2f}",
            f"{info.get('debtToEquity', np.nan):.2f}"
        ]
    })
    print(f"\n🔎 {info.get('longName', 'N/A')} ({ticker})")
    print(df.to_string(index=False))

def print_historical_data(hist: pd.DataFrame, days: int = 5):
    if hist is None or hist.empty:
        print("❗ No historical data.")
        return
    df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].tail(days)
    print(f"\n📈 Last {days} Days of Prices:")
    print(df.to_string())

def print_kpi_summary(analysis: dict):
    print(f"\n🔑 KPI Summary:")
    for category, ratios in analysis.get('fundamentals', {}).items():
        print(f"   {category.title():12}: {evaluate_category(ratios, THRESHOLDS)}")

def print_headlines_sentiment(headlines: list[str], sentiment: str):
    print(f"\n📰 Top Headlines:")
    for h in headlines:
        print(f"   • {h}")
    print(f"\n📰 News Sentiment : {sentiment}\n")

def predict_next_day_price(ticker: str):
    print(f"\n🧠 Predicting next-day price using Random Forest...")
    predictor = RandomForestPredictor(ticker)
    predictor.fetch_data()
    predictor.feature_engineering()
    predictor.train_test()
    next_price = predictor.predict_next_day()
    print(f"\n📊 Predicted Next-Day Close for {ticker}: ${next_price:.2f}")
    print(f"📈 Model R² Score: {predictor.r2:.2f}")
    print(f"📉 Model MSE: {predictor.mse:.2f}")

def print_recommendation(style: str, risk_score: float, st_signal: str, lt_signal: str, recommendation: str):
    df = pd.DataFrame({
        'Metric': ['Style', 'Risk Score', 'Short Signal', 'Long Signal', 'Recommendation'],
        'Value': [style, f"{risk_score:.2f}", st_signal, lt_signal, recommendation]
    })
    print(f"\n💡 Recommendation Details:")
    print(df.to_string(index=False))
    print("\n🔍 What it means:")
    if recommendation == 'Buy':
        print("• BUY means the model and trend analysis suggest the stock price is likely to rise. Consider adding to your position.")
    elif recommendation == 'Sell':
        print("• SELL means the analysis forecasts a downward trend. It may be prudent to reduce exposure.")
    else:
        print("• HOLD means the outlook is neutral. Maintain your current position and watch for clearer signals.")

def generate_stock_summary_prompt(ticker, info, analysis, headlines, sentiment, style, risk_score, st_signal, lt_signal, recommendation) -> str:
    fundamentals = analysis.get("fundamentals", {})
    prompt = f"""You are a helpful investment assistant.

A user asked for advice on stock '{ticker}'. Here is the analysis:

📊 Basic Info:
- Price: ${info.get('currentPrice', 'N/A')}
- Beta: {info.get('beta', 'N/A')}
- PE Ratio: {info.get('trailingPE', 'N/A')}
- Dividend Yield: {info.get('dividendYield', 0) * 100:.2f}%

📈 Recommendation Overview:
- Style: {style}
- Risk Score: {risk_score:.2f}
- Short-Term Signal: {st_signal}
- Long-Term Signal: {lt_signal}
- Final Recommendation: {recommendation}

🔑 KPI Summary:"""

    for category, data in fundamentals.items():
        prompt += f"\n- {category.capitalize()}:"
        for metric, val in data["ratios"].items():
            if isinstance(val, float):
                val = round(val, 4)
            prompt += f"\n   • {metric.replace('_', ' ').title()}: {val}"

    prompt += "\n\n📰 News Sentiment: " + sentiment
    prompt += "\nTop Headlines:\n" + "\n".join([f"• {h}" for h in headlines])
    prompt += "\n\nNow answer the user's investment questions about this stock."
    return prompt

# Main CLI Logic
if __name__ == '__main__':
    ticker = input("Enter a stock ticker (e.g. AAPL): ").strip().upper()
    ask_user_profile()

    info = fetch_info(ticker)
    if not info:
        print(f"❗ No data for {ticker}.")
        sys.exit(1)

    history = fetch_history(ticker, period="5d")
    analysis = analyze_stock(ticker)
    headlines, sentiment = analyze_news_sentiment(ticker, count=5)
    trend = analysis.get('trend', {})

    style = classify_style(analysis.get('fundamentals', {}).get('valuation', {}))
    risk_score = compute_risk_score(ticker)
    st_signal = short_term_signal(ticker)
    lt_signal = long_term_signal(ticker)
    trend_val = trend.get('annual_trend', 0)
    recommendation = 'Buy' if trend_val > 0 else 'Sell' if trend_val < 0 else 'Hold'

    menu = [
        "Basic Metrics",
        "5-Day History",
        "KPI Summary",
        "Headlines & Sentiment",
        "Recommendation",
        "Next-Day Prediction",
        "Full Analysis",
        "Ask the AI Assistant",
        "Exit"
    ]

    while True:
        print("\nWhat would you like to see?")
        for i, item in enumerate(menu, 1):
            print(f"{i}) {item}")
        choice = input("Enter the number of your choice: ").strip()

        if choice == '1':
            print_company_info(info, ticker)
        elif choice == '2':
            print_historical_data(history)
        elif choice == '3':
            print_kpi_summary(analysis)
        elif choice == '4':
            print_headlines_sentiment(headlines, sentiment)
        elif choice == '5':
            print_recommendation(style, risk_score, st_signal, lt_signal, recommendation)
        elif choice == '6':
            predict_next_day_price(ticker)
        elif choice == '7':
            print_company_info(info, ticker)
            print_historical_data(history)
            print_kpi_summary(analysis)
            print_headlines_sentiment(headlines, sentiment)
            print_recommendation(style, risk_score, st_signal, lt_signal, recommendation)
            predict_next_day_price(ticker)
        elif choice == '8':
            prompt = generate_stock_summary_prompt(
                ticker, info, analysis, headlines, sentiment,
                style, risk_score, st_signal, lt_signal, recommendation
            )
            chat = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
            print("\n💬 You are now chatting with the AI assistant. Type 'exit' to leave.")
            while True:
                user_msg = input("You: ")
                if user_msg.lower() in ['exit', 'quit']:
                    break
                response = chat.send_message(user_msg)
                print(f"\n🤖: {response.text.strip()}\n")
        else:
            print("Goodbye!")
            break

    print("\nDone.")
