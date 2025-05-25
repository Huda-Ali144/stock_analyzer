# stock_analyzer/app.py

import streamlit as st
import pandas as pd
import google.generativeai as genai

from data_fetcher import fetch_info, fetch_history
from analyzer import analyze_stock
from metrics import classify_style, compute_risk_score, short_term_signal, long_term_signal
from sentiment import analyze_news_sentiment
from price_predictor import RandomForestPredictor

# Configure Gemini
genai.configure(api_key="AIzaSyD6D3-kZ-WRM9EOmaBL3TovtBG22pSeI84")
model = genai.GenerativeModel("gemini-1.5-flash")

# — TRADE SIGNAL LOGIC —
def compute_trade_signal(info: dict,
                         fundamentals: dict[str, dict[str, float]],
                         sentiment_label: str,
                         next_price: float) -> tuple[str, str, float]:
    total = len(fundamentals)
    good_count = sum(
        1 for ratios in fundamentals.values() if classify_style(ratios) == "Good"
    )
    fund_score = (good_count / total) if total else 0.0

    sent_map = {"Positive": 1.0, "Neutral": 0.5, "Negative": 0.0}
    sent_score = sent_map.get(sentiment_label, 0.5)

    current = info.get("currentPrice", None)
    if current:
        pct = (next_price - current) / current
        pred_score = min(max((pct + 0.05) / 0.10, 0.0), 1.0)
    else:
        pred_score = 0.5

    composite = 0.4 * fund_score + 0.3 * sent_score + 0.3 * pred_score
    if composite > 0.6:
        action = "Buy"
    elif composite < 0.4:
        action = "Sell"
    else:
        action = "Hold"

    rationale = (
        f"Fundamentals: {fund_score:.2f}, "
        f"Sentiment: {sent_score:.2f}, "
        f"Forecast: {pred_score:.2f} → Composite: {composite:.2f}"
    )
    return action, rationale, composite

# — AI Summary Generator —
def generate_stock_summary_prompt(ticker, info, analysis, headlines, sentiment, style, risk_score, st_signal, lt_signal, recommendation) -> str:
    fundamentals = analysis.get("fundamentals", {})
    prompt = f"You are a helpful investment assistant.\n\nStock: {ticker}\n\n📈 Basic Info:\n- Price: ${info.get('currentPrice', 'N/A')}\n- Beta: {info.get('beta', 'N/A')}\n- PE Ratio: {info.get('trailingPE', 'N/A')}\n- Dividend Yield: {info.get('dividendYield', 0) * 100:.2f}%\n\n📈 Recommendation:\n- Style: {style}\n- Risk Score: {risk_score:.2f}\n- Short-Term Signal: {st_signal}\n- Long-Term Signal: {lt_signal}\n- Final Recommendation: {recommendation}\n\n🔑 KPI Summary:"

    for category, ratios in fundamentals.items():
        prompt += f"\n- {category.capitalize()}:"
        for metric, val in ratios.items():
            if isinstance(val, float):
                val = round(val, 4)
            prompt += f"\n   • {metric.replace('_', ' ').title()}: {val}"

    prompt += "\n\n📰 News Sentiment: " + sentiment
    prompt += "\nTop Headlines:\n" + "\n".join([f"• {h}" for h in headlines])
    prompt += "\n\nNow answer the user's investment questions about this stock."
    return prompt

# — Caching expensive calls —
@st.cache_data(ttl=600)
def get_info(ticker: str):
    return fetch_info(ticker)

@st.cache_data(ttl=600)
def get_history(ticker: str):
    return fetch_history(ticker, period="5d")

@st.cache_data(ttl=600)
def get_analysis(ticker: str):
    return analyze_stock(ticker)

@st.cache_data(ttl=600)
def get_headlines_and_sentiment(ticker: str, count: int = 5):
    return analyze_news_sentiment(ticker, count=count)

# — Helper DataFrame builders —
def get_basic_metrics_df(info: dict) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "Metric": ["Price", "P/E Ratio", "Dividend Yield", "Beta", "Debt/Equity"],
            "Value": [
                info.get("currentPrice", None),
                info.get("trailingPE", None),
                info.get("dividendYield", 0) * 100,
                info.get("beta", None),
                info.get("debtToEquity", None),
            ],
        })
        .assign(
            Value=lambda df: df["Value"].map(
                lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"
            )
        )
    )

def get_kpi_summary_df(analysis: dict) -> pd.DataFrame:
    rows = []
    for cat, ratios in analysis.get("fundamentals", {}).items():
        label = classify_style(ratios)
        rows.append({"Category": cat.title(), "Status": label})
    return pd.DataFrame(rows)

# — Main App —
st.set_page_config(page_title="StockAnalyzer", layout="wide")
st.title("📈 StockAnalyzer")

with st.sidebar:
    st.header("Profile & Ticker")
    ticker = st.text_input("Ticker", "AAPL").upper()

    st.markdown("---")
    st.subheader("Investment Profile")
    horizon = st.selectbox("Investment Horizon", ["Short-term (≤1yr)", "Long-term (>1yr)"])
    risk = st.selectbox("Risk Tolerance", ["Low (preservation)", "Medium", "High (aggressive)"])
    dividend = st.selectbox("Need Dividend Income?", ["No", "Yes"])

    st.markdown("---")
    view = st.selectbox(
        "What to show?",
        [
            "Basic Metrics",
            "5-Day History",
            "KPI Summary",
            "Headlines & Sentiment",
            "Next-Day Prediction",
            "Recommendation",
            "Full Analysis",
            "Ask the AI Assistant",
        ],
    )

info = get_info(ticker)
history = get_history(ticker)
analysis = get_analysis(ticker)
headlines, sentiment_label, sentiment_score = get_headlines_and_sentiment(ticker)

style = classify_style(analysis.get("fundamentals", {}).get("valuation", {}))
risk_score = compute_risk_score(ticker)
st_sig = short_term_signal(ticker)
lt_sig = long_term_signal(ticker)

if "next_price" not in st.session_state:
    st.session_state.next_price = None

st.markdown("### 🔐 Your Profile")
col1, col2, col3 = st.columns(3)
col1.metric("Horizon", horizon)
col2.metric("Risk", risk)
col3.metric("Dividend", dividend)
st.write("---")

if view == "Basic Metrics":
    st.subheader("🔎 Company Metrics")
    if not info:
        st.error(f"No data for {ticker}")
    else:
        df = get_basic_metrics_df(info)
        st.table(df)

elif view == "5-Day History":
    st.subheader("📈 5-Day Price History")
    if history is None or history.empty:
        st.warning("No historical data.")
    else:
        st.line_chart(history["Close"])
        st.dataframe(history[["Open", "High", "Low", "Close", "Volume"]].tail(5))

elif view == "KPI Summary":
    st.subheader("🔑 KPI Summary")
    kpi_df = get_kpi_summary_df(analysis)
    st.table(kpi_df)

elif view == "Headlines & Sentiment":
    st.subheader("📰 News Headlines & Sentiment")
    for h in headlines:
        st.write(f"• {h}")
    st.markdown(f"**Overall Sentiment:** {sentiment_label}  *(avg = {sentiment_score:.2f})*")

elif view == "Next-Day Prediction":
    st.subheader("🧠 Next-Day Price Forecast")
    predictor = RandomForestPredictor(ticker)
    predictor.fetch_data()
    predictor.feature_engineering()
    predictor.train_test()
    npred = predictor.predict_next_day()
    st.session_state.next_price = npred
    st.metric("Predicted Close", f"${npred:.2f}")
    st.metric("Model R²", f"{predictor.r2:.2f}")
    st.metric("Model MSE", f"{predictor.mse:.2f}")
    fig = predictor.get_plot_figure()
    st.pyplot(fig)

elif view == "Recommendation":
    st.subheader("💡 Composite Buy/Hold/Sell")
    if st.session_state.next_price is None:
        predictor = RandomForestPredictor(ticker)
        predictor.fetch_data()
        predictor.feature_engineering()
        predictor.train_test()
        st.session_state.next_price = predictor.predict_next_day()

    action, rationale, score = compute_trade_signal(
        info,
        analysis.get("fundamentals", {}),
        sentiment_label,
        st.session_state.next_price,
    )
    st.success(f"{action}  (score: {score:.2f})")
    st.write(rationale)

elif view == "Full Analysis":
    st.subheader("📊 Full Analysis")
    st.write("**Metrics**")
    st.table(get_basic_metrics_df(info))
    st.write("**History**")
    st.line_chart(history["Close"])
    st.write("**KPIs**")
    st.table(get_kpi_summary_df(analysis))
    st.write("**News & Sentiment**")
    for h in headlines:
        st.write(f"• {h}")
    st.markdown(f"**Sentiment:** {sentiment_label}  *(avg = {sentiment_score:.2f})*")
    st.write("**Forecast**")
    if st.session_state.next_price is None:
        predictor = RandomForestPredictor(ticker)
        predictor.fetch_data()
        predictor.feature_engineering()
        predictor.train_test()
        st.session_state.next_price = predictor.predict_next_day()
    st.metric("Next-Day Close", f"${st.session_state.next_price:.2f}")
    st.write("**Recommendation**")
    action, rationale, score = compute_trade_signal(
        info,
        analysis.get("fundamentals", {}),
        sentiment_label,
        st.session_state.next_price,
    )
    st.success(f"{action}  (score: {score:.2f})")
    st.write(rationale)

elif view == "Ask the AI Assistant":
    st.subheader("💬 Ask the AI Assistant")

    # Initialize session state if not already done
    if "chat_prompt" not in st.session_state or "chat_history" not in st.session_state:
        recommendation = compute_trade_signal(
            info, analysis.get("fundamentals", {}), sentiment_label, st.session_state.next_price or 0
        )[0]
        st.session_state.chat_prompt = generate_stock_summary_prompt(
            ticker, info, analysis, headlines, sentiment_label,
            style, risk_score, st_sig, lt_sig, recommendation
        )
        st.session_state.chat_history = [
            {"role": "user", "parts": [st.session_state.chat_prompt]}
        ]
        st.session_state.chat_session = model.start_chat(history=st.session_state.chat_history)

    # Display full chat history (excluding system prompt)
    for msg in st.session_state.chat_history[1:]:
        role = "🤖" if msg["role"] == "model" else "You"
        st.markdown(f"**{role}:** {msg['parts'][0]}")

    # Input box
    user_input = st.chat_input("Ask a question about this stock")

    # Handle user message
    if user_input:
        chat = st.session_state.chat_session
        response = chat.send_message(user_input)
        st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
        st.session_state.chat_history.append({"role": "model", "parts": [response.text]})
        st.rerun()
