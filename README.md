# 📈 Stock Analyzer

**Stock Analyzer** is an AI-powered financial analysis tool that combines technical indicators, fundamental ratios, sentiment analysis, and machine learning to generate real-time stock insights. Designed for investors and students, it delivers buy/hold/sell signals and chatbot-powered financial explanations using Google Gemini.

🌐 Live App: smart-stock-tool.streamlit.app

---

## 🔍 Features

📊 Basic Metrics – P/E Ratio, Beta, Dividend Yield, Debt/Equity, etc.

🧮 KPI Summary – Financial ratio breakdown by category

📉 5-Day Price History – Chart and table of recent stock performance

🧠 Next-Day Price Forecast – Random Forest model prediction

💡 Buy/Hold/Sell Recommendation – Composite score based on fundamentals, sentiment, and predictions

📰 News Sentiment – Headline scraping with VADER-based analysis

🤖 AI Assistant – Conversational Gemini-powered Q&A on the stock

---

## 🛠️ Getting Started

### 1. Clone the Repository
git clone https://github.com/Huda-Ali144/stock_analyzer.git
cd stock_analyzer

2. Set Up a Virtual Environment
python -m venv lstm_env
lstm_env\Scripts\activate  # On Windows

3. Install Dependencies
pip install -r requirements.txt

4. Add Your Gemini API Key
Create a .env file in the root folder:
GEMINI_API_KEY=your-api-key-here

🚀 Run the App
streamlit run app.py

📁 Project Structure
stock_analyzer/
├── app.py
├── analyzer.py
├── data_fetcher.py
├── metrics.py
├── price_predictor.py
├── sentiment.py
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md

🧠 Tech Stack
Frontend: Streamlit
ML Model: Random Forest Regressor
Finance Data: yfinance
LLM Assistant: Google Gemini (generativeai)
News Parsing: RSS via feedparser + VADER

📌 Future Features
🔮 Long-term forecast using LSTM
💼 Portfolio tracking and comparison
📄 CSV Ticker Upload

👩‍💻 Author
Huda Bhayani
Aspiring AI Engineer | Finance Enthusiast | Problem Solver
GitHub




