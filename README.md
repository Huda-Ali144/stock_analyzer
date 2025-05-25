# 📈 Stock Analyzer

**Stock Analyzer** is an AI-powered financial analysis tool that combines technical indicators, fundamental ratios, sentiment analysis, and machine learning to generate real-time stock insights. Designed for investors and students, it delivers buy/hold/sell signals and chatbot-powered financial explanations using Google Gemini.

---

## 🔍 Features

- 📊 **Basic Metrics** – Price, P/E, Dividend Yield, Beta, Debt/Equity
- 🧮 **KPI Summary** – Categorized financial ratio analysis
- 📉 **5-Day Price History** – Chart and table of recent performance
- 🧠 **Next-Day Price Forecast** – Machine learning-based prediction
- 💡 **Buy/Hold/Sell Recommendation** – Based on fundamentals, sentiment, and forecast
- 📰 **News Sentiment** – Headline scraping with sentiment scoring
- 🤖 **AI Chat Assistant** – Ask investment questions, get insights

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

Update the top of app.py:
from dotenv import load_dotenv
import os
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

🚀 Run the App
streamlit run app.py

📁 Project Structure
Copy
Edit
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

📌 Future Features
🔮 Long-term forecast using LSTM

☁️ Deployment to Streamlit Cloud

💼 Portfolio tracking and comparison

📄 CSV Ticker Upload

👩‍💻 Author
Huda Bhayani
Aspiring AI Engineer | Finance Enthusiast | Problem Solver
GitHub




