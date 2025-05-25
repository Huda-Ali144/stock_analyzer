import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer once
analyzer = SentimentIntensityAnalyzer()

def fetch_headlines(ticker: str, count: int = 5) -> list[str]:
    """
    1) Try Yahoo Finance RSS feed for up to `count` headlines.
    2) Fallback to yfinance.news if the RSS is empty.
    Returns list of titles or ["No headlines available"].
    """
    rss_url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={ticker}&region=US&lang=en-US"
    )
    try:
        feed = feedparser.parse(rss_url)
        titles = [entry.title.strip() for entry in feed.entries[:count] if entry.title]
        if titles:
            return titles
    except Exception:
        pass

    # Fallback: yfinance.news
    stock = yf.Ticker(ticker)
    try:
        items = stock.news or []
    except Exception:
        items = []
    titles = [
        item.get("title", "").strip()
        for item in items
        if item.get("title", "").strip()
    ]
    return titles[:count] or ["No headlines available"]

def analyze_news_sentiment(ticker: str, count: int = 5) -> tuple[list[str], str, float]:
    """
    Returns (headlines, label, avg_score), where:
      - headlines: list of str
      - label: one of 'Positive', 'Neutral', 'Negative', or 'No news'
      - avg_score: float in [-1.0, +1.0] representing the mean polarity
    """
    headlines = fetch_headlines(ticker, count)
    if headlines == ["No headlines available"]:
        return headlines, "No news", 0.0

    # Calculate compound scores from VADER
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    avg = sum(scores) / len(scores)

    # Map into sentiment label
    if avg > 0.1:
        label = "Positive"
    elif avg < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return headlines, label, avg
