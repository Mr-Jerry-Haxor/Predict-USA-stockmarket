# Configuration file for the project

# Success Criteria
# - Model accuracy > 60% in predicting next-day stock price direction (up/down)
# - Sentiment analysis F1-score > 0.7
# - Dashboard provides real-time insights

SUCCESS_CRITERIA = {
    "model_accuracy": 0.6,
    "sentiment_f1": 0.7
}

# API Keys (set these for full functionality)
# IMPORTANT: For production, use environment variables instead of hardcoding
import os

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "85c68275be0a42c49540360a2e401c11")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "AAAAAAAAAAAAAAAAAAAAACVT4gEAAAAA3BzhTP%2BgAT%2BC4QWJEJxlWW6U704%3DOnX7xh5mXCC5nnPnpHEhTArIFvKPByNRnNDptgFjIz1QWsKCgL")

# RSS Feeds for financial news
RSS_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",  # WSJ Markets
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC
    "https://feeds.bloomberg.com/markets/news.rss",  # Bloomberg
]

# Selected Stock Tickers
STOCK_TICKERS = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "NFLX", "GOOG", "TSLA", "ORCL", "CRM", "ADBE", "INTC", "CSCO"]

DEFAULT_STOCK_TICKERS = STOCK_TICKERS

# Data Sources
ALPHA_VANTAGE_KEY = None  # For stock data if needed

# Model Parameters
MODEL_PARAMS = {
    "sentiment_model": "bert-base-uncased",
    "prediction_model": "RandomForest"
}