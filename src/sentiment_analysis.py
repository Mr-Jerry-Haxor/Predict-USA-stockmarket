import pandas as pd
from transformers import pipeline
from config import DEFAULT_STOCK_TICKERS as STOCK_TICKERS
from paths import get_data_path

# Load sentiment analysis pipeline
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment(text):
    """
    Analyze sentiment of text using BERT.
    Returns label and score.
    """
    try:
        if not text or len(text.strip()) == 0:
            return "3 stars", 0.5  # Neutral default
        # Truncate to model max length
        text = text[:512]
        result = sentiment_pipeline(text)[0]
        return result['label'], result['score']
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "3 stars", 0.5

def apply_sentiment_to_news(ticker):
    """
    Apply sentiment analysis to news titles.
    """
    try:
        filepath = get_data_path(f"{ticker}_news_clean.csv")
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"No clean news data for {ticker}, skipping sentiment analysis.")
            return pd.DataFrame()
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Clean news data for {ticker} not found or empty: {e}")
        return pd.DataFrame()
    
    try:
        print(f"Analyzing sentiment for {len(df)} news items...")
        df['sentiment'], df['confidence'] = zip(*df['clean_title'].apply(analyze_sentiment))
        output_path = get_data_path(f"{ticker}_news_sentiment.csv")
        df.to_csv(output_path, index=False)
        print(f"Successfully analyzed sentiment for {ticker}")
        return df
    except Exception as e:
        print(f"Error applying sentiment analysis for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Enter stock ticker: ")
    apply_sentiment_to_news(ticker)
    print(f"Applied sentiment analysis to {ticker} news")