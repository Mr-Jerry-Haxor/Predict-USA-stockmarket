import pandas as pd
from config import DEFAULT_STOCK_TICKERS as STOCK_TICKERS
from paths import get_data_path

def compute_rolling_sentiment(ticker):
    """
    Compute rolling sentiment indices.
    """
    try:
        filepath = get_data_path(f"{ticker}_news_sentiment.csv")
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"No sentiment data for {ticker}, skipping rolling sentiment.")
            return pd.DataFrame()
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Sentiment data for {ticker} not found or empty: {e}")
        return pd.DataFrame()
    
    try:
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')

        # Map sentiment to numeric: assuming 1-5 scale, convert to -1 to 1
        sentiment_map = {'1 star': -1, '2 stars': -0.5, '3 stars': 0, '4 stars': 0.5, '5 stars': 1}
        df['sentiment_score'] = df['sentiment'].map(sentiment_map).fillna(0)

        # Daily average sentiment
        daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()
        daily_sentiment['rolling_sentiment'] = daily_sentiment['sentiment_score'].rolling(window=7, min_periods=1).mean()

        output_path = get_data_path(f"{ticker}_daily_sentiment.csv")
        daily_sentiment.to_csv(output_path, index=False)
        print(f"Successfully computed rolling sentiment for {ticker}")
        return daily_sentiment
    except Exception as e:
        print(f"Error computing rolling sentiment for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Enter stock ticker: ")
    compute_rolling_sentiment(ticker)
    print(f"Computed rolling sentiment for {ticker}")