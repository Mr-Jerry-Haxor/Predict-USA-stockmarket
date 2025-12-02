import pandas as pd
from config import DEFAULT_STOCK_TICKERS
from paths import get_data_path

def create_labels(ticker, horizons=[1, 7, 30]):
    """
    Create labels based on stock price movements for different horizons (days).
    """
    try:
        stock_path = get_data_path(f"{ticker}_stock_data.csv")
        sentiment_path = get_data_path(f"{ticker}_daily_sentiment.csv")
        stock_df = pd.read_csv(stock_path)
        sentiment_df = pd.read_csv(sentiment_path)
    except FileNotFoundError as e:
        print(f"Data for {ticker} not found: {e}")
        return None
    
    if stock_df.empty or sentiment_df.empty:
        print(f"Data for {ticker} is empty, skipping label creation.")
        return None

    try:
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='mixed', errors='coerce')
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], format='mixed', errors='coerce')
        
        # Drop rows with invalid dates
        stock_df = stock_df.dropna(subset=['Date'])
        sentiment_df = sentiment_df.dropna(subset=['date'])
        
        # Log date ranges for debugging
        print(f"Stock data date range: {stock_df['Date'].min()} to {stock_df['Date'].max()} ({len(stock_df)} rows)")
        print(f"Sentiment data date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()} ({len(sentiment_df)} rows)")

        # Merge on date
        merged = pd.merge(stock_df, sentiment_df, left_on='Date', right_on='date', how='inner')
        
        if merged.empty:
            print(f"❌ No matching dates between stock and sentiment data for {ticker}")
            print(f"   Stock: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
            print(f"   Sentiment: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
            return None
        
        print(f"✅ Merged {len(merged)} rows with matching dates")

        # Ensure numeric types
        merged['Close'] = pd.to_numeric(merged['Close'], errors='coerce')
        merged['Volume'] = pd.to_numeric(merged['Volume'], errors='coerce')
        merged['rolling_sentiment'] = pd.to_numeric(merged['rolling_sentiment'], errors='coerce')
        
        # Features: rolling_sentiment, Close, Volume
        features = merged[['rolling_sentiment', 'Close', 'Volume']].fillna(0)

        labels = {}
        for h in horizons:
            merged[f'label_{h}d'] = (merged['Close'].shift(-h) > merged['Close']).astype(int)
            lbl = merged[f'label_{h}d'].dropna()
            # Align
            min_len = min(len(features), len(lbl))
            
            # Adjusted minimum requirements based on horizon
            # For short horizons (1-day), we can work with very limited samples (for demo purposes)
            # Note: More samples = better model accuracy
            if h == 1:
                min_required = 3  # Absolute minimum for 1-day prediction (demo only)
            elif h == 7:
                min_required = 5  # Reduced from 10 - for 7-day prediction (works with limited data)
            else:
                min_required = 10  # Reduced from 15 - for 30-day prediction
            
            if min_len < min_required:
                print(f"⚠️ Insufficient data for {ticker} {h}-day horizon ({min_len} samples, need {min_required})")
                continue
            
            features_h = features.iloc[:min_len]
            lbl_h = lbl.iloc[:min_len]
            features_path = get_data_path(f"{ticker}_features_{h}d.csv")
            labels_path = get_data_path(f"{ticker}_labels_{h}d.csv")
            features_h.to_csv(features_path, index=False)
            lbl_h.to_frame(name='label').to_csv(labels_path, index=False)
            labels[h] = (features_h, lbl_h)
            print(f"Created {min_len} labeled samples for {ticker} {h}-day horizon")

        return labels
    except Exception as e:
        print(f"Error creating labels for {ticker}: {e}")
        return None

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Enter stock ticker: ")
    create_labels(ticker)
    print(f"Created labels for {ticker}")