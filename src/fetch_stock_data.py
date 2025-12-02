import yfinance as yf
import pandas as pd
import warnings
from datetime import datetime, timedelta
from config import DEFAULT_STOCK_TICKERS as STOCK_TICKERS
from paths import get_data_path

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"No stock data found for {ticker}")
            return pd.DataFrame()
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Fix multi-index columns if present (yfinance bug)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Ensure we have the required columns
        required_cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        if not all(col in data.columns for col in required_cols):
            print(f"Warning: Missing required columns for {ticker}")
        
        return data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

def save_stock_data(ticker, data):
    """
    Save stock data to CSV.
    """
    try:
        filepath = get_data_path(f"{ticker}_stock_data.csv")
        data.to_csv(filepath, index=False)
    except Exception as e:
        print(f"Error saving stock data for {ticker}: {e}")

if __name__ == "__main__":
    import sys
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Last year

    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        data = fetch_stock_data(ticker, start_date, end_date)
        save_stock_data(ticker, data)
        print(f"Fetched and saved data for {ticker}")
    else:
        for ticker in STOCK_TICKERS:
            data = fetch_stock_data(ticker, start_date, end_date)
            save_stock_data(ticker, data)
            print(f"Fetched and saved data for {ticker}")