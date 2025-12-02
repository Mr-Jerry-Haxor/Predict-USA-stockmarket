import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import DEFAULT_STOCK_TICKERS as STOCK_TICKERS
from paths import get_data_path

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess text data.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    words = text.split()
    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

def preprocess_news_data(ticker):
    """
    Load and preprocess news data for a ticker.
    """
    try:
        filepath = get_data_path(f"{ticker}_news.csv")
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"No news data for {ticker}, skipping preprocess.")
            return pd.DataFrame()
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"News data for {ticker} not found or empty: {e}")
        return pd.DataFrame()
    
    try:
        df['clean_title'] = df['title'].apply(clean_text)
        output_path = get_data_path(f"{ticker}_news_clean.csv")
        df.to_csv(output_path, index=False)
        print(f"Successfully preprocessed {len(df)} news items for {ticker}")
        return df
    except Exception as e:
        print(f"Error preprocessing news data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Enter stock ticker: ")
    preprocess_news_data(ticker)
    print(f"Preprocessed news data for {ticker}")