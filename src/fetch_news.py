import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
from datetime import datetime, timedelta
import feedparser
from newsapi import NewsApiClient
import tweepy
from config import NEWS_API_KEY, TWITTER_BEARER_TOKEN
from paths import get_data_path

def fetch_tweets(ticker):
    """Fetch tweets related to ticker (requires X API keys)."""
    if not TWITTER_BEARER_TOKEN or TWITTER_BEARER_TOKEN == "your_twitter_bearer_token_here":
        return pd.DataFrame()
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        # Use simple keyword search instead of cashtag (cashtag not available in basic tier)
        query = f"{ticker} (stock OR shares OR price)"
        response = client.search_recent_tweets(query=query, max_results=100, tweet_fields=['created_at', 'text', 'id'])
        if not response.data:
            return pd.DataFrame()
        headlines = []
        for tweet in response.data:
            headlines.append({
                'ticker': ticker,
                'title': tweet.text,
                'link': f"https://twitter.com/i/web/status/{tweet.id}",
                'date': tweet.created_at.strftime('%Y-%m-%d'),
                'source': 'X (Twitter)'
            })
        return pd.DataFrame(headlines)
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return pd.DataFrame()

def fetch_news_from_yahoo(ticker):
    """Fetch news from Yahoo Finance."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        for item in soup.find_all('h3', class_='Mb(5px)'):
            title = item.text
            link = item.find('a')['href'] if item.find('a') else None
            date_elem = item.find_next('div', class_='C(#959595)')
            date = date_elem.text if date_elem else datetime.now().strftime('%Y-%m-%d')
            headlines.append({
                'ticker': ticker,
                'title': title,
                'link': link,
                'date': date,
                'source': 'Yahoo Finance'
            })
        return pd.DataFrame(headlines)
    except Exception as e:
        print(f"Error fetching from Yahoo Finance: {e}")
        return pd.DataFrame()

def fetch_news_from_newsapi(ticker, days_back=30):
    """Fetch news from NewsAPI with date range support."""
    if not NEWS_API_KEY or NEWS_API_KEY == "your_newsapi_key_here":
        print("⚠️ NewsAPI key not configured, skipping NewsAPI")
        return pd.DataFrame()
    
    from datetime import datetime, timedelta
    
    # Calculate date range for last N days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Format dates as YYYY-MM-DD for NewsAPI
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    print(f"📰 Fetching NewsAPI data from {from_date} to {to_date} for {ticker}...")
    
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    try:
        # Use everything endpoint with date range
        all_articles = newsapi.get_everything(
            q=ticker, 
            from_param=from_date,  # Start date
            to=to_date,             # End date
            language='en', 
            sort_by='publishedAt', 
            page_size=100
        )
    except Exception as e:
        print(f"❌ Error fetching from NewsAPI: {e}")
        return pd.DataFrame()
    
    if not all_articles or 'articles' not in all_articles:
        print("⚠️ NewsAPI returned no articles")
        return pd.DataFrame()
    
    print(f"✅ NewsAPI returned {len(all_articles['articles'])} articles")
        
    headlines = []
    for article in all_articles['articles']:
        headlines.append({
            'ticker': ticker,
            'title': article['title'],
            'link': article['url'],
            'date': article['publishedAt'][:10],  # YYYY-MM-DD
            'source': 'NewsAPI'
        })
    return pd.DataFrame(headlines)

def fetch_news_from_rss(ticker):
    """Fetch news from RSS feeds."""
    from dateutil import parser
    headlines = []
    feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            # Parse RSS date format (e.g., "Fri, 03 Oct 2024 12:00:00 GMT")
            try:
                if 'published' in entry:
                    parsed_date = parser.parse(entry.published)
                    date_str = parsed_date.strftime('%Y-%m-%d')
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')
            except:
                date_str = datetime.now().strftime('%Y-%m-%d')
            
            headlines.append({
                'ticker': ticker,
                'title': entry.title,
                'link': entry.link,
                'date': date_str,
                'source': 'RSS'
            })
    except Exception as e:
        print(f"Error fetching from RSS: {e}")
    return pd.DataFrame(headlines)

def fetch_all_news(ticker, days_back=30):
    """Fetch news from all sources with date range support."""
    df1 = fetch_news_from_yahoo(ticker)
    df2 = fetch_news_from_newsapi(ticker, days_back=days_back)  # Pass days_back parameter
    df3 = fetch_news_from_rss(ticker)
    df4 = fetch_tweets(ticker)
    combined = pd.concat([df1, df2, df3, df4], ignore_index=True)
    if combined.empty:
        print(f"Warning: No news data found for {ticker}")
        return pd.DataFrame()
    
    # Log source distribution
    if not combined.empty and 'source' in combined.columns:
        print(f"News sources breakdown: {combined['source'].value_counts().to_dict()}")
    
    return combined.drop_duplicates(subset=['title'], keep='first')

def save_news_data(ticker, df):
    """Save news data to CSV."""
    try:
        filepath = get_data_path(f"{ticker}_news.csv")
        if df.empty:
            print(f"Warning: No news data to save for {ticker}")
            # Create empty file to prevent downstream errors
            pd.DataFrame(columns=['ticker', 'title', 'link', 'date', 'source']).to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
    except Exception as e:
        print(f"Error saving news data for {ticker}: {e}")

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Enter stock ticker: ")
    df = fetch_all_news(ticker)
    save_news_data(ticker, df)
    print(f"Fetched and saved news for {ticker}")