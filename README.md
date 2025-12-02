# Real-Time Sentiment Analysis for Short-Term Stock Prediction

## Project Overview

This project analyzes market sentiment from news, RSS feeds, NewsAPI, and X (Twitter) to predict short-term stock price movements in the USA market. Users can select any USA-listed stock ticker, and the app fetches real-time data, processes it using BERT-based sentiment analysis, and provides ML-powered predictions for 1-day, 7-day, and 30-day horizons.

## Features

- **📊 Multi-Source Data Collection**: Yahoo Finance RSS, NewsAPI, X (Twitter) API
- **🧠 Advanced NLP**: BERT-based sentiment analysis
- **🤖 Machine Learning**: RandomForest classifiers for multiple prediction horizons
- **📈 Interactive Dashboard**: Beautiful Streamlit interface with real-time progress tracking
- **🔒 Secure Configuration**: Environment variable support for API keys

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional)

### Setup Steps

1. **Clone or download the repository**

   ```bash
   git clone <repository-url>
   cd "project - stockmarket AI"
   ```
2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```
3. **Activate the virtual environment**

   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
5. **Configure API Keys** (Optional but recommended)

   - Copy `.env.example` to `.env`
   - Add your API keys to `.env`:
     ```
     NEWS_API_KEY=your_newsapi_key_here
     TWITTER_BEARER_TOKEN=your_x_bearer_token_here
     ```
   - Get NewsAPI key from: https://newsapi.org/
   - Get X Bearer Token from: https://developer.x.com/
6. **Create required directories**

   ```bash
   mkdir data models
   ```

## Usage

### Running the Dashboard

1. **Start the Streamlit app**

   ```bash
   streamlit run src/dashboard.py
   ```
2. **Access the dashboard**

   - Open your browser to `http://localhost:8501`
3. **Analyze a stock**

   - Select a ticker from the dropdown (e.g., AAPL, GOOGL, MSFT)
   - Click "🚀 Analyze"
   - Wait for the pipeline to complete (2-5 minutes)
   - View predictions, charts, and statistics

### Running Individual Scripts

You can also run individual pipeline components:

```bash
# Fetch stock data
python src/fetch_stock_data.py AAPL

# Fetch news
python src/fetch_news.py AAPL

# Preprocess text
python src/preprocess_text.py AAPL

# Analyze sentiment
python src/sentiment_analysis.py AAPL

# Compute rolling sentiment
python src/rolling_sentiment.py AAPL

# Create labels
python src/create_labels.py AAPL

# Train models
python src/train_model.py AAPL
```

## Project Structure

```
project - stockmarket AI/
├── src/
│   ├── config.py              # Configuration and API keys
│   ├── dashboard.py           # Streamlit web application
│   ├── fetch_stock_data.py    # Stock price data fetcher
│   ├── fetch_news.py          # Multi-source news fetcher
│   ├── preprocess_text.py     # Text cleaning and preprocessing
│   ├── sentiment_analysis.py  # BERT sentiment analysis
│   ├── rolling_sentiment.py   # Rolling sentiment computation
│   ├── create_labels.py       # Feature and label creation
│   └── train_model.py         # ML model training
├── data/                      # CSV data files (auto-generated)
├── models/                    # Trained ML models (auto-generated)
├── requirements.txt           # Python dependencies
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Data Sources

- **Stock Prices**: Yahoo Finance via yfinance
- **News Headlines**:
  - Yahoo Finance RSS feeds (company-specific)
  - NewsAPI (requires API key)
  - X (Twitter) API (requires Bearer Token)

## Technical Details

### Sentiment Analysis

- Model: `nlptown/bert-base-multilingual-uncased-sentiment`
- Output: 1-5 star rating mapped to -1 to 1 sentiment score

### Prediction Models

- Algorithm: Random Forest Classifier
- Features: Rolling sentiment, closing price, volume
- Horizons: 1-day, 7-day, 30-day
- Train/Test Split: 80/20

### Success Criteria

- Model accuracy > 60% for price direction prediction
- Sentiment analysis F1-score > 0.7

## Production Deployment

### Environment Variables

Set these in your production environment:

- `NEWS_API_KEY`: Your NewsAPI key
- `TWITTER_BEARER_TOKEN`: Your X API Bearer Token

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repository
4. Add secrets in dashboard settings:
   ```
   NEWS_API_KEY = "your_key"
   TWITTER_BEARER_TOKEN = "your_token"
   ```
5. Deploy!

### Docker Deployment (Alternative)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "src/dashboard.py"]
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Yahoo Finance may rate-limit requests. Wait a few minutes and retry.
2. **Missing Data**: If news sources fail, the app will continue with available data from RSS feeds.
3. **Model Training Fails**: Ensure sufficient data is available (at least 10 samples per horizon).
4. **Memory Issues**: BERT model requires ~2GB RAM. Use a machine with adequate resources.

## Future Enhancements

- [ ] Add more data sources (Reddit, financial forums)
- [ ] Implement caching for API responses
- [ ] Add backtesting and performance metrics
- [ ] Support for international markets
- [ ] Real-time streaming predictions
- [ ] Portfolio-level analysis

## License

This project is for educational purposes.

## Support

For issues or questions, please open an issue in the repository.
