import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime, timedelta
from config import DEFAULT_STOCK_TICKERS
from paths import get_data_path, get_model_path

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import pipeline functions
from fetch_stock_data import fetch_stock_data, save_stock_data
from fetch_news import fetch_all_news, save_news_data
from preprocess_text import preprocess_news_data
from sentiment_analysis import apply_sentiment_to_news
from rolling_sentiment import compute_rolling_sentiment
from create_labels import create_labels
from train_model import (
    train_all_models, predict_with_model, get_model_list, 
    get_model_description, DEFAULT_MODEL, AVAILABLE_MODELS,
    get_available_models_for_ticker, train_model
)

st.set_page_config(page_title="Stock Sentiment Predictor", page_icon="📈", layout="wide")

st.title("📈 Real-Time Sentiment Analysis for Short-Term Stock Prediction")

# Check if NewsAPI is configured
from config import NEWS_API_KEY
newsapi_configured = NEWS_API_KEY and NEWS_API_KEY != "your_newsapi_key_here" and len(NEWS_API_KEY) > 20

# Important notice about data with NewsAPI status
if newsapi_configured:
    st.success("""
    ✅ **NewsAPI Configured!** You can now fetch up to 100 articles from the last 30 days.
    Select your preferred historical period below (7-30 days recommended).
    """)
else:
    st.info("""
    **ℹ️ Important Note:** This system analyzes **recent news only** (last 7-30 days).  
    News APIs don't provide historical data, so you'll see best results with:
    - 🔑 NewsAPI configured (FREE at newsapi.org) - **Recommended!**
    - 📊 60-90 day historical period
    - 📈 High-volume tickers (AAPL, TSLA, NVDA, MSFT)
    """)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This dashboard predicts stock price direction based on news sentiment analysis.")
    st.markdown("### How it works:")
    st.write("1. 📊 Fetches historical stock data")
    st.write("2. 📰 Collects news from multiple sources")
    st.write("3. 🧠 Analyzes sentiment using BERT")
    st.write("4. 🤖 Trains ML models for prediction")
    st.markdown("---")
    
    # Model Selection Section
    st.markdown("### 🤖 Model Selection")
    selected_model = st.selectbox(
        "Choose Prediction Model",
        options=get_model_list(),
        index=0,  # Gradient Boosting is first (default)
        help="Select the ML model for predictions. Gradient Boosting is recommended."
    )
    
    # Show model description
    model_desc = get_model_description(selected_model)
    if selected_model == DEFAULT_MODEL:
        st.success(f"{model_desc}")
    else:
        st.info(f"{model_desc}")
    
    st.markdown("---")
    
    # API Configuration Check
    st.markdown("### 🔑 API Status")
    from config import NEWS_API_KEY, TWITTER_BEARER_TOKEN
    
    if NEWS_API_KEY and NEWS_API_KEY != "your_newsapi_key_here":
        st.success("✅ NewsAPI configured")
    else:
        st.warning("⚠️ NewsAPI not configured")
    
    if TWITTER_BEARER_TOKEN and TWITTER_BEARER_TOKEN != "your_twitter_bearer_token_here":
        st.success("✅ X/Twitter API configured")
    else:
        st.warning("⚠️ X/Twitter API not configured")
    
    st.caption("💡 RSS & Yahoo Finance work without API keys")
    st.markdown("---")
    st.info("💡 Select a ticker and click 'Analyze' to start")

ticker = st.selectbox("🔍 Select USA Stock Ticker", options=DEFAULT_STOCK_TICKERS, index=0)

# Date range selector
col_date1, col_date2 = st.columns(2)
with col_date1:
    days_back = st.selectbox(
        "📅 Historical Period",
        options=[7, 14, 21, 28],
        index=3,  # Default to 28 days
        format_func=lambda x: f"Last {x} days",
        help="⚠️ News APIs only provide recent articles. Select 14-28 days for best results."
    )

# Info message
if days_back > 28:
    st.warning("⚠️ **Note**: News APIs typically provide only recent articles (last 28 days). Selecting longer periods may result in insufficient overlapping data.")
else:
    st.info(f"ℹ️ Analyzing last **{days_back} days** - optimal for news availability")

if ticker:
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    if analyze_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        error_occurred = False

        # Step 1: Fetch Stock Data
        status_text.text("📊 Fetching stock data...")
        progress_bar.progress(10)
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            stock_data = fetch_stock_data(ticker, start_date, end_date)
            if stock_data.empty:
                st.error(f"❌ Failed to fetch stock data for {ticker}")
                error_occurred = True
            else:
                save_stock_data(ticker, stock_data)
                st.success("✅ Stock data fetched successfully")
        except Exception as e:
            st.error(f"❌ Error fetching stock data: {e}")
            error_occurred = True

        # Step 2: Fetch News
        if not error_occurred:
            status_text.text(f"📰 Collecting news from last {days_back} days...")
            progress_bar.progress(30)
            try:
                news_data = fetch_all_news(ticker, days_back=days_back)  # Pass days_back parameter
                save_news_data(ticker, news_data)
                if news_data.empty:
                    st.warning(f"⚠️ No news data found, continuing...")
                else:
                    st.success(f"✅ News data collected ({len(news_data)} articles from last {days_back} days)")
            except Exception as e:
                st.warning(f"⚠️ Some news sources failed: {e}")

        # Step 3: Preprocess
        if not error_occurred:
            status_text.text("🔄 Processing text data...")
            progress_bar.progress(40)
            try:
                clean_data = preprocess_news_data(ticker)
                if clean_data.empty:
                    st.error("❌ No data to preprocess")
                    error_occurred = True
            except Exception as e:
                st.error(f"❌ Failed to preprocess data: {e}")
                error_occurred = True
            
            if not error_occurred:
                status_text.text("🧠 Analyzing sentiment...")
                progress_bar.progress(50)
                try:
                    sentiment_data = apply_sentiment_to_news(ticker)
                    if sentiment_data.empty:
                        st.error("❌ No data for sentiment analysis")
                        error_occurred = True
                except Exception as e:
                    st.error(f"❌ Failed sentiment analysis: {e}")
                    error_occurred = True
            
            if not error_occurred:
                status_text.text("📊 Computing rolling sentiment...")
                progress_bar.progress(60)
                try:
                    rolling_data = compute_rolling_sentiment(ticker)
                    if rolling_data is None or rolling_data.empty:
                        st.error("❌ Failed to compute rolling sentiment")
                        error_occurred = True
                    else:
                        st.success("✅ Data processed successfully")
                except Exception as e:
                    st.error(f"❌ Failed rolling sentiment: {e}")
                    error_occurred = True

        # Step 4: Create Labels and Train
        if not error_occurred:
            status_text.text("🏷️ Creating labels...")
            progress_bar.progress(70)
            try:
                labels = create_labels(ticker)
                if labels is None or len(labels) == 0:
                    # Provide helpful guidance
                    st.error("❌ **Insufficient Data for Training**")
                    
                    # Check why and provide specific advice
                    try:
                        stock_path = get_data_path(f"{ticker}_stock_data.csv")
                        sentiment_path = get_data_path(f"{ticker}_daily_sentiment.csv")
                        
                        if os.path.exists(stock_path) and os.path.exists(sentiment_path):
                            stock_df = pd.read_csv(stock_path)
                            sentiment_df = pd.read_csv(sentiment_path)
                            
                            st.warning(f"""
                            **Data Available:**
                            - 📊 Stock data: {len(stock_df)} days
                            - 📰 News sentiment: {len(sentiment_df)} days
                            - ⚠️ Only {len(sentiment_df)} days have both stock and news data
                            
                            **Why?** News APIs only provide recent articles (last 7-30 days), not historical data.
                            
                            **Solutions:**
                            1. 🔑 **Configure NewsAPI** (FREE) - Get more historical news coverage (30 days vs 7 days)
                               - Register at: https://newsapi.org/register
                               - Add key to `.env` file: `NEWS_API_KEY=your_key_here`
                            
                            2. 📈 **Try a different ticker** - Some stocks have more news coverage:
                               - AAPL, TSLA, NVDA typically have 50-200 articles/month
                            
                            3. ⏰ **Try again tomorrow** - News updates daily, you'll accumulate more data over time
                            
                            4. 📊 **Use 60-90 day period** - Gives more time for news overlap (though still limited by API)
                            
                            Need at least 5 days of overlapping data for 1-day predictions.
                            """)
                    except:
                        st.warning("Check terminal output for details about insufficient data.")
                    
                    error_occurred = True
            except Exception as e:
                st.error(f"❌ Failed to create labels: {e}")
                error_occurred = True
            
            if not error_occurred:
                status_text.text("🤖 Training ALL models (one-time)...")
                progress_bar.progress(80)
                try:
                    # Train ALL models at once - user can switch between them without re-fetching data
                    model_results = train_all_models(ticker, horizons=[1, 7])
                    if model_results and len(model_results) > 0:
                        total_models = sum(len(models) for models in model_results.values())
                        st.success(f"✅ {total_models} models trained successfully!")
                        
                        # Show which models were trained
                        with st.expander("📊 Model Training Results", expanded=False):
                            for horizon, models in model_results.items():
                                st.markdown(f"**{horizon}-Day Horizon:**")
                                for model_name, info in models.items():
                                    marker = "🏆" if model_name == DEFAULT_MODEL else ""
                                    st.write(f"  {marker} {model_name}: {info['accuracy']:.2%} accuracy")
                    else:
                        st.warning("⚠️ No models trained - check data requirements")
                except Exception as e:
                    st.warning(f"⚠️ Some models may not have trained: {e}")

        # Step 5: Display Results (show even if some steps failed)
        progress_bar.progress(100)
        status_text.text("✨ Analysis completed!")

        st.markdown("---")
        st.header(f"📊 Results for {ticker}")
        
        # Try to load and display data even if there were errors
        try:
            stock_path = get_data_path(f"{ticker}_stock_data.csv")
            sentiment_path = get_data_path(f"{ticker}_daily_sentiment.csv")
            
            if os.path.exists(stock_path) and os.path.exists(sentiment_path):
                stock_df = pd.read_csv(stock_path)
                sentiment_df = pd.read_csv(sentiment_path)

                stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                
                # Ensure numeric columns are proper types
                stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce')
                stock_df['Volume'] = pd.to_numeric(stock_df['Volume'], errors='coerce')
                sentiment_df['rolling_sentiment'] = pd.to_numeric(sentiment_df['rolling_sentiment'], errors='coerce')

                # Create two columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot stock price
                    fig_stock = px.line(stock_df, x='Date', y='Close', 
                                       title=f"{ticker} Stock Price (Last Year)",
                                       labels={'Close': 'Closing Price ($)', 'Date': 'Date'})
                    fig_stock.update_layout(hovermode='x unified')
                    st.plotly_chart(fig_stock, use_container_width=True)

                with col2:
                    # Plot sentiment
                    fig_sentiment = px.line(sentiment_df, x='date', y='rolling_sentiment', 
                                           title=f"{ticker} Sentiment Score (7-day Rolling)",
                                           labels={'rolling_sentiment': 'Sentiment Score', 'date': 'Date'})
                    fig_sentiment.update_layout(hovermode='x unified')
                    fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
                    st.plotly_chart(fig_sentiment, use_container_width=True)

                # Predictions
                st.markdown("---")
                st.subheader(f"🔮 Predictions using **{selected_model}**")
                
                # Add model switch info
                if selected_model == DEFAULT_MODEL:
                    st.caption("🏆 Using recommended model (best accuracy)")
                else:
                    st.caption(f"💡 Switch to **{DEFAULT_MODEL}** in sidebar for best results")
                
                horizons = [1, 7]  # Removed 30-day prediction
                horizon_names = {"1": "Tomorrow", "7": "Next Week"}
                
                pred_cols = st.columns(2)  # Changed from 3 to 2 columns
                for idx, h in enumerate(horizons):
                    # Use the new predict_with_model function
                    pred, confidence, accuracy = predict_with_model(ticker, selected_model, h)
                    
                    if pred is not None:
                        direction = "📈 UP" if pred == 1 else "📉 DOWN"
                        
                        with pred_cols[idx]:
                            st.metric(
                                label=horizon_names.get(str(h), f"{h}-Day"),
                                value=direction,
                                delta=f"{confidence:.1f}% confidence"
                            )
                            if accuracy:
                                st.caption(f"Model accuracy: {accuracy:.1%}")
                    else:
                        with pred_cols[idx]:
                            st.info(f"ℹ️ {horizon_names.get(str(h), f'{h}-Day')}: Not available")
                
                # Show comparison with other models (expandable)
                with st.expander("📊 Compare All Models", expanded=False):
                    st.markdown("**Compare predictions across all trained models:**")
                    
                    for h in horizons:
                        st.markdown(f"**{horizon_names.get(str(h), f'{h}-Day')} Prediction:**")
                        
                        comparison_data = []
                        for model_name in get_model_list():
                            pred, conf, acc = predict_with_model(ticker, model_name, h)
                            if pred is not None:
                                comparison_data.append({
                                    'Model': f"{'🏆 ' if model_name == DEFAULT_MODEL else ''}{model_name}",
                                    'Prediction': "📈 UP" if pred == 1 else "📉 DOWN",
                                    'Confidence': f"{conf:.1f}%",
                                    'Accuracy': f"{acc:.1%}" if acc else "N/A"
                                })
                        
                        if comparison_data:
                            df_comparison = pd.DataFrame(comparison_data)
                            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                        else:
                            st.info("No models available for comparison")

                # Display stats
                st.markdown("---")
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = float(stock_df['Close'].iloc[-1])
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    first_price = float(stock_df['Close'].iloc[0])
                    last_price = float(stock_df['Close'].iloc[-1])
                    price_change = ((last_price - first_price) / first_price * 100)
                    st.metric("Year Change", f"{price_change:+.2f}%")
                with col3:
                    st.metric("News Articles", str(len(sentiment_df)))
                with col4:
                    avg_sentiment = float(sentiment_df['rolling_sentiment'].mean())
                    sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                    st.metric("Avg Sentiment", sentiment_label)
            else:
                st.warning("⚠️ Data files not found - analysis may have failed partway through")

        except FileNotFoundError as e:
            st.error(f"❌ Data files not found: {e}")
        except Exception as e:
            st.error(f"❌ Error displaying results: {e}")

else:
    st.info("👆 Select a stock ticker from the dropdown to begin analysis.")