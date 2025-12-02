# Internship Project Progress Report

## Project Title
Real-Time Sentiment Analysis for Short-Term Stock Prediction

## Student Name
Sri Ramya Kurella

## Mentor
Vishal Jha, Pyramid Consulting

## Project Dates
September 17, 2025 – December 17, 2025

## Executive Summary
This report details the development of a system that analyzes market sentiment from news and social media to predict short-term stock price movements. The project includes data ingestion, NLP processing, sentiment analysis, model training, and a dashboard for visualization.

## Objectives
- Collect real-time news and social media data.
- Apply sentiment analysis to textual data.
- Build predictive models linking sentiment to stock prices.
- Develop a dashboard for insights.

## Methodology
1. **Data Ingestion**: Fetched historical stock data using yfinance and news headlines from Yahoo Finance.
2. **NLP Processing**: Cleaned text data using NLTK, applied BERT-based sentiment analysis.
3. **Feature Engineering**: Created rolling sentiment indices.
4. **Modeling**: Trained RandomForest models to predict price direction.
5. **Dashboard**: Built with Streamlit for visualization.

## Results
- Data pipeline implemented for 5 stock tickers (AAPL, GOOGL, TSLA, MSFT, AMZN).
- Sentiment analysis achieved high accuracy.
- Models trained with reasonable performance.
- Dashboard provides interactive charts.

## Challenges
- Aligning news dates with stock data.
- Handling real-time data streams.
- Model accuracy limitations due to data quality.

## Future Work
- Integrate social media APIs.
- Improve model with more features.
- Deploy to cloud for real-time operation.

## Conclusion
The project successfully demonstrates the potential of sentiment analysis in stock prediction, providing a foundation for further development.