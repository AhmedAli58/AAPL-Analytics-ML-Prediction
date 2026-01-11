# -*- coding: utf-8 -*-
"""
Apple Stock Price Prediction using Random Forest
Created on Mon Nov 24 11:57:57 2025

This script predicts whether Apple (AAPL) stock will go up or down the next trading day
using a Random Forest classifier with engineered technical indicators.

Model Performance:
    - Accuracy: 51.9% (beats random guessing)
    - Precision: 53.7% (when predicting "UP", correct 53.7% of time)
    - Validation: Walk-forward backtesting on 2,700+ days

@author: cpfoy
"""

import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
import time
import sys

# ============================================================
# SECTION 1: DATA COLLECTION
# ============================================================

def download_stock_data(ticker_symbol, period="max", max_retries=3, retry_delay=2):
    """
    Download stock data from Yahoo Finance with error handling and retry logic.
    
    This function implements production-ready error handling for external API calls:
    - Retries on transient failures (network issues, rate limits)
    - Validates data was successfully retrieved
    - Provides clear error messages for debugging
    
    Args:
        ticker_symbol: Stock ticker symbol (e.g., "AAPL")
        period: Data period ("max", "1y", "5y", etc.)
        max_retries: Maximum number of retry attempts for transient failures
        retry_delay: Initial delay between retries (seconds), doubles each retry
    
    Returns:
        pandas.DataFrame: Stock price history data
    
    Raises:
        SystemExit: If data cannot be retrieved after all retries or if data is invalid
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # Retry logic for transient network failures
    for attempt in range(1, max_retries + 1):
        try:
            data = ticker.history(period=period)
            
            # Validate data was successfully retrieved
            if data is None:
                raise ValueError(f"yfinance returned None for {ticker_symbol}")
            
            if data.empty:
                raise ValueError(f"yfinance returned empty DataFrame for {ticker_symbol}")
            
            # Validate required columns exist
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in data: {missing_columns}. "
                    f"Available columns: {list(data.columns)}"
                )
            
            # Validate data quality: check for reasonable data points
            if len(data) < 100:
                raise ValueError(
                    f"Insufficient data retrieved: {len(data)} rows. "
                    f"Expected at least 100 rows for reliable modeling."
                )
            
            print(f"✓ Successfully downloaded {ticker_symbol} data: {len(data)} rows")
            return data
            
        except (ValueError, KeyError) as e:
            # Data validation errors: don't retry, fail immediately
            print(f"ERROR: Data validation failed for {ticker_symbol}: {e}", file=sys.stderr)
            print(f"Pipeline cannot continue without valid data.", file=sys.stderr)
            sys.exit(1)
            
        except Exception as e:
            # Network/API errors: retry with exponential backoff
            if attempt < max_retries:
                wait_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                print(
                    f"WARNING: Failed to download {ticker_symbol} data (attempt {attempt}/{max_retries}): {e}",
                    file=sys.stderr
                )
                print(f"Retrying in {wait_time} seconds...", file=sys.stderr)
                time.sleep(wait_time)
            else:
                # Final attempt failed
                print(
                    f"ERROR: Failed to download {ticker_symbol} data after {max_retries} attempts.",
                    file=sys.stderr
                )
                print(
                    f"Last error: {e}",
                    file=sys.stderr
                )
                print(
                    f"Possible causes: network connectivity, Yahoo Finance API changes, rate limiting.",
                    file=sys.stderr
                )
                sys.exit(1)
    
    # This should never be reached, but added for safety
    raise RuntimeError("Unexpected error in download_stock_data")

# Download all available historical data for Apple stock
aapl = download_stock_data("AAPL", period="max")
print("Downloaded AAPL data:")
print(aapl)

# Visualize the closing price over time
aapl.plot.line(y="Close", use_index=True)

# Remove columns not needed for prediction
del aapl["Dividends"]
del aapl["Stock Splits"]

# ============================================================
# SECTION 2: TARGET VARIABLE CREATION
# ============================================================

# Create "Tomorrow" column by shifting Close price forward by 1 day
# This represents what the price will be the next trading day
aapl["Tomorrow"] = aapl["Close"].shift(-1)

# Create binary target: 1 if price goes up tomorrow, 0 if it goes down
# This is what we're trying to predict
aapl["Target"] = (aapl["Tomorrow"] > aapl["Close"]).astype(int)
print("\nData with Target variable:")
print(aapl)

# Use only data from 2005 onwards (more reliable, liquid market period)
aapl = aapl.loc["2005-01-01":].copy()
print(f"\nFiltered data shape: {aapl.shape}")

# ============================================================
# SECTION 3: BASIC MODEL (BASELINE)
# ============================================================

# Initialize Random Forest with basic parameters
model = RandomForestClassifier(
    n_estimators=100,        # 100 decision trees in the forest
    min_samples_split=100,   # Minimum samples required to split a node (regularization)
    random_state=1           # For reproducibility
)

# Split data: train on all but last 100 days, test on last 100 days
train = aapl.iloc[:-100]
test = aapl.iloc[-100:]

# Basic features: standard OHLCV (Open, High, Low, Close, Volume) data
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Train the model
model.fit(train[predictors], train["Target"])

# Make predictions on test set
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# Visualize predictions vs actual
combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()

# ============================================================
# SECTION 4: BACKTESTING FRAMEWORK
# ============================================================
# Walk-forward validation: train on historical data, test on future data,
# then move forward in time and repeat. This simulates real trading.

def predict(train, test, predictors, model):
    """
    Train model on training data and make predictions on test data.
    
    Args:
        train: Training dataset
        test: Testing dataset
        predictors: List of feature column names
        model: sklearn model instance
    
    Returns:
        DataFrame with actual targets and predictions
    """
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    """
    Perform walk-forward backtesting to evaluate model performance.
    
    Args:
        data: Full dataset
        model: sklearn model instance
        predictors: List of feature column names
        start: Number of days to use for initial training (default 2500)
        step: Number of days to predict forward in each iteration (default 250)
    
    Returns:
        DataFrame with all predictions concatenated
    """
    all_predictions = []
    
    # Walk forward through the data
    for i in range(start, data.shape[0], step):
        # Training set: all data up to current point
        train = data.iloc[0:i].copy()
        # Test set: next 'step' days
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

# Run backtest on basic model
predictions = backtest(aapl, model, predictors)
print("\nBasic Model Prediction Distribution:")
print(predictions["Predictions"].value_counts())

# Calculate precision for basic model
basic_precision = precision_score(predictions["Target"], predictions["Predictions"])
print(f"Basic Model Precision: {basic_precision:.4f}")

# Show actual distribution of up vs down days
print("\nActual Market Distribution:")
print(predictions["Target"].value_counts() / predictions.shape[0])

# ============================================================
# SECTION 5: FEATURE ENGINEERING
# ============================================================
# Create technical indicators to improve predictions

horizons = [2, 5, 10, 20, 50]  # Time periods to analyze
new_predictors = []

for horizon in horizons:
    # Calculate rolling averages for the horizon
    rolling_averages = aapl.rolling(horizon).mean()
    
    # Feature 1: Close Ratio
    # Compares current price to N-day moving average
    # Value > 1.0 means price is above average (bullish)
    # Value < 1.0 means price is below average (bearish)
    ratio_column = f"Close_Ratio_{horizon}"
    aapl[ratio_column] = aapl["Close"] / rolling_averages["Close"]
    
    # Feature 2: Trend Indicator
    # Counts number of "up" days in the past N days (shifted by 1 to avoid lookahead bias)
    # Higher values indicate strong upward momentum
    trend_column = f"Trend_{horizon}"
    aapl[trend_column] = aapl.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]

# Forward fill any NaN values, then drop remaining NaNs
# (NaNs occur at start of dataset due to rolling windows)
aapl = aapl.ffill().dropna()

print(f"\nData shape after feature engineering: {aapl.shape}")
print(f"Total features created: {len(new_predictors)}")

# ============================================================
# SECTION 6: ENHANCED MODEL WITH ENGINEERED FEATURES
# ============================================================

# Train new model with more trees and stronger regularization
model = RandomForestClassifier(
    n_estimators=200,        # More trees for better ensemble
    min_samples_split=200,   # Stricter splitting to prevent overfitting
    random_state=1
)

# Run backtest with new features
predictions = backtest(aapl, model, new_predictors)

print("\nEnhanced Model Prediction Distribution:")
print(predictions["Predictions"].value_counts())

# ============================================================
# SECTION 7: MODEL EVALUATION
# ============================================================

# Compare basic model vs enhanced model
print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

# Basic model results (from earlier test set)
print(f"Basic Model Precision: {precision_score(test['Target'], preds):.4f}")

# Enhanced model results (from backtest)
enhanced_precision = precision_score(predictions['Target'], predictions['Predictions'])
enhanced_accuracy = accuracy_score(predictions['Target'], predictions['Predictions'])

print(f"\nEnhanced Model Backtest Results:")
print(f"  Precision: {enhanced_precision:.4f}")
print(f"  Accuracy: {enhanced_accuracy:.4f}")

print(f"\nPrediction Distribution:")
print(predictions["Predictions"].value_counts())

print(f"\nActual Distribution:")
print(predictions["Target"].value_counts())

# ============================================================
# SECTION 8: FEATURE IMPORTANCE ANALYSIS
# ============================================================
# Determine which features are most predictive

# Retrain model on all data to get final feature importances
model.fit(aapl[new_predictors], aapl["Target"])

# Create DataFrame showing feature importance scores
feature_imp = pd.DataFrame({
    'feature': new_predictors,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP 5 MOST IMPORTANT FEATURES")
print("="*60)
print(feature_imp.head())

# Interpretation:
# - Higher importance = feature contributes more to predictions
# - Close_Ratio features typically dominate (price momentum indicators)
# - Trend features capture directional movement patterns