# Project Overview
 
This end-to-end financial analysis system combines data engineering, machine learning, and business intelligence to provide actionable stock market insights:

Data Pipeline → Automated extraction of AAPL financials from Yahoo Finance into SQL Server

Machine Learning → Random Forest model predicting next-day price movements

Visualization → Interactive Power BI dashboard with predictions and financial KPIs

# Key Results

Machine Learning Performance

Metric                Value        Benchmark  

Accuracy                51.9%            50%(random)

Precision               53.7%            50%(random)

Training Data          5,197 days        2005-present

Backtest Period        2,708 days        Walk-forward validation

Outperforms random guessing with consistent edge across 2,700+ out-of-sample predictions

# Top Predictive Features

Close_Ratio_20 (17.7%) - 20-day price momentum

Close_Ratio_10 (17.0%) - 10-day price momentum

Close_Ratio_2 (16.1%) - Short-term price action

Close_Ratio_5 (15.9%) - Weekly trends

Close_Ratio_50 (15.8%) - Long-term momentum



# System Design

Yahoo Finance API → yfinance → Python ETL → SQL Server → Power BI Dashboard
                                    ↓
                          Feature Engineering → Random Forest → Predictions

# Technology Stack

Data Collection: yfinance, pandas, SQLAlchemy

Database: SQL Server with ODBC integration

Machine Learning: scikit-learn (RandomForestClassifier)

Visualization: Power BI, matplotlib

Deployment: Serialized model (pickle) for production


 # Model Architecture
 
Feature Engineering

Created 10 engineered features across 5 time horizons (2, 5, 10, 20, 50 days):

Close Ratio Features (5 features):

Close_Ratio_N = Current_Close / N_Day_Moving_Average

Captures price momentum relative to historical averages

Trend Features (5 features):

Trend_N = Sum_of_Up_Days_in_N_Day_Window

Identifies directional momentum over rolling windows

# Model Configuration:
  
    RandomForestClassifier(
    
    n_estimators=200,       # Ensemble of 200 decision trees    
   
    min_samples_split=200,  # Regularization to prevent overfitting
  
    random_state=1 )          # Reproducibility 



# Validation Strategy


Walk-forward backtesting with 250-day increments

No lookahead bias - features only use past data

Growing training window starting at 2,500 days

Tested on 2,708 out-of-sample days



# Power BI Dashboard (time period selected: over last 4 years)

Financial KPIs Tracked:

Net Income: $402.54B

Free Cash Flow: $418.60B

Current Stock Price: $275.25

30-Day Average Cost: $270.18


# Visualizations:

Total Assets vs Total Liabilities (composition analysis)

Profit Margin trends over time

Cash Flow breakdown (Operating, Investing, Financing)

Share volume and price correlation

Volatility analysis with 300-day moving average

ML Prediction: UP/DOWN with confidence percentage


# Data Engineering

SQL Server Schema

Four normalized tables:

StockHistory - OHLCV data with dividends/splits

IncomeStatement - Revenue, expenses, net income

BalanceSheet - Assets, liabilities, equity

CashFlow - Operating, investing, financing activities


ETL Process:
# Automated pipeline unpivots financial statements
    def unpivot_financials(df):
       df = df.melt(id_vars="MetricName", var_name="Date", value_name="Value")
       return df

# Loads into SQL Server with proper data types

    income_statement_fixed.to_sql("IncomeStatement", engine, if_exists="replace")


 Key Insights

What Worked


 Multiple time horizons captured different market patterns

 Walk-forward validation prevented overfitting

 Regularization (min_samples_split=200) improved generalization

 Price ratios more predictive than absolute prices


Challenges Overcome

 Initial trend features too simplistic (binary sums)
 
 Long horizons (250, 1000 days) caused excessive data loss

 Custom probability thresholds reduced accuracy

→ Solution: Optimized horizons, standard predict(), increased regularization

 Future Enhancements

Model Improvements:

Add volume-based momentum indicators

Implement sentiment analysis from financial news

Multi-stock portfolio predictions

Infrastructure:

Real-time streaming data pipeline

Automated model retraining scheduler

API endpoint for predictions

Risk management metrics (Sharpe ratio, max drawdown)



# Disclaimer
 
FOR EDUCATIONAL AND PORTFOLIO PURPOSES ONLY
This project demonstrates technical skills in machine learning, data engineering, and business intelligence. It is not financial advice. Stock market prediction is inherently uncertain, and past performance does not guarantee future results.

# Built With

Python 3.8+ - Core programming language
scikit-learn - Machine learning framework
yfinance - Yahoo Finance API wrapper
SQLAlchemy + pyodbc - Database connectivity
Power BI - Business intelligence visualization
pandas - Data manipulation and analysis


# License
MIT License - See LICENSE file for details

⭐ Star this repo if you found it interesting!
