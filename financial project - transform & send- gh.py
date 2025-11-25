# -*- coding: utf-8 -*-
"""
Apple Financial Data ETL Pipeline
Created on Mon Nov 24 2025

This script automates the extraction, transformation, and loading (ETL) of Apple's
financial data from Yahoo Finance into a SQL Server database for analysis and reporting.

Data Sources:
    - Income Statement (quarterly revenue, expenses, net income)
    - Balance Sheet (assets, liabilities, equity)
    - Cash Flow Statement (operating, investing, financing activities)
    - Stock History (daily OHLCV data with dividends and splits)

Output:
    - 4 normalized SQL Server tables ready for Power BI integration

@author: cpfoy
"""

# ============================================================
# IMPORTS
# ============================================================

import yfinance as yf      # Yahoo Finance API for financial data
import sqlalchemy          # SQL database connectivity and ORM
import pandas as pd        # Data manipulation and analysis

# ============================================================
# SECTION 1: DATA EXTRACTION FROM YAHOO FINANCE
# ============================================================

# Initialize ticker object for Apple Inc.
ticker = yf.Ticker("AAPL")

# Download financial statements and stock history
# Note: Financial statements are typically quarterly data
income_statement = ticker.financials      # Revenue, expenses, net income, EPS
balance_sheet = ticker.balance_sheet      # Assets, liabilities, shareholder equity
cash_flow = ticker.cashflow              # Operating, investing, financing cash flows
stock_history = ticker.history(period="max")  # Complete historical price data (OHLCV)

# Preview extracted data
print("="*60)
print("DATA EXTRACTION COMPLETE")
print("="*60)
print("\nIncome Statement Preview:")
print(income_statement.head())
print("\nStock History Preview:")
print(stock_history.head())
print(f"Stock History Shape: {stock_history.shape[0]} rows × {stock_history.shape[1]} columns")

# ============================================================
# SECTION 2: SQL SERVER CONNECTION SETUP
# ============================================================

# Database connection parameters
# IMPORTANT: Replace YOUR_SERVER with your actual SQL Server instance name
# Examples: "localhost\SQLEXPRESS", "DESKTOP-ABC123\SQLEXPRESS"
server = r"YOUR_SERVER\SQLEXPRESS"
database = "Financial Project"

# Build connection string for SQL Server with Windows Authentication
conn_str = (
    "mssql+pyodbc:///?odbc_connect="
    "Driver={ODBC Driver 17 for SQL Server};"  # Requires ODBC Driver 17 installed
    f"Server={server};"
    f"Database={database};"
    "Trusted_Connection=yes;"  # Use Windows Authentication (no password needed)
)

# Create SQLAlchemy engine for database operations
engine = sqlalchemy.create_engine(conn_str)

print("\n✓ Database connection established")

# ============================================================
# SECTION 3: DATA TRANSFORMATION
# ============================================================

def unpivot_financials(df):
    """
    Transform wide-format financial statements into long-format (normalized) tables.
    
    Original Format (Wide):
        Index          | 2023-12-31 | 2023-09-30 | ...
        ----------------|------------|------------|----
        Total Revenue  | 119.6B     | 89.5B      | ...
        Net Income     | 33.9B      | 23.0B      | ...
    
    Transformed Format (Long):
        MetricName     | Date       | Value
        ---------------|------------|--------
        Total Revenue  | 2023-12-31 | 119.6B
        Total Revenue  | 2023-09-30 | 89.5B
        Net Income     | 2023-12-31 | 33.9B
        Net Income     | 2023-09-30 | 23.0B
    
    Benefits:
        - Easier to query specific metrics over time
        - Normalized database design (3NF)
        - Better for time-series analysis in Power BI
    
    Args:
        df: Wide-format financial statement DataFrame
    
    Returns:
        Long-format DataFrame with MetricName, Date, Value columns
    """
    # Convert index (metric names) to a regular column
    df = df.reset_index().rename(columns={"index": "MetricName"})
    
    # Unpivot: Convert date columns into rows
    df = df.melt(
        id_vars="MetricName",    # Keep this column fixed
        var_name="Date",         # Date columns become values in "Date" column
        value_name="Value"       # Financial values go into "Value" column
    )
    
    # Convert Date column to proper datetime format
    df["Date"] = pd.to_datetime(df["Date"])
    
    return df

# Transform all three financial statements
print("\nTransforming financial statements...")
income_statement_fixed = unpivot_financials(income_statement)
balance_sheet_fixed = unpivot_financials(balance_sheet)
cash_flow_fixed = unpivot_financials(cash_flow)

print(f"✓ Income Statement: {len(income_statement_fixed)} rows")
print(f"✓ Balance Sheet: {len(balance_sheet_fixed)} rows")
print(f"✓ Cash Flow: {len(cash_flow_fixed)} rows")

# ============================================================
# SECTION 4: LOAD FINANCIAL STATEMENTS INTO SQL SERVER
# ============================================================

print("\nLoading financial statements to SQL Server...")

# Load Income Statement
# if_exists="replace" will drop and recreate the table each time
income_statement_fixed.to_sql(
    "IncomeStatement",
    engine,
    if_exists="replace",  # Options: "fail", "replace", "append"
    index=False          # Don't include DataFrame index as a column
)
print("✓ IncomeStatement table loaded")

# Load Balance Sheet
balance_sheet_fixed.to_sql(
    "BalanceSheet",
    engine,
    if_exists="replace",
    index=False
)
print("✓ BalanceSheet table loaded")

# Load Cash Flow Statement
cash_flow_fixed.to_sql(
    "CashFlow",
    engine,
    if_exists="replace",
    index=False
)
print("✓ CashFlow table loaded")

# ============================================================
# SECTION 5: LOAD STOCK HISTORY WITH PROPER DATA TYPES
# ============================================================

print("\nLoading stock history to SQL Server...")

# Define explicit SQL data types for each column
# This ensures proper storage and prevents data type issues
dtype_stock = {
    "Date": sqlalchemy.DateTime(),      # Timestamp for each trading day
    "Open": sqlalchemy.Float(),         # Opening price
    "High": sqlalchemy.Float(),         # Highest price during the day
    "Low": sqlalchemy.Float(),          # Lowest price during the day
    "Close": sqlalchemy.Float(),        # Closing price (adjusted for splits)
    "Volume": sqlalchemy.BigInteger(),  # Number of shares traded (can be billions)
    "Dividends": sqlalchemy.Float(),    # Dividend paid on that date (usually 0)
    "Stock Splits": sqlalchemy.Float(), # Stock split ratio (e.g., 2.0 for 2-for-1 split)
}

# Convert Date from index to regular column
stock_history = stock_history.reset_index()

# Load stock history with specified data types
stock_history.to_sql(
    "StockHistory",
    engine,
    if_exists="replace",
    index=False,
    dtype=dtype_stock  # Apply our custom data types
)

print(f"✓ StockHistory table loaded ({len(stock_history)} rows)")

# ============================================================
# COMPLETION SUMMARY
# ============================================================

print("\n" + "="*60)
print("ETL PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)
print(f"\nDatabase: {database}")
print(f"Server: {server}")
print("\nTables Created:")
print("  1. IncomeStatement   - Quarterly revenue and profit metrics")
print("  2. BalanceSheet      - Assets, liabilities, and equity")
print("  3. CashFlow          - Operating, investing, financing activities")
print("  4. StockHistory      - Daily OHLCV data with dividends/splits")
print("\n✓ Ready for Power BI integration")
print("="*60)

# ============================================================
# NEXT STEPS
# ============================================================
# 1. Open SQL Server Management Studio (SSMS)
# 2. Connect to your server instance
# 3. Verify tables exist in "Financial Project" database
# 4. Use Power BI to connect to SQL Server and build dashboards
# 5. Set up scheduled task to run this script daily/weekly for fresh data