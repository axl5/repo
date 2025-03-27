import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from docx import Document 

# Set your directory
#target_dir = r"<Insert Directory Here>"
#os.chdir(target_dir)

class ChartAnalysis:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = None
        self.end_date = None
        self.exact_start_date = None
        self.filename = None

    def download_data(self, end_date, lookback_period):
         """
        Download stock data from Yahoo Finance using yfinance and filter it to an exact lookback period.

        This function first calculates an extended start date (lookback_period in years plus extra days)
        to ensure any technical calculations have sufficient data. It then downloads the data using the
        extended start date, but filters the resulting DataFrame so that only records from the exact
        start date (lookback_period in years) onward are retained.

        Parameters:
            ticker (str): The stock ticker symbol.
            end_date (datetime): The end date for data download.
            lookback_period (float): Lookback period in years.
            extra_days (int): Additional days to subtract for the extended lookback (default is 200).

        Returns:
            df (pd.DataFrame): DataFrame containing the downloaded stock data, filtered to start at the exact lookback date.
            exact_start_date (datetime): The computed exact start date.
        """
         self.end_date = end_date
        # Define extra days to include in lookback period - for rolling windows
         extra_days = 200
        # Calculate the extended and exact start dates
         extended_start_date = end_date - timedelta(days=int(lookback_period * 365) + extra_days)
         self.exact_start_date = end_date - timedelta(days=int(lookback_period * 365))
        
        # Download data using the extended start date
         self.df = yf.download(
            self.ticker,
            start=extended_start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval="1d",
            auto_adjust=True
        )
        
        # Reset index to make the Date a column and handle potential multi-index columns
         self.df.reset_index(inplace=True)
         if hasattr(self.df.columns, 'droplevel'):
            try:
                self.df.columns = self.df.columns.droplevel(1)
            except (AttributeError, IndexError):
                pass

         self.df['Ticker'] = ticker

         print(f"Downloaded {len(self.df)} records for {ticker}")
         print(f"Data filtered from {self.exact_start_date.date()} to {end_date.date()} (exact lookback period)")
        
         return self.df, self.end_date, self.exact_start_date

    def add_technical_indicators(self):
        # ------ PRICE INDICATORS ------
        # Daily Return
        self.df['Daily_Return'] = self.df['Close'].pct_change()

        # Moving Averages
        self.df['MA50'] = self.df['Close'].rolling(window=50).mean()
        self.df['MA200'] = self.df['Close'].rolling(window=200).mean()

        # ------ VOLUME INDICATORS ------
        # Volume Moving Average
        self.df['Volume_MA20'] = self.df['Volume'].rolling(window=20).mean()

        # Calculate Volume Ratio safely
        self.df['Volume_Ratio'] = np.nan  # Initialize with NaN
        mask = self.df['Volume_MA20'] > 0  # Create a mask for non-zero denominators
        self.df.loc[mask, 'Volume_Ratio'] = self.df.loc[mask, 'Volume'] / self.df.loc[mask, 'Volume_MA20']

        # Volume change
        self.df['Volume_Change'] = self.df['Volume'].pct_change()

        # ------ VOLATILITY INDICATORS ------
        # Daily volatility (21-day rolling)
        self.df['Volatility_21d'] = self.df['Daily_Return'].rolling(window=21).std() * np.sqrt(252)

        # ------ CALCULATING ATR ------
        print("Calculating ATR...")
        # Step 1: Calculate the three components
        self.df['TR1'] = self.df['High'] - self.df['Low']  # High - Low
        self.df['TR2'] = abs(self.df['High'] - self.df['Close'].shift())  # High - Previous Close
        self.df['TR3'] = abs(self.df['Low'] - self.df['Close'].shift())  # Low - Previous Close

        # Step 2: Find the maximum value among the three
        self.df['TR'] = self.df[['TR1', 'TR2', 'TR3']].max(axis=1)

        # Step 3: Calculate ATR (14-period moving average of TR)
        self.df['ATR'] = self.df['TR'].rolling(window=14, min_periods=14).mean()

        #Step 4: Calculate ATR as percentage of price
        #Create a mask for rows where ATR is valid (not NaN and greater than 0)
        self.df['ATR_Pct'] = pd.NA

        valid_mask = self.df['ATR'].notna() & (self.df['ATR'] > 0)

        #Debug: Print number of valid ATR rows
        print("Number of rows with valid ATR (>0):", valid_mask.sum())

        if valid_mask.any():
            # Get the first index where ATR is valid
            first_valid_index = valid_mask.idxmax()
            print("First valid index:", first_valid_index)
            print("Row at first valid index:\n", self.df.loc[first_valid_index, ['ATR', 'Close']])
            # Calculate ATR_Pct starting from that index onward
            self.df.loc[first_valid_index:, 'ATR_Pct'] = ( 
                self.df.loc[first_valid_index:, 'ATR'].astype(float) / 
                self.df.loc[first_valid_index:, 'Close'].astype(float) * 100
            )
        else:
            print("No valid ATR values found.")

        # ------ MACD CALCULATION ------
        print("Calculating MACD...")
        # MACD line: 12-day EMA - 26-day EMA
        self.df['EMA12'] = self.df['Close'].ewm(span=12).mean()
        self.df['EMA26'] = self.df['Close'].ewm(span=26).mean()
        self.df['MACD'] = self.df['EMA12'] - self.df['EMA26']

        # Signal line: 9-day EMA of MACD line
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9).mean()

        # MACD histogram
        self.df['MACD_Hist'] = self.df['MACD'] -self.df['MACD_Signal']

        # MACD trend
        self.df['MACD_Trend'] = 'Neutral'
        self.df.loc[self.df['MACD_Hist'] > 0, 'MACD_Trend'] = 'Bullish'
        self.df.loc[self.df['MACD_Hist'] < 0, 'MACD_Trend'] = 'Bearish'

        # ------ RSI CALCULATION ------
        print("Calculating RSI...")
        # Calculate price changes
        delta = self.df['Close'].diff()

        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  # Make losses positive

        # Calculate average gain and loss over 14 periods
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        # Calculate RS
        rs = avg_gain / avg_loss

        # Calculate RSI
        self.df['RSI'] = 100 - (100 / (1 + rs))

        print("Technical analysis complete!")

        # Debugging: Check the values before filtering
        print(f"Exact Start Date: {self.exact_start_date}")
        print(f"DataFrame 'Date' column:\n{self.df['Date'].head()}")

        # Ensure 'Date' column is in datetime format
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')  # Convert to datetime if not already

        # Check if exact_start_date is None
        if self.exact_start_date is None:
            raise ValueError("exact_start_date is None. Please check the initialization.")

        # Filter the DataFrame to only include data from the exact start date onward
        print(f"Filtering DataFrame for dates >= {self.exact_start_date}")
        self.df = self.df[self.df['Date'] >= self.exact_start_date]
      
        # Final pre-processing data stage
        # Remove intermediate calculation columns to clean up the dataframe
        cols_to_drop = ['TR1', 'TR2', 'TR3', 'EMA12', 'EMA26']
        self.df = self.df.drop(columns=cols_to_drop)

    def plot_data(self):
        plt.figure(figsize=(14, 10))

        # Price and Moving Averages
        plt.subplot(3, 1, 1)
        plt.plot(self.df['Date'], self.df['Close'], label='Close Price', color='blue')
        plt.plot(self.df['Date'], self.df['MA50'], label='50-Day MA', color='orange')
        plt.plot(self.df['Date'], self.df['MA200'], label='200-Day MA', color='red')
        plt.title(f'{self.ticker} Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()

        # ATR
        plt.subplot(3, 1, 2)
        plt.plot(self.df['Date'], self.df['ATR'], label='ATR', color='purple')
        plt.title('Average True Range (ATR)')
        plt.xlabel('Date')
        plt.ylabel('ATR')
        plt.legend()
        plt.grid()

        # MACD
        plt.subplot(3, 1, 3)
        plt.plot(self.df['Date'], self.df['MACD'], label='MACD', color='green')
        plt.plot(self.df['Date'], self.df['MACD_Signal'], label='MACD Signal', color='red')
        plt.bar(self.df['Date'], self.df['MACD_Hist'], label='MACD Histogram', color='gray', alpha=0.5)
        plt.title('MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    def save_to_csv(self, filename):
        self.filename = filename
        self.df.to_csv(filename, index=False)
