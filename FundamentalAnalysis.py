#!pip install openpyxl
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from docx import Document
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union

class FundamentalAnalysis:
    """A class for analyzing company fundamentals from Yahoo Finance data."""
    
    def __init__(self, ticker):
        """Initialize the FundamentalAnalysis with a ticker symbol."""
        self.ticker = ticker
        self.ticker_obj = yf.Ticker(ticker)
        self.info = self.ticker_obj.info
        self.trend_years = 5  # Number of years for trend analysis
        
        # Containers for data
        self.trend_data = {}
        self.spot_data = {}
        self.calculated_ratios = {}
        self.direct_ratios = {}  # For ratios pulled directly from API
        
        # Initialize data
        self._fetch_data()
    
    def _fetch_data(self):
        """Fetch all required financial data."""
        # Get basic company info
        print(f"Fetching basic info for {self.ticker}...")
        
        # Get financial statements
        print("Fetching financial statements...")
        self.income_annual = self.ticker_obj.income_stmt
        self.balance_annual = self.ticker_obj.balance_sheet
        self.cashflow_annual = self.ticker_obj.cashflow
        
        # Get quarterly statements for spot analysis
        self.income_quarterly = self.ticker_obj.quarterly_income_stmt
        self.balance_quarterly = self.ticker_obj.quarterly_balance_sheet
        self.cashflow_quarterly = self.ticker_obj.quarterly_cashflow
        
        # Get historical data for trend analysis
        self.historical_data = self.ticker_obj.history(period=f"{self.trend_years + 1}y")
        
        # Calculate or fetch the requested metrics
        self._calculate_trend_metrics()
        self._calculate_spot_metrics()
    
    def _safe_get(self, data, key, default=np.nan):
        """Safely get a value from data source."""
        try:
            return data.get(key, default)
        except (AttributeError, KeyError, TypeError):
            return default
    
    def _safe_calculate(self, numerator, denominator, default=np.nan):
        """Safely calculate a ratio."""
        try:
            if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
                return default
            return numerator / denominator
        except (TypeError, ZeroDivisionError):
            return default
    
    def _get_statement_value(self, statement, row_name, period=0):
        """Get a value from a financial statement for a specific period."""
        try:
            if row_name in statement.index:
                if period < len(statement.columns):
                    return statement.loc[row_name, statement.columns[period]]
            return np.nan
        except (KeyError, IndexError, AttributeError):
            return np.nan
    
    def _calculate_trend_metrics(self):
        """Calculate metrics for trend analysis (last 5 years)."""
        print("Calculating trend metrics...")
        
        # Create a dictionary to store trend data
        self.trend_data = {
            'Years': [],
            'Revenue': [],
            'Revenue_Growth': [],
            'EPS': [],
            'EPS_Estimated': [],
            'Gross_Margin': [],
            'Operating_Margin': [],
            'EBITDA_Margin': [],
            'Net_Margin': [],
            'ROA': [],
            'ROE': [],
            'ROIC': [],
            'Earnings_Growth': [],
            'PS_Ratio': [],
            'PE_Ratio': [],
            'PB_Ratio': [],
            'PC_Ratio': [],
            'P_FCF': [],
            'EV_EBITDA': [],
            'EV_Revenue': [],
            'EV_FCF': [],
            'FCF_Yield': [],
            'Dividend_Yield': [],
            'Quick_Ratio': [],
            'Debt_to_Equity': [],
            'Interest_Coverage': [],
            'Asset_Turnover': [],
            'Asset_Equity_Ratio': []
        }
        
        # Process annual statements for the past 5 years
        years = min(self.trend_years, len(self.income_annual.columns))
        
        for i in range(years):
            year = self.income_annual.columns[i].year if i < len(self.income_annual.columns) else None
            if year:
                self.trend_data['Years'].append(year)
                
                # --- Company Fundamentals ---
                # Revenue
                revenue = self._get_statement_value(self.income_annual, 'Total Revenue', i)
                self.trend_data['Revenue'].append(revenue)
                
                # Revenue Growth
                if i < years - 1:
                    prev_revenue = self._get_statement_value(self.income_annual, 'Total Revenue', i + 1)
                    revenue_growth = ((revenue / prev_revenue) - 1) if prev_revenue and prev_revenue > 0 else np.nan
                    self.trend_data['Revenue_Growth'].append(revenue_growth)
                else:
                    self.trend_data['Revenue_Growth'].append(np.nan)
                
                # EPS
                net_income = self._get_statement_value(self.income_annual, 'Net Income', i)
                shares_outstanding = self._safe_get(self.info, 'sharesOutstanding')
                eps = self._safe_calculate(net_income, shares_outstanding)
                self.trend_data['EPS'].append(eps)
                
                # EPS Estimated (from API directly)
                # DIRECT API PULL - Estimated EPS
                eps_est = self._safe_get(self.info, 'forwardEps', np.nan)
                self.trend_data['EPS_Estimated'].append(eps_est if i == 0 else np.nan)  # Only for most recent year
                
                # Margins
                gross_profit = self._get_statement_value(self.income_annual, 'Gross Profit', i)
                operating_income = self._get_statement_value(self.income_annual, 'Operating Income', i)
                ebitda = self._get_statement_value(self.income_annual, 'EBITDA', i)
                
                gross_margin = self._safe_calculate(gross_profit, revenue)
                op_margin = self._safe_calculate(operating_income, revenue)
                ebitda_margin = self._safe_calculate(ebitda, revenue)
                net_margin = self._safe_calculate(net_income, revenue)
                
                self.trend_data['Gross_Margin'].append(gross_margin)
                self.trend_data['Operating_Margin'].append(op_margin)
                self.trend_data['EBITDA_Margin'].append(ebitda_margin)
                self.trend_data['Net_Margin'].append(net_margin)
                
                # Returns
                total_assets = self._get_statement_value(self.balance_annual, 'Total Assets', i)
                total_equity = self._get_statement_value(self.balance_annual, 'Total Stockholder Equity', i)
                
                # Calculate ROIC components
                total_debt = self._get_statement_value(self.balance_annual, 'Total Debt', i)
                if pd.isna(total_debt):
                    long_term_debt = self._get_statement_value(self.balance_annual, 'Long Term Debt', i)
                    short_term_debt = self._get_statement_value(self.balance_annual, 'Short Long Term Debt', i)
                    total_debt = (0 if pd.isna(long_term_debt) else long_term_debt) + (0 if pd.isna(short_term_debt) else short_term_debt)
                
                invested_capital = (0 if pd.isna(total_equity) else total_equity) + (0 if pd.isna(total_debt) else total_debt)
                
                roa = self._safe_calculate(net_income, total_assets)
                roe = self._safe_calculate(net_income, total_equity)
                roic = self._safe_calculate(operating_income * (1 - 0.21), invested_capital)  # Using standard 21% tax rate
                
                self.trend_data['ROA'].append(roa)
                self.trend_data['ROE'].append(roe)
                self.trend_data['ROIC'].append(roic)
                
                # Earnings Growth
                if i < years - 1:
                    prev_net_income = self._get_statement_value(self.income_annual, 'Net Income', i + 1)
                    earnings_growth = ((net_income / prev_net_income) - 1) if prev_net_income and prev_net_income > 0 else np.nan
                    self.trend_data['Earnings_Growth'].append(earnings_growth)
                else:
                    self.trend_data['Earnings_Growth'].append(np.nan)
                
                # --- Valuation Metrics ---
                # Get historical price data
                year_date = datetime(year, 12, 31)
                year_price = None
                
                # Find the closest trading day price
                for days_back in range(30):  # Look back up to 30 days to find a trading day
                    check_date = year_date - timedelta(days=days_back)
                    check_date_str = check_date.strftime('%Y-%m-%d')
                    if check_date_str in self.historical_data.index:
                        year_price = self.historical_data.loc[check_date_str, 'Close']
                        break
                
                if year_price is None and len(self.historical_data) > 0:
                    # If no match, use the closest date
                    closest_date = min(self.historical_data.index, key=lambda x: abs((x - year_date).days))
                    year_price = self.historical_data.loc[closest_date, 'Close']
                
                # Calculate market cap for the year
                market_cap = year_price * shares_outstanding if year_price is not None and shares_outstanding else np.nan
                
                # Enterprise Value calculation
                cash_equivalents = self._get_statement_value(self.balance_annual, 'Cash And Cash Equivalents', i)
                ev = (market_cap + total_debt - cash_equivalents) if not pd.isna(market_cap) else np.nan
                
                # Free Cash Flow calculation
                operating_cash = self._get_statement_value(self.cashflow_annual, 'Operating Cash Flow', i)
                capex = self._get_statement_value(self.cashflow_annual, 'Capital Expenditure', i)
                fcf = operating_cash + capex if not pd.isna(operating_cash) and not pd.isna(capex) else np.nan
                
                # Price multiples
                ps_ratio = self._safe_calculate(market_cap, revenue)
                pe_ratio = self._safe_calculate(market_cap, net_income)
                pb_ratio = self._safe_calculate(market_cap, total_equity)
                pc_ratio = self._safe_calculate(market_cap, cash_equivalents)
                p_fcf = self._safe_calculate(market_cap, fcf)
                
                self.trend_data['PS_Ratio'].append(ps_ratio)
                self.trend_data['PE_Ratio'].append(pe_ratio)
                self.trend_data['PB_Ratio'].append(pb_ratio)
                self.trend_data['PC_Ratio'].append(pc_ratio)
                self.trend_data['P_FCF'].append(p_fcf)
                
                # Enterprise Value multiples
                ev_ebitda = self._safe_calculate(ev, ebitda)
                ev_revenue = self._safe_calculate(ev, revenue)
                ev_fcf = self._safe_calculate(ev, fcf)
                
                self.trend_data['EV_EBITDA'].append(ev_ebitda)
                self.trend_data['EV_Revenue'].append(ev_revenue)
                self.trend_data['EV_FCF'].append(ev_fcf)
                
                # Cash Flow Yield
                fcf_yield = self._safe_calculate(fcf, market_cap)
                self.trend_data['FCF_Yield'].append(fcf_yield)
                
                # Dividend Yield
                dividends_paid = abs(self._get_statement_value(self.cashflow_annual, 'Dividends Paid', i))
                div_yield = self._safe_calculate(dividends_paid, market_cap)
                self.trend_data['Dividend_Yield'].append(div_yield)
                
                # --- Financial Statement Analysis Metrics ---
                # Liquidity & Solvency
                current_assets = self._get_statement_value(self.balance_annual, 'Current Assets', i)
                current_liabilities = self._get_statement_value(self.balance_annual, 'Current Liabilities', i)
                inventory = self._get_statement_value(self.balance_annual, 'Inventory', i)
                
                quick_ratio = self._safe_calculate((current_assets - (0 if pd.isna(inventory) else inventory)), current_liabilities)
                debt_to_equity = self._safe_calculate(total_debt, total_equity)
                
                interest_expense = abs(self._get_statement_value(self.income_annual, 'Interest Expense', i))
                interest_coverage = self._safe_calculate(operating_income, interest_expense)
                
                self.trend_data['Quick_Ratio'].append(quick_ratio)
                self.trend_data['Debt_to_Equity'].append(debt_to_equity)
                self.trend_data['Interest_Coverage'].append(interest_coverage)
                
                # Efficiency
                asset_turnover = self._safe_calculate(revenue, total_assets)
                asset_equity_ratio = self._safe_calculate(total_assets, total_equity)
                
                self.trend_data['Asset_Turnover'].append(asset_turnover)
                self.trend_data['Asset_Equity_Ratio'].append(asset_equity_ratio)
        
        # Check for direct API ratios for trend metrics
        # DIRECT API PULL SECTION - Trend Metrics
        self.direct_ratios['Trend'] = {}
        
        api_ratios = {
            'PE_Ratio': 'trailingPE',
            'PE_Forward': 'forwardPE',
            'PB_Ratio': 'priceToBook',
            'PS_Ratio': 'priceToSalesTrailing12Months',
            'EV_EBITDA': 'enterpriseToEbitda',
            'EV_Revenue': 'enterpriseToRevenue',
            'ROE': 'returnOnEquity',
            'ROA': 'returnOnAssets',
            'Dividend_Yield': 'dividendYield',
            'Debt_to_Equity': 'debtToEquity'
        }
        
        for ratio_name, api_key in api_ratios.items():
            value = self._safe_get(self.info, api_key, np.nan)
            if not pd.isna(value):
                self.direct_ratios['Trend'][ratio_name] = value
        
        # Replace calculated values with API values if calculated ones are missing
        for i in range(len(self.trend_data['Years'])):
            for ratio_name in api_ratios.keys():
                if ratio_name in self.trend_data and i == 0:  # Only for most recent year
                    if pd.isna(self.trend_data[ratio_name][i]) and ratio_name in self.direct_ratios['Trend']:
                        self.trend_data[ratio_name][i] = self.direct_ratios['Trend'][ratio_name]
    
    def _calculate_spot_metrics(self):
        """Calculate metrics for spot analysis (most recent quarter)."""
        print("Calculating spot metrics...")
        
        # Create a dictionary to store spot data
        self.spot_data = {}
        
        # Get most recent quarterly data
        if len(self.income_quarterly.columns) > 0 and len(self.balance_quarterly.columns) > 0 and len(self.cashflow_quarterly.columns) > 0:
            # Date of most recent quarter
            recent_quarter = self.income_quarterly.columns[0]
            self.spot_data['Quarter_End_Date'] = recent_quarter
            
            # Fundamental Numbers
            self.spot_data['Revenue'] = self._get_statement_value(self.income_quarterly, 'Total Revenue')
            self.spot_data['Net_Income'] = self._get_statement_value(self.income_quarterly, 'Net Income')
            self.spot_data['Gross_Profit'] = self._get_statement_value(self.income_quarterly, 'Gross Profit')
            self.spot_data['EBITDA'] = self._get_statement_value(self.income_quarterly, 'EBITDA')
            
            self.spot_data['Equity'] = self._get_statement_value(self.balance_quarterly, 'Total Stockholder Equity')
            self.spot_data['Assets'] = self._get_statement_value(self.balance_quarterly, 'Total Assets')
            
            # Debt calculation
            total_debt = self._get_statement_value(self.balance_quarterly, 'Total Debt')
            if pd.isna(total_debt):
                long_term_debt = self._get_statement_value(self.balance_quarterly, 'Long Term Debt')
                short_term_debt = self._get_statement_value(self.balance_quarterly, 'Short Long Term Debt')
                total_debt = (0 if pd.isna(long_term_debt) else long_term_debt) + (0 if pd.isna(short_term_debt) else short_term_debt)
            self.spot_data['Debt'] = total_debt
            
            self.spot_data['Operating_Cash_Flow'] = self._get_statement_value(self.cashflow_quarterly, 'Operating Cash Flow')
            self.spot_data['Dividends'] = abs(self._get_statement_value(self.cashflow_quarterly, 'Dividends Paid'))
            
            # Calculated ratios
            self.spot_data['Payout_Ratio'] = self._safe_calculate(self.spot_data['Dividends'], self.spot_data['Net_Income'])
            
            self.spot_data['Capital_Expenditures'] = self._get_statement_value(self.cashflow_quarterly, 'Capital Expenditure')
            
            # FCF calculation
            fcf = (self.spot_data['Operating_Cash_Flow'] + self.spot_data['Capital_Expenditures'] 
                  if not pd.isna(self.spot_data['Operating_Cash_Flow']) and not pd.isna(self.spot_data['Capital_Expenditures']) 
                  else np.nan)
            self.spot_data['Free_Cash_Flow'] = fcf
            
            # FCF/OCF Ratio
            self.spot_data['FCF_OCF_Ratio'] = self._safe_calculate(fcf, self.spot_data['Operating_Cash_Flow'])
        else:
            print("Warning: Not enough quarterly data available for spot analysis.")
        
        # Check for direct API ratios for spot metrics
        # DIRECT API PULL SECTION - Spot Metrics
        self.direct_ratios['Spot'] = {}
        
        api_spot_metrics = {
            'Total_Revenue': 'totalRevenue',
            'Gross_Profits': 'grossProfits',
            'EBITDA': 'ebitda',
            'Total_Cash': 'totalCash',
            'Total_Debt': 'totalDebt',
            'Operating_Cash_Flow': 'operatingCashflow',
            'Free_Cash_Flow': 'freeCashflow',
            'Payout_Ratio': 'payoutRatio'
        }
        
        for metric_name, api_key in api_spot_metrics.items():
            value = self._safe_get(self.info, api_key, np.nan)
            if not pd.isna(value):
                self.direct_ratios['Spot'][metric_name] = value
        
        # Replace calculated values with API values if calculated ones are missing
        for metric_name in self.spot_data:
            # Map to API keys if possible
            api_key = None
            for k, v in api_spot_metrics.items():
                if k.lower().replace('_', '') == metric_name.lower().replace('_', ''):
                    api_key = k
                    break
            
            if api_key and pd.isna(self.spot_data[metric_name]) and api_key in self.direct_ratios['Spot']:
                self.spot_data[metric_name] = self.direct_ratios['Spot'][api_key]
    
    def get_trend_analysis(self):
        """Return formatted trend analysis data."""
        # Convert trend data to DataFrame
        trend_df = pd.DataFrame(self.trend_data)
        if not trend_df.empty and 'Years' in trend_df.columns:
            trend_df.set_index('Years', inplace=True)
        
        return {
            'trend_data': trend_df,
            'direct_ratios': self.direct_ratios['Trend'] if 'Trend' in self.direct_ratios else {}
        }
    
    def get_spot_analysis(self):
        """Return formatted spot analysis data."""
        return {
            'spot_data': self.spot_data,
            'direct_ratios': self.direct_ratios['Spot'] if 'Spot' in self.direct_ratios else {}
        }
    
    def display_trend_analysis(self):
        """Display trend analysis data."""
        trend_data = self.get_trend_analysis()
        trend_df = trend_data['trend_data']
        
        # Format the output
        print(f"\n{'='*80}")
        print(f"TREND ANALYSIS FOR {self.ticker} - Last {len(trend_df)} years")
        print(f"{'='*80}")
        
        # Format dataframe for display
        display_df = trend_df.copy()
        
        # Format percentage values
        pct_columns = [
            'Revenue_Growth', 'Gross_Margin', 'Operating_Margin', 'EBITDA_Margin', 
            'Net_Margin', 'ROA', 'ROE', 'ROIC', 'Earnings_Growth', 'FCF_Yield', 
            'Dividend_Yield'
        ]
        
        for col in pct_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
                )
        
        # Format currency values
        currency_columns = ['Revenue', 'EPS', 'EPS_Estimated']
        for col in currency_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"${x:,.2f}" if not pd.isna(x) else "N/A"
                )
        
        # Format ratio values
        ratio_columns = [col for col in display_df.columns if col not in pct_columns + currency_columns]
        for col in ratio_columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
            )
        
        # Display by groups
        groups = {
            "Company Fundamentals": [
                'Revenue', 'Revenue_Growth', 'EPS', 'EPS_Estimated', 'Gross_Margin', 
                'Operating_Margin', 'EBITDA_Margin', 'Net_Margin', 'ROA', 'ROE', 'ROIC', 
                'Earnings_Growth'
            ],
            "Valuation Metrics": [
                'PS_Ratio', 'PE_Ratio', 'PB_Ratio', 'PC_Ratio', 'P_FCF', 'EV_EBITDA', 
                'EV_Revenue', 'EV_FCF', 'FCF_Yield', 'Dividend_Yield'
            ],
            "Financial Statement Analysis": [
                'Quick_Ratio', 'Debt_to_Equity', 'Interest_Coverage', 'Asset_Turnover', 
                'Asset_Equity_Ratio'
            ]
        }
        
        for group_name, metrics in groups.items():
            print(f"\n{group_name}:")
            print("-" * 80)
            group_df = display_df[[col for col in metrics if col in display_df.columns]]
            if not group_df.empty:
                print(group_df.T)
            else:
                print("No data available")
        
        # Display direct ratios from API
        if trend_data['direct_ratios']:
            print("\nDirect Ratios from API (Most Recent):")
            print("-" * 80)
            for ratio, value in trend_data['direct_ratios'].items():
                if ratio in pct_columns:
                    print(f"{ratio}: {value*100:.2f}%")
                else:
                    print(f"{ratio}: {value:.2f}")
    
    def display_spot_analysis(self):
        """Display spot analysis data."""
        spot_data = self.get_spot_analysis()
        
        print(f"\n{'='*80}")
        print(f"SPOT ANALYSIS FOR {self.ticker} - Most Recent Quarter")
        print(f"{'='*80}")
        
        if 'Quarter_End_Date' in spot_data['spot_data']:
            print(f"Quarter End Date: {spot_data['spot_data']['Quarter_End_Date']}")
        
        # Format and display spot metrics
        metrics = {
            'Revenue': 'Revenue',
            'Net_Income': 'Net Income',
            'Gross_Profit': 'Gross Profit',
            'EBITDA': 'EBITDA',
            'Equity': 'Total Equity',
            'Assets': 'Total Assets',
            'Debt': 'Total Debt',
            'Operating_Cash_Flow': 'Operating Cash Flow',
            'Dividends': 'Dividends Paid',
            'Payout_Ratio': 'Payout Ratio',
            'Capital_Expenditures': 'Capital Expenditures',
            'Free_Cash_Flow': 'Free Cash Flow',
            'FCF_OCF_Ratio': 'FCF/OCF Ratio'
        }
        
        print("\nFundamental Numbers:")
        print("-" * 80)
        for key, label in metrics.items():
            if key in spot_data['spot_data']:
                value = spot_data['spot_data'][key]
                if key in ['Payout_Ratio', 'FCF_OCF_Ratio']:
                    value_str = f"{value*100:.2f}%" if not pd.isna(value) else "N/A"
                elif key in ['Capital_Expenditures']:
                    # Capital expenditures are usually negative in statements
                    value_str = f"${abs(value):,.2f}" if not pd.isna(value) else "N/A"
                else:
                    value_str = f"${value:,.2f}" if not pd.isna(value) else "N/A"
                print(f"{label}: {value_str}")
        
        # Display direct ratios from API
        if spot_data['direct_ratios']:
            print("\nDirect Metrics from API:")
            print("-" * 80)
            for metric, value in spot_data['direct_ratios'].items():
                if metric in ['Payout_Ratio']:
                    print(f"{metric}: {value*100:.2f}%")
                else:
                    print(f"{metric}: ${value:,.2f}")
    
    def plot_trend_analysis(self):
        """Plot key trend metrics."""
        trend_data = self.get_trend_analysis()
        trend_df = trend_data['trend_data']
        
        if trend_df.empty:
            print("No trend data available for plotting.")
            return
        
        # Create plots for key metrics
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Trend Analysis for {self.ticker}', fontsize=16)
        
        # Revenue and Growth
        ax1 = axes[0, 0]
        trend_df['Revenue'].plot(kind='bar', ax=ax1, color='blue')
        ax1.set_title('Revenue (in millions)')
        ax1.set_ylabel('Revenue')
        ax1.set_xlabel('Year')
        
        ax1r = ax1.twinx()
        trend_df['Revenue_Growth'].plot(kind='line', ax=ax1r, color='red', marker='o')
        ax1r.set_ylabel('Growth Rate (%)')
        ax1r.grid(False)
        
        # Margins
        ax2 = axes[0, 1]
        for col in ['Gross_Margin', 'Operating_Margin', 'Net_Margin']:
            if col in trend_df.columns:
                trend_df[col].plot(kind='line', ax=ax2, marker='o')
        ax2.set_title('Margins')
        ax2.set_ylabel('Margin (%)')
        ax2.set_xlabel('Year')
        ax2.legend()
        
        # Returns
        ax3 = axes[1, 0]
        for col in ['ROA', 'ROE', 'ROIC']:
            if col in trend_df.columns:
                trend_df[col].plot(kind='line', ax=ax3, marker='o')
        ax3.set_title('Returns')
        ax3.set_ylabel('Return (%)')
        ax3.set_xlabel('Year')
        ax3.legend()
        
        # Valuation Multiples
        ax4 = axes[1, 1]
        for col in ['PE_Ratio', 'PB_Ratio', 'PS_Ratio']:
            if col in trend_df.columns:
                trend_df[col].plot(kind='line', ax=ax4, marker='o')
        ax4.set_title('Valuation Multiples')
        ax4.set_ylabel('Multiple')
        ax4.set_xlabel('Year')
        ax4.legend()
        
        # EV Multiples
        ax5 = axes[2, 0]
        for col in ['EV_EBITDA', 'EV_Revenue']:
            if col in trend_df.columns:
                trend_df[col].plot(kind='line', ax=ax5, marker='o')
        ax5.set_title('Enterprise Value Multiples')
        ax5.set_ylabel('Multiple')
        ax5.set_xlabel('Year')
        ax5.legend()
        
        # Financial Health Metrics
        ax6 = axes[2, 1]
        for col in ['Quick_Ratio', 'Debt_to_Equity', 'Interest_Coverage']:
            if col in trend_df.columns:
                trend_df[col].plot(kind='line', ax=ax6, marker='o')
        ax6.set_title('Financial Health')
        ax6.set_ylabel('Ratio')
        ax6.set_xlabel('Year')
        ax6.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
        plt.show()

    def plot_spot_analysis(self):
        """Plot key spot metrics from the most recent quarter."""
        spot_data = self.get_spot_analysis()
        
        if not spot_data['spot_data']:
            print("No spot data available for plotting.")
            return
        
        # Extract data
        data = spot_data['spot_data']
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Spot Analysis for {self.ticker} - Recent Quarter', fontsize=16)
        
        # 1. Revenue, Profit, and Income
        financial_data = {
            'Revenue': data.get('Revenue', np.nan),
            'Gross Profit': data.get('Gross_Profit', np.nan),
            'EBITDA': data.get('EBITDA', np.nan),
            'Net Income': data.get('Net_Income', np.nan)
        }
        
        # Filter out NaN values
        financial_data = {k: v for k, v in financial_data.items() if not pd.isna(v)}
        
        if financial_data:
            ax1 = axes[0, 0]
            ax1.bar(financial_data.keys(), financial_data.values(), color=['blue', 'green', 'orange', 'red'])
            ax1.set_title('Revenue & Profit Metrics')
            ax1.set_ylabel('Amount ($)')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for i, v in enumerate(financial_data.values()):
                ax1.text(i, v * 1.01, f'${v:,.0f}', ha='center')
        
        # 2. Balance Sheet Metrics
        balance_data = {
            'Assets': data.get('Assets', np.nan),
            'Debt': data.get('Debt', np.nan),
            'Equity': data.get('Equity', np.nan)
        }
        
        # Filter out NaN values
        balance_data = {k: v for k, v in balance_data.items() if not pd.isna(v)}
        
        if balance_data:
            ax2 = axes[0, 1]
            ax2.bar(balance_data.keys(), balance_data.values(), color=['blue', 'red', 'green'])
            ax2.set_title('Balance Sheet Metrics')
            ax2.set_ylabel('Amount ($)')
            
            # Add value labels
            for i, v in enumerate(balance_data.values()):
                ax2.text(i, v * 1.01, f'${v:,.0f}', ha='center')
        
        # 3. Cash Flow Metrics
        cashflow_data = {
            'Operating CF': data.get('Operating_Cash_Flow', np.nan),
            'CapEx': abs(data.get('Capital_Expenditures', 0)) if not pd.isna(data.get('Capital_Expenditures', np.nan)) else np.nan,
            'FCF': data.get('Free_Cash_Flow', np.nan),
            'Dividends': data.get('Dividends', np.nan)
        }
        
        # Filter out NaN values
        cashflow_data = {k: v for k, v in cashflow_data.items() if not pd.isna(v)}
        
        if cashflow_data:
            ax3 = axes[1, 0]
            ax3.bar(cashflow_data.keys(), cashflow_data.values(), color=['blue', 'red', 'green', 'purple'])
            ax3.set_title('Cash Flow Metrics')
            ax3.set_ylabel('Amount ($)')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for i, v in enumerate(cashflow_data.values()):
                ax3.text(i, v * 1.01, f'${v:,.0f}', ha='center')
        
        # 4. Ratios
        ratio_data = {
            'Payout Ratio': data.get('Payout_Ratio', np.nan),
            'FCF/OCF Ratio': data.get('FCF_OCF_Ratio', np.nan)
        }
        
        # Filter out NaN values
        ratio_data = {k: v for k, v in ratio_data.items() if not pd.isna(v)}
        
        if ratio_data:
            ax4 = axes[1, 1]
            ax4.bar(ratio_data.keys(), ratio_data.values(), color=['blue', 'green'])
            ax4.set_title('Financial Ratios')
            ax4.set_ylabel('Ratio')
            
            # Add percentage labels for ratios
            for i, v in enumerate(ratio_data.values()):
                ax4.text(i, v * 1.01, f'{v*100:.1f}%', ha='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
        plt.show()

    def save_to_excel(self, filename=None):
        """Save trend and spot analysis data to Excel file."""
        if filename is None:
            filename = f"{self.ticker}_fundamental_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx"
        
        with pd.ExcelWriter(filename) as writer:
            # Get trend data
            trend_data = self.get_trend_analysis()
            trend_df = trend_data['trend_data']
            
            # Save trend data
            if not trend_df.empty:
                trend_df.to_excel(writer, sheet_name='Trend Analysis')
            
            # Save direct trend ratios
            if trend_data['direct_ratios']:
                pd.DataFrame.from_dict(trend_data['direct_ratios'], orient='index', columns=['Value']).to_excel(
                    writer, sheet_name='API Direct Trend Ratios')
            
            # Get spot data
            spot_data = self.get_spot_analysis()
            
            # Save spot data
            if spot_data['spot_data']:
                spot_df = pd.DataFrame.from_dict(spot_data['spot_data'], orient='index', columns=['Value'])
                spot_df.to_excel(writer, sheet_name='Spot Analysis')
            
            # Save direct spot metrics
            if spot_data['direct_ratios']:
                pd.DataFrame.from_dict(spot_data['direct_ratios'], orient='index', columns=['Value']).to_excel(
                    writer, sheet_name='API Direct Spot Metrics')
        
        print(f"Fundamental analysis data saved to {filename}")

    def generate_report(self):
        """Generate a comprehensive report of the fundamental analysis."""
        print(f"\n{'='*80}")
        print(f"FUNDAMENTAL ANALYSIS REPORT FOR {self.ticker}")
        print(f"{'='*80}")
        print(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Display company info
        print("\nCOMPANY INFORMATION:")
        print("-" * 80)
        print(f"Name: {self._safe_get(self.info, 'longName', 'N/A')}")
        print(f"Sector: {self._safe_get(self.info, 'sector', 'N/A')}")
        print(f"Industry: {self._safe_get(self.info, 'industry', 'N/A')}")
        print(f"Website: {self._safe_get(self.info, 'website', 'N/A')}")
        
        # Display trend analysis
        self.display_trend_analysis()
        
        # Display spot analysis
        self.display_spot_analysis()
        
        print(f"\n{'='*80}")
        print("END OF REPORT")
        print(f"{'='*80}")