## TechnicalAnalysis

This module grabs daily price and volume data for U.S. stocks, then runs a suite of standard indicators (moving averages, MACD, ATR, RSI) on rolling windows. The code is structured in the following way:

- **Data Ingestion Pipeline**: Pull, normalize, and store daily US equity price & volume datawithout writing a single SQL query. 
- **Signal Detection Thresholds**: Engineered trailing indicators to automatically flag trend shift — "alert triggers" for faster, data-driven decision-making. Easily swap in different formulas or add new metrics  
- **Dashboard Visualizations**: Generates self-updating plots where price history and overlaid indicators stay in sync; exportable to PNG or embedded in Jupyter/HTML with no extra work.

## FundamentalAnalysis

This toolkit pulls financial statements, calculates key health metrics, and packages results into exportable Excel formats. Highlights include:

- **Automated KPI Assembly**: Streamline fetching and normalization of revenue, margin, profitability, and valuation data — aggregating granular data for holistic business assessments.  
- **Trend vs. Spot Analysis Views**: Delivers multi-year growth trends alongside recent quarterly “spot” deep dives, enabling users to identify accelerating or decelerating signals at a glance.
- **Auto-formatted Reporting**: Generates ready-to-share Excel exports and static plots with minimal configuration, plus customizable text summaries for “one-page” insights.

- **Unified data fetcher**: one function to grab income, balance sheet, and cash flow data for any public company—and keep it versioned by date.  
- **Trend vs. snapshot views**: out-of-the-box scripts that compare multi-year growth patterns to the most recent quarter, so you can spot accelerating or decelerating signals at a glance.  
- **Auto-formatted outputs**: ready-to-share Excel workbooks and static/interactive plots with minimal configuration, plus customizable text summaries you can drop right into a deck.
