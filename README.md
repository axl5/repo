# Analytics Toolkit Showcase

Welcome to my portfolio of end-to-end data engines—designed to turn raw numbers into clear, actionable stories. Dive in to see how I’ve built plug-and-play pipelines, flexible metric engines, and visual layers that adapt to any domain, whether it's market trends or operational insights. 


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


**Why this matters:** both modules are designed as adaptable building blocks—whether you’re analyzing customer usage patterns or business metrics, users get a turn-key data pipeline, metric engine, and visualization layer that scale with their needs. 😉
