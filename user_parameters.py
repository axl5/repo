# User Input
ticker = "AAPL"
lookback_period = float("3")
end_date = datetime.now()

# Create instance of TechnicalAnalysis and download data
tech_analysis = TechnicalAnalysis(ticker)
tech_analysis.download_data(end_date, lookback_period)
tech_analysis.add_technical_indicators()
tech_analysis.plot_data()

# Create instance of FundamentalAnalysis and download data
fund_analysis = FundamentalAnalysis(ticker)
# Generate a detailed report
fund_analysis.generate_report()
# Visualize trend analysis
fund_analysis.plot_trend_analysis()
# Visualize spot analysis
fund_analysis.plot_spot_analysis()
# Save data to Excel for further analysis
fund_analysis.save_to_excel()