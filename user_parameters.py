# User Input
ticker = "AAPL"
lookback_period = float("3")
end_date = datetime.now()

# Create instance of ChartAnalysis and download data
analysis = ChartAnalysis(ticker)
analysis.download_data(end_date, lookback_period)
analysis.add_technical_indicators()
analysis.plot_data()
