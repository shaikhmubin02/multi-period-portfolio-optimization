import pandas as pd
import yfinance as yf

def fetch_historical_data(tickers, start_date, end_date):
    """
    Fetches historical price data for the given tickers between start_date and end_date.
    
    Args:
        tickers (list): List of asset tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: Historical price data.
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']