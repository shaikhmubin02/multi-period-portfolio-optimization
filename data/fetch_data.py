# data/fetch_data.py

import pandas as pd
import yfinance as yf
import logging

def fetch_historical_data(assets, start_date, end_date):
    """
    Fetches historical adjusted close price data for given assets.
    
    Args:
        assets (list): List of asset tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: DataFrame containing adjusted close prices.
    """
    try:
        data = yf.download(assets, start=start_date, end=end_date, auto_adjust=True)
        price_data = data['Close']
        price_data = price_data.dropna()
        return price_data
    except Exception as e:
        logging.getLogger('PortfolioOptimization').error(f"Error fetching data: {e}")
        raise