# data/preprocess_data.py

import pandas as pd
import numpy as np
import logging

def preprocess_data(price_data):
    """
    Preprocesses the raw price data to calculate returns and technical indicators.
    
    Args:
        price_data (pd.DataFrame): Raw adjusted close price data.
        
    Returns:
        tuple: (returns, features)
            - returns (pd.DataFrame): Preprocessed data with returns
            - features (pd.DataFrame): Technical indicators and features
    """
    logger = logging.getLogger('PortfolioOptimization')
    
    # Handle missing values
    if price_data.isnull().values.any():
        logger.warning("Missing values detected in price data. Applying forward fill.")
        price_data = price_data.fillna(method='ffill').dropna()
    
    # Calculate daily returns
    returns = price_data.pct_change().dropna()
    
    # Verify there are no infinite or NaN returns
    if np.isinf(returns.values).any() or returns.isnull().values.any():
        logger.error("Returns data contains invalid values after preprocessing.")
        raise ValueError("Invalid values in returns data.")
    
    # Initialize features DataFrame
    features = pd.DataFrame(index=price_data.index)
    
    # Calculate technical indicators for each ticker
    for ticker in price_data.columns:
        # Moving Averages
        features[f'{ticker}_SMA_5'] = price_data[ticker].rolling(window=5).mean().pct_change()
        features[f'{ticker}_SMA_10'] = price_data[ticker].rolling(window=10).mean().pct_change()
        
        # RSI
        features[f'{ticker}_RSI_14'] = compute_rsi(price_data[ticker], window=14)
        
        # MACD
        macd, signal = compute_macd(price_data[ticker])
        features[f'{ticker}_MACD'] = macd
        features[f'{ticker}_MACD_Signal'] = signal
        
        # ATR (Average True Range) - modified to use only adjusted close prices
        features[f'{ticker}_ATR_14'] = compute_atr_from_close(price_data[ticker], window=14)
    
    # Drop rows with NaN values resulted from rolling calculations
    features = features.dropna()
    returns = returns.loc[features.index]  # Align returns with features
    
    return returns, features

def compute_rsi(series, window):
    """
    Computes the Relative Strength Index (RSI) for a given series.
    
    Args:
        series (pd.Series): Price series
        window (int): Window size for RSI calculation
        
    Returns:
        pd.Series: RSI values
    """
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    """
    Computes the Moving Average Convergence Divergence (MACD) and Signal line.
    
    Args:
        series (pd.Series): Price series
        span_short (int): Short-term EMA span
        span_long (int): Long-term EMA span
        span_signal (int): Signal line EMA span
        
    Returns:
        tuple: (macd, signal)
    """
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    return macd, signal

def compute_atr_from_close(close_prices, window=14):
    """
    Computes a simplified Average True Range (ATR) using only closing prices.
    
    Args:
        close_prices (pd.Series): Series of closing prices
        window (int): Window size for ATR calculation
    
    Returns:
        pd.Series: ATR values
    """
    # Calculate daily price ranges using close-to-close
    ranges = close_prices.diff().abs()
    
    # Calculate ATR as the rolling mean of the ranges
    atr = ranges.rolling(window=window).mean()
    
    return atr