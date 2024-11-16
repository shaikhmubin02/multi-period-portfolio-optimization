# data/preprocess_data.py

import pandas as pd
import numpy as np

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
    # Calculate daily returns
    returns = price_data.pct_change().dropna()
    
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
        
        # ATR (Average True Range)
        features[f'{ticker}_ATR_14'] = compute_atr(price_data[ticker], window=14)
    
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

def compute_macd(series, fast=12, slow=26, signal=9):
    """
    Computes the Moving Average Convergence Divergence (MACD) for a given series.
    
    Args:
        series (pd.Series): Price series
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line period
        
    Returns:
        tuple: (MACD line, Signal line)
    """
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_atr(series, window=14):
    """
    Computes the Average True Range (ATR) for a given series.
    
    Args:
        series (pd.Series): Price series
        window (int): Window size for ATR calculation
        
    Returns:
        pd.Series: ATR values
    """
    high = series
    low = series
    close = series
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr