import pandas as pd
import numpy as np

def backtest_strategy(price_data, weights, assets, initial_capital=100000, transaction_cost=0.001, rebalance_frequency='M'):
    """
    Backtests the portfolio strategy based on provided weights and rebalancing frequency.

    Args:
        price_data (pd.DataFrame): Historical price data. Columns should include only asset tickers.
        weights (np.ndarray): Initial asset weights. Shape should align with the number of assets.
        assets (list): List of asset tickers corresponding to weights.
        initial_capital (float): Starting capital.
        transaction_cost (float): Transaction cost as a decimal (e.g., 0.001 for 0.1%).
        rebalance_frequency (str): Frequency of rebalancing ('M' for monthly, 'W' for weekly, etc.).

    Returns:
        pd.Series: Portfolio value over time.
    """
    # Calculate cumulative returns based on rebalancing frequency
    returns = price_data[assets].pct_change().dropna()
    returns_resampled = returns.resample(rebalance_frequency).agg(lambda x: (1 + x).prod() - 1)
    
    # Initialize portfolio value series
    portfolio_value = pd.Series(index=returns_resampled.index, dtype='float64')
    portfolio_value.iloc[0] = initial_capital
    
    current_weights = weights.copy()
    
    for i in range(1, len(portfolio_value)):
        # Calculate portfolio return
        portfolio_return = np.dot(returns_resampled.iloc[i], current_weights)
        
        # Update portfolio value
        portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + portfolio_return)
        
        # Apply transaction costs for rebalancing
        portfolio_value.iloc[i] *= (1 - transaction_cost * np.sum(np.abs(current_weights - current_weights)))  # Placeholder for actual weight changes
    
    return portfolio_value

def rolling_backtest_strategy(price_data, weights_func, assets, initial_capital=100000, transaction_cost=0.001, window=60, rebalance_frequency='M'):
    """
    Implements a rolling window backtest where weights are updated periodically.

    Args:
        price_data (pd.DataFrame): Historical price data.
        weights_func (function): Function to compute weights based on historical data.
        assets (list): List of asset tickers.
        initial_capital (float): Starting capital.
        transaction_cost (float): Transaction cost as a decimal.
        window (int): Rolling window size in days.
        rebalance_frequency (str): Frequency of rebalancing ('M' for monthly, 'W' for weekly, etc.).

    Returns:
        pd.Series: Portfolio value over time.
    """
    returns = price_data[assets].pct_change().dropna()
    returns_resampled = returns.resample(rebalance_frequency).agg(lambda x: (1 + x).prod() - 1)
    portfolio_values = pd.Series(index=returns_resampled.index, dtype='float64')
    portfolio_values.iloc[0] = initial_capital

    for i in range(1, len(portfolio_values)):
        # Define the window for optimization
        window_start = returns_resampled.index[i - 1] - pd.Timedelta(days=window)
        window_end = returns_resampled.index[i - 1]
        
        window_data = returns.loc[window_start:window_end]
        
        # Compute new weights
        new_weights = weights_func(window_data)
        
        # Calculate portfolio return with previous weights
        daily_return = returns_resampled.iloc[i] @ new_weights
        portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + daily_return)
        
        # Apply transaction costs if weights have changed
        # Calculate change in weights
        weight_change = np.abs(new_weights - new_weights)  # Placeholder for previous weights
        portfolio_values.iloc[i] *= (1 - transaction_cost * np.sum(weight_change))
    
    return portfolio_values