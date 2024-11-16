import pandas as pd

def backtest_strategy(price_data, weights, assets, initial_capital=100000, transaction_cost=0.001):
    """
    Backtests the portfolio strategy based on provided weights.

    Args:
        price_data (pd.DataFrame): Historical price data. Columns should include only asset tickers.
        weights (np.ndarray): Asset weights. Shape should align with the number of assets.
        assets (list): List of asset tickers corresponding to weights.
        initial_capital (float): Starting capital.
        transaction_cost (float): Transaction cost as a decimal (e.g., 0.001 for 0.1%).

    Returns:
        pd.Series: Portfolio value over time.
    """
    # Calculate daily returns for assets only
    returns = price_data[assets].pct_change().dropna()
    
    # Initialize portfolio value series
    portfolio_values = pd.Series(index=returns.index, dtype='float64')
    portfolio_values.iloc[0] = initial_capital
    
    # Iterate over each day to simulate portfolio performance
    for i in range(1, len(portfolio_values)):
        # Calculate daily portfolio return
        daily_return = returns.iloc[i].dot(weights)
        
        # Apply transaction costs if rebalancing is performed
        # For simplicity, assuming rebalancing is done daily
        portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + daily_return) * (1 - transaction_cost)
    
    return portfolio_values

def backtest_strategy_with_rolling_window(price_data, weights_func, assets, initial_capital=100000, transaction_cost=0.001, window=60):
    """
    Backtests the portfolio strategy using a rolling window approach.

    Args:
        price_data (pd.DataFrame): Historical price data.
        weights_func (function): Function to calculate weights based on current window data.
        assets (list): List of asset tickers.
        initial_capital (float): Starting capital.
        transaction_cost (float): Transaction cost.
        window (int): Rolling window size in days.

    Returns:
        pd.Series: Portfolio value over time.
    """
    returns = price_data[assets].pct_change().dropna()
    portfolio_values = pd.Series(index=returns.index, dtype='float64')
    portfolio_values.iloc[0] = initial_capital
    
    current_weights = weights_func(returns.iloc[:window])
    
    for i in range(window, len(portfolio_values)):
        current_weights = weights_func(returns.iloc[i-window:i])
        daily_return = returns.iloc[i].dot(current_weights)
        portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + daily_return) * (1 - transaction_cost)
    
    return portfolio_values