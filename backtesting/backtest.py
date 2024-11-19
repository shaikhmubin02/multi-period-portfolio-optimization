# backtesting/backtest.py

import pandas as pd
import numpy as np
import logging
import warnings

def backtest_strategy(returns, weights, rebalance_frequency='BM', transaction_cost_rate=0.001, logger=None):
    """
    Backtests the portfolio strategy over the given returns.

    Args:
        returns (pd.DataFrame): DataFrame of asset returns.
        weights (np.ndarray): Initial asset weights.
        rebalance_frequency (str): Frequency for rebalancing (e.g., 'BM' for business month end).
        transaction_cost_rate (float): Transaction cost rate per trade (as a decimal).
        logger (logging.Logger, optional): Logger instance for logging.

    Returns:
        tuple: (portfolio_values, performance_metrics)
            - portfolio_values (pd.Series): Portfolio value over time.
            - performance_metrics (dict): Dictionary containing CAGR, Sharpe Ratio, Max Drawdown, Calmar Ratio.
    """
    if logger is None:
        logger = logging.getLogger()

    # Suppress the 'BM' FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*'BM' is deprecated.*")

    # Ensure returns.index is timezone-naive
    if returns.index.tz is not None:
        returns = returns.tz_convert(None)
        logger.info("Converted returns index to timezone-naive.")

    # Initialize portfolio
    portfolio_values = pd.Series(index=returns.index, dtype=float)
    portfolio_values.iloc[0] = 100000  # Starting with $100,000

    current_weights = weights.copy()

    # Normalize weights to sum to 1
    if not np.isclose(np.sum(current_weights), 1.0):
        current_weights = current_weights / np.sum(current_weights)
        logger.info("Normalized portfolio weights to sum to 1.")

    # Check for NaN in weights
    if np.isnan(current_weights).any():
        logger.error("Initial portfolio weights contain NaN values. Aborting backtest.")
        return portfolio_values, {'CAGR': np.nan, 'Sharpe Ratio': 0.0, 'Max Drawdown': np.nan, 'Calmar Ratio': np.nan}

    # Determine rebalance dates
    rebalance_dates = returns.resample(rebalance_frequency).first().index
    rebalance_dates = rebalance_dates.intersection(returns.index)

    prev_date = returns.index[0]

    for date in returns.index:
        if date in rebalance_dates:
            try:
                logger.info(f"Rebalancing portfolio on {date.date()}")

                # Calculate portfolio value up to the previous date
                if date != prev_date:
                    portfolio_value = portfolio_values.loc[prev_date] * (1 + returns.loc[prev_date].dot(current_weights))
                else:
                    portfolio_value = portfolio_values.loc[prev_date]

                # Calculate new weights (assuming weights are predefined and not changing)
                new_weights = weights.copy()

                # Calculate transaction costs
                weight_diff = new_weights - current_weights
                transaction_cost = np.sum(np.abs(weight_diff)) * portfolio_value * transaction_cost_rate

                # Validate transaction_cost
                if np.isnan(transaction_cost) or np.isinf(transaction_cost):
                    logger.error(f"Invalid transaction cost calculated: {transaction_cost}. Setting to 0.")
                    transaction_cost = 0.0
                else:
                    logger.info(f"Transaction costs applied: {transaction_cost:.2f}")

                # Update portfolio value after transaction costs
                portfolio_value -= transaction_cost

                # Update weights
                current_weights = new_weights.copy()

                # Assign updated portfolio value
                portfolio_values.loc[date] = portfolio_value

            except Exception as e:
                logger.error(f"Error during rebalancing on {date.date()}: {e}")
                # Carry forward the previous portfolio value
                portfolio_values.loc[date] = portfolio_values.loc[prev_date]

        else:
            # Update portfolio value based on returns
            portfolio_value = portfolio_values.loc[prev_date] * (1 + returns.loc[prev_date].dot(current_weights))
            portfolio_values.loc[date] = portfolio_value

        prev_date = date

    # Handle potential NaN values in portfolio_values
    portfolio_values.ffill(inplace=True)

    # Calculate performance metrics
    metrics = {}
    metrics['CAGR'] = calculate_cagr(portfolio_values) * 100  # Convert to percentage
    metrics['Sharpe Ratio'] = calculate_sharpe_ratio(portfolio_values)
    metrics['Max Drawdown'] = calculate_max_drawdown(portfolio_values) * 100  # Convert to percentage
    metrics['Calmar Ratio'] = calculate_calmar_ratio(metrics['CAGR'], metrics['Max Drawdown'])

    return portfolio_values, metrics

def calculate_cagr(portfolio_values):
    """
    Calculates the Compound Annual Growth Rate (CAGR) of the portfolio.

    Args:
        portfolio_values (pd.Series): Portfolio value over time.

    Returns:
        float: CAGR as a decimal.
    """
    start_value = portfolio_values.iloc[0]
    end_value = portfolio_values.iloc[-1]
    num_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    return (end_value / start_value) ** (1 / num_years) - 1

def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.0):
    """
    Calculates the Sharpe Ratio of the portfolio.

    Args:
        portfolio_values (pd.Series): Portfolio value over time.
        risk_free_rate (float, optional): Risk-free rate as a decimal.

    Returns:
        float: Sharpe Ratio.
    """
    portfolio_values_filled = portfolio_values.ffill()
    daily_returns = portfolio_values_filled.pct_change().dropna()
    excess_returns = daily_returns - risk_free_rate / 252
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) != 0 else 0

def calculate_max_drawdown(portfolio_values):
    """
    Calculates the Maximum Drawdown of the portfolio.

    Args:
        portfolio_values (pd.Series): Portfolio value over time.

    Returns:
        float: Maximum Drawdown as a decimal.
    """
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    return drawdown.min()

def calculate_calmar_ratio(cagr, max_drawdown):
    """
    Calculates the Calmar Ratio of the portfolio.

    Args:
        cagr (float): Compound Annual Growth Rate in percentage.
        max_drawdown (float): Maximum Drawdown in percentage.

    Returns:
        float: Calmar Ratio.
    """
    if max_drawdown != 0:
        return cagr / abs(max_drawdown)
    else:
        return np.nan