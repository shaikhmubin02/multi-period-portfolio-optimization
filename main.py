from data.fetch_data import fetch_historical_data
from data.preprocess_data import preprocess_data
from models.predictive_models import PredictiveModel
from optimization.optimization_algorithms import maximize_utility_optimization
from backtesting.backtest import backtest_strategy
import numpy as np
import pandas as pd

# Define asset sectors
sectors = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'AMZN': 'Technology',
    'META': 'Technology',
    'JPM': 'Finance',           # JPMorgan Chase & Co.
    'XOM': 'Energy',            # Exxon Mobil Corporation
    'JNJ': 'Healthcare',        # Johnson & Johnson
    'VZ': 'Telecommunications', # Verizon Communications
    'PG': 'Consumer Staples'    # Procter & Gamble
}

def main():
    # Step 1: Data Collection
    ASSETS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'XOM', 'JNJ', 'VZ', 'PG']  # Define asset tickers
    start_date = '2015-01-01'
    end_date = '2024-11-15'
    price_data = fetch_historical_data(ASSETS, start_date, end_date)
    
    # Verify if all tickers were downloaded successfully
    missing_tickers = [ticker for ticker in ASSETS if ticker not in price_data.columns]
    if missing_tickers:
        print(f"Warning: Missing data for tickers: {missing_tickers}")
    
    # Step 2: Data Preprocessing
    returns, features = preprocess_data(price_data)
    
    # Check if returns DataFrame is empty
    if returns.empty:
        print("Error: Returns DataFrame is empty. Exiting program.")
        return
    
    # Step 3: Predictive Modeling (Improved)
    model = PredictiveModel()
    # Define feature columns (all technical indicators and lagged features)
    feature_columns = [col for col in features.columns if 'SMA' in col or 'RSI' in col or 'MACD' in col or 'ATR' in col]
    X = features[feature_columns].values
    
    # Initialize expected_returns dictionary
    expected_returns_dict = {}
    
    for asset in ASSETS:
        y = returns[asset].values  # Target variable for the asset
        model.train(X, y, asset)
        predictions = model.predict(X, asset)
        
        # Reassess Expected Returns Calculation
        # Annualize expected returns assuming daily predictions
        daily_return = np.mean(predictions)
        annual_return = daily_return * 252  # 252 trading days
        expected_returns_dict[asset] = annual_return
    
    # Aggregate expected returns into a NumPy array
    expected_returns = np.array([expected_returns_dict[asset] for asset in ASSETS])
    
    print(f"Expected Returns (Annualized): {expected_returns}\n")
    
    # Step 4: Optimization
    asset_returns = returns[ASSETS]  # Extract asset returns
    cov_matrix = asset_returns.cov().values  # Shape: (5, 5)
    
    lambda_param = 0.7  # Adjusted Risk Aversion Parameter
    max_weight = 0.2    # Maximum 20% in any single asset
    min_weight = 0.05   # Minimum 5% in any single asset
    
    # Get sector constraints and asset sectors list
    sector_constraints, asset_sectors_list = map_sectors_to_constraints(ASSETS, sectors)
    
    try:
        weights = maximize_utility_optimization(
            expected_returns=expected_returns, 
            cov_matrix=cov_matrix, 
            lambda_param=lambda_param, 
            max_weight=max_weight,
            min_weight=min_weight,
            sector_constraints=sector_constraints,
            asset_sectors=asset_sectors_list
        )
        print(f"Optimized Weights: {weights}\n")
    except ValueError as e:
        print(str(e))
        return
    
    # Step 5: Backtesting
    portfolio_value = backtest_strategy(price_data, weights, ASSETS)
    print("Portfolio Value Over Time:")
    print(portfolio_value.tail())
    
    # Step 6: Enhanced Performance Metrics
    calculate_performance_metrics(portfolio_value)

def map_sectors_to_constraints(assets, sectors_dict):
    """
    Maps asset sectors to their cumulative weight constraints.
    
    Args:
        assets (list): List of asset tickers.
        sectors_dict (dict): Dictionary mapping asset tickers to sectors.
        
    Returns:
        tuple: (sector_constraints, asset_sectors_list)
            - sector_constraints: Dictionary mapping sectors to their max cumulative weight.
            - asset_sectors_list: List of sectors in the same order as assets.
    """
    # Define sector constraints
    sector_constraints = {
        'Technology': 0.6,      # Max 60% in Technology sector
        'Finance': 0.3,         # Max 30% in Finance sector
        'Energy': 0.1,          # Max 10% in Energy sector
        'Healthcare': 0.2,      # Max 20% in Healthcare sector
        'Telecommunications': 0.2, # Max 20% in Telecommunications sector
        'Consumer Staples': 0.2  # Max 20% in Consumer Staples sector
    }
    
    # Create list of sectors in same order as assets
    asset_sectors_list = [sectors_dict.get(asset, 'Other') for asset in assets]
    
    return sector_constraints, asset_sectors_list

def calculate_performance_metrics(portfolio_value):
    """
    Calculates and prints additional performance metrics.

    Args:
        portfolio_value (pd.Series): Portfolio value over time.
    """
    # Calculate daily returns
    daily_returns = portfolio_value.pct_change().dropna()
    
    # Calculate CAGR
    total_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
    cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (365.25 / total_days) - 1
    
    # Calculate Sharpe Ratio (assuming risk-free rate = 0)
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    
    # Calculate Maximum Drawdown
    cumulative_returns = portfolio_value.cummax()
    drawdown = (portfolio_value - cumulative_returns) / cumulative_returns
    max_drawdown = drawdown.min()
    
    print("\nPerformance Metrics:")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

if __name__ == "__main__":
    main()