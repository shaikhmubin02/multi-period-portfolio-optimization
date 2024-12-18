import logging
from datetime import datetime
from data.fetch_data import fetch_historical_data
from data.preprocess_data import preprocess_data
from models.predictive_models import PredictiveModel
from optimization.optimization_algorithms import maximize_utility_optimization
from backtesting.backtest import backtest_strategy
import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def map_sectors_to_constraints(assets, sectors_dict, config):
    """
    Maps asset sectors to their cumulative weight constraints.
    
    Args:
        assets (list): List of asset tickers.
        sectors_dict (dict): Dictionary mapping asset tickers to sectors.
        config (dict): Configuration dictionary containing sector constraints.
        
    Returns:
        tuple: (sector_constraints, asset_sectors_list)
            - sector_constraints: Dictionary mapping sectors to their max cumulative weight.
            - asset_sectors_list: List of sectors in the same order as assets.
    """
    sector_constraints = config.get('sector_constraints', {})
    asset_sectors_list = [sectors_dict.get(asset, 'Other') for asset in assets]
    return sector_constraints, asset_sectors_list

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("portfolio_optimization.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    logger.info("Starting portfolio optimization process.")

    # Load configuration
    config = load_config()
    
    # Define asset sectors
    sectors = config.get('asset_sectors', {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'AMZN': 'Technology',
        'META': 'Technology',
        'JPM': 'Finance',
        'XOM': 'Energy',
        'JNJ': 'Healthcare',
        'VZ': 'Telecommunications',
        'PG': 'Consumer Staples'
    })
    
    ASSETS = list(sectors.keys())
    
    # Fetch historical data
    logger.info("Fetching historical data.")
    price_data = fetch_historical_data(
        assets=ASSETS,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    logger.info("Historical data fetched successfully.")
    
    # Preprocess data
    logger.info("Preprocessing data.")
    returns, features = preprocess_data(price_data)
    logger.info("Data preprocessing completed.")
    
    # Train predictive models
    logger.info("Training predictive models.")
    model = PredictiveModel(config=config.get('model_config', {}))
    model.train(returns, features, ASSETS, logger)
    expected_returns = model.get_expected_returns()
    logger.info("Predictive modeling completed.")
    
    # Portfolio Optimization
    logger.info("Starting portfolio optimization.")
    cov_matrix = returns.cov().values
    
    # Mapping sectors to constraints
    sector_constraints, asset_sectors_list = map_sectors_to_constraints(ASSETS, sectors, config)
    
    optimization_config = config.get('optimization', {})
    lambda_param = optimization_config.get('lambda_param', 0.7)
    max_weight = optimization_config.get('max_weight', 0.2)
    min_weight = optimization_config.get('min_weight', 0.05)
    
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
        logger.info(f"Optimized Weights: {weights}\n")
    except ValueError as e:
        logger.error(f"Optimization failed: {str(e)}")
        return
    
    # Backtesting
    logger.info("Starting backtesting.")
    backtesting_config = config.get('backtesting', {})
    portfolio_value, metrics = backtest_strategy(
        returns=returns,
        weights=weights,
        rebalance_frequency='BM',
        transaction_cost_rate=0.001,
        logger=logger  # Optional: Pass logger if needed for additional logging within backtest_strategy
    )
    
    logger.info("Backtesting completed.")
    
    # Save Portfolio Value
    portfolio_value.to_csv('portfolio_value_over_time.csv')
    logger.info("Portfolio Value Over Time saved to 'portfolio_value_over_time.csv'.")
    
    # Log Performance Metrics
    logger.info("Performance Metrics:")
    logger.info(f"CAGR: {metrics['CAGR']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['Max Drawdown']:.2f}%")
    logger.info(f"Calmar Ratio: {metrics['Calmar Ratio']:.2f}")
    
    # After backtesting and before plotting
    logger.info(f"Portfolio Value Statistics:\n{portfolio_value.describe()}")
    
    # Create Plots (Ensure this is called only once)
    create_plots(portfolio_value, metrics, returns, weights, logger)
    
    logger.info("Portfolio optimization process completed.")

def create_plots(portfolio_value, metrics, returns, weights, logger):
    """
    Creates and saves plots for portfolio performance metrics.
    """
    # Ensure the 'plots' directory exists
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # --- Plot 1: Portfolio Value Over Time ---
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_value.index, portfolio_value.values, label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    portfolio_plot_path = os.path.join(plots_dir, 'portfolio_value_over_time.png')
    plt.savefig(portfolio_plot_path)
    plt.close()
    logger.info(f"Portfolio Value Over Time plot saved to '{portfolio_plot_path}'.")
    
    # --- Plot 2: Performance Metrics Bar Chart ---
    metrics_keys = ['CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio']
    metrics_values = [metrics.get(key, np.nan) for key in metrics_keys]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics_keys, metrics_values, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Performance Metrics')
    plt.ylabel('Values')
    plt.title('Portfolio Performance Metrics')
    
    # Annotate bars with their values
    for bar, value, key in zip(bars, metrics_values, metrics_keys):
        height = bar.get_height()
        if np.isnan(value):
            text = 'N/A'
        else:
            if key in ['CAGR', 'Max Drawdown']:
                text = f"{value:.2f}%"
            else:
                text = f"{value:.2f}"
        plt.text(bar.get_x() + bar.get_width() / 2, height, text, ha='center', va='bottom')
    
    plt.ylim(0, max(metrics_values) * 1.2 if not all(np.isnan(metrics_values)) else 1)
    plt.grid(axis='y')
    metrics_plot_path = os.path.join(plots_dir, 'performance_metrics.png')
    plt.savefig(metrics_plot_path)
    plt.close()
    logger.info(f"Performance Metrics plot saved to '{metrics_plot_path}'.")
    
    # --- Plot 3: Portfolio Drawdown Over Time ---
    drawdown = (portfolio_value / portfolio_value.cummax()) - 1
    plt.figure(figsize=(10, 6))
    plt.plot(drawdown.index, drawdown.values * 100, label='Drawdown', color='red')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.title('Portfolio Drawdown Over Time')
    plt.legend()
    plt.grid(True)
    drawdown_plot_path = os.path.join(plots_dir, 'portfolio_drawdown_over_time.png')
    plt.savefig(drawdown_plot_path)
    plt.close()
    logger.info(f"Portfolio Drawdown Over Time plot saved to '{drawdown_plot_path}'.")
    
    # --- Plot 4: Yearly Percentage Returns ---
    # Calculate yearly returns correctly using 'YE' for year-end
    yearly_returns = portfolio_value.resample('YE').last().pct_change() * 100
    yearly_returns = yearly_returns.dropna()
    
    # Log yearly returns with proper formatting
    logger.info("Yearly Percentage Returns:")
    for year, value in yearly_returns.items():
        logger.info(f"{year.year}: {value:.2f}%")
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(yearly_returns.index.year, yearly_returns.values, color='skyblue')
    
    # Annotate bars with their values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height, 
                 f'{height:.1f}%', 
                 ha='center', 
                 va='bottom' if height >= 0 else 'top')
    
    plt.xlabel('Year')
    plt.ylabel('Annual Return (%)')
    plt.title('Yearly Percentage Returns of Portfolio')
    plt.grid(axis='y')
    plt.xticks(yearly_returns.index.year, rotation=45)
    
    plt.tight_layout()
    yearly_returns_plot_path = os.path.join(plots_dir, 'yearly_returns.png')
    plt.savefig(yearly_returns_plot_path)
    plt.close()
    logger.info(f"Yearly Returns plot saved to '{yearly_returns_plot_path}'.")
    
    # --- Plot 5: Individual Companies' Annual Returns ---
    weighted_returns = returns.mul(weights, axis=1)
    annual_asset_returns = weighted_returns.resample('YE').sum() * 100  # 'YE' stands for year-end
    
    plt.figure(figsize=(12, 8))
    years = annual_asset_returns.index.year
    
    for column in annual_asset_returns.columns:
        plt.plot(years, annual_asset_returns[column], marker='o', label=column, linewidth=2)
    
    plt.xlabel('Year')
    plt.ylabel('Annual Return Contribution (%)')
    plt.title("Individual Companies' Annual Return Contribution to Portfolio")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xticks(years, rotation=45)
    
    plt.tight_layout()
    individual_returns_plot_path = os.path.join(plots_dir, 'individual_companies_annual_returns.png')
    plt.savefig(individual_returns_plot_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Individual Companies' Annual Returns plot saved to '{individual_returns_plot_path}'.")

def calculate_and_plot_yearly_returns(portfolio_value, logger):
    """
    Calculates yearly percentage returns and creates a plot to visualize them.

    Args:
        portfolio_value (pd.Series): Portfolio value over time.
        logger (logging.Logger): Logger instance for logging.
    """
    # Calculate Yearly Returns
    yearly_returns = portfolio_value.resample('YE').last().pct_change().dropna() * 100  # Convert to percentage
    
    logger.info("Yearly Percentage Returns:")
    for year, ret in yearly_returns.items():
        logger.info(f"{year}: {ret:.2%}")
    
    # Create Yearly Returns Plot
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in yearly_returns.values]
    bars = plt.bar(yearly_returns.index.astype(str), yearly_returns.values * 100, color=colors)
    
    plt.title('Yearly Returns', fontsize=14, pad=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Return (%)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    yearly_returns_plot_path = os.path.join(plots_dir, 'yearly_returns.png')
    plt.savefig(yearly_returns_plot_path)
    plt.close()
    logger.info(f"Yearly Returns plot saved to '{yearly_returns_plot_path}'.")

def plot_portfolio_value(portfolio_value):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value.index, portfolio_value.values, linewidth=2)
    plt.title('Portfolio Value Over Time', fontsize=14, pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/portfolio_value_over_time.png')
    plt.close()

def plot_performance_metrics(cagr, sharpe, max_drawdown, calmar):
    metrics = ['CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio']
    values = [cagr, sharpe, max_drawdown, calmar]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#2ecc71', '#3498db', '#e74c3c', '#f1c40f'])
    plt.title('Performance Metrics', fontsize=14, pad=15)
    plt.ylabel('Value', fontsize=12)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}' if height < 0 else f'+{height:.2%}',
                ha='center', va='bottom')
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/performance_metrics.png')
    plt.close()

def plot_drawdown(drawdown):
    plt.figure(figsize=(12, 6))
    plt.plot(drawdown.index, drawdown.values * 100, linewidth=2, color='#e74c3c')
    plt.title('Portfolio Drawdown Over Time', fontsize=14, pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    plt.fill_between(drawdown.index, drawdown.values * 100, 0, color='#e74c3c', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/portfolio_drawdown_over_time.png')
    plt.close()

if __name__ == "__main__":
    main()