# optimization/optimization_algorithms.py

import numpy as np
from scipy.optimize import minimize

def maximize_utility_optimization(expected_returns, cov_matrix, lambda_param=1.0, max_weight=0.2, min_weight=0.05, sector_constraints=None, asset_sectors=None):
    """
    Maximizes the utility function: Expected Return - lambda * Portfolio Variance
    
    Args:
        expected_returns (np.ndarray): Expected returns for each asset.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        lambda_param (float): Risk aversion parameter.
        max_weight (float): Maximum weight per asset.
        min_weight (float): Minimum weight per asset.
        sector_constraints (dict): Maximum cumulative weight per sector.
        asset_sectors (list): List of sector assignments for each asset.
    
    Returns:
        np.ndarray: Optimized asset weights.
    """
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix, lambda_param)
    
    # Initial guess (equal distribution)
    initial_weights = num_assets * [1. / num_assets,]
    
    # Constraints: Sum of weights = 1
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]
    
    # Adding sector constraints if provided
    if sector_constraints and asset_sectors:
        unique_sectors = set(asset_sectors)
        for sector in unique_sectors:
            # Get indices of assets in this sector
            sector_indices = [i for i, s in enumerate(asset_sectors) if s == sector]
            if sector in sector_constraints:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, indices=sector_indices, sector=sector: sector_constraints[sector] - np.sum(x[indices])
                })
            else:
                print(f"Warning: No sector constraint defined for sector '{sector}'. Skipping.")
    
    # Bounds: min_weight <= weight <= max_weight
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    
    # Optimize
    result = minimize(
        utility_function,
        initial_weights,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    
    return result.x

def utility_function(weights, expected_returns, cov_matrix, lambda_param):
    """
    Utility function to maximize: Expected Return - lambda * Portfolio Variance
    
    Args:
        weights (np.ndarray): Asset weights.
        expected_returns (np.ndarray): Expected returns.
        cov_matrix (np.ndarray): Covariance matrix.
        lambda_param (float): Risk aversion parameter.
    
    Returns:
        float: Negative utility (for minimization).
    """
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    utility = portfolio_return - lambda_param * portfolio_variance
    return -utility  # Negative for minimization