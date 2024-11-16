# optimization/optimization_algorithms.py

import numpy as np
import cvxpy as cp

def maximize_utility_optimization(expected_returns, cov_matrix, lambda_param=1.0, max_weight=0.3):
    """
    Maximizes a utility function balancing return and variance with diversification constraints.
    
    Args:
        expected_returns (np.ndarray): Expected returns of assets.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        lambda_param (float): Risk aversion parameter.
        max_weight (float): Maximum allowable weight for any single asset.
        
    Returns:
        np.ndarray: Optimized asset weights.
    """
    num_assets = len(expected_returns)
    weights = cp.Variable(num_assets)
    
    portfolio_return = expected_returns @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    # Objective: Maximize utility (return - lambda * variance)
    objective = cp.Maximize(portfolio_return - lambda_param * portfolio_variance)
    
    # Constraints
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if weights.value is not None:
        return weights.value
    else:
        raise ValueError("Optimization failed. Please check input parameters.")