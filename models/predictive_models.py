# models/predictive_models.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import logging

class PredictiveModel:
    def __init__(self, config=None):
        """
        Initializes the PredictiveModel with optional configuration.
        
        Args:
            config (dict, optional): Configuration dictionary for model parameters.
        """
        self.config = config or {}
        self.models = {}
        self.best_params = {}
        self.expected_returns = {}
    
    def train(self, returns, features, assets, logger):
        """
        Trains predictive models for each asset.
        
        Args:
            returns (pd.DataFrame): DataFrame of asset returns.
            features (pd.DataFrame): DataFrame of features.
            assets (list): List of asset tickers.
            logger (logging.Logger): Logger instance for logging.
        """
        for asset in assets:
            logger.info(f"Training model for {asset}.")
            y = returns[asset]
            X = features
            
            # Handle missing or infinite values
            if y.isnull().any() or np.isinf(y).any():
                logger.error(f"Returns for {asset} contain NaN or infinite values. Skipping this asset.")
                continue
            
            if X.isnull().values.any() or np.isinf(X.values).any():
                logger.error(f"Features for {asset} contain NaN or infinite values. Skipping this asset.")
                continue
            
            # Define the model
            model = RandomForestRegressor(random_state=42)
            
            # Define hyperparameters
            param_grid = self.config.get('model_params', {
                'n_estimators': [100, 200],
                'min_samples_split': [10, 5],
                'min_samples_leaf': [4, 2],
                'max_features': ['sqrt'],
                'max_depth': [30, 20]
            })
            
            # TimeSeries Cross-Validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Grid Search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # Save the best model and parameters
            best_model = grid_search.best_estimator_
            self.models[asset] = best_model
            self.best_params[asset] = grid_search.best_params_
            
            # Evaluate the model
            y_pred = best_model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            logger.info(f"Asset: {asset}")
            logger.info(f"Best Parameters: {self.best_params[asset]}")
            logger.info(f"Average MAE: {mae:.6f}")
            logger.info(f"Average RMSE: {rmse:.6f}")
            logger.info(f"Average R^2 Score: {r2:.6f}\n")
            
            # Store expected returns for optimization
            self.expected_returns[asset] = np.mean(y_pred)
    
    def get_expected_returns(self):
        """
        Retrieves the expected returns for each asset.
        
        Returns:
            np.ndarray: Array of expected returns.
        """
        return np.array(list(self.expected_returns.values()))