import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

class PredictiveModel:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        
    def train(self, X, y, asset_name):
        """
        Trains the predictive model for a specific asset using Time Series Cross-Validation.
        
        Args:
            X (np.ndarray): Feature set
            y (np.ndarray): Target variable
            asset_name (str): Name of the asset being trained
        """
        # Initialize metrics lists
        maes, rmses, r2s = [], [], []
        predictions_all = []
        y_test_all = []
        
        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Define parameter distribution for RandomizedSearch
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }
        
        # Initialize RandomForestRegressor
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=10,
            cv=tscv,
            scoring='neg_mean_squared_error',
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit RandomizedSearchCV
        random_search.fit(X, y)
        
        best_model = random_search.best_estimator_
        
        # Iterate over TimeSeriesSplit
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train the best model
            best_model.fit(X_train, y_train)
            
            # Make predictions
            predictions = best_model.predict(X_test)
            
            # Store predictions and actual values
            predictions_all.extend(predictions)
            y_test_all.extend(y_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)
        
        # Store the final trained model
        self.models[asset_name] = best_model
        
        # Store the average metrics
        self.metrics[asset_name] = {
            'MAE': np.mean(maes),
            'RMSE': np.mean(rmses),
            'R2': np.mean(r2s)
        }
        
        # Print metrics
        print(f"\nAsset: {asset_name}")
        print(f"Best Parameters: {random_search.best_params_}")
        print(f"Average MAE: {self.metrics[asset_name]['MAE']:.6f}")
        print(f"Average RMSE: {self.metrics[asset_name]['RMSE']:.6f}")
        print(f"Average R^2 Score: {self.metrics[asset_name]['R2']:.6f}")
        
        # Plot Actual vs Predicted and save to file
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_all, label='Actual')
            plt.plot(predictions_all, label='Predicted', alpha=0.7)
            plt.legend()
            plt.title(f'Actual vs Predicted Returns for {asset_name}')
            plt.xlabel('Time')
            plt.ylabel('Returns')
            plt.tight_layout()
            plt.savefig(f'actual_vs_predicted_{asset_name}.png')
        except Exception as e:
            print(f"Error plotting for {asset_name}: {e}")
        finally:
            plt.close()
                
    def predict(self, X, asset_name):
        """
        Makes predictions using the trained model for a specific asset.
        
        Args:
            X (np.ndarray): Feature set
            asset_name (str): Name of the asset
            
        Returns:
            np.ndarray: Predictions
        """
        if asset_name not in self.models:
            raise ValueError(f"Model for {asset_name} has not been trained.")
        return self.models[asset_name].predict(X)