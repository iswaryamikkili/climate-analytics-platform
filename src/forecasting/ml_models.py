"""
Machine Learning Forecasters
Implements XGBoost, LightGBM, and Random Forest for time series forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import timedelta
import warnings

# ML models
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Import base classes
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting.base import BaseForecaster, FeatureEngineering

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLForecasterBase(BaseForecaster):
    """
    Base class for ML-based forecasters
    
    Transforms time series into supervised learning problem
    """
    
    def __init__(self, name: str, model):
        super().__init__(name=name)
        self.model = model
        self.target_col = None
        self.feature_cols = None
        self.last_known_values = None
    
    def fit(self, data: pd.DataFrame, target_col: str = 'temperature') -> 'MLForecasterBase':
        """Fit ML model to time series data"""
        logger.info(f"Fitting {self.name} for {target_col}...")
        
        # Validate data
        self.validate_data(data, target_col)
        
        self.target_col = target_col
        self.train_data = data.copy()
        
        # Create features
        logger.info("Creating features...")
        df_features = FeatureEngineering.create_all_features(data, target_col)
        
        # Separate features and target
        self.feature_cols = [col for col in df_features.columns 
                            if col != target_col and col != 'city_name']
        
        X = df_features[self.feature_cols].values
        y = df_features[target_col].values
        
        logger.info(f"Training with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Store last known values for prediction
        self.last_known_values = df_features.iloc[-1:].copy()
        
        # Train model
        self.model.fit(X, y)
        
        self.is_fitted = True
        logger.info(f"✓ {self.name} fitted successfully")
        
        return self
    
    def predict(self, steps: int, confidence_level: float = 0.95) -> pd.DataFrame:
        """Generate multi-step forecasts"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        logger.info(f"Generating {steps}-step forecast...")
        
        predictions = []
        current_values = self.last_known_values.copy()
        
        # Determine frequency
        freq = pd.infer_freq(self.train_data.index) or 'H'
        last_date = self.train_data.index[-1]
        
        for step in range(steps):
            # Prepare features
            X_pred = current_values[self.feature_cols].values
            
            # Predict
            y_pred = self.model.predict(X_pred)[0]
            predictions.append(y_pred)
            
            # Update features for next step
            current_values = self._update_features(
                current_values, 
                y_pred, 
                last_date + timedelta(hours=step+1)
            )
        
        # Create result dataframe
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=steps,
            freq=freq
        )
        
        # Calculate approximate confidence intervals
        predictions_array = np.array(predictions)
        std_pred = np.std(predictions_array) if len(predictions_array) > 1 else 1.0
        
        z_score = 1.96 if confidence_level == 0.95 else 1.645
        margin = z_score * std_pred
        
        result_df = pd.DataFrame({
            'timestamp': future_dates,
            'forecast': predictions,
            'lower_bound': predictions_array - margin,
            'upper_bound': predictions_array + margin
        })
        
        result_df.set_index('timestamp', inplace=True)
        
        logger.info(f"✓ Generated {steps}-step forecast")
        
        return result_df
    
    def _update_features(self, current_values: pd.DataFrame, 
                        new_value: float, 
                        new_timestamp: pd.Timestamp) -> pd.DataFrame:
        """Update feature values for next prediction step"""
        updated = current_values.copy()
        
        # Update lag features
        lag_cols = [col for col in self.feature_cols if '_lag_' in col]
        for col in sorted(lag_cols, reverse=True):
            lag_num = int(col.split('_lag_')[-1])
            if lag_num > 1:
                prev_lag_col = col.replace(f'_lag_{lag_num}', f'_lag_{lag_num-1}')
                if prev_lag_col in updated.columns:
                    updated[col] = updated[prev_lag_col].values
        
        # Update lag_1 with new prediction
        lag_1_col = f'{self.target_col}_lag_1'
        if lag_1_col in updated.columns:
            updated[lag_1_col] = new_value
        
        # Update temporal features
        updated.index = pd.DatetimeIndex([new_timestamp])
        
        if 'hour' in updated.columns:
            updated['hour'] = new_timestamp.hour
            updated['hour_sin'] = np.sin(2 * np.pi * new_timestamp.hour / 24)
            updated['hour_cos'] = np.cos(2 * np.pi * new_timestamp.hour / 24)
        
        if 'day_of_week' in updated.columns:
            updated['day_of_week'] = new_timestamp.dayofweek
            updated['dow_sin'] = np.sin(2 * np.pi * new_timestamp.dayofweek / 7)
            updated['dow_cos'] = np.cos(2 * np.pi * new_timestamp.dayofweek / 7)
        
        if 'month' in updated.columns:
            updated['month'] = new_timestamp.month
            updated['month_sin'] = np.sin(2 * np.pi * new_timestamp.month / 12)
            updated['month_cos'] = np.cos(2 * np.pi * new_timestamp.month / 12)
        
        return updated
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Get importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            logger.warning("Model does not support feature importance")
            return {}
        
        # Create dictionary
        importance_dict = {
            feature: float(importance) 
            for feature, importance in zip(self.feature_cols, importances)
        }
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return importance_dict

class XGBoostForecaster(MLForecasterBase):
    """
    XGBoost Forecaster
    
    Best for:
    - Complex patterns
    - Non-linear relationships
    - Large datasets
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1):
        """
        Initialize XGBoost forecaster
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
        """
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        
        super().__init__(name="XGBoost", model=model)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
    
    def get_name(self) -> str:
        """Return model identifier"""
        return f"XGBoost(n={self.n_estimators},d={self.max_depth})"

class LightGBMForecaster(MLForecasterBase):
    """
    LightGBM Forecaster
    
    Best for:
    - Faster training than XGBoost
    - Large datasets
    - Production deployments
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = -1,
                 learning_rate: float = 0.1):
        """
        Initialize LightGBM forecaster
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 = no limit)
            learning_rate: Step size
        """
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        super().__init__(name="LightGBM", model=model)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def get_name(self) -> str:
        """Return model identifier"""
        return f"LightGBM(n={self.n_estimators})"

class RandomForestForecaster(MLForecasterBase):
    """
    Random Forest Forecaster
    
    Best for:
    - Robust baseline
    - When you want stability
    - Medium-sized datasets
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None):
        """
        Initialize Random Forest forecaster
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None = unlimited)
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        super().__init__(name="RandomForest", model=model)
        
        self.n_estimators = n_estimators
    
    def get_name(self) -> str:
        """Return model identifier"""
        return f"RandomForest(n={self.n_estimators})"
