"""
Weather Forecasting Module - Base Classes and Utilities
Implements foundation for multiple forecasting algorithms
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForecastResult:
    """
    Container for forecast results
    Makes it easy to pass around predictions with metadata
    """
    predictions: pd.DataFrame  # Contains: timestamp, forecast, lower_bound, upper_bound
    model_name: str
    variable: str
    city: str
    train_end_date: datetime
    forecast_horizon: int
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    residuals: Optional[np.ndarray] = None
    
    def __repr__(self) -> str:
        return (f"ForecastResult(model={self.model_name}, "
                f"horizon={self.forecast_horizon}, "
                f"RMSE={self.metrics.get('rmse', 'N/A'):.3f})")

class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models
    
    All forecasters must implement:
    - fit(): Train on historical data
    - predict(): Generate forecasts
    - get_name(): Return model identifier
    """
    
    def __init__(self, name: str = "BaseForecaster"):
        self.name = name
        self.is_fitted = False
        self.train_data = None
        self.feature_names = None
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, target_col: str = 'temperature') -> 'BaseForecaster':
        """Train the forecasting model"""
        pass
    
    @abstractmethod
    def predict(self, steps: int, confidence_level: float = 0.95) -> pd.DataFrame:
        """Generate forecasts"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return model name for identification"""
        pass
    
    def validate_data(self, data: pd.DataFrame, target_col: str):
        """Validate input data quality"""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Check for missing values
        missing_pct = data[target_col].isnull().sum() / len(data) * 100
        if missing_pct > 20:
            logger.warning(f"High percentage of missing values: {missing_pct:.1f}%")
        
        # Check minimum data points
        min_points = 30
        if len(data) < min_points:
            raise ValueError(f"Insufficient data: need at least {min_points} points, got {len(data)}")
        
        logger.info(f"Data validation passed: {len(data)} points, {missing_pct:.1f}% missing")

class ModelEvaluator:
    """
    Comprehensive model evaluation for time series forecasting
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive forecast accuracy metrics"""
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {'error': 'No valid data points'}
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE (handle division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        # SMAPE (Symmetric MAPE)
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        # R²
        r2 = r2_score(y_true, y_pred)
        
        # Bias (mean error)
        bias = np.mean(y_pred - y_true)
        
        return {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2),
            'smape': round(smape, 2),
            'r2': round(r2, 4),
            'bias': round(bias, 4),
            'n_points': len(y_true)
        }
    
    @staticmethod
    def residual_diagnostics(residuals: np.ndarray) -> Dict[str, any]:
        """Diagnostic tests on forecast residuals"""
        # Remove NaN
        residuals = residuals[~np.isnan(residuals)]
        
        if len(residuals) < 3:
            return {'error': 'Insufficient residuals for diagnostics'}
        
        diagnostics = {}
        
        # 1. Normality test (Shapiro-Wilk)
        if len(residuals) < 5000:
            _, p_norm = stats.shapiro(residuals)
            diagnostics['normality'] = {
                'test': 'Shapiro-Wilk',
                'p_value': round(p_norm, 4),
                'is_normal': p_norm > 0.05,
                'interpretation': 'Residuals are normally distributed' if p_norm > 0.05 
                                 else 'Residuals deviate from normality'
            }
        
        # 2. Zero mean test
        mean_resid = np.mean(residuals)
        std_resid = np.std(residuals)
        diagnostics['zero_mean'] = {
            'mean': round(mean_resid, 4),
            'std': round(std_resid, 4),
            'is_zero_mean': abs(mean_resid) < 0.1 * std_resid,
            'interpretation': 'Mean is approximately zero' if abs(mean_resid) < 0.1 * std_resid
                             else 'Residuals show bias (non-zero mean)'
        }
        
        # 3. Autocorrelation (simple check - lag 1)
        if len(residuals) > 1:
            acf_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            diagnostics['autocorrelation'] = {
                'lag_1_acf': round(acf_lag1, 4),
                'is_white_noise': abs(acf_lag1) < 0.2,
                'interpretation': 'No significant autocorrelation' if abs(acf_lag1) < 0.2
                                 else 'Residuals show autocorrelation'
            }
        
        return diagnostics

class FeatureEngineering:
    """
    Feature engineering utilities for ML-based forecasting
    """
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, 
                           target_col: str,
                           lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lag features (past values as predictors)"""
        df_features = df.copy()
        
        for lag in lags:
            df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
        
        return df_features
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame,
                               target_col: str,
                               windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """Create rolling window statistics"""
        df_features = df.copy()
        
        for window in windows:
            df_features[f'{target_col}_rolling_mean_{window}'] = (
                df_features[target_col].rolling(window, min_periods=1).mean()
            )
            
            df_features[f'{target_col}_rolling_std_{window}'] = (
                df_features[target_col].rolling(window, min_periods=1).std()
            )
        
        return df_features
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features with cyclical encoding"""
        df_features = df.copy()
        
        # Extract components
        df_features['hour'] = df_features.index.hour
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['month'] = df_features.index.month
        
        # Cyclical encoding - IMPORTANT!
        # Hour (0-23)
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        
        # Day of week (0-6)
        df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # Month (1-12)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        # Binary features
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        df_features['is_daytime'] = df_features['hour'].between(6, 18).astype(int)
        
        return df_features
    
    @staticmethod
    def create_all_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create comprehensive feature set for ML forecasting"""
        df_features = df.copy()
        
        # Lag features
        df_features = FeatureEngineering.create_lag_features(df_features, target_col)
        
        # Rolling features
        df_features = FeatureEngineering.create_rolling_features(df_features, target_col)
        
        # Temporal features
        df_features = FeatureEngineering.create_temporal_features(df_features)
        
        # Drop rows with NaN (from lag/rolling)
        df_features = df_features.dropna()
        
        logger.info(f"Created {len(df_features.columns)} features")
        
        return df_features
