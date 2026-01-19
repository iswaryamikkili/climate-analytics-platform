"""
Ensemble Forecasting Models
Combines multiple forecasters for improved accuracy and robustness
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import warnings

# ML models for stacking
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Import base classes
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting.base import BaseForecaster, ForecastResult, ModelEvaluator
from forecasting.statistical_models import ProphetForecaster, ARIMAForecaster
from forecasting.ml_models import XGBoostForecaster, LightGBMForecaster, RandomForestForecaster

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedEnsembleForecaster(BaseForecaster):
    """
    Weighted Average Ensemble
    
    Combines predictions from multiple models using weighted averaging.
    Weights can be:
    - Equal: Simple average
    - Performance-based: Based on historical accuracy
    - Custom: User-defined weights
    
    Best for:
    - Reducing variance
    - Robust predictions
    - When models have different strengths
    """
    
    def __init__(self, 
                 forecasters: Optional[List[BaseForecaster]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 weight_method: str = 'equal'):
        """
        Initialize weighted ensemble
        
        Args:
            forecasters: List of forecaster instances
            weights: Custom weights dict {model_name: weight}
            weight_method: 'equal', 'performance', or 'custom'
        """
        super().__init__(name="WeightedEnsemble")
        
        self.forecasters = forecasters or []
        self.weights = weights or {}
        self.weight_method = weight_method
        
        self.target_col = None
        self.model_performances = {}
    
    def add_forecaster(self, forecaster: BaseForecaster) -> 'WeightedEnsembleForecaster':
        """Add a forecaster to the ensemble"""
        self.forecasters.append(forecaster)
        logger.info(f"Added {forecaster.get_name()} to ensemble")
        return self
    
    def fit(self, data: pd.DataFrame, target_col: str = 'temperature') -> 'WeightedEnsembleForecaster':
        """Fit all forecasters in the ensemble"""
        logger.info(f"Fitting ensemble with {len(self.forecasters)} models...")
        
        # Validate data
        self.validate_data(data, target_col)
        
        self.target_col = target_col
        self.train_data = data.copy()
        
        # Fit each forecaster
        for i, forecaster in enumerate(self.forecasters):
            try:
                logger.info(f"[{i+1}/{len(self.forecasters)}] Fitting {forecaster.get_name()}...")
                forecaster.fit(data, target_col)
            except Exception as e:
                logger.error(f"Failed to fit {forecaster.get_name()}: {e}")
                continue
        
        # Calculate weights based on method
        if self.weight_method == 'performance':
            self._calculate_performance_weights(data, target_col)
        elif self.weight_method == 'equal':
            self._set_equal_weights()
        
        self.is_fitted = True
        logger.info(f"✓ Ensemble fitted with weights: {self.weights}")
        
        return self
    
    def predict(self, steps: int, confidence_level: float = 0.95) -> pd.DataFrame:
        """Generate ensemble forecast"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")
        
        logger.info(f"Generating {steps}-step ensemble forecast...")
        
        all_predictions = []
        all_lower = []
        all_upper = []
        model_names = []
        
        # Get predictions from each model
        for forecaster in self.forecasters:
            try:
                forecast_df = forecaster.predict(steps, confidence_level)
                
                model_name = forecaster.get_name()
                weight = self.weights.get(model_name, 1.0 / len(self.forecasters))
                
                all_predictions.append(forecast_df['forecast'].values * weight)
                all_lower.append(forecast_df['lower_bound'].values * weight)
                all_upper.append(forecast_df['upper_bound'].values * weight)
                
                model_names.append(model_name)
                
                logger.info(f"  {model_name}: weight={weight:.3f}")
                
            except Exception as e:
                logger.warning(f"Prediction failed for {forecaster.get_name()}: {e}")
                continue
        
        if len(all_predictions) == 0:
            raise RuntimeError("All forecasters failed to generate predictions")
        
        # Combine predictions
        ensemble_forecast = np.sum(all_predictions, axis=0)
        ensemble_lower = np.sum(all_lower, axis=0)
        ensemble_upper = np.sum(all_upper, axis=0)
        
        # Create result dataframe
        last_date = self.train_data.index[-1]
        freq = pd.infer_freq(self.train_data.index) or 'H'
        
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=steps,
            freq=freq
        )
        
        result_df = pd.DataFrame({
            'timestamp': future_dates,
            'forecast': ensemble_forecast,
            'lower_bound': ensemble_lower,
            'upper_bound': ensemble_upper
        })
        
        result_df.set_index('timestamp', inplace=True)
        
        logger.info(f"✓ Ensemble forecast complete (used {len(all_predictions)} models)")
        
        return result_df
    
    def get_name(self) -> str:
        """Return ensemble identifier"""
        return f"WeightedEnsemble({len(self.forecasters)}models)"
    
    def _set_equal_weights(self):
        """Set equal weights for all models"""
        weight = 1.0 / len(self.forecasters)
        for forecaster in self.forecasters:
            self.weights[forecaster.get_name()] = weight
    
    def _calculate_performance_weights(self, data: pd.DataFrame, target_col: str):
        """Calculate weights based on cross-validation performance"""
        logger.info("Calculating performance-based weights...")
        
        # Use last 20% of data for validation
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        model_scores = {}
        
        for forecaster in self.forecasters:
            try:
                # Fit on training data
                forecaster.fit(train_data, target_col)
                
                # Predict validation period
                steps = len(val_data)
                predictions = forecaster.predict(steps)
                
                # Calculate RMSE
                y_true = val_data[target_col].values
                y_pred = predictions['forecast'].values[:len(y_true)]
                
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                model_scores[forecaster.get_name()] = rmse
                
                logger.info(f"  {forecaster.get_name()}: RMSE={rmse:.3f}")
                
            except Exception as e:
                logger.warning(f"Validation failed for {forecaster.get_name()}: {e}")
                model_scores[forecaster.get_name()] = np.inf
        
        # Convert RMSE to weights (inverse of error)
        # Better models (lower RMSE) get higher weights
        inverse_errors = {name: 1.0 / (score + 1e-6) 
                         for name, score in model_scores.items()}
        
        total_inverse = sum(inverse_errors.values())
        
        self.weights = {name: inv / total_inverse 
                       for name, inv in inverse_errors.items()}
        
        self.model_performances = model_scores


class StackingEnsembleForecaster(BaseForecaster):
    """
    Stacking Ensemble
    
    Uses a meta-learner to combine base model predictions.
    The meta-learner learns optimal combination weights.
    
    Architecture:
    - Base models: Generate predictions
    - Meta-learner: Combines predictions (Ridge, XGBoost, etc.)
    
    Best for:
    - When models have complex interactions
    - Maximum accuracy
    - Large datasets
    """
    
    def __init__(self,
                 base_forecasters: Optional[List[BaseForecaster]] = None,
                 meta_learner: str = 'ridge'):
        """
        Initialize stacking ensemble
        
        Args:
            base_forecasters: List of base forecaster instances
            meta_learner: 'ridge', 'lasso', 'rf', or 'xgboost'
        """
        super().__init__(name="StackingEnsemble")
        
        self.base_forecasters = base_forecasters or []
        self.meta_learner_type = meta_learner
        self.meta_learner = None
        
        self.target_col = None
    
    def add_base_forecaster(self, forecaster: BaseForecaster) -> 'StackingEnsembleForecaster':
        """Add a base forecaster"""
        self.base_forecasters.append(forecaster)
        logger.info(f"Added base forecaster: {forecaster.get_name()}")
        return self
    
    def fit(self, data: pd.DataFrame, target_col: str = 'temperature') -> 'StackingEnsembleForecaster':
        """Fit stacking ensemble"""
        logger.info(f"Fitting stacking ensemble with {len(self.base_forecasters)} base models...")
        
        # Validate data
        self.validate_data(data, target_col)
        
        self.target_col = target_col
        self.train_data = data.copy()
        
        # Split data for meta-learner training
        split_idx = int(len(data) * 0.7)
        base_train = data.iloc[:split_idx]
        meta_train = data.iloc[split_idx:]
        
        # Step 1: Train base forecasters
        logger.info("Step 1: Training base forecasters...")
        for i, forecaster in enumerate(self.base_forecasters):
            try:
                logger.info(f"  [{i+1}/{len(self.base_forecasters)}] Training {forecaster.get_name()}...")
                forecaster.fit(base_train, target_col)
            except Exception as e:
                logger.error(f"Failed to train {forecaster.get_name()}: {e}")
                continue
        
        # Step 2: Generate meta-features
        logger.info("Step 2: Generating meta-features...")
        meta_features = self._generate_meta_features(meta_train)
        
        if meta_features is None or len(meta_features) == 0:
            raise RuntimeError("Failed to generate meta-features")
        
        # Step 3: Train meta-learner
        logger.info(f"Step 3: Training meta-learner ({self.meta_learner_type})...")
        self._train_meta_learner(meta_features, meta_train[target_col].values)
        
        # Step 4: Retrain base models on full data
        logger.info("Step 4: Retraining base models on full data...")
        for forecaster in self.base_forecasters:
            try:
                forecaster.fit(data, target_col)
            except Exception as e:
                logger.warning(f"Retraining failed for {forecaster.get_name()}: {e}")
        
        self.is_fitted = True
        logger.info("✓ Stacking ensemble fitted successfully")
        
        return self
    
    def predict(self, steps: int, confidence_level: float = 0.95) -> pd.DataFrame:
        """Generate stacking ensemble forecast"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")
        
        logger.info(f"Generating {steps}-step stacking forecast...")
        
        # Get base predictions
        base_predictions = []
        
        for forecaster in self.base_forecasters:
            try:
                forecast_df = forecaster.predict(steps, confidence_level)
                base_predictions.append(forecast_df['forecast'].values)
            except Exception as e:
                logger.warning(f"Prediction failed for {forecaster.get_name()}: {e}")
                continue
        
        if len(base_predictions) == 0:
            raise RuntimeError("All base forecasters failed")
        
        # Stack predictions as features
        X_meta = np.column_stack(base_predictions)
        
        # Meta-learner prediction
        ensemble_forecast = self.meta_learner.predict(X_meta)
        
        # Estimate confidence intervals (use spread of base predictions)
        base_std = np.std(base_predictions, axis=0)
        z_score = 1.96 if confidence_level == 0.95 else 1.645
        margin = z_score * base_std
        
        # Create result
        last_date = self.train_data.index[-1]
        freq = pd.infer_freq(self.train_data.index) or 'H'
        
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=steps,
            freq=freq
        )
        
        result_df = pd.DataFrame({
            'timestamp': future_dates,
            'forecast': ensemble_forecast,
            'lower_bound': ensemble_forecast - margin,
            'upper_bound': ensemble_forecast + margin
        })
        
        result_df.set_index('timestamp', inplace=True)
        
        logger.info(f"✓ Stacking forecast complete")
        
        return result_df
    
    def get_name(self) -> str:
        """Return ensemble identifier"""
        return f"StackingEnsemble({self.meta_learner_type},{len(self.base_forecasters)}base)"
    
    def _generate_meta_features(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions from base models as meta-features"""
        meta_features = []
        
        steps = len(data)
        
        for forecaster in self.base_forecasters:
            try:
                predictions = forecaster.predict(steps)
                meta_features.append(predictions['forecast'].values[:steps])
            except Exception as e:
                logger.warning(f"Meta-feature generation failed for {forecaster.get_name()}: {e}")
                continue
        
        if len(meta_features) == 0:
            return None
        
        return np.column_stack(meta_features)
    
    def _train_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Train the meta-learner"""
        if self.meta_learner_type == 'ridge':
            self.meta_learner = Ridge(alpha=1.0)
        elif self.meta_learner_type == 'lasso':
            self.meta_learner = Lasso(alpha=0.1)
        elif self.meta_learner_type == 'rf':
            self.meta_learner = RandomForestRegressor(n_estimators=50, random_state=42)
        elif self.meta_learner_type == 'xgboost':
            self.meta_learner = xgb.XGBRegressor(n_estimators=50, random_state=42)
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner_type}")
        
        self.meta_learner.fit(X, y)
        
        # Log meta-learner weights if available
        if hasattr(self.meta_learner, 'coef_'):
            logger.info(f"Meta-learner weights: {self.meta_learner.coef_}")


class DynamicEnsembleForecaster(BaseForecaster):
    """
    Dynamic Weighted Ensemble
    
    Adjusts weights dynamically based on recent performance.
    Uses sliding window to evaluate model performance and
    updates weights accordingly.
    
    Best for:
    - Non-stationary data
    - Adapting to regime changes
    - Long-term forecasting
    """
    
    def __init__(self,
                 forecasters: Optional[List[BaseForecaster]] = None,
                 window_size: int = 24,
                 update_frequency: int = 6):
        """
        Initialize dynamic ensemble
        
        Args:
            forecasters: List of forecaster instances
            window_size: Hours to consider for performance evaluation
            update_frequency: How often to update weights (hours)
        """
        super().__init__(name="DynamicEnsemble")
        
        self.forecasters = forecasters or []
        self.window_size = window_size
        self.update_frequency = update_frequency
        
        self.weights = {}
        self.target_col = None
        self.performance_history = {f.get_name(): [] for f in forecasters}
    
    def add_forecaster(self, forecaster: BaseForecaster) -> 'DynamicEnsembleForecaster':
        """Add a forecaster"""
        self.forecasters.append(forecaster)
        self.performance_history[forecaster.get_name()] = []
        logger.info(f"Added {forecaster.get_name()} to dynamic ensemble")
        return self
    
    def fit(self, data: pd.DataFrame, target_col: str = 'temperature') -> 'DynamicEnsembleForecaster':
        """Fit all forecasters"""
        logger.info(f"Fitting dynamic ensemble with {len(self.forecasters)} models...")
        
        self.validate_data(data, target_col)
        
        self.target_col = target_col
        self.train_data = data.copy()
        
        # Fit each forecaster
        for i, forecaster in enumerate(self.forecasters):
            try:
                logger.info(f"[{i+1}/{len(self.forecasters)}] Fitting {forecaster.get_name()}...")
                forecaster.fit(data, target_col)
            except Exception as e:
                logger.error(f"Failed to fit {forecaster.get_name()}: {e}")
                continue
        
        # Initialize weights with rolling validation
        self._initialize_dynamic_weights(data, target_col)
        
        self.is_fitted = True
        logger.info(f"✓ Dynamic ensemble fitted with initial weights: {self.weights}")
        
        return self
    
    def predict(self, steps: int, confidence_level: float = 0.95) -> pd.DataFrame:
        """Generate dynamic ensemble forecast"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")
        
        logger.info(f"Generating {steps}-step dynamic forecast...")
        
        # Get current weights
        current_weights = self._get_current_weights()
        
        all_predictions = []
        all_lower = []
        all_upper = []
        
        # Get predictions from each model
        for forecaster in self.forecasters:
            try:
                forecast_df = forecaster.predict(steps, confidence_level)
                
                model_name = forecaster.get_name()
                weight = current_weights.get(model_name, 0.0)
                
                all_predictions.append(forecast_df['forecast'].values * weight)
                all_lower.append(forecast_df['lower_bound'].values * weight)
                all_upper.append(forecast_df['upper_bound'].values * weight)
                
                logger.info(f"  {model_name}: weight={weight:.3f}")
                
            except Exception as e:
                logger.warning(f"Prediction failed for {forecaster.get_name()}: {e}")
                continue
        
        if len(all_predictions) == 0:
            raise RuntimeError("All forecasters failed")
        
        # Combine predictions
        ensemble_forecast = np.sum(all_predictions, axis=0)
        ensemble_lower = np.sum(all_lower, axis=0)
        ensemble_upper = np.sum(all_upper, axis=0)
        
        # Create result
        last_date = self.train_data.index[-1]
        freq = pd.infer_freq(self.train_data.index) or 'H'
        
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=steps,
            freq=freq
        )
        
        result_df = pd.DataFrame({
            'timestamp': future_dates,
            'forecast': ensemble_forecast,
            'lower_bound': ensemble_lower,
            'upper_bound': ensemble_upper
        })
        
        result_df.set_index('timestamp', inplace=True)
        
        logger.info(f"✓ Dynamic forecast complete")
        
        return result_df
    
    def get_name(self) -> str:
        """Return ensemble identifier"""
        return f"DynamicEnsemble({len(self.forecasters)}models,w={self.window_size})"
    
    def _initialize_dynamic_weights(self, data: pd.DataFrame, target_col: str):
        """Initialize weights using recent performance"""
        logger.info("Initializing dynamic weights...")
        
        # Use last window_size hours for evaluation
        eval_data = data.iloc[-self.window_size:]
        
        if len(eval_data) < 10:
            # Not enough data, use equal weights
            equal_weight = 1.0 / len(self.forecasters)
            for forecaster in self.forecasters:
                self.weights[forecaster.get_name()] = equal_weight
            return
        
        # Split into train/val
        split_idx = int(len(eval_data) * 0.7)
        train_data = eval_data.iloc[:split_idx]
        val_data = eval_data.iloc[split_idx:]
        
        model_errors = {}
        
        for forecaster in self.forecasters:
            try:
                forecaster.fit(train_data, target_col)
                predictions = forecaster.predict(len(val_data))
                
                y_true = val_data[target_col].values
                y_pred = predictions['forecast'].values[:len(y_true)]
                
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                model_errors[forecaster.get_name()] = rmse
                
            except Exception as e:
                logger.warning(f"Weight initialization failed for {forecaster.get_name()}: {e}")
                model_errors[forecaster.get_name()] = np.inf
        
        # Convert to weights
        inverse_errors = {name: 1.0 / (error + 1e-6) 
                         for name, error in model_errors.items()}
        total_inverse = sum(inverse_errors.values())
        
        self.weights = {name: inv / total_inverse 
                       for name, inv in inverse_errors.items()}
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Get current weights (can be updated based on performance history)"""
        # For now, return stored weights
        # In production, this could analyze performance_history to adjust
        return self.weights


def create_default_ensemble(ensemble_type: str = 'weighted') -> BaseForecaster:
    """
    Factory function to create ensemble with all 5 models
    
    Args:
        ensemble_type: 'weighted', 'stacking', or 'dynamic'
    
    Returns:
        Configured ensemble forecaster
    """
    logger.info(f"Creating {ensemble_type} ensemble with all models...")
    
    # Initialize all base models
    prophet = ProphetForecaster(
        seasonality_mode='additive',
        daily_seasonality=True,
        weekly_seasonality=True
    )
    
    arima = ARIMAForecaster(auto_select=True)
    
    xgboost = XGBoostForecaster(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    lightgbm = LightGBMForecaster(
        n_estimators=100,
        learning_rate=0.1
    )
    
    rf = RandomForestForecaster(
        n_estimators=100,
        max_depth=10
    )
    
    # Create ensemble based on type
    if ensemble_type == 'weighted':
        ensemble = WeightedEnsembleForecaster(weight_method='performance')
        ensemble.add_forecaster(prophet)
        ensemble.add_forecaster(arima)
        ensemble.add_forecaster(xgboost)
        ensemble.add_forecaster(lightgbm)
        ensemble.add_forecaster(rf)
        
    elif ensemble_type == 'stacking':
        ensemble = StackingEnsembleForecaster(meta_learner='ridge')
        ensemble.add_base_forecaster(prophet)
        ensemble.add_base_forecaster(arima)
        ensemble.add_base_forecaster(xgboost)
        ensemble.add_base_forecaster(lightgbm)
        ensemble.add_base_forecaster(rf)
        
    elif ensemble_type == 'dynamic':
        ensemble = DynamicEnsembleForecaster(window_size=24, update_frequency=6)
        ensemble.add_forecaster(prophet)
        ensemble.add_forecaster(arima)
        ensemble.add_forecaster(xgboost)
        ensemble.add_forecaster(lightgbm)
        ensemble.add_forecaster(rf)
        
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    logger.info(f"✓ Created {ensemble.get_name()}")
    
    return ensemble
