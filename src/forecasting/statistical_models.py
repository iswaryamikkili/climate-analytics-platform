"""
Statistical Forecasting Models
Implements Prophet and ARIMA forecasters
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import timedelta
import warnings

# Statistical models
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Import base classes from our base module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting.base import BaseForecaster, ForecastResult

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet Forecaster
    
    Best for:
    - Data with strong seasonal patterns
    - Handling missing data
    - Quick, robust results
    """
    
    def __init__(self, 
                 seasonality_mode: str = 'additive',
                 yearly_seasonality: bool = False,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = True):
        """
        Initialize Prophet forecaster
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Include yearly patterns
            weekly_seasonality: Include weekly patterns
            daily_seasonality: Include daily patterns
        """
        super().__init__(name="Prophet")
        
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
        self.model = None
        self.target_col = None
        self.city = None
    
    def fit(self, data: pd.DataFrame, target_col: str = 'temperature') -> 'ProphetForecaster':
        """Fit Prophet model to data"""
        logger.info(f"Fitting Prophet model for {target_col}...")
        
        # Validate data
        self.validate_data(data, target_col)
        
        self.target_col = target_col
        self.train_data = data.copy()
        
        # Extract city if available
        if 'city_name' in data.columns:
            self.city = data['city_name'].iloc[0]
        
        # Prophet requires specific column names: 'ds' and 'y'
        prophet_df = pd.DataFrame({
            'ds': data.index,
            'y': data[target_col].values
        })
        
        # Initialize Prophet
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=0.95
        )
        
        # Fit model
        self.model.fit(prophet_df)
        
        self.is_fitted = True
        logger.info(f"✓ Prophet model fitted on {len(data)} data points")
        
        return self
    
    def predict(self, steps: int, confidence_level: float = 0.95) -> pd.DataFrame:
        """Generate forecasts with Prophet"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        logger.info(f"Generating {steps}-step forecast...")
        
        # Determine frequency
        freq = pd.infer_freq(self.train_data.index)
        if freq is None:
            freq = 'H'
            logger.warning("Could not infer frequency, using hourly")
        
        # Create future dataframe
        last_date = self.train_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=steps,
            freq=freq
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Generate forecast
        forecast = self.model.predict(future_df)
        
        # Extract predictions and intervals
        result_df = pd.DataFrame({
            'timestamp': forecast['ds'],
            'forecast': forecast['yhat'],
            'lower_bound': forecast['yhat_lower'],
            'upper_bound': forecast['yhat_upper']
        })
        
        result_df.set_index('timestamp', inplace=True)
        
        logger.info(f"✓ Generated {steps}-step forecast")
        
        return result_df
    
    def get_name(self) -> str:
        """Return model identifier"""
        return f"Prophet({self.seasonality_mode})"

class ARIMAForecaster(BaseForecaster):
    """
    ARIMA Forecaster
    
    Best for:
    - Short-term forecasts
    - Statistical rigor
    - When you need confidence intervals
    """
    
    def __init__(self, 
                 order: Optional[tuple] = None,
                 auto_select: bool = True):
        """
        Initialize ARIMA forecaster
        
        Args:
            order: (p, d, q) tuple. If None, will auto-select
            auto_select: Automatically select best order
        """
        super().__init__(name="ARIMA")
        
        self.order = order
        self.auto_select = auto_select
        
        self.model = None
        self.model_fit = None
        self.target_col = None
    
    def fit(self, data: pd.DataFrame, target_col: str = 'temperature') -> 'ARIMAForecaster':
        """Fit ARIMA model to data"""
        logger.info(f"Fitting ARIMA model for {target_col}...")
        
        # Validate data
        self.validate_data(data, target_col)
        
        self.target_col = target_col
        self.train_data = data.copy()
        
        # Get time series
        ts = data[target_col].dropna()
        
        # Check stationarity
        is_stationary = self._check_stationarity(ts)
        
        # Auto-select order if needed
        if self.auto_select and self.order is None:
            self.order = self._auto_select_order(ts, is_stationary)
            logger.info(f"Auto-selected order: {self.order}")
        elif self.order is None:
            self.order = (1, 1 if not is_stationary else 0, 1)
            logger.info(f"Using default order: {self.order}")
        
        # Fit model
        try:
            self.model = ARIMA(
                ts,
                order=self.order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.model_fit = self.model.fit()
            
            self.is_fitted = True
            logger.info(f"✓ ARIMA{self.order} fitted successfully")
            logger.info(f"  AIC: {self.model_fit.aic:.2f}")
            
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            raise
        
        return self
    
    def predict(self, steps: int, confidence_level: float = 0.95) -> pd.DataFrame:
        """Generate forecasts with ARIMA"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        logger.info(f"Generating {steps}-step forecast...")
        
        # Generate forecast
        forecast_result = self.model_fit.get_forecast(steps=steps, alpha=1-confidence_level)
        
        # Get predictions and confidence intervals
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        # Create timestamps
        last_date = self.train_data.index[-1]
        freq = pd.infer_freq(self.train_data.index) or 'H'
        
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=steps,
            freq=freq
        )
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'timestamp': future_dates,
            'forecast': forecast_mean.values,
            'lower_bound': forecast_ci.iloc[:, 0].values,
            'upper_bound': forecast_ci.iloc[:, 1].values
        })
        
        result_df.set_index('timestamp', inplace=True)
        
        logger.info(f"✓ Generated {steps}-step forecast")
        
        return result_df
    
    def get_name(self) -> str:
        """Return model identifier"""
        return f"ARIMA{self.order}"
    
    def _check_stationarity(self, ts: pd.Series, alpha: float = 0.05) -> bool:
        """Check if time series is stationary using ADF test"""
        try:
            result = adfuller(ts.dropna())
            p_value = result[1]
            
            is_stationary = p_value < alpha
            
            logger.info(f"Stationarity test: p-value={p_value:.4f}, "
                       f"stationary={'Yes' if is_stationary else 'No'}")
            
            return is_stationary
            
        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}")
            return False
    
    def _auto_select_order(self, ts: pd.Series, is_stationary: bool) -> tuple:
        """Automatically select ARIMA order using AIC"""
        logger.info("Auto-selecting ARIMA order...")
        
        # Search space
        p_values = [0, 1, 2]
        d_values = [0] if is_stationary else [1]
        q_values = [0, 1, 2]
        
        best_aic = np.inf
        best_order = None
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    if p == 0 and d == 0 and q == 0:
                        continue
                    
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        model_fit = model.fit()
                        
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                    
                    except:
                        continue
        
        if best_order is None:
            logger.warning("Auto-selection failed, using default (1,1,1)")
            best_order = (1, 1, 1)
        
        logger.info(f"Best order: {best_order} (AIC={best_aic:.2f})")
        
        return best_order
