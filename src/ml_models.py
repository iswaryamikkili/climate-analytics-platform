"""
Machine Learning & Forecasting Module
Implements various ML models for weather prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherPredictor:
    """Machine Learning models for weather prediction"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize predictor with weather data
        
        Args:
            df: DataFrame with weather data
        """
        self.df = df.copy()
        self.models = {}
        self.results = {}
        self._prepare_data()
        logger.info(f"Initialized predictor with {len(self.df)} records")
    
    def _prepare_data(self):
        """Prepare data for ML models"""
        # Convert timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp')
        
        # Add time-based features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['day_of_year'] = self.df['timestamp'].dt.dayofyear
        
        # Add lag features
        for city in self.df['city_name'].unique():
            city_mask = self.df['city_name'] == city
            self.df.loc[city_mask, 'temp_lag_1'] = self.df.loc[city_mask, 'temperature'].shift(1)
            self.df.loc[city_mask, 'temp_lag_2'] = self.df.loc[city_mask, 'temperature'].shift(2)
        
        # Drop rows with NaN (from lag features)
        self.df = self.df.dropna()
    
    # ========== REGRESSION MODELS ==========
    
    def train_linear_regression(self, city: str, target: str = 'temperature'):
        """
        Train Linear Regression model
        
        Args:
            city: City name
            target: Target variable to predict
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training Linear Regression for {city}...")
        
        # Filter data for city
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 10:
            return {'error': 'Insufficient data for training'}
        
        # Features
        features = ['humidity', 'wind_speed', 'pressure', 'cloudiness', 
                   'hour', 'day_of_week', 'month', 'temp_lag_1', 'temp_lag_2']
        
        X = city_data[features]
        y = city_data[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        results = {
            'model_name': 'Linear Regression',
            'city': city,
            'target': target,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'model': model,
            'scaler': scaler,
            'features': features,
            'feature_importance': dict(zip(features, model.coef_)),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred_test
        }
        
        self.models[f'lr_{city}'] = results
        logger.info(f"Linear Regression - Test R²: {results['test_r2']:.3f}, RMSE: {results['test_rmse']:.3f}")
        
        return results
    
    def train_random_forest(self, city: str, target: str = 'temperature', 
                           n_estimators: int = 100):
        """
        Train Random Forest model
        
        Args:
            city: City name
            target: Target variable
            n_estimators: Number of trees
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training Random Forest for {city}...")
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 10:
            return {'error': 'Insufficient data for training'}
        
        # Features
        features = ['humidity', 'wind_speed', 'pressure', 'cloudiness', 
                   'hour', 'day_of_week', 'month', 'temp_lag_1', 'temp_lag_2']
        
        X = city_data[features]
        y = city_data[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                    scoring='r2', n_jobs=-1)
        
        # Metrics
        results = {
            'model_name': 'Random Forest',
            'city': city,
            'target': target,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'model': model,
            'features': features,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred_test
        }
        
        self.models[f'rf_{city}'] = results
        logger.info(f"Random Forest - Test R²: {results['test_r2']:.3f}, RMSE: {results['test_rmse']:.3f}")
        
        return results
    
    def train_gradient_boosting(self, city: str, target: str = 'temperature'):
        """
        Train Gradient Boosting model
        
        Args:
            city: City name
            target: Target variable
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training Gradient Boosting for {city}...")
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 10:
            return {'error': 'Insufficient data for training'}
        
        features = ['humidity', 'wind_speed', 'pressure', 'cloudiness', 
                   'hour', 'day_of_week', 'month', 'temp_lag_1', 'temp_lag_2']
        
        X = city_data[features]
        y = city_data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        results = {
            'model_name': 'Gradient Boosting',
            'city': city,
            'target': target,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'model': model,
            'features': features,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred_test
        }
        
        self.models[f'gb_{city}'] = results
        logger.info(f"Gradient Boosting - Test R²: {results['test_r2']:.3f}, RMSE: {results['test_rmse']:.3f}")
        
        return results
    
    # ========== TIME SERIES FORECASTING ==========
    
    def forecast_arima(self, city: str, variable: str = 'temperature', 
                      steps: int = 24, order=(1,1,1)):
        """
        ARIMA time series forecasting
        
        Args:
            city: City name
            variable: Variable to forecast
            steps: Number of steps to forecast
            order: ARIMA order (p,d,q)
            
        Returns:
            Dictionary with forecast and metrics
        """
        logger.info(f"Forecasting {variable} for {city} using ARIMA...")
        
        city_data = self.df[self.df['city_name'] == city].copy()
        city_data = city_data.sort_values('timestamp')
        
        if len(city_data) < 20:
            return {'error': 'Insufficient data for time series forecasting (need at least 20 points)'}
        
        # Prepare time series
        ts_data = city_data.set_index('timestamp')[variable]
        
        # Train-test split (80-20)
        train_size = int(len(ts_data) * 0.8)
        train = ts_data[:train_size]
        test = ts_data[train_size:]
        
        try:
            # Fit ARIMA model
            model = ARIMA(train, order=order)
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(test))
            
            # Calculate metrics on test set
            test_rmse = np.sqrt(mean_squared_error(test, forecast[:len(test)]))
            test_mae = mean_absolute_error(test, forecast[:len(test)])
            
            # Forecast future
            future_forecast = fitted_model.forecast(steps=steps)
            
            # Create future timestamps
            last_timestamp = city_data['timestamp'].max()
            future_timestamps = pd.date_range(
                start=last_timestamp, 
                periods=steps + 1, 
                freq='6H'
            )[1:]  # Skip first as it's the last known point
            
            results = {
                'model_name': 'ARIMA',
                'city': city,
                'variable': variable,
                'order': order,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'forecast': future_forecast.tolist(),
                'forecast_timestamps': future_timestamps.tolist(),
                'model_summary': str(fitted_model.summary()),
                'train_data': train,
                'test_data': test,
                'test_forecast': forecast
            }
            
            logger.info(f"ARIMA Forecast - Test RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}")
            return {'error': str(e)}
    
    # ========== MODEL COMPARISON ==========
    
    def compare_models(self, city: str):
        """
        Train and compare all models for a city
        
        Args:
            city: City name
            
        Returns:
            DataFrame with model comparison
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Comparing all models for {city}")
        logger.info(f"{'='*60}\n")
        
        # Train all models
        lr_results = self.train_linear_regression(city)
        rf_results = self.train_random_forest(city)
        gb_results = self.train_gradient_boosting(city)
        
        # Check if any model failed
        if 'error' in lr_results or 'error' in rf_results or 'error' in gb_results:
            logger.error("⚠️ One or more models failed to train due to insufficient data")
            return None
        # Create comparison DataFrame
        comparison = pd.DataFrame([
            {
                'Model': lr_results['model_name'],
                'Train R²': lr_results['train_r2'],
                'Test R²': lr_results['test_r2'],
                'Train RMSE': lr_results['train_rmse'],
                'Test RMSE': lr_results['test_rmse'],
                'Test MAE': lr_results['test_mae']
            },
            {
                'Model': rf_results['model_name'],
                'Train R²': rf_results['train_r2'],
                'Test R²': rf_results['test_r2'],
                'Train RMSE': rf_results['train_rmse'],
                'Test RMSE': rf_results['test_rmse'],
                'Test MAE': rf_results['test_mae']
            },
            {
                'Model': gb_results['model_name'],
                'Train R²': gb_results['train_r2'],
                'Test R²': gb_results['test_r2'],
                'Train RMSE': gb_results['train_rmse'],
                'Test RMSE': gb_results['test_rmse'],
                'Test MAE': gb_results['test_mae']
            }
        ])
        
        # Round for display
        comparison = comparison.round(4)
        
        # Sort by test R²
        comparison = comparison.sort_values('Test R²', ascending=False)
        
        return comparison
    
    def get_feature_importance(self, city: str, model_type: str = 'rf'):
        """
        Get feature importance from trained model
        
        Args:
            city: City name
            model_type: 'rf' for Random Forest or 'gb' for Gradient Boosting
            
        Returns:
            DataFrame with feature importance
        """
        model_key = f'{model_type}_{city}'
        
        if model_key not in self.models:
            return None
        
        importance = self.models[model_key]['feature_importance']
        
        # Convert to DataFrame and sort
        df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        
        df = df.sort_values('Importance', ascending=False)
        df['Importance'] = df['Importance'].round(4)
        
        return df


def main():
    """Test ML models"""
    
    from database import WeatherDatabase
    
    # Load data
    print("\n🔄 Loading data from database...")
    db = WeatherDatabase()
    conn = db.get_connection()
    df = pd.read_sql_query("SELECT * FROM weather_data", conn)
    db.close()
    
    
    
    # Initialize predictor
    print("\n🤖 Initializing ML predictor...")
    predictor = WeatherPredictor(df)
    # Get first city
    cities = predictor.df['city_name'].unique()
    city = cities[0]
    
    print(f"\n{'='*60}")
    print(f"🤖 MACHINE LEARNING ANALYSIS FOR {city.upper()}")
    print(f"{'='*60}\n")
    
    # Compare all models
    print("📊 Model Comparison:")
    comparison = predictor.compare_models(city)
    if comparison is None:
        print("\n❌ Model training failed due to insufficient data")
        print(f"💡 Current records for {city}: {len(predictor.df[predictor.df['city_name'] == city])}")
        print("   Collect more data points for better model training")
        return
    
    print("\n📊 Model Comparison:")
    print(comparison.to_string(index=False))
    
    # Feature importance
    print(f"\n📈 Feature Importance (Random Forest):")
    importance = predictor.get_feature_importance(city, 'rf')
    if importance is not None:
        print(importance.to_string(index=False))
    
    # Time series forecast
    print(f"\n🔮 Time Series Forecast (ARIMA):")
    forecast_results = predictor.forecast_arima(city, steps=12)
    
    if 'error' not in forecast_results:
        print(f"   Test RMSE: {forecast_results['test_rmse']:.3f}°C")
        print(f"   Test MAE: {forecast_results['test_mae']:.3f}°C")
        print(f"   Next 5 forecasted values: {[f'{x:.1f}°C' for x in forecast_results['forecast'][:5]]}")
    else:
        print(f"   ⚠️  {forecast_results['error']}")
        print(f"   💡 Time series forecasting needs at least 20 data points")
    
    print(f"\n{'='*60}")
    print("✅ ML Analysis Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
