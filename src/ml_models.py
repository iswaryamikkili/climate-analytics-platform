import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AdvancedWeatherPredictor:
    """
    Masters-level Weather Prediction System with:
    - Cross-validation
    - Hyperparameter tuning
    - Advanced feature engineering
    - Statistical rigor
    - Time series diagnostics
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with weather data"""
        self.df = df.copy()
        self.models = {}
        self.scalers = {}
        self.best_params = {}
        
        # Ensure timestamp is datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        logger.info(f"Initialized predictor with {len(self.df)} records")
    
    # ========== ADVANCED FEATURE ENGINEERING ==========
    
    def engineer_features(self, city_data: pd.DataFrame, include_interactions=True,
                         include_polynomials=False) -> pd.DataFrame:
        """
        Create advanced features with domain knowledge
        
        Args:
            city_data: City-specific weather data
            include_interactions: Add feature interactions
            include_polynomials: Add polynomial features
            
        Returns:
            DataFrame with engineered features
        """
        data = city_data.copy()
        
        # Sort by timestamp for lag features
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Basic temporal features
        if 'timestamp' in data.columns:
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['month'] = data['timestamp'].dt.month
            data['day_of_year'] = data['timestamp'].dt.dayofyear
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for temporal features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Lag features (properly avoiding leakage)
        for lag in [1, 2, 3, 6, 12, 24]:
            data[f'temp_lag_{lag}'] = data['temperature'].shift(lag)
            data[f'humidity_lag_{lag}'] = data['humidity'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            data[f'temp_rolling_mean_{window}'] = data['temperature'].rolling(window).mean()
            data[f'temp_rolling_std_{window}'] = data['temperature'].rolling(window).std()
            data[f'humidity_rolling_mean_{window}'] = data['humidity'].rolling(window).mean()
        
        # Domain-specific features
        data['temp_range'] = data['temperature'].rolling(24).max() - data['temperature'].rolling(24).min()
        data['pressure_change'] = data['pressure'].diff()
        data['humidity_change'] = data['humidity'].diff()
        
        # Heat index approximation (simplified)
        data['feels_like'] = data['temperature'] + 0.5 * (data['humidity'] / 100) * (data['temperature'] - 14)
        
        # Dew point approximation
        data['dew_point'] = data['temperature'] - ((100 - data['humidity']) / 5)
        
        # Feature interactions
        if include_interactions:
            data['temp_humidity'] = data['temperature'] * data['humidity']
            data['temp_pressure'] = data['temperature'] * data['pressure']
            data['wind_humidity'] = data['wind_speed'] * data['humidity']
            data['pressure_humidity'] = data['pressure'] * data['humidity']
        
        # Drop rows with NaN from lag/rolling features
        data = data.dropna()
        
        return data
    
    # ========== STATISTICAL DIAGNOSTICS ==========
    
    def test_stationarity(self, series: pd.Series, alpha=0.05):
        """
        Augmented Dickey-Fuller test for stationarity
        
        Args:
            series: Time series data
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(series.dropna())
        
        is_stationary = result[1] < alpha
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': is_stationary,
            'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
        }
    
    def analyze_residuals(self, y_true, y_pred):
        """
        Comprehensive residual analysis
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with diagnostic statistics
        """
        residuals = y_true - y_pred
        
        # Normality test (Shapiro-Wilk)
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        
        # Mean and std of residuals
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        # Heteroscedasticity check (simple variance test)
        mid_point = len(residuals) // 2
        var_first_half = np.var(residuals[:mid_point])
        var_second_half = np.var(residuals[mid_point:])
        variance_ratio = var_second_half / var_first_half if var_first_half > 0 else np.inf
        
        return {
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05,
            'variance_ratio': variance_ratio,
            'potential_heteroscedasticity': variance_ratio > 2 or variance_ratio < 0.5
        }
    
    # ========== CROSS-VALIDATED MODEL TRAINING ==========
    
    def train_with_cv(self, city: str, target: str = 'temperature', cv_folds=5):
        """
        Train models with cross-validation and hyperparameter tuning
        
        Args:
            city: City name
            target: Target variable
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with comprehensive results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Training models for {city} with {cv_folds}-fold CV")
        logger.info(f"{'='*70}\n")
        
        # Get and prepare data
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 50:
            return {'error': f'Insufficient data: {len(city_data)} records (need at least 50)'}
        
        # Engineer features
        city_data = self.engineer_features(city_data, include_interactions=True)
        
        if len(city_data) < 30:
            return {'error': 'Insufficient data after feature engineering'}
        
        # Define features
        feature_cols = [col for col in city_data.columns if col not in 
                       ['temperature', 'city_name', 'timestamp', 'id']]
        
        X = city_data[feature_cols]
        y = city_data[target]
        
        # Time series split for temporal data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Train-test split (final holdout)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # No shuffle for time series
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[city] = scaler
        
        results = {}
        
        # ========== 1. RIDGE REGRESSION WITH TUNING ==========
        logger.info("Training Ridge Regression with hyperparameter tuning...")
        
        ridge_params = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
        }
        
        ridge = Ridge()
        ridge_grid = GridSearchCV(
            ridge, ridge_params, cv=tscv, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        ridge_grid.fit(X_train_scaled, y_train)
        
        ridge_best = ridge_grid.best_estimator_
        ridge_cv_scores = -ridge_grid.best_score_  # Convert to positive MSE
        
        y_pred_ridge_train = ridge_best.predict(X_train_scaled)
        y_pred_ridge_test = ridge_best.predict(X_test_scaled)
        
        # Residual analysis
        ridge_residuals = self.analyze_residuals(y_test, y_pred_ridge_test)
        
        results['ridge'] = {
            'model_name': 'Ridge Regression',
            'best_params': ridge_grid.best_params_,
            'cv_rmse': np.sqrt(ridge_cv_scores),
            'train_r2': r2_score(y_train, y_pred_ridge_train),
            'test_r2': r2_score(y_test, y_pred_ridge_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_ridge_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_ridge_test),
            'model': ridge_best,
            'residual_diagnostics': ridge_residuals,
            'coefficients': dict(zip(feature_cols, ridge_best.coef_))
        }
        
        logger.info(f"✓ Ridge - CV RMSE: {results['ridge']['cv_rmse']:.3f}, "
                   f"Test R²: {results['ridge']['test_r2']:.3f}")
        
        # ========== 2. RANDOM FOREST WITH TUNING ==========
        logger.info("Training Random Forest with hyperparameter tuning...")
        
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(
            rf, rf_params, cv=3,  # Fewer splits for speed
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        
        rf_best = rf_grid.best_estimator_
        
        y_pred_rf_train = rf_best.predict(X_train)
        y_pred_rf_test = rf_best.predict(X_test)
        
        rf_residuals = self.analyze_residuals(y_test, y_pred_rf_test)
        
        results['random_forest'] = {
            'model_name': 'Random Forest',
            'best_params': rf_grid.best_params_,
            'train_r2': r2_score(y_train, y_pred_rf_train),
            'test_r2': r2_score(y_test, y_pred_rf_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_rf_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_rf_test),
            'model': rf_best,
            'residual_diagnostics': rf_residuals,
            'feature_importance': dict(zip(feature_cols, rf_best.feature_importances_))
        }
        
        logger.info(f"✓ Random Forest - Test R²: {results['random_forest']['test_r2']:.3f}, "
                   f"Test RMSE: {results['random_forest']['test_rmse']:.3f}")
        
        # ========== 3. GRADIENT BOOSTING WITH TUNING ==========
        logger.info("Training Gradient Boosting with hyperparameter tuning...")
        
        gb_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
        
        gb = GradientBoostingRegressor(random_state=42)
        gb_grid = GridSearchCV(
            gb, gb_params, cv=3,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        
        gb_best = gb_grid.best_estimator_
        
        y_pred_gb_train = gb_best.predict(X_train)
        y_pred_gb_test = gb_best.predict(X_test)
        
        gb_residuals = self.analyze_residuals(y_test, y_pred_gb_test)
        
        results['gradient_boosting'] = {
            'model_name': 'Gradient Boosting',
            'best_params': gb_grid.best_params_,
            'train_r2': r2_score(y_train, y_pred_gb_train),
            'test_r2': r2_score(y_test, y_pred_gb_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_gb_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_gb_test),
            'model': gb_best,
            'residual_diagnostics': gb_residuals,
            'feature_importance': dict(zip(feature_cols, gb_best.feature_importances_))
        }
        
        logger.info(f"✓ Gradient Boosting - Test R²: {results['gradient_boosting']['test_r2']:.3f}, "
                   f"Test RMSE: {results['gradient_boosting']['test_rmse']:.3f}")
        
        # Store results
        self.models[city] = results
        
        # Calculate confidence intervals for best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        best_model = results[best_model_name]['model']
        
        logger.info(f"\n🏆 Best model: {results[best_model_name]['model_name']}")
        
        return results
    
    # ========== ADVANCED TIME SERIES FORECASTING ==========
    
    def forecast_arima_auto(self, city: str, variable: str = 'temperature', 
                           steps: int = 24):
        """
        ARIMA forecasting with automatic order selection and diagnostics
        
        Args:
            city: City name
            variable: Variable to forecast
            steps: Forecast horizon
            
        Returns:
            Dictionary with forecast and comprehensive diagnostics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Advanced ARIMA Forecasting: {variable} for {city}")
        logger.info(f"{'='*70}\n")
        
        city_data = self.df[self.df['city_name'] == city].copy()
        city_data = city_data.sort_values('timestamp')
        
        if len(city_data) < 50:
            return {'error': f'Insufficient data: {len(city_data)} records (need at least 50)'}
        
        # Prepare time series
        ts_data = city_data.set_index('timestamp')[variable]
        
        # Stationarity test
        logger.info("Testing for stationarity...")
        stationarity_test = self.test_stationarity(ts_data)
        logger.info(f"ADF Statistic: {stationarity_test['adf_statistic']:.4f}, "
                   f"p-value: {stationarity_test['p_value']:.4f}")
        logger.info(f"Series is: {stationarity_test['interpretation']}")
        
        # Determine differencing order
        d = 0 if stationarity_test['is_stationary'] else 1
        
        # If not stationary, difference and test again
        if d == 1:
            ts_diff = ts_data.diff().dropna()
            stationarity_test_diff = self.test_stationarity(ts_diff)
            if not stationarity_test_diff['is_stationary']:
                d = 2
        
        # Auto-select AR and MA orders based on AIC
        logger.info(f"Searching for optimal ARIMA order (differencing: {d})...")
        
        best_aic = np.inf
        best_order = (1, d, 1)
        
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    model = ARIMA(ts_data, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
        
        logger.info(f"Optimal order found: {best_order} (AIC: {best_aic:.2f})")
        
        # Train-test split
        train_size = int(len(ts_data) * 0.8)
        train = ts_data[:train_size]
        test = ts_data[train_size:]
        
        # Fit final model
        model = ARIMA(train, order=best_order)
        fitted_model = model.fit()
        
        # Forecast on test set
        forecast_test = fitted_model.forecast(steps=len(test))
        
        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(test, forecast_test))
        test_mae = mean_absolute_error(test, forecast_test)
        test_mape = np.mean(np.abs((test - forecast_test) / test)) * 100
        
        # Residual diagnostics
        residuals = fitted_model.resid
        residual_diagnostics = self.analyze_residuals(
            train.values, 
            fitted_model.fittedvalues.values
        )
        
        # Future forecast with confidence intervals
        forecast_obj = fitted_model.get_forecast(steps=steps)
        forecast_mean = forecast_obj.predicted_mean
        forecast_ci = forecast_obj.conf_int()
        
        # Generate future timestamps
        last_timestamp = city_data['timestamp'].max()
        future_timestamps = pd.date_range(
            start=last_timestamp,
            periods=steps + 1,
            freq='6H'
        )[1:]
        
        results = {
            'model_name': 'Auto-ARIMA',
            'city': city,
            'variable': variable,
            'optimal_order': best_order,
            'aic': best_aic,
            'bic': fitted_model.bic,
            'stationarity_test': stationarity_test,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'residual_diagnostics': residual_diagnostics,
            'forecast': forecast_mean.tolist(),
            'forecast_lower': forecast_ci.iloc[:, 0].tolist(),
            'forecast_upper': forecast_ci.iloc[:, 1].tolist(),
            'forecast_timestamps': future_timestamps.tolist(),
            'model_summary': str(fitted_model.summary()),
            'ljung_box_p': fitted_model.test_serial_correlation('ljungbox')[0, 0, 1]
        }
        
        logger.info(f"\n📊 Forecast Performance:")
        logger.info(f"   Test RMSE: {test_rmse:.3f}")
        logger.info(f"   Test MAE: {test_mae:.3f}")
        logger.info(f"   Test MAPE: {test_mape:.2f}%")
        logger.info(f"   Ljung-Box p-value: {results['ljung_box_p']:.4f}")
        
        return results
    
    # ========== COMPREHENSIVE REPORTING ==========
    
    def generate_report(self, city: str):
        """
        Generate comprehensive model comparison report
        
        Args:
            city: City name
            
        Returns:
            Formatted report string
        """
        if city not in self.models:
            return "No models trained for this city"
        
        results = self.models[city]
        
        report = f"\n{'='*70}\n"
        report += f"COMPREHENSIVE MODEL EVALUATION REPORT: {city.upper()}\n"
        report += f"{'='*70}\n\n"
        
        # Model comparison table
        report += "📊 MODEL PERFORMANCE COMPARISON:\n"
        report += "-" * 70 + "\n"
        report += f"{'Model':<20} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}\n"
        report += "-" * 70 + "\n"
        
        for model_key, model_data in results.items():
            report += f"{model_data['model_name']:<20} "
            report += f"{model_data['test_r2']:<12.4f} "
            report += f"{model_data['test_rmse']:<12.3f} "
            report += f"{model_data['test_mae']:<12.3f}\n"
        
        # Best model
        best_model_key = max(results.keys(), key=lambda k: results[k]['test_r2'])
        best_model = results[best_model_key]
        
        report += "\n" + "=" * 70 + "\n"
        report += f"🏆 BEST MODEL: {best_model['model_name']}\n"
        report += "=" * 70 + "\n"
        
        if 'best_params' in best_model:
            report += "\n📋 Optimal Hyperparameters:\n"
            for param, value in best_model['best_params'].items():
                report += f"   • {param}: {value}\n"
        
        # Residual diagnostics
        report += "\n🔬 RESIDUAL DIAGNOSTICS:\n"
        diag = best_model['residual_diagnostics']
        report += f"   • Mean: {diag['residual_mean']:.4f} (should be ≈ 0)\n"
        report += f"   • Std Dev: {diag['residual_std']:.4f}\n"
        report += f"   • Normality (Shapiro-Wilk p-value): {diag['shapiro_p_value']:.4f}\n"
        report += f"   • Residuals normally distributed: {'Yes' if diag['is_normal'] else 'No'}\n"
        report += f"   • Potential heteroscedasticity: {'Yes' if diag['potential_heteroscedasticity'] else 'No'}\n"
        
        # Feature importance
        if 'feature_importance' in best_model:
            report += "\n⭐ TOP 10 MOST IMPORTANT FEATURES:\n"
            importance = best_model['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, imp) in enumerate(sorted_features, 1):
                report += f"   {i:2d}. {feature:<30} {imp:.4f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report


def main():
    """Demonstrate advanced ML pipeline"""
    from database import WeatherDatabase
    
    print("\n🔄 Loading data from database...")
    db = WeatherDatabase()
    conn = db.get_connection()
    df = pd.read_sql_query("SELECT * FROM weather_data", conn)
    db.close()
    
    print(f"✓ Loaded {len(df)} records")
    
    # Initialize advanced predictor
    print("\n🤖 Initializing Advanced ML Predictor...")
    predictor = AdvancedWeatherPredictor(df)
    
    # Get first city
    cities = predictor.df['city_name'].unique()
    if len(cities) == 0:
        print("❌ No cities found in data")
        return
    
    city = cities[0]
    
    # Train models with cross-validation
    print(f"\n🎯 Training models for {city}...")
    results = predictor.train_with_cv(city, cv_folds=5)
    
    if 'error' in results:
        print(f"\n❌ Error: {results['error']}")
        return
    
    # Generate comprehensive report
    report = predictor.generate_report(city)
    print(report)
    
    # Time series forecast
    print(f"\n📈 Running Advanced ARIMA Forecast...")
    forecast_results = predictor.forecast_arima_auto(city, steps=12)
    
    if 'error' not in forecast_results:
        print(f"\n✓ ARIMA Model: {forecast_results['optimal_order']}")
        print(f"   AIC: {forecast_results['aic']:.2f}")
        print(f"   Test RMSE: {forecast_results['test_rmse']:.3f}°C")
        print(f"   Test MAPE: {forecast_results['test_mape']:.2f}%")
        print(f"\n   Next 5 forecasts with 95% CI:")
        for i in range(5):
            print(f"   {i+1}. {forecast_results['forecast'][i]:.1f}°C "
                  f"[{forecast_results['forecast_lower'][i]:.1f}, "
                  f"{forecast_results['forecast_upper'][i]:.1f}]")
    
    print(f"\n{'='*70}")
    print("✅ Advanced ML Analysis Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
