"""
Statistical Analysis Module
Performs comprehensive statistical analyses on weather data with proper assumption checking
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAnalyzer:
    """
    Performs comprehensive statistical analysis on weather data.
    
    Features:
    - Descriptive statistics with advanced metrics
    - Time series analysis with decomposition
    - Correlation analysis with significance tests
    - Statistical hypothesis testing with assumption checking
    - Anomaly detection (multiple methods)
    - Multivariate analysis (PCA)
    - Feature engineering for ML
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with weather data
        
        Args:
            df: DataFrame with weather data
        """
        self.df = df.copy()
        self._prepare_data()
        logger.info(f"Initialized analyzer with {len(self.df)} records")
    
    def _prepare_data(self):
        """Prepare data for analysis"""
        # Convert timestamp to datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Sort by timestamp
            self.df = self.df.sort_values('timestamp')
            
            # Add derived temporal features
            self.df['hour'] = self.df['timestamp'].dt.hour
            self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
            self.df['month'] = self.df['timestamp'].dt.month
            self.df['day_of_year'] = self.df['timestamp'].dt.dayofyear
    
    # ========== DESCRIPTIVE STATISTICS ==========
    
    def get_descriptive_stats(self, city: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate comprehensive descriptive statistics
        
        Args:
            city: Optional city name to filter by
            
        Returns:
            DataFrame with descriptive statistics
        """
        logger.info(f"Calculating descriptive statistics{' for ' + city if city else ''}...")
        
        df = self.df[self.df['city_name'] == city] if city else self.df
        
        numeric_cols = ['temperature', 'feels_like', 'humidity', 
                       'wind_speed', 'pressure', 'cloudiness']
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        # Basic statistics
        stats_df = df[numeric_cols].describe()
        
        # Additional statistics
        stats_df.loc['variance'] = df[numeric_cols].var()
        stats_df.loc['skewness'] = df[numeric_cols].skew()
        stats_df.loc['kurtosis'] = df[numeric_cols].kurtosis()
        stats_df.loc['range'] = df[numeric_cols].max() - df[numeric_cols].min()
        stats_df.loc['iqr'] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
        stats_df.loc['cv'] = (df[numeric_cols].std() / df[numeric_cols].mean()) * 100  # Coefficient of variation
        
        return stats_df.round(2)
    
    def get_city_comparison(self) -> pd.DataFrame:
        """Compare statistics across cities"""
        logger.info("Generating city comparison...")
        
        comparison = self.df.groupby('city_name').agg({
            'temperature': ['mean', 'min', 'max', 'std'],
            'humidity': ['mean', 'std'],
            'wind_speed': ['mean', 'max'],
            'pressure': ['mean', 'std']
        }).round(2)
        
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
        
        # Add rankings
        comparison['temp_rank'] = comparison['temperature_mean'].rank(ascending=False)
        
        return comparison.sort_values('temperature_mean', ascending=False)
    
    # ========== TIME SERIES ANALYSIS ==========
    
    def detect_trends(self, city: str, variable: str = 'temperature') -> Dict:
        """
        Detect trends in time series data with statistical validation
        
        Args:
            city: City name
            variable: Variable to analyze
            
        Returns:
            Dictionary with trend information and diagnostics
        """
        logger.info(f"Detecting trends for {variable} in {city}...")
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 3:
            return {'error': 'Insufficient data for trend analysis (need n≥3)'}
        
        # Prepare data
        city_data = city_data.sort_values('timestamp')
        x = np.arange(len(city_data))
        y = city_data[variable].values
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate residuals
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Determine trend direction with practical significance
        # A trend is "increasing" only if slope is meaningful
        practical_threshold = 0.01  # Adjust based on your domain
        if abs(slope) < practical_threshold:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # Additional diagnostics
        diagnostics = self._validate_linear_regression(residuals, x)
        
        return {
            'variable': variable,
            'city': city,
            'trend_direction': trend_direction,
            'slope': round(slope, 6),
            'intercept': round(intercept, 2),
            'r_squared': round(r_value ** 2, 4),
            'p_value': round(p_value, 4),
            'std_error': round(std_err, 6),
            'is_significant': p_value < 0.05,
            'is_practically_significant': abs(slope) >= practical_threshold,
            'data_points': len(city_data),
            'diagnostics': diagnostics,
            'interpretation': self._interpret_trend(slope, p_value, r_value**2)
        }
    
    def _validate_linear_regression(self, residuals: np.ndarray, x: np.ndarray) -> Dict:
        """
        Validate linear regression assumptions
        
        Returns:
            Dictionary with assumption test results
        """
        diagnostics = {}
        
        # 1. Normality of residuals (Shapiro-Wilk test)
        if len(residuals) >= 3 and len(residuals) < 5000:
            _, p_norm = stats.shapiro(residuals)
            diagnostics['normality'] = {
                'test': 'Shapiro-Wilk',
                'p_value': round(p_norm, 4),
                'assumption_met': p_norm > 0.05,
                'interpretation': 'Residuals are normally distributed' if p_norm > 0.05 
                                 else 'Residuals deviate from normality'
            }
        
        # 2. Independence (Durbin-Watson test)
        dw_stat = durbin_watson(residuals)
        diagnostics['independence'] = {
            'test': 'Durbin-Watson',
            'statistic': round(dw_stat, 3),
            'assumption_met': 1.5 < dw_stat < 2.5,
            'interpretation': 'No autocorrelation' if 1.5 < dw_stat < 2.5 
                             else 'Possible autocorrelation detected'
        }
        
        # 3. Homoscedasticity (constant variance)
        # Split residuals and compare variances
        if len(residuals) >= 4:
            mid = len(residuals) // 2
            _, p_var = stats.levene(residuals[:mid], residuals[mid:])
            diagnostics['homoscedasticity'] = {
                'test': 'Levene',
                'p_value': round(p_var, 4),
                'assumption_met': p_var > 0.05,
                'interpretation': 'Constant variance' if p_var > 0.05 
                                 else 'Heteroscedasticity detected'
            }
        
        return diagnostics
    
    def _interpret_trend(self, slope: float, p_value: float, r_squared: float) -> str:
        """Generate plain language interpretation of trend"""
        if p_value >= 0.05:
            return "No statistically significant trend detected"
        
        direction = "increasing" if slope > 0 else "decreasing"
        strength = "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.3 else "weak"
        
        return f"Statistically significant {direction} trend detected with {strength} fit (R²={r_squared:.3f})"
    
    def decompose_time_series(self, city: str, 
                             variable: str = 'temperature',
                             period: int = 24,
                             model: str = 'additive') -> Dict:
        """
        Perform seasonal decomposition of time series
        
        Args:
            city: City name
            variable: Variable to decompose
            period: Seasonal period (24 for hourly data with daily seasonality)
            model: 'additive' or 'multiplicative'
            
        Returns:
            Dictionary with decomposition components
        """
        logger.info(f"Decomposing time series for {variable} in {city}...")
        
        city_data = self.df[self.df['city_name'] == city].copy()
        city_data = city_data.sort_values('timestamp').set_index('timestamp')
        
        if len(city_data) < 2 * period:
            return {'error': f'Need at least {2*period} observations for decomposition'}
        
        # Resample to regular intervals if needed
        city_data = city_data.resample('H').mean()
        
        # Handle missing values
        city_data[variable] = city_data[variable].interpolate(method='linear')
        
        try:
            # Perform STL decomposition (more robust than classical)
            stl = STL(city_data[variable], seasonal=period, robust=True)
            result = stl.fit()
            
            # Calculate strength metrics
            trend_strength = max(0, 1 - (result.resid.var() / (result.trend + result.resid).var()))
            seasonal_strength = max(0, 1 - (result.resid.var() / (result.seasonal + result.resid).var()))
            
            return {
                'success': True,
                'model': 'STL (Seasonal-Trend decomposition using LOESS)',
                'period': period,
                'n_observations': len(city_data),
                'trend_strength': round(trend_strength, 3),
                'seasonal_strength': round(seasonal_strength, 3),
                'trend_data': result.trend.dropna().to_dict(),
                'seasonal_data': result.seasonal.dropna().to_dict(),
                'residual_data': result.resid.dropna().to_dict(),
                'residual_std': round(result.resid.std(), 3),
                'interpretation': self._interpret_decomposition(
                    result.trend, result.seasonal, result.resid, 
                    trend_strength, seasonal_strength
                )
            }
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return {'error': str(e)}
    
    def _interpret_decomposition(self, trend, seasonal, residual, 
                                 trend_strength: float, seasonal_strength: float) -> str:
        """Interpret time series decomposition results"""
        trend_direction = "increasing" if trend.iloc[-1] > trend.iloc[0] else "decreasing"
        seasonal_range = seasonal.max() - seasonal.min()
        
        interpretation = []
        
        # Trend
        if trend_strength > 0.6:
            interpretation.append(f"Strong {trend_direction} trend (strength: {trend_strength:.2f})")
        elif trend_strength > 0.3:
            interpretation.append(f"Moderate {trend_direction} trend (strength: {trend_strength:.2f})")
        else:
            interpretation.append(f"Weak trend component (strength: {trend_strength:.2f})")
        
        # Seasonality
        if seasonal_strength > 0.6:
            interpretation.append(f"Strong seasonal pattern with range of ±{seasonal_range/2:.1f} units")
        elif seasonal_strength > 0.3:
            interpretation.append(f"Moderate seasonal pattern with range of ±{seasonal_range/2:.1f} units")
        else:
            interpretation.append(f"Weak seasonality")
        
        # Residuals
        interpretation.append(f"Residual noise level: {residual.std():.2f}")
        
        return ". ".join(interpretation)
    
    def calculate_moving_average(self, city: str, 
                                 variable: str = 'temperature',
                                 window: int = 3) -> pd.DataFrame:
        """
        Calculate moving average with multiple statistics
        
        Args:
            city: City name
            variable: Variable to analyze
            window: Window size for moving average
            
        Returns:
            DataFrame with original values and moving statistics
        """
        city_data = self.df[self.df['city_name'] == city].copy()
        city_data = city_data.sort_values('timestamp')
        
        # Calculate various rolling statistics
        city_data[f'{variable}_ma'] = city_data[variable].rolling(
            window=window, center=True
        ).mean()
        
        city_data[f'{variable}_std'] = city_data[variable].rolling(
            window=window, center=True
        ).std()
        
        city_data[f'{variable}_min'] = city_data[variable].rolling(
            window=window, center=True
        ).min()
        
        city_data[f'{variable}_max'] = city_data[variable].rolling(
            window=window, center=True
        ).max()
        
        return city_data[['timestamp', variable, f'{variable}_ma', 
                         f'{variable}_std', f'{variable}_min', f'{variable}_max']]
    
    # ========== CORRELATION ANALYSIS ==========
    
    def calculate_correlations(self, city: Optional[str] = None, 
                              method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix with specified method
        
        Args:
            city: Optional city name to filter by
            method: 'pearson', 'spearman', or 'kendall'
            
        Returns:
            Correlation matrix
        """
        logger.info(f"Calculating {method} correlations{' for ' + city if city else ''}...")
        
        df = self.df[self.df['city_name'] == city] if city else self.df
        
        numeric_cols = ['temperature', 'feels_like', 'humidity', 
                       'wind_speed', 'pressure', 'cloudiness']
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        correlation_matrix = df[numeric_cols].corr(method=method)
        return correlation_matrix.round(3)
    
    def find_strong_correlations(self, threshold: float = 0.7,
                                method: str = 'pearson') -> List[Dict]:
        """
        Find pairs of variables with strong correlations including significance tests
        
        Args:
            threshold: Correlation threshold (0-1)
            method: Correlation method
            
        Returns:
            List of strongly correlated variable pairs with p-values
        """
        corr_matrix = self.calculate_correlations(method=method)
        
        numeric_cols = ['temperature', 'feels_like', 'humidity', 
                       'wind_speed', 'pressure', 'cloudiness']
        numeric_cols = [c for c in numeric_cols if c in self.df.columns]
        
        strong_correlations = []
        
        # Get upper triangle of correlation matrix
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    # Calculate p-value for correlation
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    
                    if method == 'pearson':
                        _, p_value = stats.pearsonr(
                            self.df[var1].dropna(), 
                            self.df[var2].dropna()
                        )
                    elif method == 'spearman':
                        _, p_value = stats.spearmanr(
                            self.df[var1].dropna(), 
                            self.df[var2].dropna()
                        )
                    else:  # kendall
                        _, p_value = stats.kendalltau(
                            self.df[var1].dropna(), 
                            self.df[var2].dropna()
                        )
                    
                    strong_correlations.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'correlation': round(corr_value, 3),
                        'p_value': round(p_value, 6),
                        'is_significant': p_value < 0.05,
                        'strength': 'strong positive' if corr_value > 0 else 'strong negative',
                        'method': method
                    })
        
        return strong_correlations
    
    # ========== STATISTICAL HYPOTHESIS TESTING ==========
    
    def test_temperature_difference(self, city1: str, city2: str) -> Dict:
        """
        Test if temperature difference between cities is significant.
        Automatically selects appropriate test based on data properties.
        
        Args:
            city1: First city name
            city2: Second city name
            
        Returns:
            Dictionary with test results, assumptions, and interpretation
        """
        logger.info(f"Testing temperature difference between {city1} and {city2}...")
        
        temp1 = self.df[self.df['city_name'] == city1]['temperature'].dropna()
        temp2 = self.df[self.df['city_name'] == city2]['temperature'].dropna()
        
        if len(temp1) < 3 or len(temp2) < 3:
            return {'error': 'Insufficient data for statistical test (need n≥3 for each city)'}
        
        # 1. Check normality (Shapiro-Wilk test)
        _, p_norm1 = stats.shapiro(temp1) if len(temp1) < 5000 else (None, 1.0)
        _, p_norm2 = stats.shapiro(temp2) if len(temp2) < 5000 else (None, 1.0)
        
        is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
        
        # 2. Check variance equality (Levene's test)
        _, p_levene = stats.levene(temp1, temp2)
        equal_var = p_levene > 0.05
        
        # 3. Select and perform appropriate test
        if is_normal and equal_var:
            # Standard independent t-test
            t_stat, p_value = stats.ttest_ind(temp1, temp2)
            test_used = "Independent t-test (parametric)"
            test_statistic_name = "t-statistic"
        elif is_normal and not equal_var:
            # Welch's t-test (unequal variances)
            t_stat, p_value = stats.ttest_ind(temp1, temp2, equal_var=False)
            test_used = "Welch's t-test (unequal variances)"
            test_statistic_name = "t-statistic"
        else:
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = stats.mannwhitneyu(temp1, temp2, alternative='two-sided')
            t_stat = u_stat
            test_used = "Mann-Whitney U test (non-parametric)"
            test_statistic_name = "U-statistic"
        
        # 4. Calculate effect size
        if is_normal:
            # Cohen's d for parametric tests
            pooled_std = np.sqrt(
                ((len(temp1) - 1) * temp1.std()**2 + (len(temp2) - 1) * temp2.std()**2) / 
                (len(temp1) + len(temp2) - 2)
            )
            effect_size = (temp1.mean() - temp2.mean()) / pooled_std if pooled_std > 0 else 0
            effect_size_name = "Cohen's d"
        else:
            # Rank-biserial correlation for non-parametric
            effect_size = 1 - (2 * t_stat) / (len(temp1) * len(temp2))
            effect_size_name = "Rank-biserial correlation"
        
        # 5. Interpret effect size
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            effect_interpretation = "negligible"
        elif abs_effect < 0.5:
            effect_interpretation = "small"
        elif abs_effect < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        # 6. Calculate confidence interval for difference
        diff_mean = temp1.mean() - temp2.mean()
        diff_se = np.sqrt(temp1.var()/len(temp1) + temp2.var()/len(temp2))
        ci_95 = stats.t.interval(0.95, len(temp1) + len(temp2) - 2, 
                                 loc=diff_mean, scale=diff_se)
        
        return {
            'city_1': city1,
            'city_2': city2,
            'n_city1': len(temp1),
            'n_city2': len(temp2),
            'mean_temp_city1': round(temp1.mean(), 2),
            'std_city1': round(temp1.std(), 2),
            'mean_temp_city2': round(temp2.mean(), 2),
            'std_city2': round(temp2.std(), 2),
            'difference': round(diff_mean, 2),
            'ci_95_lower': round(ci_95[0], 2),
            'ci_95_upper': round(ci_95[1], 2),
            'test_used': test_used,
            'test_statistic_name': test_statistic_name,
            'test_statistic': round(t_stat, 4),
            'p_value': round(p_value, 4),
            'is_significant': p_value < 0.05,
            'alpha': 0.05,
            'assumptions': {
                'normality_city1': {
                    'p_value': round(p_norm1, 4) if p_norm1 else 'N/A',
                    'met': 'yes' if p_norm1 > 0.05 else 'no'
                },
                'normality_city2': {
                    'p_value': round(p_norm2, 4) if p_norm2 else 'N/A',
                    'met': 'yes' if p_norm2 > 0.05 else 'no'
                },
                'equal_variance': {
                    'test': 'Levene',
                    'p_value': round(p_levene, 4),
                    'met': 'yes' if equal_var else 'no'
                }
            },
            'effect_size': round(effect_size, 3),
            'effect_size_type': effect_size_name,
            'effect_interpretation': effect_interpretation,
            'interpretation': self._interpret_test_result(
                p_value, effect_interpretation, diff_mean, city1, city2
            )
        }
    
    def _interpret_test_result(self, p_value: float, effect_size: str, 
                              diff: float, city1: str, city2: str) -> str:
        """Generate plain language interpretation of test results"""
        if p_value >= 0.05:
            return (f"No statistically significant temperature difference detected "
                   f"between {city1} and {city2} (p={p_value:.3f}). "
                   f"The observed difference of {abs(diff):.2f}°C could be due to random chance.")
        
        direction = "warmer" if diff > 0 else "cooler"
        return (f"{city1} is significantly {direction} than {city2} "
               f"(difference: {abs(diff):.2f}°C, p={p_value:.4f}). "
               f"This is a {effect_size} effect size, meaning the difference is "
               f"{'practically meaningful' if effect_size in ['medium', 'large'] else 'statistically detectable but small'}.")
    
    # ========== ANOMALY DETECTION ==========
    
    def detect_anomalies_zscore(self, city: str, 
                               variable: str = 'temperature',
                               threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies using z-score method
        
        Args:
            city: City name
            variable: Variable to check for anomalies
            threshold: Z-score threshold (typically 2-3)
            
        Returns:
            DataFrame with anomalous records
        """
        logger.info(f"Detecting anomalies in {variable} for {city} using z-score...")
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 3:
            return pd.DataFrame()
        
        # Calculate z-scores
        mean = city_data[variable].mean()
        std = city_data[variable].std()
        
        if std == 0:
            logger.warning(f"Zero standard deviation for {variable} in {city}")
            return pd.DataFrame()
        
        city_data['z_score'] = (city_data[variable] - mean) / std
        
        # Flag anomalies
        city_data['is_anomaly'] = abs(city_data['z_score']) > threshold
        
        anomalies = city_data[city_data['is_anomaly']]
        
        logger.info(f"Found {len(anomalies)} anomalies using z-score method")
        
        return anomalies[['timestamp', 'city_name', variable, 'z_score']]
    
    def detect_anomalies_iqr(self, city: str, 
                            variable: str = 'temperature',
                            multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Args:
            city: City name
            variable: Variable to check
            multiplier: IQR multiplier (1.5 for mild, 3.0 for extreme outliers)
            
        Returns:
            DataFrame with outlier records
        """
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 4:
            return pd.DataFrame()
        
        Q1 = city_data[variable].quantile(0.25)
        Q3 = city_data[variable].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = city_data[
            (city_data[variable] < lower_bound) | 
            (city_data[variable] > upper_bound)
        ].copy()
        
        # Add outlier score
        outliers['outlier_score'] = outliers[variable].apply(
            lambda x: abs(x - lower_bound) if x < lower_bound else abs(x - upper_bound)
        )
        
        logger.info(f"Found {len(outliers)} outliers using IQR method")
        
        return outliers[['timestamp', 'city_name', variable, 'outlier_score']]
    
    def detect_anomalies_isolation_forest(self, city: str,
                                         contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest (ML-based)
        
        Args:
            city: City name
            contamination: Expected proportion of anomalies
            
        Returns:
            DataFrame with anomalies and scores
        """
        logger.info(f"Detecting anomalies using Isolation Forest for {city}...")
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 10:
            logger.warning(f"Insufficient data for Isolation Forest (need n≥10)")
            return pd.DataFrame()
        
        # Select features
        features = ['temperature', 'humidity', 'wind_speed', 'pressure']
        features = [f for f in features if f in city_data.columns]
        
        X = city_data[features].values
        
        # Handle missing values
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.decision_function(X_scaled)
        
        # Extract anomalies
        city_data['is_anomaly'] = predictions == -1
        city_data['anomaly_score'] = anomaly_scores
        
        anomalies = city_data[city_data['is_anomaly']]
        
        logger.info(f"Found {len(anomalies)} anomalies using Isolation Forest")
        
        return anomalies[['timestamp', 'city_name'] + features + ['anomaly_score']]
    
    # ========== MULTIVARIATE ANALYSIS ==========
    
    def perform_pca(self, n_components: int = 3) -> Dict:
        """
        Principal Component Analysis for dimensionality reduction
        
        Args:
            n_components: Number of principal components
            
        Returns:
            Dictionary with PCA results and interpretation
        """
        logger.info(f"Performing PCA with {n_components} components...")
        
        features = ['temperature', 'humidity', 'wind_speed', 'pressure']
        features = [f for f in features if f in self.df.columns]
        
        X = self.df[features].dropna()
        
        if len(X) < n_components:
            return {'error': f'Need at least {n_components} samples for PCA'}
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate feature importance
        feature_importance = {}
        for i, feature in enumerate(features):
            # Importance is the absolute loading on first PC
            feature_importance[feature] = abs(pca.components_[0][i])
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return {
            'n_components': n_components,
            'n_features': len(features),
            'features': features,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'total_variance_explained': round(sum(pca.explained_variance_ratio_), 3),
            'component_loadings': pca.components_.tolist(),
            'feature_importance': {k: round(v, 3) for k, v in feature_importance.items()},
            'interpretation': self._interpret_pca(
                pca.explained_variance_ratio_, 
                feature_importance
            )
        }
    
    def _interpret_pca(self, explained_var: np.ndarray, 
                      feature_importance: Dict) -> str:
        """Interpret PCA results"""
        cumsum = np.cumsum(explained_var)
        
        interpretation = []
        interpretation.append(
            f"First component explains {explained_var[0]*100:.1f}% of variance"
        )
        interpretation.append(
            f"First {len(explained_var)} components explain {cumsum[-1]*100:.1f}% total variance"
        )
        
        top_feature = list(feature_importance.keys())[0]
        interpretation.append(f"Most important feature: {top_feature}")
        
        return ". ".join(interpretation)
    
    # ========== FEATURE ENGINEERING ==========
    
    def engineer_features(self, city: Optional[str] = None) -> pd.DataFrame:
        """
        Create advanced features for ML modeling
        
        Args:
            city: Optional city filter
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features for ML...")
        
        df = self.df[self.df['city_name'] == city].copy() if city else self.df.copy()
        df = df.sort_values(['city_name', 'timestamp'])
        
        # 1. Temporal features (cyclical encoding)
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'day_of_week' in df.columns:
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 2. Lag features
        for lag in [1, 3, 6, 12, 24]:
            if 'temperature' in df.columns:
                df[f'temp_lag_{lag}'] = df.groupby('city_name')['temperature'].shift(lag)
            if 'humidity' in df.columns:
                df[f'humidity_lag_{lag}'] = df.groupby('city_name')['humidity'].shift(lag)
        
        # 3. Rolling statistics
        for window in [3, 6, 12, 24]:
            if 'temperature' in df.columns:
                df[f'temp_rolling_mean_{window}'] = (
                    df.groupby('city_name')['temperature']
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                df[f'temp_rolling_std_{window}'] = (
                    df.groupby('city_name')['temperature']
                    .rolling(window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )
        
        # 4. Rate of change
        if 'temperature' in df.columns:
            df['temp_change_1h'] = df.groupby('city_name')['temperature'].diff(1)
            df['temp_change_3h'] = df.groupby('city_name')['temperature'].diff(3)
        
        # 5. Interaction features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        
        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            df['wind_chill_index'] = df['temperature'] - (df['wind_speed'] * 0.3)
        
        # 6. Domain-specific features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = self._calculate_heat_index(df['temperature'], df['humidity'])
        
        # 7. Binary features
        if 'hour' in df.columns:
            df['is_daytime'] = df['hour'].between(6, 18).astype(int)
        if 'day_of_week' in df.columns:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Engineered {len(df.columns) - len(self.df.columns)} new features")
        
        return df
    
    def _calculate_heat_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index using simplified formula"""
        # Convert to Fahrenheit
        T = temp * 9/5 + 32
        RH = humidity
        
        # Simplified heat index
        HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
        
        # Convert back to Celsius
        return (HI - 32) * 5/9
    
    # ========== COMPARATIVE ANALYSIS ==========
    
    def compare_cities(self, variable: str = 'temperature') -> pd.DataFrame:
        """
        Compare a variable across all cities
        
        Args:
            variable: Variable to compare
            
        Returns:
            Comparison DataFrame with statistics
        """
        comparison = self.df.groupby('city_name')[variable].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        # Add IQR
        comparison['iqr'] = (
            self.df.groupby('city_name')[variable].quantile(0.75) - 
            self.df.groupby('city_name')[variable].quantile(0.25)
        ).round(2)
        
        comparison = comparison.sort_values('mean', ascending=False)
        
        return comparison
    
    def find_extremes(self) -> Dict:
        """Find cities with extreme weather conditions"""
        extremes = {}
        
        if 'temperature' in self.df.columns:
            extremes['hottest'] = self.df.loc[self.df['temperature'].idxmax()].to_dict()
            extremes['coldest'] = self.df.loc[self.df['temperature'].idxmin()].to_dict()
        
        if 'humidity' in self.df.columns:
            extremes['most_humid'] = self.df.loc[self.df['humidity'].idxmax()].to_dict()
            extremes['least_humid'] = self.df.loc[self.df['humidity'].idxmin()].to_dict()
        
        if 'wind_speed' in self.df.columns:
            extremes['windiest'] = self.df.loc[self.df['wind_speed'].idxmax()].to_dict()
            extremes['calmest'] = self.df.loc[self.df['wind_speed'].idxmin()].to_dict()
        
        if 'pressure' in self.df.columns:
            extremes['highest_pressure'] = self.df.loc[self.df['pressure'].idxmax()].to_dict()
            extremes['lowest_pressure'] = self.df.loc[self.df['pressure'].idxmin()].to_dict()
        
        return extremes
    
    # ========== SUMMARY REPORT ==========
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive analysis summary"""
        logger.info("Generating comprehensive summary report...")
        
        report = {
            'dataset_info': {
                'total_records': len(self.df),
                'cities': self.df['city_name'].nunique() if 'city_name' in self.df.columns else 0,
                'date_range': {
                    'start': str(self.df['timestamp'].min()) if 'timestamp' in self.df.columns else 'N/A',
                    'end': str(self.df['timestamp'].max()) if 'timestamp' in self.df.columns else 'N/A',
                    'duration_days': (self.df['timestamp'].max() - self.df['timestamp'].min()).days 
                                    if 'timestamp' in self.df.columns else 0
                }
            }
        }
        
        # Temperature summary
        if 'temperature' in self.df.columns:
            report['temperature_summary'] = {
                'overall_mean': round(self.df['temperature'].mean(), 2),
                'overall_std': round(self.df['temperature'].std(), 2),
                'overall_min': round(self.df['temperature'].min(), 2),
                'overall_max': round(self.df['temperature'].max(), 2),
                'range': round(self.df['temperature'].max() - self.df['temperature'].min(), 2)
            }
        
        # City rankings
        if 'city_name' in self.df.columns and 'temperature' in self.df.columns:
            city_means = self.df.groupby('city_name')['temperature'].mean()
            report['city_rankings'] = {
                'warmest': city_means.idxmax(),
                'warmest_temp': round(city_means.max(), 2),
                'coldest': city_means.idxmin(),
                'coldest_temp': round(city_means.min(), 2)
            }
            
            if 'humidity' in self.df.columns:
                humidity_means = self.df.groupby('city_name')['humidity'].mean()
                report['city_rankings']['most_humid'] = humidity_means.idxmax()
                report['city_rankings']['most_humid_value'] = round(humidity_means.max(), 2)
        
        # Correlations
        report['correlations'] = self.find_strong_correlations(threshold=0.6)
        
        # Data quality
        report['data_quality'] = {
            'missing_values': self.df.isnull().sum().to_dict(),
            'completeness_pct': round((1 - self.df.isnull().sum().sum() / 
                                      (len(self.df) * len(self.df.columns))) * 100, 2)
        }
        
        return report
    
    def print_summary_report(self):
        """Print formatted summary report to console"""
        report = self.generate_summary_report()
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE WEATHER ANALYSIS REPORT")
        print("=" * 70)
        
        # Dataset info
        print("\n📁 Dataset Information:")
        print(f"   Total Records: {report['dataset_info']['total_records']:,}")
        print(f"   Cities Analyzed: {report['dataset_info']['cities']}")
        if report['dataset_info']['date_range']['start'] != 'N/A':
            print(f"   Date Range: {report['dataset_info']['date_range']['start']} to "
                  f"{report['dataset_info']['date_range']['end']}")
            print(f"   Duration: {report['dataset_info']['date_range']['duration_days']} days")
        
        # Temperature summary
        if 'temperature_summary' in report:
            print("\n🌡️  Temperature Summary:")
            print(f"   Mean: {report['temperature_summary']['overall_mean']}°C")
            print(f"   Std Dev: {report['temperature_summary']['overall_std']}°C")
            print(f"   Range: {report['temperature_summary']['overall_min']}°C to "
                  f"{report['temperature_summary']['overall_max']}°C")
        
        # City rankings
        if 'city_rankings' in report:
            print("\n🏆 City Rankings:")
            print(f"   Warmest: {report['city_rankings']['warmest']} "
                  f"({report['city_rankings']['warmest_temp']}°C)")
            print(f"   Coldest: {report['city_rankings']['coldest']} "
                  f"({report['city_rankings']['coldest_temp']}°C)")
            if 'most_humid' in report['city_rankings']:
                print(f"   Most Humid: {report['city_rankings']['most_humid']} "
                      f"({report['city_rankings']['most_humid_value']}%)")
        
        # Correlations
        print("\n🔗 Strong Correlations:")
        if report['correlations']:
            for corr in report['correlations'][:5]:  # Show top 5
                print(f"   {corr['variable_1']} ↔ {corr['variable_2']}: "
                      f"{corr['correlation']} ({corr['strength']}, p={corr['p_value']})")
        else:
            print("   No strong correlations found (threshold: 0.6)")
        
        # Data quality
        print("\n✅ Data Quality:")
        print(f"   Completeness: {report['data_quality']['completeness_pct']}%")
        missing = {k: v for k, v in report['data_quality']['missing_values'].items() if v > 0}
        if missing:
            print(f"   Missing Values: {missing}")
        else:
            print("   No missing values detected")
        
        print("\n" + "=" * 70 + "\n")


# ========== USAGE EXAMPLE ==========

def main():
    """Test statistical analysis functionality"""
    from database import WeatherDatabase
    
    print("=" * 70)
    print("STATISTICAL ANALYSIS MODULE")
    print("=" * 70)
    
    # Load data from database
    db = WeatherDatabase()
    conn = db.get_connection()
    df = pd.read_sql_query("SELECT * FROM weather_data", conn)
    db.close()
    
    if df.empty:
        print("\n❌ No data available for analysis")
        print("Run data collection first: python src/data_ingestion.py")
        return
    
    print(f"\n✓ Loaded {len(df)} records for analysis")
    
    # Initialize analyzer
    analyzer = WeatherAnalyzer(df)
    
    # Generate comprehensive report
    analyzer.print_summary_report()
    
    # Get cities
    cities = df['city_name'].unique()
    
    if len(cities) > 0:
        first_city = cities[0]
        
        # Descriptive statistics
        print(f"\n📊 Descriptive Statistics for {first_city}:")
        stats = analyzer.get_descriptive_stats(first_city)
        print(stats)
        
        # Trend analysis
        print(f"\n📈 Trend Analysis for {first_city}:")
        trend = analyzer.detect_trends(first_city, 'temperature')
        if 'error' not in trend:
            print(f"   Direction: {trend['trend_direction']}")
            print(f"   R²: {trend['r_squared']}")
            print(f"   P-value: {trend['p_value']}")
            print(f"   Significant: {trend['is_significant']}")
            print(f"   Interpretation: {trend['interpretation']}")
        
        # Anomaly detection
        print(f"\n⚠️  Anomaly Detection for {first_city}:")
        anomalies_z = analyzer.detect_anomalies_zscore(first_city, 'temperature')
        anomalies_iqr = analyzer.detect_anomalies_iqr(first_city, 'temperature')
        print(f"   Z-score method: {len(anomalies_z)} anomalies")
        print(f"   IQR method: {len(anomalies_iqr)} outliers")
    
    # Compare two cities if available
    if len(cities) >= 2:
        print(f"\n🔬 Statistical Test: {cities[0]} vs {cities[1]}:")
        test_result = analyzer.test_temperature_difference(cities[0], cities[1])
        if 'error' not in test_result:
            print(f"   Test Used: {test_result['test_used']}")
            print(f"   Difference: {test_result['difference']}°C")
            print(f"   P-value: {test_result['p_value']}")
            print(f"   Significant: {test_result['is_significant']}")
            print(f"   Effect Size: {test_result['effect_size']} ({test_result['effect_interpretation']})")
            print(f"   Interpretation: {test_result['interpretation']}")
    
    # Correlation analysis
    print("\n🔗 Correlation Matrix:")
    corr = analyzer.calculate_correlations()
    print(corr)
    
    # PCA
    print("\n🎯 Principal Component Analysis:")
    pca_result = analyzer.perform_pca(n_components=3)
    if 'error' not in pca_result:
        print(f"   Components: {pca_result['n_components']}")
        print(f"   Variance Explained: {[f'{v*100:.1f}%' for v in pca_result['explained_variance']]}")
        print(f"   Total Variance: {pca_result['total_variance_explained']*100:.1f}%")
        print(f"   Feature Importance: {pca_result['feature_importance']}")
    
    print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()
