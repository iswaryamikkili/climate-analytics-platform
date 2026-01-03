"""
Statistical Analysis Module
Performs various statistical analyses on weather data
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAnalyzer:
    """Performs statistical analysis on weather data"""
    
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
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp')
        
        # Add derived features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
    
    # ========== DESCRIPTIVE STATISTICS ==========
    
    def get_descriptive_stats(self, city: str = None) -> pd.DataFrame:
        """
        Calculate descriptive statistics
        
        Args:
            city: Optional city name to filter by
            
        Returns:
            DataFrame with descriptive statistics
        """
        logger.info(f"Calculating descriptive statistics{' for ' + city if city else ''}...")
        
        df = self.df[self.df['city_name'] == city] if city else self.df
        
        numeric_cols = ['temperature', 'feels_like', 'humidity', 
                       'wind_speed', 'pressure', 'cloudiness']
        
        stats_df = df[numeric_cols].describe()
        
        # Add additional statistics
        stats_df.loc['variance'] = df[numeric_cols].var()
        stats_df.loc['skewness'] = df[numeric_cols].skew()
        stats_df.loc['kurtosis'] = df[numeric_cols].kurtosis()
        
        return stats_df.round(2)
    
    def get_city_comparison(self) -> pd.DataFrame:
        """Compare statistics across cities"""
        logger.info("Generating city comparison...")
        
        comparison = self.df.groupby('city_name').agg({
            'temperature': ['mean', 'min', 'max', 'std'],
            'humidity': ['mean'],
            'wind_speed': ['mean'],
            'pressure': ['mean']
        }).round(2)
        
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
        return comparison
    
    # ========== TIME SERIES ANALYSIS ==========
    
    def detect_trends(self, city: str, variable: str = 'temperature') -> Dict:
        """
        Detect trends in time series data
        
        Args:
            city: City name
            variable: Variable to analyze (temperature, humidity, etc.)
            
        Returns:
            Dictionary with trend information
        """
        logger.info(f"Detecting trends for {variable} in {city}...")
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Prepare data
        city_data = city_data.sort_values('timestamp')
        x = np.arange(len(city_data))
        y = city_data[variable].values
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        return {
            'variable': variable,
            'city': city,
            'trend_direction': trend_direction,
            'slope': round(slope, 4),
            'r_squared': round(r_value ** 2, 4),
            'p_value': round(p_value, 4),
            'is_significant': p_value < 0.05,
            'data_points': len(city_data)
        }
    
    def calculate_moving_average(self, city: str, 
                                 variable: str = 'temperature',
                                 window: int = 3) -> pd.DataFrame:
        """
        Calculate moving average
        
        Args:
            city: City name
            variable: Variable to analyze
            window: Window size for moving average
            
        Returns:
            DataFrame with original values and moving average
        """
        city_data = self.df[self.df['city_name'] == city].copy()
        city_data = city_data.sort_values('timestamp')
        
        city_data[f'{variable}_ma'] = city_data[variable].rolling(
            window=window, center=True
        ).mean()
        
        return city_data[['timestamp', variable, f'{variable}_ma']]
    
    # ========== CORRELATION ANALYSIS ==========
    
    def calculate_correlations(self, city: str = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between variables
        
        Args:
            city: Optional city name to filter by
            
        Returns:
            Correlation matrix
        """
        logger.info(f"Calculating correlations{' for ' + city if city else ''}...")
        
        df = self.df[self.df['city_name'] == city] if city else self.df
        
        numeric_cols = ['temperature', 'feels_like', 'humidity', 
                       'wind_speed', 'pressure', 'cloudiness']
        
        correlation_matrix = df[numeric_cols].corr()
        return correlation_matrix.round(3)
    
    def find_strong_correlations(self, threshold: float = 0.7) -> List[Dict]:
        """
        Find pairs of variables with strong correlations
        
        Args:
            threshold: Correlation threshold (0-1)
            
        Returns:
            List of strongly correlated variable pairs
        """
        corr_matrix = self.calculate_correlations()
        
        strong_correlations = []
        
        # Get upper triangle of correlation matrix
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        'variable_1': corr_matrix.columns[i],
                        'variable_2': corr_matrix.columns[j],
                        'correlation': round(corr_value, 3),
                        'strength': 'strong positive' if corr_value > 0 else 'strong negative'
                    })
        
        return strong_correlations
    
    # ========== ANOMALY DETECTION ==========
    
    def detect_anomalies(self, city: str, 
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
        logger.info(f"Detecting anomalies in {variable} for {city}...")
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        # Calculate z-scores
        mean = city_data[variable].mean()
        std = city_data[variable].std()
        city_data['z_score'] = (city_data[variable] - mean) / std
        
        # Flag anomalies
        city_data['is_anomaly'] = abs(city_data['z_score']) > threshold
        
        anomalies = city_data[city_data['is_anomaly']]
        
        logger.info(f"Found {len(anomalies)} anomalies")
        
        return anomalies[['timestamp', 'city_name', variable, 'z_score']]
    
    def detect_outliers_iqr(self, city: str, 
                           variable: str = 'temperature') -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Args:
            city: City name
            variable: Variable to check
            
        Returns:
            DataFrame with outlier records
        """
        city_data = self.df[self.df['city_name'] == city].copy()
        
        Q1 = city_data[variable].quantile(0.25)
        Q3 = city_data[variable].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = city_data[
            (city_data[variable] < lower_bound) | 
            (city_data[variable] > upper_bound)
        ]
        
        logger.info(f"Found {len(outliers)} outliers using IQR method")
        
        return outliers[['timestamp', 'city_name', variable]]
    
    # ========== COMPARATIVE ANALYSIS ==========
    
    def compare_cities(self, variable: str = 'temperature') -> pd.DataFrame:
        """
        Compare a variable across all cities
        
        Args:
            variable: Variable to compare
            
        Returns:
            Comparison DataFrame
        """
        comparison = self.df.groupby('city_name')[variable].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        comparison = comparison.sort_values('mean', ascending=False)
        
        return comparison
    
    def find_extremes(self) -> Dict:
        """Find cities with extreme weather conditions"""
        extremes = {
            'hottest': self.df.loc[self.df['temperature'].idxmax()],
            'coldest': self.df.loc[self.df['temperature'].idxmin()],
            'most_humid': self.df.loc[self.df['humidity'].idxmax()],
            'windiest': self.df.loc[self.df['wind_speed'].idxmax()],
            'highest_pressure': self.df.loc[self.df['pressure'].idxmax()],
            'lowest_pressure': self.df.loc[self.df['pressure'].idxmin()]
        }
        
        return extremes
    
    # ========== STATISTICAL TESTS ==========
    
    def test_temperature_difference(self, city1: str, city2: str) -> Dict:
        """
        Perform t-test to check if temperature difference is significant
        
        Args:
            city1: First city name
            city2: Second city name
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing temperature difference between {city1} and {city2}...")
        
        temp1 = self.df[self.df['city_name'] == city1]['temperature']
        temp2 = self.df[self.df['city_name'] == city2]['temperature']
        
        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(temp1, temp2)
        
        # Calculate effect size (Cohen's d)
        cohens_d = (temp1.mean() - temp2.mean()) / np.sqrt(
            ((len(temp1) - 1) * temp1.std()**2 + (len(temp2) - 1) * temp2.std()**2) / 
            (len(temp1) + len(temp2) - 2)
        )
        
        return {
            'city_1': city1,
            'city_2': city2,
            'mean_temp_city1': round(temp1.mean(), 2),
            'mean_temp_city2': round(temp2.mean(), 2),
            'difference': round(temp1.mean() - temp2.mean(), 2),
            't_statistic': round(t_stat, 4),
            'p_value': round(p_value, 4),
            'is_significant': p_value < 0.05,
            'cohens_d': round(cohens_d, 3),
            'interpretation': 'Significantly different' if p_value < 0.05 else 'Not significantly different'
        }
    
    # ========== SUMMARY REPORT ==========
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive analysis summary"""
        logger.info("Generating comprehensive summary report...")
        
        report = {
            'dataset_info': {
                'total_records': len(self.df),
                'cities': self.df['city_name'].nunique(),
                'date_range': {
                    'start': str(self.df['timestamp'].min()),
                    'end': str(self.df['timestamp'].max())
                }
            },
            'temperature_summary': {
                'overall_mean': round(self.df['temperature'].mean(), 2),
                'overall_std': round(self.df['temperature'].std(), 2),
                'min': round(self.df['temperature'].min(), 2),
                'max': round(self.df['temperature'].max(), 2)
            },
            'city_rankings': {
                'warmest': self.df.groupby('city_name')['temperature'].mean().idxmax(),
                'coldest': self.df.groupby('city_name')['temperature'].mean().idxmin(),
                'most_humid': self.df.groupby('city_name')['humidity'].mean().idxmax()
            },
            'correlations': self.find_strong_correlations(threshold=0.6)
        }
        
        return report
    
    def print_summary_report(self):
        """Print formatted summary report"""
        report = self.generate_summary_report()
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE WEATHER ANALYSIS REPORT")
        print("=" * 70)
        
        print("\n📁 Dataset Information:")
        print(f"   Total Records: {report['dataset_info']['total_records']}")
        print(f"   Cities Analyzed: {report['dataset_info']['cities']}")
        print(f"   Date Range: {report['dataset_info']['date_range']['start']} to {report['dataset_info']['date_range']['end']}")
        
        print("\n🌡️  Temperature Summary:")
        print(f"   Mean: {report['temperature_summary']['overall_mean']}°C")
        print(f"   Std Dev: {report['temperature_summary']['overall_std']}°C")
        print(f"   Range: {report['temperature_summary']['min']}°C to {report['temperature_summary']['max']}°C")
        
        print("\n🏆 City Rankings:")
        print(f"   Warmest City: {report['city_rankings']['warmest']}")
        print(f"   Coldest City: {report['city_rankings']['coldest']}")
        print(f"   Most Humid City: {report['city_rankings']['most_humid']}")
        
        print("\n🔗 Strong Correlations:")
        if report['correlations']:
            for corr in report['correlations']:
                print(f"   {corr['variable_1']} ↔ {corr['variable_2']}: {corr['correlation']} ({corr['strength']})")
        else:
            print("   No strong correlations found (threshold: 0.6)")
        
        print("\n" + "=" * 70 + "\n")


def main():
    """Test statistical analysis functionality"""
    from database import WeatherDatabase
    
    # Load data from database
    db = WeatherDatabase()
    
    # Get all data
    conn = db.get_connection()
    df = pd.read_sql_query("SELECT * FROM weather_data", conn)
    db.close()
    
    if df.empty:
        print("❌ No data available for analysis")
        print("Run data collection first: python src/data_ingestion.py")
        return
    
    print(f"\n✅ Loaded {len(df)} records for analysis")
    
    # Initialize analyzer
    analyzer = WeatherAnalyzer(df)
    
    # Generate comprehensive report
    analyzer.print_summary_report()
    
    # Descriptive statistics
    print("\n📊 Descriptive Statistics (All Cities):")
    print(analyzer.get_descriptive_stats())
    
    # City comparison
    print("\n🌍 City Comparison:")
    print(analyzer.get_city_comparison())
    
    # Correlations
    print("\n🔗 Correlation Matrix:")
    print(analyzer.calculate_correlations())
    
    # Trends for first city
    cities = df['city_name'].unique()
    if len(cities) > 0:
        first_city = cities[0]
        print(f"\n📈 Trend Analysis for {first_city}:")
        trend = analyzer.detect_trends(first_city, 'temperature')
        for key, value in trend.items():
            print(f"   {key}: {value}")
        
        # Anomalies
        print(f"\n⚠️  Anomalies in {first_city}:")
        anomalies = analyzer.detect_anomalies(first_city, 'temperature')
        if len(anomalies) > 0:
            print(anomalies)
        else:
            print("   No anomalies detected")
    
    # Compare two cities if we have at least 2
    if len(cities) >= 2:
        print(f"\n🔬 Statistical Test: {cities[0]} vs {cities[1]}:")
        test_result = analyzer.test_temperature_difference(cities[0], cities[1])
        for key, value in test_result.items():
            print(f"   {key}: {value}")
    
    print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()
