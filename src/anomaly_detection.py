"""
Anomaly Detection Module
Simple, well-documented methods for detecting unusual weather patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class AnomalyDetector:
    """Detects anomalies in weather data"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with weather data
        
        Args:
            df: DataFrame with weather data
        """
        self.df = df.copy()
    
    def detect_zscore(self, city: str, variable: str = 'temperature', 
                     threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies using Z-score method
        
        HOW IT WORKS:
        1. Calculate mean and standard deviation
        2. For each value, calculate: z = (value - mean) / std
        3. If |z| > threshold (usually 3), it's an anomaly
        
        Args:
            city: City name
            variable: Variable to check (temperature, humidity, etc.)
            threshold: Z-score threshold (default 3.0 = 99.7% confidence)
            
        Returns:
            DataFrame with only anomalous records
        """
        # Filter for city
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 3:
            return pd.DataFrame()  # Not enough data
        
        # Calculate statistics
        mean = city_data[variable].mean()
        std = city_data[variable].std()
        
        # Calculate z-scores
        city_data['z_score'] = (city_data[variable] - mean) / std
        
        # Flag anomalies
        city_data['is_anomaly'] = abs(city_data['z_score']) > threshold
        
        # Return only anomalies
        anomalies = city_data[city_data['is_anomaly']]
        
        return anomalies[['timestamp', 'city_name', variable, 'z_score']]
    
    def detect_iqr(self, city: str, variable: str = 'temperature') -> pd.DataFrame:
        """
        Detect anomalies using IQR (Interquartile Range) method
        
        HOW IT WORKS:
        1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
        2. Calculate IQR = Q3 - Q1
        3. Anything < Q1 - 1.5*IQR or > Q3 + 1.5*IQR is an outlier
        
        This is the "box plot" method
        
        Args:
            city: City name
            variable: Variable to check
            
        Returns:
            DataFrame with outliers
        """
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 4:
            return pd.DataFrame()
        
        # Calculate quartiles
        Q1 = city_data[variable].quantile(0.25)
        Q3 = city_data[variable].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outliers = city_data[
            (city_data[variable] < lower_bound) | 
            (city_data[variable] > upper_bound)
        ]
        
        return outliers[['timestamp', 'city_name', variable]]
    
    def get_summary(self, city: str, variable: str = 'temperature') -> Dict:
        """
        Get summary of anomalies for a city
        
        Returns:
            Dictionary with anomaly statistics
        """
        zscore_anomalies = self.detect_zscore(city, variable)
        iqr_anomalies = self.detect_iqr(city, variable)
        
        total_points = len(self.df[self.df['city_name'] == city])
        
        return {
            'total_data_points': total_points,
            'zscore_anomalies': len(zscore_anomalies),
            'iqr_outliers': len(iqr_anomalies),
            'zscore_percentage': (len(zscore_anomalies) / total_points * 100) if total_points > 0 else 0,
            'iqr_percentage': (len(iqr_anomalies) / total_points * 100) if total_points > 0 else 0
        }


# TEST IT
if __name__ == "__main__":
    from database import WeatherDatabase
    
    # Load data
    db = WeatherDatabase()
    conn = db.get_connection()
    df = pd.read_sql_query("SELECT * FROM weather_data", conn)
    db.close()
    
    # Detect anomalies
    detector = AnomalyDetector(df)
    
    # Test on first city
    city = df['city_name'].unique()[0]
    print(f"\n🔍 Testing Anomaly Detection on {city}")
    print("="*60)
    
    # Z-score method
    print("\n📊 Z-Score Method (threshold=3):")
    zscore_anomalies = detector.detect_zscore(city, 'temperature', threshold=3)
    print(f"Found {len(zscore_anomalies)} anomalies")
    if len(zscore_anomalies) > 0:
        print(zscore_anomalies)
    
    # IQR method
    print("\n📦 IQR Method:")
    iqr_anomalies = detector.detect_iqr(city, 'temperature')
    print(f"Found {len(iqr_anomalies)} outliers")
    if len(iqr_anomalies) > 0:
        print(iqr_anomalies)
    
    # Summary
    print("\n📈 Summary:")
    summary = detector.get_summary(city)
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Anomaly detection working!")
