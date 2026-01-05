"""
Create sample data for ML testing
Generates realistic weather data with temporal patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_ml_sample_data(num_records=50):
    """Generate sample weather data with temporal patterns"""
    
    print("\n" + "="*60)
    print("🔄 Generating Sample Weather Data for ML Training")
    print("="*60 + "\n")
    
    cities = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060, "base_temp": 10},
        {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "base_temp": 20},
        {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "base_temp": 5},
        {"name": "Houston", "lat": 29.7604, "lon": -95.3698, "base_temp": 23},
        {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "base_temp": 25}
    ]
    
    records = []
    start_time = datetime.utcnow() - timedelta(hours=num_records * 6)
    
    for i in range(num_records):
        timestamp = start_time + timedelta(hours=i * 6)
        hour = timestamp.hour
        
        for city in cities:
            # Add realistic patterns
            daily_variation = 5 * np.sin(2 * np.pi * hour / 24)
            
            seasonal_variation = 3 * np.sin(2 * np.pi * timestamp.timetuple().tm_yday/ 365)
            noise = np.random.normal(0, 2)
            
            temp = city["base_temp"] + daily_variation + seasonal_variation + noise
            
            # Correlate other variables with temperature
            humidity = max(20, min(100, 70 - temp * 0.5 + np.random.normal(0, 5)))
            wind_speed = max(0, 5 + np.random.normal(0, 2))
            pressure = 1013 + np.random.normal(0, 10)
            cloudiness = max(0, min(100, humidity * 0.8 + np.random.normal(0, 10)))
            
            record = {
                'timestamp': timestamp,
                'city_name': city['name'],
                'latitude': city['lat'],
                'longitude': city['lon'],
                'temperature': round(temp, 2),
                'feels_like': round(temp - 1 + np.random.normal(0, 1), 2),
                'temp_min': round(temp - 2, 2),
                'temp_max': round(temp + 2, 2),
                'pressure': int(pressure),
                'humidity': int(humidity),
                'wind_speed': round(wind_speed, 2),
                'wind_direction': np.random.randint(0, 360),
                'cloudiness': int(cloudiness),
                'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain'], p=[0.5, 0.3, 0.2]),
                'weather_description': 'sample data',
                'visibility': np.random.randint(5000, 10000),
                'rain_1h': round(np.random.exponential(0.5) if np.random.random() > 0.7 else 0, 2),
                'snow_1h': 0
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    
    print(f"✅ Generated {len(df)} sample records")
    print(f"   - {num_records} time points")
    print(f"   - {len(cities)} cities")
    print(f"   - Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Save to database
    from src.database import WeatherDatabase
    db = WeatherDatabase()
    inserted = db.insert_weather_data(df)
    db.close()
    
    print(f"\n📝 Inserted {inserted} records into database")
    print(f"\n{'='*60}")
    print("✅ Sample data generation complete!")
    print("💡 Now you can use ML & Forecasting features in the dashboard")
    print("='*60 + '\n")
    
    return df

if __name__ == "__main__":
    create_ml_sample_data(num_records=50)  # 50 time points = 250 total records
