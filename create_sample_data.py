"""
Create sample weather data for testing
Use this while waiting for API key activation
"""

import pandas as pd
from datetime import datetime, timedelta
import random

def create_sample_data():
    """Generate realistic sample weather data"""
    
    cities = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
        {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
        {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740}
    ]
    
    # Temperature ranges by city (in Celsius)
    temp_ranges = {
        "New York": (5, 15),
        "Los Angeles": (15, 25),
        "Chicago": (0, 10),
        "Houston": (18, 28),
        "Phoenix": (20, 30)
    }
    
    records = []
    
    # Generate 50 records (10 per city across 5 time points)
    for hours_ago in [0, 6, 12, 18, 24]:
        timestamp = datetime.utcnow() - timedelta(hours=hours_ago)
        
        for city in cities:
            temp_min, temp_max = temp_ranges[city["name"]]
            temp = random.uniform(temp_min, temp_max)
            
            record = {
                'timestamp': timestamp,
                'city_name': city['name'],
                'latitude': city['lat'],
                'longitude': city['lon'],
                'temperature': round(temp, 2),
                'feels_like': round(temp - random.uniform(0, 2), 2),
                'temp_min': round(temp - random.uniform(1, 3), 2),
                'temp_max': round(temp + random.uniform(1, 3), 2),
                'pressure': random.randint(1000, 1020),
                'humidity': random.randint(40, 80),
                'wind_speed': round(random.uniform(0, 10), 2),
                'wind_direction': random.randint(0, 360),
                'cloudiness': random.randint(0, 100),
                'weather_main': random.choice(['Clear', 'Clouds', 'Rain']),
                'weather_description': random.choice(['clear sky', 'few clouds', 'light rain']),
                'visibility': random.randint(5000, 10000),
                'rain_1h': round(random.uniform(0, 2), 2) if random.random() > 0.7 else 0,
                'snow_1h': 0
            }
            records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save to CSV
    import os
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/weather_data.csv', index=False)
    
    print("=" * 60)
    print("✅ Sample data created successfully!")
    print("=" * 60)
    print(f"\nRecords: {len(df)}")
    print(f"Cities: {df['city_name'].unique().tolist()}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nSaved to: data/raw/weather_data.csv")
    print("\n📊 Preview:")
    print(df[['timestamp', 'city_name', 'temperature', 'humidity']].head(10))
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    create_sample_data()
