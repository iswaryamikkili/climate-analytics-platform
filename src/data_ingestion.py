"""
Data Ingestion Module
Collects weather data from OpenWeatherMap API
"""

import requests
import pandas as pd
from datetime import datetime
import yaml
import time
import os
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeatherDataCollector:
    """Collects weather data from OpenWeatherMap API"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize collector with configuration"""
        logger.info("Initializing Weather Data Collector...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.api_key = self.config['api']['openweather_key']
        self.base_url = self.config['api']['base_url']
        self.cities = self.config['cities']
        
        logger.info(f"Loaded configuration for {len(self.cities)} cities")
    
    def get_current_weather(self, lat: float, lon: float, city_name: str) -> Dict:
        """Fetch current weather data for a location"""
        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'  # Celsius
        }
        
        try:
            logger.info(f"Fetching weather data for {city_name}...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✅ Successfully fetched data for {city_name}")
            return data
            
        except requests.exceptions.HTTPError as e:
            # Special handling for 401 errors
            if response.status_code == 401:
                logger.error(f"❌ Error fetching weather data for {city_name}: {e}")
                logger.error("=" * 60)
                logger.error("🔑 API KEY AUTHENTICATION FAILED!")
                logger.error("=" * 60)
                logger.error("Possible causes:")
                logger.error("  1. API key is new (needs 10-15 min to activate)")
                logger.error("  2. API key is invalid or deactivated")
                logger.error("  3. Wrong API key in config/config.yaml")
                logger.error(f"\nYour key starts with: {self.api_key[:8]}...")
                logger.error(f"Your key ends with: ...{self.api_key[-8:]}")
                logger.error("\nVerify at: https://openweathermap.org/api_keys")
                logger.error("=" * 60)
            else:
                logger.error(f"❌ Error fetching weather data for {city_name}: {e}")
            return None
        except requests.exceptions.RequestException as e:
           logger.error(f"❌ Network error for {city_name}: {e}")
           return None
    def parse_weather_data(self, weather_data: Dict, city_name: str, 
                          lat: float, lon: float) -> Dict:
        """Parse raw API response into structured format"""
        if not weather_data:
            return None
        
        try:
            record = {
                'timestamp': datetime.utcnow(),
                'city_name': city_name,
                'latitude': lat,
                'longitude': lon,
                'temperature': weather_data['main']['temp'],
                'feels_like': weather_data['main']['feels_like'],
                'temp_min': weather_data['main']['temp_min'],
                'temp_max': weather_data['main']['temp_max'],
                'pressure': weather_data['main']['pressure'],
                'humidity': weather_data['main']['humidity'],
                'wind_speed': weather_data['wind']['speed'],
                'wind_direction': weather_data['wind'].get('deg', 0),
                'cloudiness': weather_data['clouds']['all'],
                'weather_main': weather_data['weather'][0]['main'],
                'weather_description': weather_data['weather'][0]['description'],
                'visibility': weather_data.get('visibility', 0),
                'rain_1h': weather_data.get('rain', {}).get('1h', 0),
                'snow_1h': weather_data.get('snow', {}).get('1h', 0)
            }
            return record
        except KeyError as e:
            logger.error(f"Error parsing weather data: Missing key {e}")
            return None
    
    def collect_all_cities(self) -> pd.DataFrame:
        """Collect current weather data for all configured cities"""
        logger.info("=" * 60)
        logger.info("Starting data collection for all cities...")
        logger.info("=" * 60)
        
        data_records = []
        
        for city in self.cities:
            # Fetch raw data
            weather_data = self.get_current_weather(
                city['lat'], 
                city['lon'],
                city['name']
            )
            
            # Parse into structured format
            if weather_data:
                record = self.parse_weather_data(
                    weather_data,
                    city['name'],
                    city['lat'],
                    city['lon']
                )
                if record:
                    data_records.append(record)
            
            # Respect API rate limits (free tier: 60 calls/min)
            time.sleep(1)
        
        # Convert to DataFrame
        df = pd.DataFrame(data_records)
        
        logger.info("=" * 60)
        logger.info(f"✅ Data collection complete! Collected {len(df)} records")
        logger.info("=" * 60)
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, 
                    filepath: str = "data/raw/weather_data.csv"):
        """Save data to CSV (append mode for continuous collection)"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Append if file exists, otherwise create new
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
            logger.info(f"📝 Appended {len(df)} records to {filepath}")
        else:
            df.to_csv(filepath, index=False)
            logger.info(f"📝 Created new file {filepath} with {len(df)} records")
    
    def display_summary(self, df: pd.DataFrame):
        """Display summary statistics of collected data"""
        if df.empty:
            logger.warning("No data to display")
            return
        
        print("\n" + "=" * 60)
        print("📊 DATA COLLECTION SUMMARY")
        print("=" * 60)
        print(f"\n🌍 Cities: {df['city_name'].tolist()}")
        print(f"\n🌡️  Temperature Statistics (°C):")
        print(f"   Average: {df['temperature'].mean():.1f}°C")
        print(f"   Min: {df['temperature'].min():.1f}°C ({df.loc[df['temperature'].idxmin(), 'city_name']})")
        print(f"   Max: {df['temperature'].max():.1f}°C ({df.loc[df['temperature'].idxmax(), 'city_name']})")
        
        print(f"\n💨 Wind Speed: {df['wind_speed'].mean():.1f} m/s (average)")
        print(f"💧 Humidity: {df['humidity'].mean():.0f}% (average)")
        print(f"☁️  Cloudiness: {df['cloudiness'].mean():.0f}% (average)")
        
        print("\n📋 Sample Data:")
        print(df[['city_name', 'temperature', 'humidity', 'weather_description']].to_string(index=False))
        print("=" * 60 + "\n")


def main():
    """Main function to run data collection"""
    try:
        # Initialize collector
        collector = WeatherDataCollector()
        
        # Collect data
        weather_df = collector.collect_all_cities()
        
        # Display summary
        collector.display_summary(weather_df)
        
        # Save to CSV
        collector.save_to_csv(weather_df)
        
        print("✅ SUCCESS! Data collection and storage complete.\n")
        
        return weather_df
        
    except FileNotFoundError:
        logger.error("❌ Config file not found! Make sure config/config.yaml exists with your API key.")
    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
