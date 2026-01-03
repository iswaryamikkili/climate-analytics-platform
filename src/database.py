"""
Database Module
Handles SQLite database operations for weather data storage
"""

import sqlite3
import pandas as pd
from datetime import datetime
import yaml
import logging
from typing import List, Dict, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherDatabase:
    """Manages SQLite database for weather data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize database connection"""
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.db_path = config['database']['sqlite_path']
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self.connection = None
        self.create_tables()
        
        logger.info(f"Database initialized at: {self.db_path}")
    
    def get_connection(self):
        """Get database connection"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
        return self.connection
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Main weather data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            city_name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            temperature REAL,
            feels_like REAL,
            temp_min REAL,
            temp_max REAL,
            pressure INTEGER,
            humidity INTEGER,
            wind_speed REAL,
            wind_direction INTEGER,
            cloudiness INTEGER,
            weather_main TEXT,
            weather_description TEXT,
            visibility INTEGER,
            rain_1h REAL,
            snow_1h REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, city_name)
        )
        ''')
        
        # Index for faster queries
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_city_timestamp 
        ON weather_data(city_name, timestamp)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON weather_data(timestamp)
        ''')
        
        conn.commit()
        logger.info("✅ Database tables created/verified")
    
    def insert_weather_data(self, df: pd.DataFrame) -> int:
        """
        Insert weather data from DataFrame
        Returns number of records inserted
        """
        conn = self.get_connection()
        
        # Insert data, ignore duplicates
        try:
            records_before = self.get_record_count()
            
            df.to_sql('weather_data', conn, if_exists='append', 
                     index=False, method='multi')
            
            records_after = self.get_record_count()
            inserted = records_after - records_before
            
            logger.info(f"📝 Inserted {inserted} new records into database")
            return inserted
            
        except sqlite3.IntegrityError as e:
            # Handle duplicate records
            logger.warning(f"Some records already exist (duplicates skipped)")
            return 0
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return 0
    
    def get_record_count(self) -> int:
        """Get total number of records"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        count = cursor.fetchone()[0]
        return count
    
    def get_latest_data(self, limit: int = 10) -> pd.DataFrame:
        """Get most recent weather records"""
        conn = self.get_connection()
        query = f"""
        SELECT * FROM weather_data 
        ORDER BY timestamp DESC 
        LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)
        return df
    
    def get_city_data(self, city_name: str, 
                      days: int = 7) -> pd.DataFrame:
        """Get weather data for specific city"""
        conn = self.get_connection()
        query = f"""
        SELECT * FROM weather_data 
        WHERE city_name = ?
        AND timestamp >= datetime('now', '-{days} days')
        ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn, params=(city_name,))
        return df
    
    def get_all_cities(self) -> List[str]:
        """Get list of all cities in database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT city_name FROM weather_data")
        cities = [row[0] for row in cursor.fetchall()]
        return cities
    
    def get_temperature_stats(self, city_name: Optional[str] = None) -> Dict:
        """Get temperature statistics"""
        conn = self.get_connection()
        
        if city_name:
            query = """
            SELECT 
                AVG(temperature) as avg_temp,
                MIN(temperature) as min_temp,
                MAX(temperature) as max_temp,
                COUNT(*) as record_count
            FROM weather_data 
            WHERE city_name = ?
            """
            cursor = conn.cursor()
            cursor.execute(query, (city_name,))
        else:
            query = """
            SELECT 
                AVG(temperature) as avg_temp,
                MIN(temperature) as min_temp,
                MAX(temperature) as max_temp,
                COUNT(*) as record_count
            FROM weather_data
            """
            cursor = conn.cursor()
            cursor.execute(query)
        
        result = cursor.fetchone()
        return {
            'avg_temp': round(result[0], 2) if result[0] else None,
            'min_temp': round(result[1], 2) if result[1] else None,
            'max_temp': round(result[2], 2) if result[2] else None,
            'record_count': result[3]
        }
    
    def get_data_summary(self):
        """Display database summary"""
        total_records = self.get_record_count()
        cities = self.get_all_cities()
        
        print("\n" + "=" * 60)
        print("📊 DATABASE SUMMARY")
        print("=" * 60)
        print(f"\n📁 Database: {self.db_path}")
        print(f"📝 Total Records: {total_records}")
        print(f"🌍 Cities: {len(cities)}")
        print(f"   {', '.join(cities)}")
        
        if total_records > 0:
            # Get latest record
            latest = self.get_latest_data(1)
            print(f"\n⏰ Latest Data: {latest.iloc[0]['timestamp']}")
            
            # Temperature stats
            stats = self.get_temperature_stats()
            print(f"\n🌡️  Overall Temperature Stats:")
            print(f"   Average: {stats['avg_temp']}°C")
            print(f"   Min: {stats['min_temp']}°C")
            print(f"   Max: {stats['max_temp']}°C")
        
        print("=" * 60 + "\n")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


def main():
    """Test database functionality"""
    # Initialize database
    db = WeatherDatabase()
    
    # Import data from CSV if it exists
    csv_path = "data/raw/weather_data.csv"
    if os.path.exists(csv_path):
        print("\n📥 Importing data from CSV to database...")
        df = pd.read_csv(csv_path)
        inserted = db.insert_weather_data(df)
        print(f"✅ Imported {inserted} records")
    
    # Display summary
    db.get_data_summary()
    
    # Show latest data
    print("\n📋 Latest 5 Records:")
    latest = db.get_latest_data(5)
    print(latest[['timestamp', 'city_name', 'temperature', 'humidity']])
    
    # Close connection
    db.close()


if __name__ == "__main__":
    main()
