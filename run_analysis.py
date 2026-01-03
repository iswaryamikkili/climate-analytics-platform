"""
Quick analysis runner
Loads data and performs comprehensive analysis
"""

import pandas as pd
from src.database import WeatherDatabase
from src.statistical_analysis import WeatherAnalyzer


def main():
    print("\n🔬 Starting Weather Data Analysis...\n")
    
    # Load data
    db = WeatherDatabase()
    conn = db.get_connection()
    df = pd.read_sql_query("SELECT * FROM weather_data", conn)
    db.close()
    
    if df.empty:
        print("❌ No data available!")
        print("💡 Collect data first: python src/data_ingestion.py")
        return
    
    # Analyze
    analyzer = WeatherAnalyzer(df)
    analyzer.print_summary_report()
    
    print("✅ Analysis complete! See above for results.\n")


if __name__ == "__main__":
    main()
