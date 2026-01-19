"""
Test script for forecasting models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import database
from src.database import WeatherDatabase

# Import forecasters
from src.forecasting.statistical_models import ProphetForecaster, ARIMAForecaster

print("=" * 70)
print("FORECASTING MODEL TEST")
print("=" * 70)

# ========== STEP 1: Load Data ==========
print("\n📊 Step 1: Loading data from database...")

db = WeatherDatabase()

try:
    with db.get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM weather_data ORDER BY timestamp", conn)
    
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Cities: {df['city_name'].unique().tolist()}")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("\nMake sure you have data in your database!")
    print("Run: python src/data_ingestion.py")
    exit(1)

# Check if we have enough data
if len(df) < 50:
    print(f"\n⚠️  Warning: Only {len(df)} records available")
    print("This might not be enough for good forecasts")
    print("Consider collecting more data first")
    response = input("\nContinue anyway? (yes/no): ")
    if response.lower() != 'yes':
        exit(0)

# ========== STEP 2: Prepare Data ==========
print("\n🔧 Step 2: Preparing data...")

# Select a city (use the first one)
city = df['city_name'].unique()[0]
print(f"✓ Using city: {city}")

# Filter for this city
city_data = df[df['city_name'] == city].copy()

# Convert timestamp to datetime
city_data['timestamp'] = pd.to_datetime(city_data['timestamp'])

# Set timestamp as index
city_data = city_data.set_index('timestamp')

# Sort by time
city_data = city_data.sort_index()

print(f"✓ City data: {len(city_data)} records")
print(f"✓ Time range: {city_data.index.min()} to {city_data.index.max()}")

# ========== STEP 3: Split Data ==========
print("\n✂️  Step 3: Splitting into train/test...")

# Use 80% for training, 20% for testing
split_idx = int(len(city_data) * 0.8)

train_data = city_data.iloc[:split_idx]
test_data = city_data.iloc[split_idx:]

print(f"✓ Train: {len(train_data)} records")
print(f"✓ Test: {len(test_data)} records")

# ========== STEP 4: Test Prophet ==========
print("\n🔮 Step 4: Testing Prophet Forecaster...")

try:
    # Initialize Prophet
    prophet_model = ProphetForecaster(
        seasonality_mode='additive',
        daily_seasonality=True,
        weekly_seasonality=True
    )
    
    # Fit model
    print("   Training Prophet...")
    prophet_model.fit(train_data, target_col='temperature')
    
    # Generate forecast
    forecast_steps = len(test_data)
    print(f"   Generating {forecast_steps}-step forecast...")
    prophet_forecast = prophet_model.predict(steps=forecast_steps)
    
    # Calculate metrics
    from src.forecasting.base import ModelEvaluator
    
    y_true = test_data['temperature'].values
    y_pred = prophet_forecast['forecast'].values[:len(y_true)]
    
    prophet_metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
    
    print("\n   📊 Prophet Results:")
    print(f"      RMSE: {prophet_metrics['rmse']:.3f}°C")
    print(f"      MAE:  {prophet_metrics['mae']:.3f}°C")
    print(f"      MAPE: {prophet_metrics['mape']:.2f}%")
    print(f"      R²:   {prophet_metrics['r2']:.3f}")
    
    print("   ✅ Prophet test successful!")
    
except Exception as e:
    print(f"   ❌ Prophet failed: {e}")
    prophet_forecast = None
    prophet_metrics = None

# ========== STEP 5: Test ARIMA ==========
print("\n📈 Step 5: Testing ARIMA Forecaster...")

try:
    # Initialize ARIMA
    arima_model = ARIMAForecaster(auto_select=True)
    
    # Fit model
    print("   Training ARIMA (this may take a minute)...")
    arima_model.fit(train_data, target_col='temperature')
    
    # Generate forecast
    print(f"   Generating {forecast_steps}-step forecast...")
    arima_forecast = arima_model.predict(steps=forecast_steps)
    
    # Calculate metrics
    y_pred_arima = arima_forecast['forecast'].values[:len(y_true)]
    arima_metrics = ModelEvaluator.calculate_metrics(y_true, y_pred_arima)
    
    print("\n   📊 ARIMA Results:")
    print(f"      RMSE: {arima_metrics['rmse']:.3f}°C")
    print(f"      MAE:  {arima_metrics['mae']:.3f}°C")
    print(f"      MAPE: {arima_metrics['mape']:.2f}%")
    print(f"      R²:   {arima_metrics['r2']:.3f}")
    
    print("   ✅ ARIMA test successful!")
    
except Exception as e:
    print(f"   ❌ ARIMA failed: {e}")
    arima_forecast = None
    arima_metrics = None

# ========== STEP 6: Compare Models ==========
print("\n🏆 Step 6: Model Comparison...")

if prophet_metrics and arima_metrics:
    print("\n   Metric    | Prophet  | ARIMA    | Winner")
    print("   " + "-" * 50)
    
    # RMSE
    prophet_rmse = prophet_metrics['rmse']
    arima_rmse = arima_metrics['rmse']
    winner = "Prophet" if prophet_rmse < arima_rmse else "ARIMA"
    print(f"   RMSE      | {prophet_rmse:8.3f} | {arima_rmse:8.3f} | {winner}")
    
    # MAE
    prophet_mae = prophet_metrics['mae']
    arima_mae = arima_metrics['mae']
    winner = "Prophet" if prophet_mae < arima_mae else "ARIMA"
    print(f"   MAE       | {prophet_mae:8.3f} | {arima_mae:8.3f} | {winner}")
    
    # MAPE
    prophet_mape = prophet_metrics['mape']
    arima_mape = arima_metrics['mape']
    winner = "Prophet" if prophet_mape < arima_mape else "ARIMA"
    print(f"   MAPE      | {prophet_mape:8.2f} | {arima_mape:8.2f} | {winner}")
    
    # Overall winner (by RMSE)
    overall_winner = "Prophet" if prophet_rmse < arima_rmse else "ARIMA"
    print(f"\n   🏆 Overall Winner: {overall_winner}")

# ========== STEP 7: Visualize (Optional) ==========
print("\n📊 Step 7: Visualization...")

try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(test_data.index, test_data['temperature'], 
             label='Actual', color='black', linewidth=2, marker='o')
    
    # Plot Prophet forecast
    if prophet_forecast is not None:
        plt.plot(prophet_forecast.index, prophet_forecast['forecast'], 
                label='Prophet', color='blue', linewidth=2, linestyle='--')
        plt.fill_between(prophet_forecast.index, 
                        prophet_forecast['lower_bound'], 
                        prophet_forecast['upper_bound'],
                        alpha=0.2, color='blue')
    
    # Plot ARIMA forecast
    if arima_forecast is not None:
        plt.plot(arima_forecast.index, arima_forecast['forecast'], 
                label='ARIMA', color='red', linewidth=2, linestyle='--')
        plt.fill_between(arima_forecast.index, 
                        arima_forecast['lower_bound'], 
                        arima_forecast['upper_bound'],
                        alpha=0.2, color='red')
    
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Forecast Comparison - {city}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('forecast_comparison.png', dpi=150)
    print("   ✓ Plot saved as 'forecast_comparison.png'")
    
    # Show plot (optional - comment out if running on server)
    # plt.show()
    
except Exception as e:
    print(f"   ⚠️  Visualization skipped: {e}")

# ========== FINAL SUMMARY ==========
print("\n" + "=" * 70)
print("✅ TESTING COMPLETE!")
print("=" * 70)

if prophet_metrics and arima_metrics:
    print(f"\n✓ Both models trained successfully")
    print(f"✓ Prophet RMSE: {prophet_metrics['rmse']:.3f}°C")
    print(f"✓ ARIMA RMSE: {arima_metrics['rmse']:.3f}°C")
    print(f"\nNext steps:")
    print(f"  1. Check 'forecast_comparison.png' for visual results")
    print(f"  2. Try collecting more data for better accuracy")
    print(f"  3. Continue to ML models (XGBoost, etc.)")
else:
    print("\n⚠️  Some models failed - check errors above")

print("\n")
