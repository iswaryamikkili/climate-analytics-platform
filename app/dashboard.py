import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import WeatherDatabase
from src.statistical_analysis import WeatherAnalyzer
from src.data_ingestion import WeatherDataCollector
from src.ml_models import AdvancedWeatherPredictor
# Page configuration
st.set_page_config(
    page_title="Climate Analytics Platform",
    page_icon="cloud",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #00FF00;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)


# ========== DATA LOADING ==========

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from database"""
    db = WeatherDatabase()
    conn = db.get_connection()
    df = pd.read_sql_query("SELECT * FROM weather_data ORDER BY timestamp DESC", conn)
    db.close()
    return df


def collect_new_data():
    """Collect fresh weather data"""
    with st.spinner("Collecting fresh weather data..."):
        collector = WeatherDataCollector()
        weather_df = collector.collect_all_cities()
        
        # Save to database
        from src.database import WeatherDatabase
        db = WeatherDatabase()
        inserted = db.insert_weather_data(weather_df)
        db.close()
        
        return inserted


# ========== SIDEBAR ==========

st.sidebar.title("Climate Analytics")
st.sidebar.markdown("---")

# Data refresh button
if st.sidebar.button("Collect New Data", width="stretch"):
    records = collect_new_data()
    st.sidebar.success(f" Collected {records} new records!")
    st.cache_data.clear()  # Clear cache to reload data
    st.rerun()

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "City Comparison", "Time Series", 
     "Correlations", "️Anomalies", "ML & Forecasting", "Raw Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Climate Analytics Platform**\n\n"
    "Real-time weather data analysis with statistical insights.\n\n"
    "Built with Python, Streamlit, Pandas, and Plotly."
)


# ========== LOAD DATA ==========

df = load_data()

if df.empty:
    st.error("No data available!")
    st.info(" Click 'Collect New Data' in the sidebar to get started.")
    st.stop()

# Initialize analyzer
analyzer = WeatherAnalyzer(df)


# ========== PAGE: OVERVIEW ==========

if page == "Overview":
    st.title("Weather Analytics Overview")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Cities Tracked",
            df['city_name'].nunique()
        )
    
    with col3:
        avg_temp = df['temperature'].mean()
        st.metric(
            "Avg Temperature",
            f"{avg_temp:.1f}C",
            delta=None
        )
    
    with col4:
        latest = df.iloc[0]['timestamp']
        st.metric(
            "Latest Update",
            pd.to_datetime(latest).strftime("%H:%M")
        )
    
    st.markdown("---")
    
    # Two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("️Current Temperature by City")
        
        # Get latest temperature for each city
        latest_temps = df.groupby('city_name').first().reset_index()
        
        fig = px.bar(
            latest_temps,
            x='city_name',
            y='temperature',
            color='temperature',
            color_continuous_scale='RdYlBu_r',
            labels={'temperature': 'Temperature (C)', 'city_name': 'City'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.subheader("Current Humidity by City")
        
        fig = px.bar(
            latest_temps,
            x='city_name',
            y='humidity',
            color='humidity',
            color_continuous_scale='Blues',
            labels={'humidity': 'Humidity (%)', 'city_name': 'City'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    
    # Full width chart
    st.subheader("Weather Conditions Across Cities")
    
    # Create summary DataFrame
    summary = df.groupby('city_name').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean',
        'pressure': 'mean'
    }).round(2).reset_index()
    
    # Display as table
    st.dataframe(summary, width="stretch")
    
    # Quick stats
    st.markdown("---")
    st.subheader("Quick Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("** Hottest City**")
        hottest = df.loc[df['temperature'].idxmax()]
        st.write(f"{hottest['city_name']}: {hottest['temperature']:.1f}C")
    
    with col2:
        st.write("**️ Coldest City**")
        coldest = df.loc[df['temperature'].idxmin()]
        st.write(f"{coldest['city_name']}: {coldest['temperature']:.1f}C")
    
    with col3:
        st.write("** Windiest City**")
        windiest = df.loc[df['wind_speed'].idxmax()]
        st.write(f"{windiest['city_name']}: {windiest['wind_speed']:.1f} m/s")


# ========== PAGE: CITY COMPARISON ==========

elif page == "City Comparison":
    st.title("City-by-City Comparison")
    st.markdown("---")
    
    # City selector
    cities = df['city_name'].unique().tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        city1 = st.selectbox("Select First City", cities, index=0)
    with col2:
        city2 = st.selectbox("Select Second City", cities, index=1 if len(cities) > 1 else 0)
    
    if city1 == city2:
        st.warning(" Please select different cities for comparison")
        st.stop()
    
    # Get data for selected cities
    city1_data = df[df['city_name'] == city1]
    city2_data = df[df['city_name'] == city2]
    
    # Comparison metrics
    st.subheader("Key Metrics Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"{city1}",
            f"{city1_data['temperature'].mean():.1f}C",
            delta=f"{city1_data['temperature'].mean() - city2_data['temperature'].mean():.1f}C"
        )
    
    with col2:
        st.metric(
            f"{city2}",
            f"{city2_data['temperature'].mean():.1f}C"
        )
    
    with col3:
        st.metric(
            f"{city1} Humidity",
            f"{city1_data['humidity'].mean():.0f}%",
            delta=f"{city1_data['humidity'].mean() - city2_data['humidity'].mean():.0f}%"
        )
    
    with col4:
        st.metric(
            f"{city2} Humidity",
            f"{city2_data['humidity'].mean():.0f}%"
        )
    
    st.markdown("---")
    
    # Statistical test
    st.subheader("Statistical Significance Test")
    test_result = analyzer.test_temperature_difference(city1, city2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Temperature Difference:** {abs(test_result['difference']):.2f}C")
        st.write(f"**P-value:** {test_result['p_value']:.4f}")
    with col2:
        if test_result['is_significant']:
            st.success("Difference is statistically significant (p < 0.05)")
        else:
            st.info("Difference is not statistically significant (p ≥ 0.05)")
    
    # Visualization
    st.markdown("---")
    st.subheader("Side-by-Side Comparison")
    
    # Prepare data for comparison
    comparison_df = pd.DataFrame({
        'City': [city1] * len(city1_data) + [city2] * len(city2_data),
        'Temperature': pd.concat([city1_data['temperature'], city2_data['temperature']]),
        'Humidity': pd.concat([city1_data['humidity'], city2_data['humidity']]),
        'Wind Speed': pd.concat([city1_data['wind_speed'], city2_data['wind_speed']])
    })
    
    # Box plots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Temperature (C)', 'Humidity (%)', 'Wind Speed (m/s)')
    )
    
    # Temperature
    for city in [city1, city2]:
        city_temp = comparison_df[comparison_df['City'] == city]['Temperature']
        fig.add_trace(
            go.Box(y=city_temp, name=city, showlegend=False),
            row=1, col=1
        )
    
    # Humidity
    for city in [city1, city2]:
        city_hum = comparison_df[comparison_df['City'] == city]['Humidity']
        fig.add_trace(
            go.Box(y=city_hum, name=city, showlegend=False),
            row=1, col=2
        )
    
    # Wind Speed
    for city in [city1, city2]:
        city_wind = comparison_df[comparison_df['City'] == city]['Wind Speed']
        fig.add_trace(
            go.Box(y=city_wind, name=city),
            row=1, col=3
        )
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, width="stretch")


# ========== PAGE: TIME SERIES ==========

elif page == "Time Series":
    st.title("Time Series Analysis")
    st.markdown("---")
    
    # City and variable selectors
    col1, col2 = st.columns(2)
    
    with col1:
        selected_city = st.selectbox(
            "Select City",
            df['city_name'].unique()
        )
    
    with col2:
        variable = st.selectbox(
            "Select Variable",
            ['temperature', 'humidity', 'wind_speed', 'pressure']
        )
    
    # Get city data
    city_data = df[df['city_name'] == selected_city].sort_values('timestamp')
    
    # Trend analysis
    st.subheader("Trend Analysis")
    trend = analyzer.detect_trends(selected_city, variable)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Trend Direction", trend['trend_direction'].upper())
    with col2:
        st.metric("R2 Score", f"{trend['r_squared']:.3f}")
    with col3:
        status = "Significant" if trend['is_significant'] else "Not Significant"
        st.metric("Statistical Significance", status)
    
    # Time series plot
    st.subheader(f"{variable.replace('_', ' ').title()} Over Time")
    
    fig = go.Figure()
    
    # Actual data
    fig.add_trace(go.Scatter(
        x=city_data['timestamp'],
        y=city_data[variable],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Moving average if enough data
    if len(city_data) >= 3:
        ma_data = analyzer.calculate_moving_average(selected_city, variable, window=3)
        fig.add_trace(go.Scatter(
            x=ma_data['timestamp'],
            y=ma_data[f'{variable}_ma'],
            mode='lines',
            name='Moving Average (3)',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
    
    fig.update_layout(
        height=500,
        xaxis_title="Time",
        yaxis_title=variable.replace('_', ' ').title(),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Recent values table
    st.subheader(" Recent Values")
    recent = city_data.head(10)[['timestamp', variable]].copy()
    recent['timestamp'] = pd.to_datetime(recent['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(recent, width="stretch")


# ========== PAGE: CORRELATIONS ==========

elif page == "Correlations":
    st.title("Correlation Analysis")
    st.markdown("---")
    
    # City selector
    analysis_type = st.radio(
        "Analysis Type",
        ["All Cities Combined", "Individual City"],
        horizontal=True
    )
    
    if analysis_type == "Individual City":
        selected_city = st.selectbox("Select City", df['city_name'].unique())
        corr_matrix = analyzer.calculate_correlations(city=selected_city)
        title_suffix = f" - {selected_city}"
    else:
        corr_matrix = analyzer.calculate_correlations()
        title_suffix = " - All Cities"
    
    # Correlation heatmap
    st.subheader(f" Correlation Heatmap{title_suffix}")
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        labels=dict(color="Correlation")
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")
    
    # Strong correlations
    st.subheader("Strong Correlations")
    strong_corrs = analyzer.find_strong_correlations(threshold=0.5)
    
    if strong_corrs:
        for corr in strong_corrs:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{corr['variable_1']}**")
            with col2:
                st.write(f"**{corr['variable_2']}**")
            with col3:
                st.write(f"**{corr['correlation']}**")
    else:
        st.info("No strong correlations found (threshold: 0.5)")


# ========== PAGE: ANOMALIES ==========

elif page == "Anomalies":
    st.title("️Anomaly Detection")
    st.markdown("---")
    
    # Selectors
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_city = st.selectbox("Select City", df['city_name'].unique())
    
    with col2:
        variable = st.selectbox(
            "Select Variable",
            ['temperature', 'humidity', 'wind_speed', 'pressure']
        )
    
    with col3:
        threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 3.0, 0.5)
    
    # Detect anomalies
    anomalies = analyzer.detect_anomalies(selected_city, variable, threshold)
    
    # Display results
    st.subheader("Anomaly Detection Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Data Points", len(df[df['city_name'] == selected_city]))
    with col2:
        st.metric("Anomalies Detected", len(anomalies))
    
    if len(anomalies) > 0:
        st.warning(f"️ Found {len(anomalies)} anomalous data points!")
        
        # Show anomalies
        st.subheader("Anomalous Records")
        anomalies_display = anomalies.copy()
        anomalies_display['timestamp'] = pd.to_datetime(anomalies_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(anomalies_display, width="stretch")
        
        # Visualization
        st.subheader("Visualization with Anomalies Highlighted")
        
        city_data = df[df['city_name'] == selected_city].sort_values('timestamp')
        
        fig = go.Figure()
        
        # Normal data
        fig.add_trace(go.Scatter(
            x=city_data['timestamp'],
            y=city_data[variable],
            mode='lines+markers',
            name='Normal',
            line=dict(color='#1f77b4'),
            marker=dict(size=6)
        ))
        
        # Anomalies
        fig.add_trace(go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies[variable],
            mode='markers',
            name='Anomaly',
            marker=dict(size=12, color='red', symbol='x')
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Time",
            yaxis_title=variable.replace('_', ' ').title()
        )
        
        st.plotly_chart(fig, width="stretch")
        
    else:
        st.success("No anomalies detected in the data!")


# ========== PAGE: ML & FORECASTING ==========
elif page == "ML & Forecasting":
    st.title("Machine Learning & Forecasting")
    st.markdown("---")
    
    # Check if enough data
    if len(df) < 50:
        st.warning("️Insufficient data for advanced ML models (need at least 50 records)")
        st.info("Collect more data using the 'Data Collection' page")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Records", len(df))
        with col2:
            st.metric("Required Records", 50)
        with col3:
            st.metric("Deficit", max(0, 50 - len(df)))
        
        st.stop()
    
    # City selector
    st.sidebar.markdown("### Model Settings")
    selected_city = st.sidebar.selectbox("Select City for Analysis", sorted(df['city_name'].unique()))
    cv_folds = st.sidebar.slider("Cross-Validation Folds", 3, 10, 5)
    
    st.markdown("---")
    
    # Initialize predictor
    @st.cache_resource
    def get_predictor(data):
        return AdvancedWeatherPredictor(data)
    
    try:
        predictor = get_predictor(df)
        city_count = len(predictor.df[predictor.df['city_name'] == selected_city])
        
        st.info(f" Loaded {city_count} records for {selected_city}")
        
        if city_count < 50:
            st.error(f" {selected_city} has only {city_count} records. Need at least 50 for training.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error initializing predictor: {e}")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        " Model Comparison",
        " Feature Importance",
        " Time Series Forecasting",
        " Model Diagnostics"
    ])
    
    # ========== TAB 1: MODEL COMPARISON ==========
    with tab1:
        st.subheader(" Model Performance Comparison")
        st.markdown("Train multiple models with hyperparameter tuning and cross-validation")
        
        if st.button(" Train All Models", type="primary", use_container_width=True):
            with st.spinner(" Training models with cross-validation and hyperparameter tuning..."):
                try:
                    # Train models
                    results = predictor.train_with_cv(selected_city, cv_folds=cv_folds)
                    
                    if 'error' in results:
                        st.error(f" Training failed: {results['error']}")
                    else:
                        st.success(" Models trained successfully!")
                        
                        # Create comparison DataFrame
                        comparison_data = []
                        for model_key, model_info in results.items():
                            comparison_data.append({
                                'Model': model_info['model_name'],
                                'Train R²': model_info['train_r2'],
                                'Test R²': model_info['test_r2'],
                                'Train RMSE': model_info['train_rmse'],
                                'Test RMSE': model_info['test_rmse'],
                                'Test MAE': model_info['test_mae']
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        comparison_df = comparison_df.sort_values('Test R²', ascending=False)
                        
                        # Display table
                        st.dataframe(
                            comparison_df.style.highlight_max(
                                subset=['Train R²', 'Test R²'], 
                                color='lightgreen'
                            ).highlight_min(
                                subset=['Train RMSE', 'Test RMSE', 'Test MAE'], 
                                color='lightgreen'
                            ),
                            use_container_width=True
                        )
                        
                        # Visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # R² Comparison
                            fig_r2 = go.Figure()
                            fig_r2.add_trace(go.Bar(
                                name='Train R²',
                                x=comparison_df['Model'],
                                y=comparison_df['Train R²'],
                                marker_color='lightblue'
                            ))
                            fig_r2.add_trace(go.Bar(
                                name='Test R²',
                                x=comparison_df['Model'],
                                y=comparison_df['Test R²'],
                                marker_color='darkblue'
                            ))
                            fig_r2.update_layout(
                                title="R² Score Comparison",
                                barmode='group',
                                height=400,
                                yaxis_title="R² Score"
                            )
                            st.plotly_chart(fig_r2, use_container_width=True)
                        
                        with col2:
                            # RMSE Comparison
                            fig_rmse = go.Figure()
                            fig_rmse.add_trace(go.Bar(
                                name='Train RMSE',
                                x=comparison_df['Model'],
                                y=comparison_df['Train RMSE'],
                                marker_color='lightcoral'
                            ))
                            fig_rmse.add_trace(go.Bar(
                                name='Test RMSE',
                                x=comparison_df['Model'],
                                y=comparison_df['Test RMSE'],
                                marker_color='darkred'
                            ))
                            fig_rmse.update_layout(
                                title="RMSE Comparison",
                                barmode='group',
                                height=400,
                                yaxis_title="RMSE"
                            )
                            st.plotly_chart(fig_rmse, use_container_width=True)
                        
                        # Best model info
                        best_model = comparison_df.iloc[0]
                        st.success(f" Best Model: **{best_model['Model']}** (Test R² = {best_model['Test R²']:.4f})")
                        
                        # Show hyperparameters
                        best_model_key = list(results.keys())[0]  # Best model is first after sorting
                        for key, info in results.items():
                            if info['model_name'] == best_model['Model']:
                                best_model_key = key
                                break
                        
                        if 'best_params' in results[best_model_key]:
                            st.markdown("#### 🔧 Optimal Hyperparameters")
                            params_df = pd.DataFrame([results[best_model_key]['best_params']])
                            st.dataframe(params_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f" Error during training: {str(e)}")
                    st.exception(e)
    
    # ========== TAB 2: FEATURE IMPORTANCE ==========
    with tab2:
        st.subheader(" Feature Importance Analysis")
        st.markdown("Understand which features are most important for predictions")
        
        if selected_city in predictor.models:
            results = predictor.models[selected_city]
            
            # Find model with feature importance
            model_with_importance = None
            for model_key, model_info in results.items():
                if 'feature_importance' in model_info:
                    model_with_importance = model_info
                    break
            
            if model_with_importance:
                # Create feature importance DataFrame
                importance_dict = model_with_importance['feature_importance']
                importance_df = pd.DataFrame({
                    'Feature': list(importance_dict.keys()),
                    'Importance': list(importance_dict.values())
                })
                importance_df = importance_df.sort_values('Importance', ascending=False).head(15)
                
                # Display
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Bar chart
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f"Top 15 Features - {model_with_importance['model_name']}",
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("####  Top 10 Features")
                    for i, row in importance_df.head(10).iterrows():
                        st.metric(
                            label=row['Feature'],
                            value=f"{row['Importance']:.4f}"
                        )
            else:
                st.info("Train models first to see feature importance")
        else:
            st.info("Train models first to see feature importance")
    
    # ========== TAB 3: TIME SERIES FORECASTING ==========
    with tab3:
        st.subheader(" Advanced Time Series Forecasting")
        st.markdown("ARIMA forecasting with automatic order selection and diagnostics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_variable = st.selectbox(
                "Select Variable to Forecast",
                ['temperature', 'humidity', 'wind_speed', 'pressure']
            )
        
        with col2:
            forecast_steps = st.slider("Forecast Horizon (steps)", 6, 48, 24)
        
        if st.button(" Generate Forecast", type="primary", use_container_width=True):
            with st.spinner(" Running ARIMA analysis with automatic order selection..."):
                try:
                    forecast_results = predictor.forecast_arima_auto(
                        selected_city, 
                        variable=forecast_variable,
                        steps=forecast_steps
                    )
                    
                    if 'error' in forecast_results:
                        st.error(f" Forecasting failed: {forecast_results['error']}")
                    else:
                        st.success(" Forecast generated successfully!")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("ARIMA Order", str(forecast_results['optimal_order']))
                        with col2:
                            st.metric("Test RMSE", f"{forecast_results['test_rmse']:.3f}")
                        with col3:
                            st.metric("Test MAE", f"{forecast_results['test_mae']:.3f}")
                        with col4:
                            st.metric("Test MAPE", f"{forecast_results['test_mape']:.2f}%")
                        
                        # Stationarity test
                        st.markdown("#### 🔬 Stationarity Test")
                        stationarity = forecast_results['stationarity_test']
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("ADF Statistic", f"{stationarity['adf_statistic']:.4f}")
                        with col2:
                            st.metric("p-value", f"{stationarity['p_value']:.4f}")
                        with col3:
                            status = "Stationary" if stationarity['is_stationary'] else "⚠️ Non-stationary"
                            st.metric("Status", status)
                        
                        # Forecast plot
                        st.markdown("#### 📊 Forecast Visualization")
                        
                        fig = go.Figure()
                        
                        # Forecast line
                        fig.add_trace(go.Scatter(
                            x=forecast_results['forecast_timestamps'],
                            y=forecast_results['forecast'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Confidence intervals
                        fig.add_trace(go.Scatter(
                            x=forecast_results['forecast_timestamps'],
                            y=forecast_results['forecast_upper'],
                            mode='lines',
                            name='Upper 95% CI',
                            line=dict(color='lightblue', dash='dash'),
                            showlegend=True
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_results['forecast_timestamps'],
                            y=forecast_results['forecast_lower'],
                            mode='lines',
                            name='Lower 95% CI',
                            line=dict(color='lightblue', dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(173, 216, 230, 0.3)',
                            showlegend=True
                        ))
                        
                        fig.update_layout(
                            title=f"{forecast_variable.title()} Forecast with 95% Confidence Intervals",
                            xaxis_title="Timestamp",
                            yaxis_title=forecast_variable.title(),
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table
                        st.markdown("#### 📋 Detailed Forecast")
                        forecast_df = pd.DataFrame({
                            'Timestamp': forecast_results['forecast_timestamps'],
                            'Forecast': forecast_results['forecast'],
                            'Lower CI': forecast_results['forecast_lower'],
                            'Upper CI': forecast_results['forecast_upper']
                        })
                        st.dataframe(forecast_df.head(10), use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error during forecasting: {str(e)}")
                    st.exception(e)
    
    # ========== TAB 4: MODEL DIAGNOSTICS ==========
    with tab4:
        st.subheader("🎯 Model Diagnostics & Validation")
        st.markdown("Detailed residual analysis and model validation metrics")
        
        if selected_city in predictor.models:
            results = predictor.models[selected_city]
            
            # Model selector
            model_names = [info['model_name'] for info in results.values()]
            selected_model_name = st.selectbox("Select Model", model_names)
            
            # Find selected model
            selected_model_info = None
            for model_info in results.values():
                if model_info['model_name'] == selected_model_name:
                    selected_model_info = model_info
                    break
            
            if selected_model_info and 'residual_diagnostics' in selected_model_info:
                diag = selected_model_info['residual_diagnostics']
                
                st.markdown("#### 📊 Residual Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Residual Mean", f"{diag['residual_mean']:.6f}")
                    st.caption("Should be close to 0")
                
                with col2:
                    st.metric("Residual Std Dev", f"{diag['residual_std']:.4f}")
                
                with col3:
                    normality_status = "✅ Yes" if diag['is_normal'] else "⚠️ No"
                    st.metric("Normal Distribution", normality_status)
                    st.caption(f"p-value: {diag['shapiro_p_value']:.4f}")
                
                with col4:
                    hetero_status = "⚠️ Yes" if diag['potential_heteroscedasticity'] else "✅ No"
                    st.metric("Heteroscedasticity", hetero_status)
                    st.caption(f"Variance ratio: {diag['variance_ratio']:.2f}")
                
                # Interpretation
                st.markdown("#### 🔍 Interpretation")
                
                if abs(diag['residual_mean']) < 0.01:
                    st.success("✅ Residual mean is close to zero - model is unbiased")
                else:
                    st.warning("⚠️ Residual mean deviates from zero - model may have systematic bias")
                
                if diag['is_normal']:
                    st.success("✅ Residuals are normally distributed - model assumptions satisfied")
                else:
                    st.info("ℹ️ Residuals are not normally distributed - consider model transformation")
                
                if not diag['potential_heteroscedasticity']:
                    st.success("✅ No signs of heteroscedasticity - variance is constant")
                else:
                    st.warning("⚠️ Potential heteroscedasticity detected - variance may not be constant")
            
            else:
                st.info("Train models first to see diagnostics")
        else:
            st.info("Train models first to see diagnostics")






            
# ========== PAGE: RAW DATA ==========

elif page == " Raw Data":
    st.title(" Raw Data Explorer")
    st.markdown("---")
    
    # Filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_cities = st.multiselect(
            "Filter by Cities",
            df['city_name'].unique(),
            default=df['city_name'].unique()
        )
    
    with col2:
        num_records = st.slider("Number of Records", 10, len(df), min(100, len(df)))
    
    # Filter data
    filtered_df = df[df['city_name'].isin(selected_cities)].head(num_records)
    
    # Display data
    st.subheader(f" Showing {len(filtered_df)} records")
    st.dataframe(filtered_df, width="stretch")
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label=" Download as CSV",
        data=csv,
        file_name="weather_data.csv",
        mime="text/csv"
    )
    
    # Database stats
    st.markdown("---")
    st.subheader(" Database Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Cities", df['city_name'].nunique())
    with col3:
        st.metric("Date Range", f"{(pd.to_datetime(df['timestamp'].max()) - pd.to_datetime(df['timestamp'].min())).days} days")
    with col4:
        st.metric("Variables", len(df.columns))
