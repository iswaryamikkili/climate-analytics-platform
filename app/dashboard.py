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
from src.anomaly_detection import AnomalyDetector
from src.data_ingestion import WeatherDataCollector

# Page configuration
st.set_page_config(
    page_title="Climate Analytics Platform",
    page_icon="🌤️",
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
        background-color: #f0f2f6;
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
    try:
        db = WeatherDatabase()
        with db.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM weather_data ORDER BY timestamp DESC", conn)
        return df
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()


def collect_new_data():
    """Collect fresh weather data"""
    with st.spinner("Collecting fresh weather data..."):
        try:
            collector = WeatherDataCollector()
            weather_df = collector.collect_all_cities()
            
            # Save to database
            db = WeatherDatabase()
            result = db.insert_weather_data(weather_df)
            
            if isinstance(result, dict):
                inserted = result.get('inserted', 0)
            else:
                inserted = result
            
            return inserted
        except Exception as e:
            st.error(f"Collection error: {str(e)}")
            return 0


# ========== SIDEBAR ==========

st.sidebar.title("🌤️ Climate Analytics")
st.sidebar.markdown("---")

# Data refresh button
if st.sidebar.button("🔄 Collect New Data", use_container_width=True):
    records = collect_new_data()
    st.sidebar.success(f"✅ Collected {records} new records!")
    st.cache_data.clear()  # Clear cache to reload data
    st.rerun()

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "City Comparison", "Time Series", 
     "Correlations", "Advanced Analysis", "Anomalies", "Raw Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Climate Analytics Platform**\n\n"
    "Real-time weather data analysis with advanced statistical insights and ML.\n\n"
    "Built with Python, Streamlit, Pandas, Plotly, and Scikit-learn."
)


# ========== LOAD DATA ==========

df = load_data()

if df.empty:
    st.error("❌ No data available!")
    st.info("👉 Click 'Collect New Data' in the sidebar to get started.")
    st.stop()

# Initialize analyzers
analyzer = WeatherAnalyzer(df)
detector = AnomalyDetector(df)

# ========== PAGE: OVERVIEW ==========

if page == "Overview":
    st.title("🌍 Weather Analytics Overview")
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
            f"{avg_temp:.1f}°C",
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
        st.subheader("🌡️ Current Temperature by City")
        
        # Get latest temperature for each city
        latest_temps = df.groupby('city_name').first().reset_index()
        
        fig = px.bar(
            latest_temps,
            x='city_name',
            y='temperature',
            color='temperature',
            color_continuous_scale='RdYlBu_r',
            labels={'temperature': 'Temperature (°C)', 'city_name': 'City'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💧 Current Humidity by City")
        
        fig = px.bar(
            latest_temps,
            x='city_name',
            y='humidity',
            color='humidity',
            color_continuous_scale='Blues',
            labels={'humidity': 'Humidity (%)', 'city_name': 'City'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Full width chart
    st.subheader("📊 Weather Conditions Across Cities")
    
    # Create summary DataFrame
    summary = df.groupby('city_name').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean',
        'pressure': 'mean'
    }).round(2).reset_index()
    
    # Display as table
    st.dataframe(summary, use_container_width=True)
    
    # Quick stats
    st.markdown("---")
    st.subheader("🏆 Quick Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🔥 Hottest City**")
        hottest = df.loc[df['temperature'].idxmax()]
        st.write(f"{hottest['city_name']}: {hottest['temperature']:.1f}°C")
    
    with col2:
        st.write("**❄️ Coldest City**")
        coldest = df.loc[df['temperature'].idxmin()]
        st.write(f"{coldest['city_name']}: {coldest['temperature']:.1f}°C")
    
    with col3:
        st.write("**💨 Windiest City**")
        windiest = df.loc[df['wind_speed'].idxmax()]
        st.write(f"{windiest['city_name']}: {windiest['wind_speed']:.1f} m/s")


# ========== PAGE: CITY COMPARISON ==========

elif page == "City Comparison":
    st.title("🏙️ City-by-City Comparison")
    st.markdown("---")
    
    # City selector
    cities = df['city_name'].unique().tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        city1 = st.selectbox("Select First City", cities, index=0)
    with col2:
        city2 = st.selectbox("Select Second City", cities, index=1 if len(cities) > 1 else 0)
    
    if city1 == city2:
        st.warning("⚠️ Please select different cities for comparison")
        st.stop()
    
    # Get data for selected cities
    city1_data = df[df['city_name'] == city1]
    city2_data = df[df['city_name'] == city2]
    
    # Comparison metrics
    st.subheader("📊 Key Metrics Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"{city1}",
            f"{city1_data['temperature'].mean():.1f}°C",
            delta=f"{city1_data['temperature'].mean() - city2_data['temperature'].mean():.1f}°C"
        )
    
    with col2:
        st.metric(
            f"{city2}",
            f"{city2_data['temperature'].mean():.1f}°C"
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
    
    # Statistical test - UPDATED SECTION
    st.subheader("🔬 Statistical Significance Test")
    
    test_result = analyzer.test_temperature_difference(city1, city2)
    
    if 'error' in test_result:
        st.error(test_result['error'])
    else:
        # Show which test was used
        st.info(f"**Test Used:** {test_result['test_used']}")
        
        # Main results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Difference", f"{test_result['difference']:.2f}°C")
        with col2:
            st.metric("P-value", f"{test_result['p_value']:.4f}")
        with col3:
            st.metric("Effect Size", 
                     f"{test_result['effect_size']:.3f}",
                     delta=test_result['effect_interpretation'])
        
        # Confidence interval
        st.write(f"**95% Confidence Interval:** [{test_result['ci_95_lower']:.2f}°C, {test_result['ci_95_upper']:.2f}°C]")
        
        # Interpretation
        if test_result['is_significant']:
            st.success(f"✅ {test_result['interpretation']}")
        else:
            st.info(f"ℹ️ {test_result['interpretation']}")
        
        # Show assumption checks (expandable)
        with st.expander("📋 View Statistical Assumptions"):
            st.write("**Normality Tests:**")
            st.write(f"- {city1}: p={test_result['assumptions']['normality_city1']['p_value']} "
                    f"({'✓ Met' if test_result['assumptions']['normality_city1']['met'] == 'yes' else '✗ Violated'})")
            st.write(f"- {city2}: p={test_result['assumptions']['normality_city2']['p_value']} "
                    f"({'✓ Met' if test_result['assumptions']['normality_city2']['met'] == 'yes' else '✗ Violated'})")
            
            st.write(f"\n**Equal Variance (Levene's Test):**")
            st.write(f"- p={test_result['assumptions']['equal_variance']['p_value']} "
                    f"({'✓ Met' if test_result['assumptions']['equal_variance']['met'] == 'yes' else '✗ Violated'})")
            
            st.info("💡 The test automatically selects the most appropriate method based on these assumptions.")
    
    # Visualization
    st.markdown("---")
    st.subheader("📈 Side-by-Side Comparison")
    
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
        subplot_titles=('Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)')
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
    st.plotly_chart(fig, use_container_width=True)


# ========== PAGE: TIME SERIES ==========

elif page == "Time Series":
    st.title("📈 Time Series Analysis")
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
    
    # Trend analysis - UPDATED SECTION
    st.subheader("📊 Trend Analysis")
    
    trend = analyzer.detect_trends(selected_city, variable)
    
    if 'error' in trend:
        st.warning(trend['error'])
    else:
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trend Direction", trend['trend_direction'].upper())
        with col2:
            st.metric("R² Score", f"{trend['r_squared']:.3f}")
        with col3:
            st.metric("Slope", f"{trend['slope']:.6f}")
        with col4:
            sig_icon = "✓" if trend['is_significant'] else "✗"
            st.metric("Significant", f"{sig_icon} (p={trend['p_value']:.3f})")
        
        # Interpretation
        st.info(f"**Interpretation:** {trend['interpretation']}")
        
        # Show diagnostics (expandable)
        if 'diagnostics' in trend and trend['diagnostics']:
            with st.expander("🔍 View Regression Diagnostics"):
                diag = trend['diagnostics']
                
                if 'normality' in diag:
                    st.write("**Normality of Residuals:**")
                    st.write(f"- {diag['normality']['interpretation']} (p={diag['normality']['p_value']})")
                
                if 'independence' in diag:
                    st.write("\n**Independence (No Autocorrelation):**")
                    st.write(f"- {diag['independence']['interpretation']} (DW={diag['independence']['statistic']})")
                
                if 'homoscedasticity' in diag:
                    st.write("\n**Homoscedasticity (Constant Variance):**")
                    st.write(f"- {diag['homoscedasticity']['interpretation']} (p={diag['homoscedasticity']['p_value']})")
                
                st.caption("✓ = Assumption met | ✗ = Assumption violated")
    
    # Time series plot
    st.subheader(f"📉 {variable.replace('_', ' ').title()} Over Time")
    
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent values table
    st.subheader("📋 Recent Values")
    recent = city_data.head(10)[['timestamp', variable]].copy()
    recent['timestamp'] = pd.to_datetime(recent['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(recent, use_container_width=True)


# ========== PAGE: CORRELATIONS ==========

elif page == "Correlations":
    st.title("🔗 Correlation Analysis")
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
    st.subheader(f"🔥 Correlation Heatmap{title_suffix}")
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        labels=dict(color="Correlation")
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Strong correlations
    st.subheader("💪 Strong Correlations")
    strong_corrs = analyzer.find_strong_correlations(threshold=0.5)
    
    if strong_corrs:
        for corr in strong_corrs:
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            with col1:
                st.write(f"**{corr['variable_1']}**")
            with col2:
                st.write(f"**{corr['variable_2']}**")
            with col3:
                st.write(f"**{corr['correlation']}**")
            with col4:
                sig = "✓" if corr['is_significant'] else "✗"
                st.write(f"**{sig}**")
    else:
        st.info("No strong correlations found (threshold: 0.5)")


# ========== PAGE: ADVANCED ANALYSIS ==========

elif page == "Advanced Analysis":
    st.title("🎯 Advanced Statistical Analysis")
    st.markdown("---")
    
    # Time Series Decomposition
    st.subheader("📊 Time Series Decomposition")
    st.caption("Separate trend, seasonal, and residual components")
    
    col1, col2 = st.columns(2)
    with col1:
        decomp_city = st.selectbox("Select City", df['city_name'].unique(), key='decomp')
    with col2:
        decomp_var = st.selectbox("Variable", ['temperature', 'humidity', 'wind_speed'], key='decomp_var')
    
    if st.button("🔍 Run Decomposition"):
        with st.spinner("Decomposing time series..."):
            result = analyzer.decompose_time_series(decomp_city, decomp_var, period=24)
            
            if 'error' in result:
                st.error(result['error'])
            else:
                st.success("✓ Decomposition complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Trend Strength", f"{result['trend_strength']:.3f}")
                with col2:
                    st.metric("Seasonal Strength", f"{result['seasonal_strength']:.3f}")
                with col3:
                    st.metric("Observations", result['n_observations'])
                
                st.info(f"**Interpretation:** {result['interpretation']}")
                
                st.caption("💡 Trend strength >0.6 = strong trend, >0.3 = moderate trend")
    
    st.markdown("---")
    
    # Principal Component Analysis
    st.subheader("🎯 Principal Component Analysis (PCA)")
    st.caption("Reduce dimensionality and identify most important features")
    
    n_components = st.slider("Number of Components", 2, 4, 3)
    
    if st.button("🔍 Run PCA"):
        with st.spinner("Performing PCA..."):
            pca_result = analyzer.perform_pca(n_components=n_components)
            
            if 'error' in pca_result:
                st.error(pca_result['error'])
            else:
                st.success("✓ PCA complete!")
                
                # Explained variance
                st.write("**Variance Explained by Each Component:**")
                var_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(n_components)],
                    'Variance Explained': [f"{v*100:.1f}%" for v in pca_result['explained_variance']],
                    'Cumulative': [f"{v*100:.1f}%" for v in pca_result['cumulative_variance']]
                })
                st.dataframe(var_df, use_container_width=True)
                
                # Feature importance
                st.write("**Feature Importance (First Component):**")
                importance_df = pd.DataFrame(
                    list(pca_result['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df, x='Feature', y='Importance',
                           title='Feature Importance in First Principal Component',
                           color='Importance',
                           color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"**Interpretation:** {pca_result['interpretation']}")
    
    st.markdown("---")
    
    # Feature Engineering Preview
    st.subheader("⚙️ Feature Engineering")
    st.caption("Generate advanced features for machine learning")
    
    city_for_features = st.selectbox("Select City", df['city_name'].unique(), key='features')
    
    if st.button("🔧 Generate Features"):
        with st.spinner("Engineering features..."):
            features_df = analyzer.engineer_features(city=city_for_features)
            
            st.success(f"✓ Generated {len(features_df.columns)} total features!")
            
            # Show sample
            st.write("**Sample of Engineered Features:**")
            feature_cols = [c for c in features_df.columns if c not in df.columns][:10]
            
            if feature_cols:
                st.dataframe(features_df[['timestamp'] + feature_cols].head(10), use_container_width=True)
                
                st.info(f"💡 Created {len(feature_cols)} new features including:\n"
                       f"- Lag features (past values)\n"
                       f"- Rolling statistics (trends)\n"
                       f"- Cyclical time encoding\n"
                       f"- Domain-specific features (heat index, wind chill)")
            else:
                st.warning("No new features were created")

# ========== PAGE: ANOMALIES ==========

elif page == "Anomalies":
    st.title("⚠️ Anomaly Detection")
    st.markdown("---")
    
    # User controls
    st.subheader("🔍 Detection Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_city = st.selectbox("Select City", df['city_name'].unique())
    
    with col2:
        variable = st.selectbox(
            "Select Variable",
            ['temperature', 'humidity', 'wind_speed', 'pressure']
        )
    
    with col3:
        method = st.radio(
            "Detection Method", 
            ["Z-Score", "IQR", "Isolation Forest (ML)"], 
            horizontal=True
        )
    
    # Show method explanation
    if method == "Z-Score":
        st.info("**Z-Score Method**: Detects values that are unusually far from the average (>3 standard deviations)")
        threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 3.0, 0.5)
    elif method == "IQR":
        st.info("**IQR Method**: Detects outliers using the box plot rule (outside 1.5×IQR from quartiles)")
        threshold = None
    else:
        st.info("**Isolation Forest (ML)**: Uses machine learning to detect anomalies based on multiple features")
        threshold = None
    
    st.markdown("---")
    
    # Detect anomalies - UPDATED SECTION
    if method == "Z-Score":
        anomalies = analyzer.detect_anomalies_zscore(selected_city, variable, threshold)
    elif method == "IQR":
        anomalies = analyzer.detect_anomalies_iqr(selected_city, variable)
    else:  # Isolation Forest
        st.caption("🤖 Using Machine Learning-based anomaly detection on multiple features")
        anomalies = analyzer.detect_anomalies_isolation_forest(selected_city, contamination=0.1)
    
    # Display results
    st.subheader("📊 Detection Results")
    
    total_points = len(df[df['city_name'] == selected_city])
    anomaly_count = len(anomalies)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Data Points", total_points)
    
    with col2:
        st.metric("Anomalies Detected", anomaly_count)
    
    with col3:
        percentage = (anomaly_count / total_points * 100) if total_points > 0 else 0
        st.metric("Anomaly Rate", f"{percentage:.1f}%")
    
    # Show anomalies if found
    if len(anomalies) > 0:
        st.warning(f"⚠️ Found {len(anomalies)} anomalous data points!")
        
        # Show anomalies table
        st.subheader("📋 Anomalous Records")
        anomalies_display = anomalies.copy()
        if 'timestamp' in anomalies_display.columns:
            anomalies_display['timestamp'] = pd.to_datetime(anomalies_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(anomalies_display, use_container_width=True)
        
        # Visualization
        st.subheader("📈 Visualization")
        
        city_data = df[df['city_name'] == selected_city].sort_values('timestamp')
        
        fig = go.Figure()
        
        # Normal data points
        fig.add_trace(go.Scatter(
            x=city_data['timestamp'],
            y=city_data[variable],
            mode='lines+markers',
            name='Normal Data',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        # Highlight anomalies
        if 'timestamp' in anomalies.columns and variable in anomalies.columns:
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies[variable],
                mode='markers',
                name='Anomalies',
                marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='darkred'))
            ))
        
        fig.update_layout(
            title=f"Anomaly Detection: {variable.title()} in {selected_city}",
            xaxis_title="Time",
            yaxis_title=variable.replace('_', ' ').title(),
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download anomalies
        csv = anomalies_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Anomalies as CSV",
            data=csv,
            file_name=f"anomalies_{selected_city}_{variable}.csv",
            mime="text/csv"
        )
        
    else:
        st.success(f"✅ No anomalies detected in {variable} for {selected_city}!")
        st.info("All data points fall within normal range.")





# ========== PAGE: RAW DATA ==========

elif page == "Raw Data":
    st.title("📁 Raw Data Explorer")
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_cities = st.multiselect(
            "Filter by Cities",
            df['city_name'].unique(),
            default=df['city_name'].unique()
        )
    
    with col2:
        num_records = st.slider("Number of Records", 10, min(500, len(df)), min(100, len(df)))
    
    # Filter data
    filtered_df = df[df['city_name'].isin(selected_cities)].head(num_records)
    
    # Display data
    st.subheader(f"📊 Showing {len(filtered_df)} records")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="weather_data.csv",
        mime="text/csv"
    )
    
    # Database stats
    st.markdown("---")
    st.subheader("💾 Database Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Cities", df['city_name'].nunique())
    with col3:
        if 'timestamp' in df.columns:
            date_range = (pd.to_datetime(df['timestamp'].max()) - pd.to_datetime(df['timestamp'].min())).days
            st.metric("Date Range (days)", f"{date_range}")
        else:
            st.metric("Date Range", "N/A")
    with col4:
        st.metric("Variables", len(df.columns))
