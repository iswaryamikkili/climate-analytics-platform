# Climate Analytics Platform

An end-to-end data pipeline for climate data analysis with statistical modeling and interactive visualizations.

## 🎯 Project Overview

This project demonstrates skills in:
- **Data Engineering**: Automated ETL pipeline with API integration
- **Statistical Analysis**: Time series analysis, trend detection, correlation studies
- **Data Visualization**: Interactive dashboard with real-time insights
- **Cloud Deployment**: AWS infrastructure with scheduled data collection
- **Software Engineering**: Clean code, testing, documentation, version control

## 🏗️ Architecture
```
[OpenWeather API] → [Data Ingestion] → [SQLite Database] → [Statistical Analysis] → [Streamlit Dashboard]
```

## 📊 Features

- Multi-city climate data collection
- Real-time weather monitoring
- Statistical trend analysis
- Anomaly detection
- Interactive visualizations
- Automated data pipeline

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **Data Processing**: Pandas, NumPy
- **Statistical Analysis**: SciPy, Statsmodels
- **Visualization**: Plotly, Streamlit
- **Database**: SQLite (local) / PostgreSQL (production)
- **Cloud**: AWS (EC2, RDS, S3)
- **Version Control**: Git, GitHub

## 📁 Project Structure
```
climate-analytics-platform/
├── data/               # Data storage
├── src/                # Source code modules
├── app/                # Dashboard application
├── tests/              # Unit tests
├── config/             # Configuration files
├── notebooks/          # Jupyter notebooks for EDA
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenWeatherMap API key (free tier)

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/climate-analytics-platform.git
cd climate-analytics-platform

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp config/config.example.yaml config/config.yaml
# Add your API key to config/config.yaml
```

## 📈 Usage

(Coming soon - will be updated as project develops)

## 🧪 Testing
```bash
pytest tests/
```

## 📝 Development Progress

- [x] Project setup and structure
- [ ] Data ingestion module
- [ ] Database schema and setup
- [ ] Data processing pipeline
- [ ] Statistical analysis module
- [ ] Dashboard development
- [ ] AWS deployment
- [ ] Documentation

## 👨‍💻 Author

Iswarya Mikkili
- Masters in Computer Science, University of Cincinnati
- [LinkedIn](https://www.linkedin.com/in/iswaryamikkili/)
- [GitHub](https://github.com/iswaryamikkili)

## 📄 License

This project is open source and available for educational purposes.


This project is part of my portfolio demonstrating data engineering, statistical analysis, and cloud deployment skills.
