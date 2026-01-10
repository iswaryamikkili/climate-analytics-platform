"""
Advanced Anomaly Detection Module
Implements multiple ML-based anomaly detection algorithms with proper validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time

# ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyMethod(Enum):
    """Available anomaly detection methods"""
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    DBSCAN = "dbscan"


@dataclass
class AnomalyResult:
    """Structure for anomaly detection results"""
    method: str
    anomalies: pd.DataFrame
    scores: np.ndarray
    threshold: float
    n_anomalies: int
    percentage: float
    execution_time_ms: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy display"""
        return {
            'method': self.method,
            'n_anomalies': self.n_anomalies,
            'percentage': round(self.percentage, 2),
            'execution_time_ms': round(self.execution_time_ms, 2)
        }
    
    def __repr__(self) -> str:
        return (f"AnomalyResult(method={self.method}, "
                f"n_anomalies={self.n_anomalies}, "
                f"percentage={self.percentage:.2f}%)")


class AnomalyDetector:
    """
    Advanced anomaly detection with multiple ML algorithms.
    
    Implements:
    - Statistical methods (Z-score, IQR)
    - ML-based methods (Isolation Forest, LOF, One-Class SVM, Elliptic Envelope)
    - Ensemble methods (voting-based consensus)
    - Temporal anomaly detection
    
    Features:
    - Automatic method selection based on data characteristics
    - Performance tracking and comparison
    - Robust error handling
    - Method recommendations
    """
    
    def __init__(self, df: pd.DataFrame, contamination: float = 0.1):
        """
        Initialize detector with weather data
        
        Args:
            df: DataFrame with weather data
            contamination: Expected proportion of anomalies (0.01-0.5)
                          0.1 = expect 10% of data to be anomalies
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        
        self.df = df.copy()
        self.contamination = contamination
        self.scaler = StandardScaler()
        
        # Prepare data
        self._prepare_data()
        
        logger.info(f"Initialized AnomalyDetector with {len(self.df)} records, "
                   f"contamination={contamination}")
    
    def _prepare_data(self):
        """Prepare and validate data for anomaly detection"""
        # Convert timestamp
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Identify numeric features for ML methods
        self.numeric_features = [
            'temperature', 'feels_like', 'humidity', 
            'wind_speed', 'pressure', 'cloudiness'
        ]
        
        # Filter to available features
        self.numeric_features = [
            f for f in self.numeric_features if f in self.df.columns
        ]
        
        if not self.numeric_features:
            logger.warning("No numeric features found for anomaly detection")
    
    # ========== STATISTICAL METHODS ==========
    
    def detect_zscore(self, city: str, variable: str = 'temperature', 
                     threshold: float = 3.0) -> AnomalyResult:
        """
        Z-score anomaly detection with performance metrics
        
        How it works:
        - Calculate mean (μ) and standard deviation (σ)
        - For each value x: z = (x - μ) / σ
        - Values with |z| > threshold are anomalies
        
        Best for:
        - Normally distributed data
        - Quick detection
        - Single variable analysis
        
        Args:
            city: City name
            variable: Variable to check
            threshold: Z-score threshold (typically 2-4)
                      2.0 = 95% confidence, 3.0 = 99.7% confidence
            
        Returns:
            AnomalyResult object with detailed results
        """
        start = time.time()
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 3:
            logger.warning(f"Insufficient data for {city}: need at least 3 samples")
            return self._empty_result('Z-Score')
        
        # Calculate z-scores
        mean = city_data[variable].mean()
        std = city_data[variable].std()
        
        if std == 0:
            logger.warning(f"Zero standard deviation for {variable} in {city}")
            return self._empty_result('Z-Score')
        
        z_scores = (city_data[variable] - mean) / std
        is_anomaly = np.abs(z_scores) > threshold
        
        anomalies = city_data[is_anomaly].copy()
        anomalies['anomaly_score'] = z_scores[is_anomaly].values
        
        execution_time = (time.time() - start) * 1000
        
        logger.info(f"Z-score detection: {len(anomalies)} anomalies in {execution_time:.2f}ms")
        
        return AnomalyResult(
            method='Z-Score',
            anomalies=anomalies[['timestamp', 'city_name', variable, 'anomaly_score']],
            scores=z_scores.values,
            threshold=threshold,
            n_anomalies=len(anomalies),
            percentage=(len(anomalies) / len(city_data)) * 100,
            execution_time_ms=execution_time
        )
    
    def detect_iqr(self, city: str, variable: str = 'temperature',
                   multiplier: float = 1.5) -> AnomalyResult:
        """
        IQR outlier detection with configurable sensitivity
        
        How it works:
        - Calculate Q1 (25th percentile) and Q3 (75th percentile)
        - IQR = Q3 - Q1
        - Lower bound = Q1 - multiplier × IQR
        - Upper bound = Q3 + multiplier × IQR
        - Values outside bounds are outliers
        
        Best for:
        - Skewed distributions
        - Robust to extreme values
        - Visual "box plot" method
        
        Args:
            city: City name
            variable: Variable to check
            multiplier: IQR multiplier
                       1.5 = mild outliers (default)
                       3.0 = extreme outliers only
            
        Returns:
            AnomalyResult object
        """
        start = time.time()
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 4:
            logger.warning(f"Insufficient data for IQR: need at least 4 samples")
            return self._empty_result('IQR')
        
        Q1 = city_data[variable].quantile(0.25)
        Q3 = city_data[variable].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        is_anomaly = (city_data[variable] < lower_bound) | (city_data[variable] > upper_bound)
        anomalies = city_data[is_anomaly].copy()
        
        # Calculate outlier scores (distance from bounds)
        anomalies['anomaly_score'] = anomalies[variable].apply(
            lambda x: abs(x - lower_bound) if x < lower_bound else abs(x - upper_bound)
        )
        
        execution_time = (time.time() - start) * 1000
        
        logger.info(f"IQR detection: {len(anomalies)} anomalies in {execution_time:.2f}ms")
        
        return AnomalyResult(
            method='IQR',
            anomalies=anomalies[['timestamp', 'city_name', variable, 'anomaly_score']],
            scores=np.zeros(len(city_data)),  # Placeholder
            threshold=multiplier,
            n_anomalies=len(anomalies),
            percentage=(len(anomalies) / len(city_data)) * 100,
            execution_time_ms=execution_time
        )
    
    # ========== MACHINE LEARNING METHODS ==========
    
    def detect_isolation_forest(self, city: str, 
                                contamination: Optional[float] = None) -> AnomalyResult:
        """
        Isolation Forest - Efficient tree-based anomaly detection
        
        How it works:
        - Builds random decision trees
        - Anomalies are isolated with fewer splits
        - Measures average path length to isolate each point
        - Shorter paths = more anomalous
        
        Best for:
        - Large datasets (scales well)
        - High-dimensional data
        - No assumption about data distribution
        - Fast training and prediction
        
        Args:
            city: City name
            contamination: Expected anomaly proportion (uses class default if None)
            
        Returns:
            AnomalyResult with anomalies and scores
        """
        start = time.time()
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 10:
            logger.warning(f"Insufficient data for Isolation Forest: need at least 10 samples")
            return self._empty_result('Isolation Forest')
        
        # Prepare features
        X = city_data[self.numeric_features].values
        
        # Handle missing values
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Standardize features (important for ML models)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        contamination_rate = contamination or self.contamination
        model = IsolationForest(
            contamination=contamination_rate,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1  # Use all CPU cores
        )
        
        predictions = model.fit_predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        
        # Extract anomalies (-1 = anomaly, 1 = normal)
        is_anomaly = predictions == -1
        anomalies = city_data[is_anomaly].copy()
        anomalies['anomaly_score'] = anomaly_scores[is_anomaly]
        
        execution_time = (time.time() - start) * 1000
        
        logger.info(f"Isolation Forest: {len(anomalies)} anomalies in {execution_time:.2f}ms")
        
        return AnomalyResult(
            method='Isolation Forest',
            anomalies=anomalies[['timestamp', 'city_name'] + self.numeric_features + ['anomaly_score']],
            scores=anomaly_scores,
            threshold=-0.1,  # Typical IF threshold
            n_anomalies=len(anomalies),
            percentage=(len(anomalies) / len(city_data)) * 100,
            execution_time_ms=execution_time
        )
    
    def detect_local_outlier_factor(self, city: str,
                                   n_neighbors: int = 20) -> AnomalyResult:
        """
        Local Outlier Factor - Density-based anomaly detection
        
        How it works:
        - Compares local density of each point to its neighbors
        - Calculates Local Reachability Density (LRD)
        - Points in sparse regions have low density = anomalies
        - Good for finding local outliers in clusters
        
        Best for:
        - Data with clusters of varying density
        - Detecting local anomalies
        - Spatial/geographic data
        
        Args:
            city: City name
            n_neighbors: Number of neighbors to consider (typically 10-30)
            
        Returns:
            AnomalyResult with anomalies
        """
        start = time.time()
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < n_neighbors + 1:
            logger.warning(f"Insufficient data: need at least {n_neighbors + 1} samples")
            return self._empty_result('Local Outlier Factor')
        
        # Prepare features
        X = city_data[self.numeric_features].values
        
        # Handle missing values
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Train LOF
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=False  # For training data anomaly detection
        )
        
        predictions = lof.fit_predict(X_scaled)
        anomaly_scores = lof.negative_outlier_factor_
        
        # Extract anomalies
        is_anomaly = predictions == -1
        anomalies = city_data[is_anomaly].copy()
        anomalies['anomaly_score'] = anomaly_scores[is_anomaly]
        
        execution_time = (time.time() - start) * 1000
        
        logger.info(f"LOF: {len(anomalies)} anomalies in {execution_time:.2f}ms")
        
        return AnomalyResult(
            method='Local Outlier Factor',
            anomalies=anomalies[['timestamp', 'city_name'] + self.numeric_features + ['anomaly_score']],
            scores=anomaly_scores,
            threshold=-1.5,
            n_anomalies=len(anomalies),
            percentage=(len(anomalies) / len(city_data)) * 100,
            execution_time_ms=execution_time
        )
    
    def detect_one_class_svm(self, city: str, kernel: str = 'rbf') -> AnomalyResult:
        """
        One-Class SVM - Boundary-based anomaly detection
        
        How it works:
        - Learns a decision boundary around normal data
        - Uses kernel trick to map data to higher dimensions
        - Points outside the boundary are anomalies
        - Creates hyperplane in feature space
        
        Best for:
        - Well-defined normal region
        - Non-linear decision boundaries
        - High-dimensional feature spaces
        
        Note: Computationally expensive for large datasets
        
        Args:
            city: City name
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
                   'rbf' (Radial Basis Function) is most common
            
        Returns:
            AnomalyResult with anomalies
        """
        start = time.time()
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        if len(city_data) < 10:
            logger.warning("Insufficient data for One-Class SVM")
            return self._empty_result('One-Class SVM')
        
        # Prepare features
        X = city_data[self.numeric_features].values
        
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Train One-Class SVM
        svm = OneClassSVM(
            nu=self.contamination,  # nu ≈ upper bound on fraction of outliers
            kernel=kernel,
            gamma='auto'
        )
        
        predictions = svm.fit_predict(X_scaled)
        anomaly_scores = svm.decision_function(X_scaled)
        
        # Extract anomalies
        is_anomaly = predictions == -1
        anomalies = city_data[is_anomaly].copy()
        anomalies['anomaly_score'] = anomaly_scores[is_anomaly]
        
        execution_time = (time.time() - start) * 1000
        
        logger.info(f"One-Class SVM: {len(anomalies)} anomalies in {execution_time:.2f}ms")
        
        return AnomalyResult(
            method='One-Class SVM',
            anomalies=anomalies[['timestamp', 'city_name'] + self.numeric_features + ['anomaly_score']],
            scores=anomaly_scores,
            threshold=0.0,
            n_anomalies=len(anomalies),
            percentage=(len(anomalies) / len(city_data)) * 100,
            execution_time_ms=execution_time
        )
    
    def detect_elliptic_envelope(self, city: str) -> AnomalyResult:
        """
        Elliptic Envelope - Gaussian assumption-based detection
        
        How it works:
        - Assumes data follows multivariate Gaussian distribution
        - Fits robust covariance estimate (Minimum Covariance Determinant)
        - Computes Mahalanobis distance to center
        - Points far from center are anomalies
        
        Best for:
        - Data that is approximately Gaussian
        - Detecting global outliers
        - When you have domain knowledge of normal distribution
        
        Note: Sensitive to non-Gaussian data
        
        Args:
            city: City name
            
        Returns:
            AnomalyResult with anomalies
        """
        start = time.time()
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        # Need enough samples for covariance estimation
        if len(city_data) < len(self.numeric_features) * 2:
            logger.warning(f"Insufficient data for Elliptic Envelope: "
                          f"need at least {len(self.numeric_features) * 2} samples")
            return self._empty_result('Elliptic Envelope')
        
        # Prepare features
        X = city_data[self.numeric_features].values
        
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Elliptic Envelope
        try:
            envelope = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
            
            predictions = envelope.fit_predict(X_scaled)
            anomaly_scores = envelope.decision_function(X_scaled)
            
            # Extract anomalies
            is_anomaly = predictions == -1
            anomalies = city_data[is_anomaly].copy()
            anomalies['anomaly_score'] = anomaly_scores[is_anomaly]
            
            execution_time = (time.time() - start) * 1000
            
            logger.info(f"Elliptic Envelope: {len(anomalies)} anomalies in {execution_time:.2f}ms")
            
            return AnomalyResult(
                method='Elliptic Envelope',
                anomalies=anomalies[['timestamp', 'city_name'] + self.numeric_features + ['anomaly_score']],
                scores=anomaly_scores,
                threshold=0.0,
                n_anomalies=len(anomalies),
                percentage=(len(anomalies) / len(city_data)) * 100,
                execution_time_ms=execution_time
            )
        except Exception as e:
            logger.error(f"Elliptic Envelope failed: {e}")
            return self._empty_result('Elliptic Envelope')
    
    # ========== ENSEMBLE METHODS ==========
    
    def detect_ensemble(self, city: str, 
                       methods: Optional[List[str]] = None,
                       voting_threshold: int = 2) -> Dict:
        """
        Ensemble anomaly detection - combines multiple methods
        
        How it works:
        - Runs multiple anomaly detection methods
        - Each method "votes" on whether a point is an anomaly
        - Points flagged by ≥ voting_threshold methods are consensus anomalies
        - More robust than any single method
        
        Best for:
        - High-confidence anomaly detection
        - Reducing false positives
        - Production systems
        
        Args:
            city: City name
            methods: List of methods to use (None = use all fast methods)
            voting_threshold: Minimum votes to classify as anomaly (2-3 recommended)
            
        Returns:
            Dictionary with ensemble results and consensus anomalies
        """
        logger.info(f"Running ensemble detection for {city}...")
        
        if methods is None:
            # Use fast, reliable methods by default
            methods = ['zscore', 'iqr', 'isolation_forest', 'local_outlier_factor']
        
        city_data = self.df[self.df['city_name'] == city].copy()
        
        # Run all methods
        results = {}
        anomaly_indices = {}
        
        for method in methods:
            try:
                if method == 'zscore':
                    result = self.detect_zscore(city)
                elif method == 'iqr':
                    result = self.detect_iqr(city)
                elif method == 'isolation_forest':
                    result = self.detect_isolation_forest(city)
                elif method == 'local_outlier_factor':
                    result = self.detect_local_outlier_factor(city)
                elif method == 'one_class_svm':
                    result = self.detect_one_class_svm(city)
                elif method == 'elliptic_envelope':
                    result = self.detect_elliptic_envelope(city)
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue
                
                results[method] = result
                anomaly_indices[method] = set(result.anomalies.index)
                
            except Exception as e:
                logger.error(f"Method {method} failed: {e}")
                continue
        
        if not results:
            return {
                'error': 'All methods failed',
                'individual_results': {},
                'consensus_anomalies': pd.DataFrame(),
                'n_consensus': 0
            }
        
        # Voting: count how many methods flagged each point
        all_indices = set()
        for indices in anomaly_indices.values():
            all_indices.update(indices)
        
        vote_counts = {}
        for idx in all_indices:
            votes = sum(1 for indices in anomaly_indices.values() if idx in indices)
            vote_counts[idx] = votes
        
        # Get high-confidence anomalies (consensus)
        consensus_indices = [idx for idx, votes in vote_counts.items() 
                           if votes >= voting_threshold]
        
        consensus_anomalies = city_data.loc[consensus_indices].copy()
        consensus_anomalies['vote_count'] = [vote_counts[idx] for idx in consensus_indices]
        consensus_anomalies['max_votes'] = len(methods)
        
        logger.info(f"Ensemble: {len(consensus_anomalies)} consensus anomalies "
                   f"(threshold: {voting_threshold}/{len(methods)} votes)")
        
        return {
            'individual_results': {
                name: result.to_dict() for name, result in results.items()
            },
            'consensus_anomalies': consensus_anomalies,
            'n_consensus': len(consensus_anomalies),
            'consensus_percentage': (len(consensus_anomalies) / len(city_data)) * 100,
            'voting_threshold': voting_threshold,
            'methods_used': methods,
            'vote_distribution': vote_counts
        }
    
    # ========== EVALUATION & COMPARISON ==========
    
    def compare_methods(self, city: str) -> pd.DataFrame:
        """
        Compare all anomaly detection methods
        
        Runs all available methods and compares:
        - Number of anomalies detected
        - Percentage of data flagged
        - Execution time
        - Success/failure status
        
        Args:
            city: City name
            
        Returns:
            DataFrame with method comparison metrics
        """
        logger.info(f"Comparing anomaly detection methods for {city}...")
        
        methods_to_test = [
            ('Z-Score', lambda: self.detect_zscore(city)),
            ('IQR', lambda: self.detect_iqr(city)),
            ('Isolation Forest', lambda: self.detect_isolation_forest(city)),
            ('Local Outlier Factor', lambda: self.detect_local_outlier_factor(city)),
            ('One-Class SVM', lambda: self.detect_one_class_svm(city)),
            ('Elliptic Envelope', lambda: self.detect_elliptic_envelope(city))
        ]
        
        comparison_data = []
        
        for method_name, method_func in methods_to_test:
            try:
                result = method_func()
                comparison_data.append({
                    'Method': method_name,
                    'Anomalies': result.n_anomalies,
                    'Percentage': f"{result.percentage:.2f}%",
                    'Execution Time (ms)': f"{result.execution_time_ms:.2f}",
                    'Status': '✓ Success'
                })
            except Exception as e:
                comparison_data.append({
                    'Method': method_name,
                    'Anomalies': 0,
                    'Percentage': '0.00%',
                    'Execution Time (ms)': 'N/A',
                    'Status': f'✗ Failed: {str(e)[:30]}'
                })
        
        return pd.DataFrame(comparison_data)
    
    def get_method_recommendations(self, city: str) -> Dict:
        """
        Recommend best methods based on data characteristics
        
        Analyzes:
        - Sample size
        - Number of features
        - Data distribution
        
        Returns:
            Dictionary with recommendations and reasoning
        """
        city_data = self.df[self.df['city_name'] == city]
        n_samples = len(city_data)
        n_features = len(self.numeric_features)
        
        recommendations = {
            'data_characteristics': {
                'n_samples': n_samples,
                'n_features': n_features,
                'sample_to_feature_ratio': n_samples / n_features if n_features > 0 else 0
            },
            'recommended_methods': []
        }
        
        # Recommendation logic based on data size
        if n_samples < 30:
            recommendations['recommended_methods'].append({
                'method': 'Z-Score or IQR',
                'reason': 'Small dataset - use simple statistical methods',
                'priority': 'High'
            })
        elif n_samples < 100:
            recommendations['recommended_methods'].append({
                'method': 'Isolation Forest',
                'reason': 'Medium dataset - efficient and robust ML method',
                'priority': 'High'
            })
            recommendations['recommended_methods'].append({
                'method': 'Local Outlier Factor',
                'reason': 'Good for detecting local anomalies in medium datasets',
                'priority': 'Medium'
            })
        else:
            recommendations['recommended_methods'].append({
                'method': 'Ensemble (Isolation Forest + LOF + Z-Score)',
                'reason': 'Large dataset - combine multiple ML methods for robust detection',
                'priority': 'High'
            })
            recommendations['recommended_methods'].append({
                'method': 'Isolation Forest',
                'reason': 'Scales well to large datasets, fast execution',
                'priority': 'High'
            })
        
        # Additional recommendations
        if n_features >= 4:
            recommendations['recommended_methods'].append({
                'method': 'PCA + Isolation Forest',
                'reason': 'Multiple features - dimensionality reduction may help',
                'priority': 'Low'
            })
        
        return recommendations
    
    def get_summary(self, city: str, variable: str = 'temperature') -> Dict:
        """
        Get summary of anomalies for a city across multiple methods
        
        Args:
            city: City name
            variable: Variable to analyze
            
        Returns:
            Dictionary with anomaly statistics
        """
        zscore_anomalies = self.detect_zscore(city, variable)
        iqr_anomalies = self.detect_iqr(city, variable)
        
        total_points = len(self.df[self.df['city_name'] == city])
        
        return {
            'total_data_points': total_points,
            'zscore_anomalies': zscore_anomalies.n_anomalies,
            'iqr_outliers': iqr_anomalies.n_anomalies,
            'zscore_percentage': zscore_anomalies.percentage,
            'iqr_percentage': iqr_anomalies.percentage,
            'zscore_execution_ms': zscore_anomalies.execution_time_ms,
            'iqr_execution_ms': iqr_anomalies.execution_time_ms
        }
    
    # ========== HELPER METHODS ==========
    
    def _empty_result(self, method: str) -> AnomalyResult:
        """Return empty result for failed detection"""
        return AnomalyResult(
            method=method,
            anomalies=pd.DataFrame(),
            scores=np.array([]),
            threshold=0.0,
            n_anomalies=0,
            percentage=0.0,
            execution_time_ms=0.0
        )


# ========== USAGE EXAMPLE ==========

def main():
    """Test anomaly detection functionality"""
    from database import WeatherDatabase
    
    print("=" * 70)
    print("ADVANCED ANOMALY DETECTION MODULE")
    print("=" * 70)
    
    # Load data
    db = WeatherDatabase()
    with db.get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM weather_data", conn)
    
    if df.empty:
        print("\n❌ No data available for analysis")
        print("Run data collection first: python src/data_ingestion.py")
        return
    
    print(f"\n✓ Loaded {len(df)} records for analysis")
    
    # Initialize detector
    detector = AnomalyDetector(df, contamination=0.1)
    
    # Get first city
    cities = df['city_name'].unique()
    if len(cities) == 0:
        print("No cities found in data")
        return
    
    city = cities[0]
    print(f"\n🔍 Testing Anomaly Detection on {city}")
    print("=" * 70)
    
    # Get recommendations
    print("\n💡 Method Recommendations:")
    recommendations = detector.get_method_recommendations(city)
    print(f"   Data: {recommendations['data_characteristics']['n_samples']} samples, "
          f"{recommendations['data_characteristics']['n_features']} features")
    for rec in recommendations['recommended_methods']:
        print(f"   [{rec['priority']}] {rec['method']}")
        print(f"       → {rec['reason']}")
    
    # Test statistical methods
    print("\n📊 Statistical Methods:")
    print("-" * 70)
    
    # Z-score
    print("\n1️⃣  Z-Score Method (threshold=3):")
    zscore_result = detector.detect_zscore(city, 'temperature', threshold=3)
    print(f"   Anomalies: {zscore_result.n_anomalies} ({zscore_result.percentage:.2f}%)")
    print(f"   Execution time: {zscore_result.execution_time_ms:.2f}ms")
    if len(zscore_result.anomalies) > 0:
        print(f"   Sample anomalies:\n{zscore_result.anomalies.head(3)}")
    
    # IQR
    print("\n2️⃣  IQR Method:")
    iqr_result = detector.detect_iqr(city, 'temperature')
    print(f"   Outliers: {iqr_result.n_anomalies} ({iqr_result.percentage:.2f}%)")
    print(f"   Execution time: {iqr_result.execution_time_ms:.2f}ms")
    
    # Test ML methods
    print("\n🤖 Machine Learning Methods:")
    print("-" * 70)
    
    # Isolation Forest
    print("\n3️⃣  Isolation Forest:")
    if_result = detector.detect_isolation_forest(city)
    print(f"   Anomalies: {if_result.n_anomalies} ({if_result.percentage:.2f}%)")
    print(f"   Execution time: {if_result.execution_time_ms:.2f}ms")
    
    # Local Outlier Factor
    print("\n4️⃣  Local Outlier Factor:")
    lof_result = detector.detect_local_outlier_factor(city, n_neighbors=20)
    print(f"   Anomalies: {lof_result.n_anomalies} ({lof_result.percentage:.2f}%)")
    print(f"   Execution time: {lof_result.execution_time_ms:.2f}ms")
    
    # One-Class SVM
    print("\n5️⃣  One-Class SVM:")
    svm_result = detector.detect_one_class_svm(city)
    print(f"   Anomalies: {svm_result.n_anomalies} ({svm_result.percentage:.2f}%)")
    print(f"   Execution time: {svm_result.execution_time_ms:.2f}ms")
    
    # Elliptic Envelope
    print("\n6️⃣  Elliptic Envelope:")
    ee_result = detector.detect_elliptic_envelope(city)
    print(f"   Anomalies: {ee_result.n_anomalies} ({ee_result.percentage:.2f}%)")
    print(f"   Execution time: {ee_result.execution_time_ms:.2f}ms")
    
    # Method comparison
    print("\n📈 Method Comparison:")
    print("-" * 70)
    comparison = detector.compare_methods(city)
    print(comparison.to_string(index=False))
    
    # Ensemble detection
    print("\n🎯 Ensemble Detection (Voting Threshold = 2):")
    print("-" * 70)
    ensemble_result = detector.detect_ensemble(city, voting_threshold=2)
    
    if 'error' not in ensemble_result:
        print(f"   Consensus anomalies: {ensemble_result['n_consensus']} "
              f"({ensemble_result['consensus_percentage']:.2f}%)")
        print(f"   Methods used: {', '.join(ensemble_result['methods_used'])}")
        
        if len(ensemble_result['consensus_anomalies']) > 0:
            print(f"\n   High-confidence anomalies (flagged by ≥2 methods):")
            print(ensemble_result['consensus_anomalies'][['timestamp', 'temperature', 'vote_count', 'max_votes']].head())
    else:
        print(f"   Error: {ensemble_result['error']}")
    
    # Summary
    print("\n📋 Summary:")
    print("-" * 70)
    summary = detector.get_summary(city, 'temperature')
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Anomaly detection complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
