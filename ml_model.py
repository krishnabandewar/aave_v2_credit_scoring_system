import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class CreditScoreModel:
    """
    Machine learning model for generating wallet credit scores based on transaction behavior
    """
    
    def __init__(self, model_type='random_forest', use_scaling=True, random_state=42):
        self.model_type = model_type
        self.use_scaling = use_scaling
        self.random_state = random_state
        self.scaler = None
        self.model = None
        self.pca = None
        self.feature_names = None
        
    def train_and_score(self, features_df: pd.DataFrame, test_size=0.2) -> tuple:
        """
        Train the model and generate credit scores
        
        Args:
            features_df: DataFrame with wallet features
            test_size: Proportion of data for testing
            
        Returns:
            tuple: (scores_df, metrics_dict, feature_importance)
        """
        # Prepare features
        X = features_df.drop('wallet', axis=1)
        self.feature_names = X.columns.tolist()
        
        # Handle missing values and infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Apply scaling if requested
        if self.use_scaling:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        # Create synthetic target variable using unsupervised methods
        target = self._create_target_variable(X_scaled)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, target, test_size=test_size, random_state=self.random_state
        )
        
        # Train model
        self.model = self._get_model()
        self.model.fit(X_train, y_train)
        
        # Generate predictions for all data
        predictions = self.model.predict(X_scaled)
        
        # Convert to credit scores (0-1000 scale)
        credit_scores = self._convert_to_credit_scores(predictions)
        
        # Create scores DataFrame
        scores_df = pd.DataFrame({
            'wallet': features_df['wallet'],
            'credit_score': credit_scores
        })
        
        # Calculate metrics
        metrics = self._calculate_metrics(X_scaled, target, y_test, self.model.predict(X_test))
        
        # Get feature importance
        feature_importance = self._get_feature_importance()
        
        return scores_df, metrics, feature_importance
    
    def _create_target_variable(self, X: pd.DataFrame) -> np.ndarray:
        """
        Create a synthetic target variable using unsupervised learning techniques
        """
        # Use PCA to capture the main patterns in the data
        self.pca = PCA(n_components=min(10, X.shape[1]), random_state=self.random_state)
        X_pca = self.pca.fit_transform(X)
        
        # Use K-means clustering to identify different behavior patterns
        kmeans = KMeans(n_clusters=5, random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(X_pca)
        
        # Create target based on cluster characteristics and key features
        target = np.zeros(len(X))
        
        for cluster_id in range(5):
            cluster_mask = clusters == cluster_id
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Score based on positive behavior indicators
            positive_indicators = []
            
            # Look for features that indicate good behavior
            for col in X.columns:
                if any(keyword in col.lower() for keyword in ['deposit', 'repay', 'diversity', 'frequency']):
                    if 'ratio' in col.lower() and 'liquidation' not in col.lower():
                        positive_indicators.append(cluster_data[col].mean())
                    elif 'liquidation' not in col.lower():
                        positive_indicators.append(cluster_data[col].mean())
            
            # Negative behavior indicators
            negative_indicators = []
            for col in X.columns:
                if any(keyword in col.lower() for keyword in ['liquidation', 'volatility', 'rapid', 'concentration']):
                    negative_indicators.append(cluster_data[col].mean())
            
            # Calculate cluster score
            positive_score = np.mean(positive_indicators) if positive_indicators else 0
            negative_score = np.mean(negative_indicators) if negative_indicators else 0
            
            # Combine scores (higher positive, lower negative = better score)
            cluster_score = positive_score - negative_score
            target[cluster_mask] = cluster_score
        
        # Normalize target to reasonable range
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        return target
    
    def _get_model(self):
        """Get the specified model"""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif self.model_type == 'ensemble':
            # Simple ensemble of multiple models
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=3,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            # Default to Random Forest
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def _convert_to_credit_scores(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert model predictions to credit scores (0-1000 scale)
        Higher scores indicate better creditworthiness
        """
        # Normalize predictions to 0-1 range
        min_pred = predictions.min()
        max_pred = predictions.max()
        normalized = (predictions - min_pred) / (max_pred - min_pred + 1e-8)
        
        # Convert to 0-1000 scale
        credit_scores = normalized * 1000
        
        # Ensure scores are within bounds
        credit_scores = np.clip(credit_scores, 0, 1000)
        
        return credit_scores
    
    def _calculate_metrics(self, X: pd.DataFrame, target: np.ndarray, y_test: np.ndarray, predictions_test: np.ndarray) -> dict:
        """Calculate model performance metrics"""
        metrics = {}
        
        # Clustering metrics for unsupervised evaluation
        try:
            # Use target as cluster labels for evaluation
            n_clusters = min(5, len(np.unique(target)))
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            metrics['silhouette_score'] = silhouette_score(X, cluster_labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, cluster_labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X, cluster_labels)
        except:
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz'] = 0.0
            metrics['davies_bouldin'] = 1.0
        
        # Prediction consistency metrics
        if len(y_test) > 0:
            mse = np.mean((y_test - predictions_test) ** 2)
            mae = np.mean(np.abs(y_test - predictions_test))
            metrics['mse'] = mse
            metrics['mae'] = mae
        
        return metrics
    
    def _get_feature_importance(self) -> pd.Series:
        """Get feature importance from the trained model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
        else:
            # For models without feature_importances_, return uniform importance
            importance = pd.Series(
                np.ones(len(self.feature_names)) / len(self.feature_names),
                index=self.feature_names
            )
        
        return importance
    
    def predict_credit_score(self, features: pd.DataFrame) -> np.ndarray:
        """Predict credit score for new wallet features"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Prepare features
        X = features.drop('wallet', axis=1) if 'wallet' in features.columns else features
        
        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)  # Use 0 for missing features in new data
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Convert to credit scores
        credit_scores = self._convert_to_credit_scores(predictions)
        
        return credit_scores
