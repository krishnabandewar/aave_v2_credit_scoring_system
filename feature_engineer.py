import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

class FeatureEngineer:
    """
    Engineers features from processed Aave V2 transaction data for credit scoring
    """
    
    def __init__(self):
        self.feature_groups = {
            'basic_stats': [],
            'behavioral': [],
            'risk_metrics': [],
            'temporal': [],
            'network': []
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive features for each wallet
        
        Args:
            df: Processed transaction DataFrame
            
        Returns:
            pd.DataFrame: Feature matrix with one row per wallet
        """
        print("Engineering features for credit scoring...")
        
        # Initialize features DataFrame
        wallets = df['user'].unique()
        features_df = pd.DataFrame({'wallet': wallets})
        
        # Basic transaction statistics
        basic_features = self._create_basic_features(df)
        features_df = features_df.merge(basic_features, on='wallet', how='left')
        
        # Behavioral features
        behavioral_features = self._create_behavioral_features(df)
        features_df = features_df.merge(behavioral_features, on='wallet', how='left')
        
        # Risk metrics
        risk_features = self._create_risk_features(df)
        features_df = features_df.merge(risk_features, on='wallet', how='left')
        
        # Temporal features
        temporal_features = self._create_temporal_features(df)
        features_df = features_df.merge(temporal_features, on='wallet', how='left')
        
        # Network and diversity features
        network_features = self._create_network_features(df)
        features_df = features_df.merge(network_features, on='wallet', how='left')
        
        # Fill missing values with 0
        features_df = features_df.fillna(0)
        
        print(f"Generated {len(features_df.columns) - 1} features for {len(features_df)} wallets")
        
        return features_df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic transaction statistics features"""
        features = df.groupby('user').agg({
            'amount': ['count', 'sum', 'mean', 'std', 'median', 'min', 'max'],
            'log_amount': ['mean', 'std'],
            'action': lambda x: x.nunique(),
            'reserve': lambda x: x.nunique()
        }).round(6)
        
        # Flatten column names
        features.columns = [
            'total_transactions', 'total_volume', 'avg_transaction_amount', 
            'amount_std', 'median_amount', 'min_amount', 'max_amount',
            'avg_log_amount', 'log_amount_std', 'unique_actions', 'unique_reserves'
        ]
        
        features = features.reset_index().rename(columns={'user': 'wallet'})
        
        # Add derived ratios
        features['amount_coefficient_variation'] = features['amount_std'] / (features['avg_transaction_amount'] + 1e-8)
        features['max_to_median_ratio'] = features['max_amount'] / (features['median_amount'] + 1e-8)
        features['volume_per_transaction'] = features['total_volume'] / features['total_transactions']
        
        self.feature_groups['basic_stats'] = features.columns.tolist()[1:]
        return features
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features"""
        behavioral_features = []
        
        for wallet in df['user'].unique():
            wallet_df = df[df['user'] == wallet].copy()
            
            # Action distribution
            action_counts = wallet_df['action'].value_counts()
            total_actions = len(wallet_df)
            
            # Calculate action ratios
            deposit_ratio = action_counts.get('deposit', 0) / total_actions
            borrow_ratio = action_counts.get('borrow', 0) / total_actions
            repay_ratio = action_counts.get('repay', 0) / total_actions
            redeem_ratio = action_counts.get('redeemunderlying', 0) / total_actions
            liquidation_ratio = action_counts.get('liquidationcall', 0) / total_actions
            
            # Behavioral patterns
            borrow_to_repay_ratio = (action_counts.get('borrow', 0) + 1) / (action_counts.get('repay', 0) + 1)
            deposit_to_borrow_ratio = (action_counts.get('deposit', 0) + 1) / (action_counts.get('borrow', 0) + 1)
            
            # Time-based patterns
            if len(wallet_df) > 1:
                wallet_df_sorted = wallet_df.sort_values('timestamp')
                time_diffs = wallet_df_sorted['timestamp'].diff().dt.total_seconds() / 3600  # hours
                avg_time_between_tx = time_diffs.mean()
                std_time_between_tx = time_diffs.std()
                min_time_between_tx = time_diffs.min()
            else:
                avg_time_between_tx = std_time_between_tx = min_time_between_tx = 0
            
            # Activity concentration
            daily_activity = wallet_df.groupby('date').size()
            activity_concentration = (daily_activity ** 2).sum() / (daily_activity.sum() ** 2)
            
            behavioral_features.append({
                'wallet': wallet,
                'deposit_ratio': deposit_ratio,
                'borrow_ratio': borrow_ratio,
                'repay_ratio': repay_ratio,
                'redeem_ratio': redeem_ratio,
                'liquidation_ratio': liquidation_ratio,
                'borrow_to_repay_ratio': borrow_to_repay_ratio,
                'deposit_to_borrow_ratio': deposit_to_borrow_ratio,
                'avg_time_between_tx_hours': avg_time_between_tx,
                'std_time_between_tx_hours': std_time_between_tx,
                'min_time_between_tx_hours': min_time_between_tx,
                'activity_concentration': activity_concentration
            })
        
        behavioral_df = pd.DataFrame(behavioral_features)
        self.feature_groups['behavioral'] = behavioral_df.columns.tolist()[1:]
        return behavioral_df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk assessment features"""
        risk_features = []
        
        for wallet in df['user'].unique():
            wallet_df = df[df['user'] == wallet].copy()
            
            # Liquidation involvement
            liquidation_count = len(wallet_df[wallet_df['action'] == 'liquidationcall'])
            liquidation_volume = wallet_df[wallet_df['action'] == 'liquidationcall']['amount'].sum()
            
            # Amount volatility
            amount_volatility = wallet_df['amount'].std() / (wallet_df['amount'].mean() + 1e-8)
            
            # Large transaction indicator
            q95_amount = wallet_df['amount'].quantile(0.95)
            large_tx_count = len(wallet_df[wallet_df['amount'] >= q95_amount])
            large_tx_ratio = large_tx_count / len(wallet_df)
            
            # Reserve concentration (Herfindahl index)
            reserve_counts = wallet_df['reserve'].value_counts()
            reserve_shares = reserve_counts / reserve_counts.sum()
            reserve_concentration = (reserve_shares ** 2).sum()
            
            # Unusual timing patterns
            hourly_activity = wallet_df['hour'].value_counts()
            night_activity_ratio = hourly_activity.get(range(0, 6), pd.Series()).sum() / len(wallet_df)
            
            # Rapid transaction sequences
            if len(wallet_df) > 1:
                wallet_df_sorted = wallet_df.sort_values('timestamp')
                time_diffs = wallet_df_sorted['timestamp'].diff().dt.total_seconds()
                rapid_tx_count = len(time_diffs[time_diffs < 60])  # Less than 1 minute
                rapid_tx_ratio = rapid_tx_count / len(wallet_df)
            else:
                rapid_tx_ratio = 0
            
            risk_features.append({
                'wallet': wallet,
                'liquidation_count': liquidation_count,
                'liquidation_volume': liquidation_volume,
                'amount_volatility': amount_volatility,
                'large_tx_ratio': large_tx_ratio,
                'reserve_concentration': reserve_concentration,
                'night_activity_ratio': night_activity_ratio,
                'rapid_tx_ratio': rapid_tx_ratio
            })
        
        risk_df = pd.DataFrame(risk_features)
        self.feature_groups['risk_metrics'] = risk_df.columns.tolist()[1:]
        return risk_df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        temporal_features = []
        
        for wallet in df['user'].unique():
            wallet_df = df[df['user'] == wallet].copy()
            
            # Activity duration
            if len(wallet_df) > 1:
                activity_duration_days = (wallet_df['timestamp'].max() - wallet_df['timestamp'].min()).days
                activity_frequency = len(wallet_df) / max(activity_duration_days, 1)
            else:
                activity_duration_days = 0
                activity_frequency = 0
            
            # Day of week patterns
            weekday_activity = len(wallet_df[wallet_df['day_of_week'] < 5])
            weekend_activity = len(wallet_df[wallet_df['day_of_week'] >= 5])
            weekend_ratio = weekend_activity / len(wallet_df) if len(wallet_df) > 0 else 0
            
            # Monthly distribution
            monthly_counts = wallet_df['month'].value_counts()
            monthly_concentration = (monthly_counts / monthly_counts.sum() ** 2).sum()
            
            # Recent activity (last 30 days from max date in dataset)
            max_date = df['timestamp'].max()
            recent_cutoff = max_date - timedelta(days=30)
            recent_activity_count = len(wallet_df[wallet_df['timestamp'] > recent_cutoff])
            recent_activity_ratio = recent_activity_count / len(wallet_df)
            
            temporal_features.append({
                'wallet': wallet,
                'activity_duration_days': activity_duration_days,
                'activity_frequency': activity_frequency,
                'weekend_ratio': weekend_ratio,
                'monthly_concentration': monthly_concentration,
                'recent_activity_ratio': recent_activity_ratio
            })
        
        temporal_df = pd.DataFrame(temporal_features)
        self.feature_groups['temporal'] = temporal_df.columns.tolist()[1:]
        return temporal_df
    
    def _create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create network and diversity features"""
        network_features = []
        
        # Global statistics for normalization
        global_avg_amount = df['amount'].mean()
        global_reserve_counts = df['reserve'].value_counts()
        
        for wallet in df['user'].unique():
            wallet_df = df[df['user'] == wallet].copy()
            
            # Diversity metrics
            action_diversity = wallet_df['action'].nunique() / len(df['action'].unique())
            reserve_diversity = wallet_df['reserve'].nunique() / len(df['reserve'].unique())
            
            # Amount patterns relative to global
            relative_avg_amount = wallet_df['amount'].mean() / global_avg_amount
            
            # Popular reserve usage
            popular_reserves = global_reserve_counts.head(10).index
            popular_reserve_usage = len(wallet_df[wallet_df['reserve'].isin(popular_reserves)]) / len(wallet_df)
            
            # Transaction size distribution
            small_tx_ratio = len(wallet_df[wallet_df['amount'] < global_avg_amount * 0.1]) / len(wallet_df)
            large_tx_ratio = len(wallet_df[wallet_df['amount'] > global_avg_amount * 10]) / len(wallet_df)
            
            network_features.append({
                'wallet': wallet,
                'action_diversity': action_diversity,
                'reserve_diversity': reserve_diversity,
                'relative_avg_amount': relative_avg_amount,
                'popular_reserve_usage': popular_reserve_usage,
                'small_tx_ratio': small_tx_ratio,
                'large_tx_ratio_network': large_tx_ratio
            })
        
        network_df = pd.DataFrame(network_features)
        self.feature_groups['network'] = network_df.columns.tolist()[1:]
        return network_df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for analysis"""
        return self.feature_groups
