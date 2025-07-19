import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import List, Dict, Any

class DataProcessor:
    """
    Processes raw Aave V2 transaction data into a structured DataFrame
    """
    
    def __init__(self):
        self.required_fields = ['user', 'reserve', 'action', 'amount', 'timestamp']
        self.valid_actions = ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']
    
    def process_transactions(self, raw_data: List[Dict[Any, Any]]) -> pd.DataFrame:
        """
        Process raw transaction data into a clean DataFrame
        
        Args:
            raw_data: List of transaction dictionaries
            
        Returns:
            pd.DataFrame: Processed transaction data
        """
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Basic data validation
        self._validate_data(df)
        
        # Clean and standardize data
        df = self._clean_data(df)
        
        # Convert data types
        df = self._convert_data_types(df)
        
        # Add derived fields
        df = self._add_derived_fields(df)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate that required fields are present"""
        missing_fields = [field for field in self.required_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Check for valid actions
        invalid_actions = df[~df['action'].isin(self.valid_actions)]['action'].unique()
        if len(invalid_actions) > 0:
            print(f"Warning: Found invalid actions: {invalid_actions}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data"""
        # Remove rows with missing critical data
        df = df.dropna(subset=['user', 'action', 'amount'])
        
        # Standardize action names
        df['action'] = df['action'].str.lower().str.strip()
        
        # Clean user addresses (remove '0x' prefix if present for consistency)
        df['user'] = df['user'].astype(str).str.lower().str.strip()
        
        # Clean reserve addresses
        df['reserve'] = df['reserve'].astype(str).str.lower().str.strip()
        
        # Remove negative amounts (invalid transactions) -- MOVED to _convert_data_types
        # df = df[df['amount'] > 0]
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data to appropriate types"""
        # Convert amount to float
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        # Remove rows with invalid or negative amounts
        df = df[df['amount'] > 0]
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            # Check if timestamp is numeric (unix timestamp)
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            else:
                # Try to parse as datetime string
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Remove rows with invalid timestamps or amounts
        df = df.dropna(subset=['timestamp', 'amount'])
        
        return df
    
    def _add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived fields for analysis"""
        # Add date components
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Add amount in log scale for better distribution
        df['log_amount'] = np.log1p(df['amount'])
        
        # Add amount quartiles
        df['amount_quartile'] = pd.qcut(df['amount'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Sort by user and timestamp for sequential analysis
        df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of the processed data"""
        return {
            'total_transactions': len(df),
            'unique_users': df['user'].nunique(),
            'unique_reserves': df['reserve'].nunique(),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'action_distribution': df['action'].value_counts().to_dict(),
            'amount_stats': {
                'mean': df['amount'].mean(),
                'median': df['amount'].median(),
                'std': df['amount'].std(),
                'min': df['amount'].min(),
                'max': df['amount'].max()
            }
        }
