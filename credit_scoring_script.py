#!/usr/bin/env python3
"""
One-step Aave V2 Wallet Credit Scoring Script

This script processes raw Aave V2 transaction data and generates wallet credit scores (0-1000).
Usage: python credit_scoring_script.py [--input INPUT_FILE] [--output OUTPUT_FILE]

Example:
    python credit_scoring_script.py --input user-transactions.json --output wallet_scores.csv
    python credit_scoring_script.py  # Uses default Google Drive download
"""

import argparse
import json
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import our custom modules
from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
from ml_model import CreditScoreModel
from analyzer import WalletAnalyzer
from utils import download_file_from_google_drive, validate_transaction_data

def flatten_transactions(data):
    """Ensure required fields like 'amount' are at the top level, copying from actionData if needed."""
    for tx in data:
        # Flatten 'amount' from actionData if not present
        if 'amount' not in tx or tx['amount'] in [None, '', 0]:
            if 'actionData' in tx and isinstance(tx['actionData'], dict):
                amt = tx['actionData'].get('amount')
                if amt is not None:
                    tx['amount'] = amt
        # Optionally flatten 'user' and 'reserve' if needed
        if 'user' not in tx:
            for k in tx.keys():
                if 'user' in k.lower() or 'wallet' in k.lower():
                    tx['user'] = tx[k]
                    break
        if 'reserve' not in tx:
            if 'actionData' in tx and isinstance(tx['actionData'], dict):
                reserve = tx['actionData'].get('assetSymbol') or tx['actionData'].get('reserve')
                if reserve is not None:
                    tx['reserve'] = reserve
    return data

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate wallet credit scores from Aave V2 transaction data'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input JSON file with transaction data'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str,
        default='wallet_credit_scores.csv',
        help='Output CSV file for credit scores (default: wallet_credit_scores.csv)'
    )
    parser.add_argument(
        '--analysis', '-a',
        type=str,
        default='analysis_report.md',
        help='Output markdown file for analysis report (default: analysis_report.md)'
    )
    parser.add_argument(
        '--model-type', '-m',
        type=str,
        default='random_forest',
        choices=['random_forest', 'gradient_boosting', 'xgboost', 'ensemble'],
        help='ML model type to use (default: random_forest)'
    )
    parser.add_argument(
        '--download-zip',
        action='store_true',
        help='Download compressed ZIP file instead of raw JSON from Google Drive'
    )
    
    args = parser.parse_args()
    
    print("🚀 Aave V2 Wallet Credit Scoring System")
    print("=" * 50)
    
    # Step 1: Load Data
    print("\n📊 Step 1: Loading transaction data...")
    data = load_transaction_data(args.input, args.download_zip)
    
    if not data:
        print("❌ Failed to load transaction data. Exiting.")
        sys.exit(1)
    
    print(f"✅ Loaded {len(data)} transactions")
    
    # Step 2: Process Data
    print("\n🔧 Step 2: Processing and cleaning data...")
    processor = DataProcessor()
    df = processor.process_transactions(data)
    print(f"✅ Processed data: {len(df)} transactions from {df['user'].nunique()} unique wallets")
    
    # Step 3: Feature Engineering
    print("\n⚙️ Step 3: Engineering features...")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.engineer_features(df)
    print(f"✅ Generated {len(features_df.columns) - 1} features for {len(features_df)} wallets")
    
    # Step 4: Train Model and Generate Scores
    print(f"\n🤖 Step 4: Training {args.model_type} model and generating credit scores...")
    model = CreditScoreModel(model_type=args.model_type.replace('-', '_'))
    scores_df, metrics, feature_importance = model.train_and_score(features_df)
    
    print(f"✅ Generated credit scores for {len(scores_df)} wallets")
    print(f"   - Average score: {scores_df['credit_score'].mean():.1f}")
    print(f"   - Score range: {scores_df['credit_score'].min():.1f} - {scores_df['credit_score'].max():.1f}")
    
    # Step 5: Analysis
    print("\n📈 Step 5: Generating comprehensive analysis...")
    analyzer = WalletAnalyzer()
    analysis_results = analyzer.analyze_wallets(scores_df, df)
    
    # Step 6: Save Results
    print(f"\n💾 Step 6: Saving results...")
    
    # Save credit scores
    scores_df.to_csv(args.output, index=False)
    print(f"✅ Credit scores saved to: {args.output}")
    
    # Save analysis report
    analysis_report = analyzer.generate_analysis_report(analysis_results, scores_df)
    with open(args.analysis, 'w') as f:
        f.write(analysis_report)
    print(f"✅ Analysis report saved to: {args.analysis}")
    
    # Save detailed results
    detailed_output = args.output.replace('.csv', '_detailed.json')
    save_detailed_results(scores_df, analysis_results, metrics, feature_importance, detailed_output)
    print(f"✅ Detailed results saved to: {detailed_output}")
    
    # Step 7: Summary
    print("\n📋 Summary:")
    print(f"   - Total wallets analyzed: {len(scores_df)}")
    print(f"   - Average credit score: {scores_df['credit_score'].mean():.1f}")
    print(f"   - Model performance (Silhouette): {metrics.get('silhouette_score', 0):.3f}")
    
    # Display score distribution
    ranges = [(0, 100), (100, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
    labels = ["Very Low Risk", "Low Risk", "Med-Low Risk", "Medium Risk", "Med-High Risk", "High Risk"]
    
    print(f"\n📊 Score Distribution:")
    for (min_score, max_score), label in zip(ranges, labels):
        count = len(scores_df[(scores_df['credit_score'] >= min_score) & (scores_df['credit_score'] < max_score)])
        percentage = count / len(scores_df) * 100
        print(f"   - {label} ({min_score}-{max_score}): {count} wallets ({percentage:.1f}%)")
    
    print(f"\n🎉 Credit scoring completed successfully!")
    print(f"📁 Output files: {args.output}, {args.analysis}, {detailed_output}")

def load_transaction_data(input_file: Optional[str], download_zip: bool = False) -> Optional[List[Dict[Any, Any]]]:
    """Load transaction data from file or Google Drive"""
    
    if input_file:
        # Load from local file
        print(f"Loading data from local file: {input_file}")
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            data = flatten_transactions(data)  # PATCH: flatten fields for validation
            if validate_transaction_data(data):
                return data
            else:
                print("❌ Data validation failed")
                return None
                
        except Exception as e:
            print(f"❌ Error loading local file: {str(e)}")
            return None
    
    else:
        # Download from Google Drive
        print("No input file specified. Downloading from Google Drive...")
        
        if download_zip:
            print("Downloading compressed ZIP file...")
            file_id = "14ceBCLQ-BTcydDrFJauVA_PKAZ7VtDor"
            data = download_file_from_google_drive(file_id, file_type="zip")
        else:
            print("Downloading raw JSON file...")
            file_id = "1ISFbAXxadMrt7Zl96rmzzZmEKZnyW7FS"
            data = download_file_from_google_drive(file_id, file_type="json")
        
        if data and validate_transaction_data(data):
            return data
        else:
            print("❌ Failed to download or validate data from Google Drive")
            return None

def save_detailed_results(scores_df: pd.DataFrame, analysis_results: Dict[str, Any], 
                         metrics: Dict[str, float], feature_importance: pd.Series, 
                         output_file: str) -> None:
    """Save detailed results in JSON format"""
    
    # Convert pandas objects to JSON-serializable format
    detailed_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_wallets': len(scores_df),
            'score_statistics': {
                'mean': float(scores_df['credit_score'].mean()),
                'median': float(scores_df['credit_score'].median()),
                'std': float(scores_df['credit_score'].std()),
                'min': float(scores_df['credit_score'].min()),
                'max': float(scores_df['credit_score'].max())
            }
        },
        'model_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                         for k, v in metrics.items()},
        'feature_importance': {
            str(k): float(v) for k, v in feature_importance.head(20).items()
        },
        'wallet_scores': scores_df.to_dict('records'),
        'analysis_summary': {
            'score_distribution': analysis_results.get('score_distribution', {}),
            'insights': analysis_results.get('insights', [])
        }
    }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)

if __name__ == "__main__":
    main()