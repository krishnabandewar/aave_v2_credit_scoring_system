import pandas as pd
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class WalletAnalyzer:
    """
    Analyzes wallet credit scores and generates comprehensive insights
    """
    
    def __init__(self):
        self.score_ranges = [
            (0, 100, "Very Low Risk"),
            (100, 200, "Low Risk"),
            (200, 400, "Medium-Low Risk"),
            (400, 600, "Medium Risk"),
            (600, 800, "Medium-High Risk"),
            (800, 1000, "High Risk")
        ]
    
    def analyze_wallets(self, scores_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of wallet scores and behaviors
        
        Args:
            scores_df: DataFrame with wallet credit scores
            transactions_df: Original transaction data
            
        Returns:
            Dict with analysis results
        """
        analysis_results = {}
        
        # Score distribution analysis
        analysis_results['score_distribution'] = self._analyze_score_distribution(scores_df)
        
        # High vs low scoring wallet behavior
        analysis_results['high_score_behavior'] = self._analyze_wallet_behavior(
            scores_df, transactions_df, score_range=(800, 1000)
        )
        analysis_results['low_score_behavior'] = self._analyze_wallet_behavior(
            scores_df, transactions_df, score_range=(0, 200)
        )
        
        # Risk category analysis
        analysis_results['risk_categories'] = self._analyze_risk_categories(scores_df, transactions_df)
        
        # Generate insights
        analysis_results['insights'] = self._generate_insights(analysis_results, scores_df, transactions_df)
        
        return analysis_results
    
    def _analyze_score_distribution(self, scores_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the distribution of credit scores"""
        distribution = {}
        
        # Basic statistics
        distribution['mean'] = scores_df['credit_score'].mean()
        distribution['median'] = scores_df['credit_score'].median()
        distribution['std'] = scores_df['credit_score'].std()
        distribution['min'] = scores_df['credit_score'].min()
        distribution['max'] = scores_df['credit_score'].max()
        
        # Score range counts
        range_counts = {}
        for min_score, max_score, label in self.score_ranges:
            count = len(scores_df[
                (scores_df['credit_score'] >= min_score) & 
                (scores_df['credit_score'] < max_score)
            ])
            percentage = count / len(scores_df) * 100
            range_counts[f"{min_score}-{max_score}"] = {
                'count': count,
                'percentage': percentage,
                'label': label
            }
        
        distribution['range_counts'] = range_counts
        
        # Percentiles
        distribution['percentiles'] = {
            'p10': scores_df['credit_score'].quantile(0.1),
            'p25': scores_df['credit_score'].quantile(0.25),
            'p75': scores_df['credit_score'].quantile(0.75),
            'p90': scores_df['credit_score'].quantile(0.9)
        }
        
        return distribution
    
    def _analyze_wallet_behavior(self, scores_df: pd.DataFrame, transactions_df: pd.DataFrame, 
                                score_range: tuple) -> Dict[str, float]:
        """Analyze behavior patterns for wallets in a specific score range"""
        min_score, max_score = score_range
        
        # Get wallets in the score range
        range_wallets = scores_df[
            (scores_df['credit_score'] >= min_score) & 
            (scores_df['credit_score'] < max_score)
        ]['wallet'].tolist()
        
        if not range_wallets:
            return {}
        
        # Filter transactions for these wallets
        range_transactions = transactions_df[transactions_df['user'].isin(range_wallets)]
        
        if len(range_transactions) == 0:
            return {}
        
        # Calculate behavioral metrics
        behavior = {}
        
        # Transaction patterns
        behavior['avg_transactions_per_wallet'] = len(range_transactions) / len(range_wallets)
        behavior['avg_transaction_amount'] = range_transactions['amount'].mean()
        behavior['median_transaction_amount'] = range_transactions['amount'].median()
        behavior['total_volume'] = range_transactions['amount'].sum()
        
        # Action distribution
        action_counts = range_transactions['action'].value_counts()
        total_actions = len(range_transactions)
        
        for action in ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']:
            behavior[f'{action}_ratio'] = action_counts.get(action, 0) / total_actions
        
        # Risk indicators
        behavior['liquidation_involvement'] = (
            range_transactions['action'] == 'liquidationcall'
        ).sum() / len(range_wallets)
        
        # Diversity metrics
        behavior['avg_unique_reserves_per_wallet'] = (
            range_transactions.groupby('user')['reserve'].nunique().mean()
        )
        behavior['avg_unique_actions_per_wallet'] = (
            range_transactions.groupby('user')['action'].nunique().mean()
        )
        
        # Time patterns
        if len(range_transactions) > 1:
            # Activity duration
            wallet_durations = []
            for wallet in range_wallets:
                wallet_tx = range_transactions[range_transactions['user'] == wallet]
                if len(wallet_tx) > 1:
                    duration = (wallet_tx['timestamp'].max() - wallet_tx['timestamp'].min()).days
                    wallet_durations.append(duration)
            
            behavior['avg_activity_duration_days'] = np.mean(wallet_durations) if wallet_durations else 0
            
            # Transaction frequency
            behavior['avg_daily_frequency'] = behavior['avg_transactions_per_wallet'] / max(
                behavior['avg_activity_duration_days'], 1
            )
        
        # Amount volatility
        wallet_volatilities = []
        for wallet in range_wallets:
            wallet_tx = range_transactions[range_transactions['user'] == wallet]
            if len(wallet_tx) > 1:
                volatility = wallet_tx['amount'].std() / (wallet_tx['amount'].mean() + 1e-8)
                wallet_volatilities.append(volatility)
        
        behavior['avg_amount_volatility'] = np.mean(wallet_volatilities) if wallet_volatilities else 0
        
        return behavior
    
    def _analyze_risk_categories(self, scores_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of different risk categories"""
        risk_analysis = {}
        
        for min_score, max_score, label in self.score_ranges:
            category_wallets = scores_df[
                (scores_df['credit_score'] >= min_score) & 
                (scores_df['credit_score'] < max_score)
            ]
            
            if len(category_wallets) == 0:
                continue
            
            category_behavior = self._analyze_wallet_behavior(
                scores_df, transactions_df, (min_score, max_score)
            )
            
            risk_analysis[label] = {
                'wallet_count': len(category_wallets),
                'percentage': len(category_wallets) / len(scores_df) * 100,
                'avg_score': category_wallets['credit_score'].mean(),
                'behavior_patterns': category_behavior
            }
        
        return risk_analysis
    
    def _generate_insights(self, analysis_results: Dict[str, Any], scores_df: pd.DataFrame, 
                          transactions_df: pd.DataFrame) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []
        
        # Score distribution insights
        score_dist = analysis_results['score_distribution']
        insights.append(
            f"The average credit score is {score_dist['mean']:.1f} with a standard deviation of {score_dist['std']:.1f}"
        )
        
        # Range distribution insights
        range_counts = score_dist['range_counts']
        largest_category = max(range_counts.items(), key=lambda x: x[1]['count'])
        insights.append(
            f"The largest group is {largest_category[1]['label']} ({largest_category[0]}) with {largest_category[1]['percentage']:.1f}% of wallets"
        )
        
        # High vs low risk comparison
        if 'high_score_behavior' in analysis_results and 'low_score_behavior' in analysis_results:
            high_behavior = analysis_results['high_score_behavior']
            low_behavior = analysis_results['low_score_behavior']
            
            if high_behavior and low_behavior:
                # Transaction patterns
                if high_behavior.get('avg_transactions_per_wallet', 0) > low_behavior.get('avg_transactions_per_wallet', 0):
                    insights.append("High-scoring wallets tend to have more transactions than low-scoring wallets")
                
                # Liquidation involvement
                high_liquidation = high_behavior.get('liquidation_involvement', 0)
                low_liquidation = low_behavior.get('liquidation_involvement', 0)
                if low_liquidation > high_liquidation:
                    insights.append("Low-scoring wallets have higher liquidation involvement")
                
                # Diversity
                high_diversity = high_behavior.get('avg_unique_reserves_per_wallet', 0)
                low_diversity = low_behavior.get('avg_unique_reserves_per_wallet', 0)
                if high_diversity > low_diversity:
                    insights.append("High-scoring wallets interact with more diverse reserves")
                
                # Volatility
                high_volatility = high_behavior.get('avg_amount_volatility', 0)
                low_volatility = low_behavior.get('avg_amount_volatility', 0)
                if low_volatility > high_volatility:
                    insights.append("Low-scoring wallets show higher transaction amount volatility")
        
        # Risk category insights
        if 'risk_categories' in analysis_results:
            risk_cats = analysis_results['risk_categories']
            
            # Find dominant risk category
            dominant_cat = max(risk_cats.items(), key=lambda x: x[1]['wallet_count'])
            insights.append(
                f"Most wallets ({dominant_cat[1]['percentage']:.1f}%) fall into the '{dominant_cat[0]}' category"
            )
            
            # Extreme categories
            very_low_risk = risk_cats.get('Very Low Risk', {})
            high_risk = risk_cats.get('High Risk', {})
            
            if very_low_risk:
                insights.append(
                    f"{very_low_risk['percentage']:.1f}% of wallets are classified as Very Low Risk"
                )
            
            if high_risk:
                insights.append(
                    f"{high_risk['percentage']:.1f}% of wallets are classified as High Risk"
                )
        
        return insights
    
    def generate_analysis_report(self, analysis_results: Dict[str, Any], scores_df: pd.DataFrame) -> str:
        """Generate a comprehensive analysis report in markdown format"""
        report = []
        
        # Header
        report.append("# Aave V2 Wallet Credit Scoring Analysis")
        report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nTotal wallets analyzed: {len(scores_df)}")
        
        # Executive Summary
        report.append("\n## Executive Summary")
        score_dist = analysis_results['score_distribution']
        report.append(f"\n- **Average Credit Score**: {score_dist['mean']:.1f}")
        report.append(f"- **Median Credit Score**: {score_dist['median']:.1f}")
        report.append(f"- **Score Range**: {score_dist['min']:.1f} - {score_dist['max']:.1f}")
        report.append(f"- **Standard Deviation**: {score_dist['std']:.1f}")
        
        # Score Distribution
        report.append("\n## Score Distribution by Ranges")
        report.append("\n| Range | Label | Count | Percentage |")
        report.append("|-------|-------|-------|------------|")
        
        for range_key, range_data in score_dist['range_counts'].items():
            report.append(
                f"| {range_key} | {range_data['label']} | {range_data['count']} | {range_data['percentage']:.1f}% |"
            )
        
        # Key Insights
        report.append("\n## Key Insights")
        insights = analysis_results.get('insights', [])
        for insight in insights:
            report.append(f"\n- {insight}")
        
        # High-Scoring Wallet Behavior
        if 'high_score_behavior' in analysis_results:
            report.append("\n## High-Scoring Wallet Behavior (800-1000)")
            high_behavior = analysis_results['high_score_behavior']
            
            if high_behavior:
                report.append(f"\n- **Average Transactions per Wallet**: {high_behavior.get('avg_transactions_per_wallet', 0):.2f}")
                report.append(f"- **Average Transaction Amount**: {high_behavior.get('avg_transaction_amount', 0):.2f}")
                report.append(f"- **Deposit Ratio**: {high_behavior.get('deposit_ratio', 0):.3f}")
                report.append(f"- **Borrow Ratio**: {high_behavior.get('borrow_ratio', 0):.3f}")
                report.append(f"- **Repay Ratio**: {high_behavior.get('repay_ratio', 0):.3f}")
                report.append(f"- **Liquidation Involvement**: {high_behavior.get('liquidation_involvement', 0):.3f}")
                report.append(f"- **Average Unique Reserves**: {high_behavior.get('avg_unique_reserves_per_wallet', 0):.2f}")
        
        # Low-Scoring Wallet Behavior
        if 'low_score_behavior' in analysis_results:
            report.append("\n## Low-Scoring Wallet Behavior (0-200)")
            low_behavior = analysis_results['low_score_behavior']
            
            if low_behavior:
                report.append(f"\n- **Average Transactions per Wallet**: {low_behavior.get('avg_transactions_per_wallet', 0):.2f}")
                report.append(f"- **Average Transaction Amount**: {low_behavior.get('avg_transaction_amount', 0):.2f}")
                report.append(f"- **Deposit Ratio**: {low_behavior.get('deposit_ratio', 0):.3f}")
                report.append(f"- **Borrow Ratio**: {low_behavior.get('borrow_ratio', 0):.3f}")
                report.append(f"- **Repay Ratio**: {low_behavior.get('repay_ratio', 0):.3f}")
                report.append(f"- **Liquidation Involvement**: {low_behavior.get('liquidation_involvement', 0):.3f}")
                report.append(f"- **Average Unique Reserves**: {low_behavior.get('avg_unique_reserves_per_wallet', 0):.2f}")
        
        # Risk Category Analysis
        if 'risk_categories' in analysis_results:
            report.append("\n## Risk Category Analysis")
            
            for category, data in analysis_results['risk_categories'].items():
                report.append(f"\n### {category}")
                report.append(f"- **Wallet Count**: {data['wallet_count']}")
                report.append(f"- **Percentage**: {data['percentage']:.1f}%")
                report.append(f"- **Average Score**: {data['avg_score']:.1f}")
        
        # Methodology
        report.append("\n## Methodology")
        report.append("\nThe credit scoring model uses machine learning techniques to analyze wallet behavior patterns:")
        report.append("\n1. **Feature Engineering**: Extracted behavioral, temporal, and risk metrics from transaction data")
        report.append("2. **Unsupervised Learning**: Used clustering and PCA to identify behavior patterns")
        report.append("3. **Score Generation**: Applied machine learning models to generate scores on a 0-1000 scale")
        report.append("4. **Validation**: Used clustering metrics and behavior analysis for validation")
        
        report.append("\n## Conclusion")
        report.append("\nThe credit scoring system successfully differentiates between different types of wallet behaviors,")
        report.append("providing a reliable measure of creditworthiness based on historical transaction patterns.")
        
        return "\n".join(report)
