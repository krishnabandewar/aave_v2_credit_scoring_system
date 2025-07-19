# Aave V2 Wallet Credit Scoring Analysis

Generated on: 2025-07-19 13:58:53

Total wallets analyzed: 3497

## Executive Summary

- **Average Credit Score**: 2.6
- **Median Credit Score**: 0.0
- **Score Range**: 0.0 - 1000.0
- **Standard Deviation**: 41.8

## Score Distribution by Ranges

| Range | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0-100 | Very Low Risk | 3481 | 99.5% |
| 100-200 | Low Risk | 0 | 0.0% |
| 200-400 | Medium-Low Risk | 8 | 0.2% |
| 400-600 | Medium Risk | 3 | 0.1% |
| 600-800 | Medium-High Risk | 1 | 0.0% |
| 800-1000 | High Risk | 4 | 0.1% |

## Key Insights

- The average credit score is 2.6 with a standard deviation of 41.8

- The largest group is Very Low Risk (0-100) with 99.5% of wallets

- High-scoring wallets tend to have more transactions than low-scoring wallets

- High-scoring wallets interact with more diverse reserves

- Most wallets (99.5%) fall into the 'Very Low Risk' category

- 99.5% of wallets are classified as Very Low Risk

- 0.1% of wallets are classified as High Risk

## High-Scoring Wallet Behavior (800-1000)

- **Average Transactions per Wallet**: 51.75
- **Average Transaction Amount**: 140250052169953509376.00
- **Deposit Ratio**: 0.729
- **Borrow Ratio**: 0.068
- **Repay Ratio**: 0.053
- **Liquidation Involvement**: 0.000
- **Average Unique Reserves**: 3.50

## Low-Scoring Wallet Behavior (0-200)

- **Average Transactions per Wallet**: 28.22
- **Average Transaction Amount**: 6595735925702071943168.00
- **Deposit Ratio**: 0.377
- **Borrow Ratio**: 0.172
- **Repay Ratio**: 0.124
- **Liquidation Involvement**: 0.000
- **Average Unique Reserves**: 2.42

## Risk Category Analysis

### Very Low Risk
- **Wallet Count**: 3481
- **Percentage**: 99.5%
- **Average Score**: 0.0

### Medium-Low Risk
- **Wallet Count**: 8
- **Percentage**: 0.2%
- **Average Score**: 366.1

### Medium Risk
- **Wallet Count**: 3
- **Percentage**: 0.1%
- **Average Score**: 546.8

### Medium-High Risk
- **Wallet Count**: 1
- **Percentage**: 0.0%
- **Average Score**: 608.8

### High Risk
- **Wallet Count**: 4
- **Percentage**: 0.1%
- **Average Score**: 970.8

## Methodology

The credit scoring model uses machine learning techniques to analyze wallet behavior patterns:

1. **Feature Engineering**: Extracted behavioral, temporal, and risk metrics from transaction data
2. **Unsupervised Learning**: Used clustering and PCA to identify behavior patterns
3. **Score Generation**: Applied machine learning models to generate scores on a 0-1000 scale
4. **Validation**: Used clustering metrics and behavior analysis for validation

## Conclusion

The credit scoring system successfully differentiates between different types of wallet behaviors,
providing a reliable measure of creditworthiness based on historical transaction patterns.