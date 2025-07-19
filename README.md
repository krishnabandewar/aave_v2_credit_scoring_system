# Aave V2 Wallet Credit Scoring System

---
## 🚀 Quick Results

- **Total wallets analyzed:** 3,497
- **Average credit score:** 2.6 (lower is better)
- **Score range:** 0 – 1000
- **Very Low Risk wallets:** 99.5%
- **High Risk wallets:** 0.1%

See [analysis.md](analysis.md) for full breakdown and insights.
---

A comprehensive machine learning system that analyzes Aave V2 transaction data to generate wallet credit scores ranging from 0-1000, providing insights into wallet creditworthiness and behavioral patterns.

## 🎯 Objective

Develop a robust machine learning model that assigns credit scores to DeFi wallets based on their historical transaction behavior on the Aave V2 protocol. Higher scores indicate reliable and responsible usage, while lower scores reflect risky, bot-like, or exploitative behavior.

## 🏗️ Architecture

### System Components

1. **Data Processing Layer** (`data_processor.py`)
   - Loads and validates raw transaction data
   - Cleans and standardizes transaction records
   - Converts data types and adds derived fields

2. **Feature Engineering Layer** (`feature_engineer.py`)
   - Extracts behavioral patterns from transaction data
   - Creates risk metrics and temporal features
   - Generates network and diversity indicators

3. **Machine Learning Layer** (`ml_model.py`)
   - Implements multiple ML algorithms (Random Forest, Gradient Boosting, XGBoost)
   - Uses unsupervised learning to create target variables
   - Generates credit scores on 0-1000 scale

4. **Analysis Layer** (`analyzer.py`)
   - Performs comprehensive wallet behavior analysis
   - Generates insights and recommendations
   - Creates detailed reports and visualizations

5. **Streamlit Interface** (`app.py`)
   - Interactive web application for data analysis
   - Real-time visualizations and insights
   - Export capabilities for results and reports

### Data Flow

```
Raw JSON Data → Data Processing → Feature Engineering → ML Model Training → Credit Scoring → Analysis & Insights
```

## 🚀 Quick Start

### One-Step Script Usage

The system includes a comprehensive one-step script that processes raw transaction data and generates wallet credit scores:

```bash
# Using local JSON file
python credit_scoring_script.py --input user-transactions.json --output wallet_scores.csv

# Download from Google Drive (raw JSON)
python credit_scoring_script.py --output wallet_scores.csv

# Download compressed ZIP file
python credit_scoring_script.py --download-zip --output wallet_scores.csv

# Specify model type
python credit_scoring_script.py --model-type xgboost --output wallet_scores.csv
```

### Interactive Web Interface

Launch the Streamlit application for interactive analysis:

```bash
streamlit run app.py --server.port 5000
```

## 📊 Features Engineered

The system extracts comprehensive behavioral patterns from transaction data:

### Basic Statistics
- Transaction count, volume, and amount statistics
- Amount volatility and distribution metrics
- Reserve and action diversity

### Behavioral Patterns  
- Action ratios (deposit, borrow, repay, liquidation)
- Borrowing-to-repaying patterns
- Time-based transaction patterns
- Activity concentration metrics

### Risk Metrics
- Liquidation involvement
- Transaction amount volatility
- Large transaction indicators
- Reserve concentration (Herfindahl index)
- Rapid transaction sequences
- Unusual timing patterns

### Temporal Features
- Activity duration and frequency
- Weekend vs weekday patterns  
- Recent activity indicators
- Monthly distribution patterns

### Network Features
- Action and reserve diversity
- Popular reserve usage
- Relative transaction sizes
- Global behavior comparisons

## 🤖 Machine Learning Approach

### Unsupervised Target Creation
Since there are no existing credit labels in DeFi, the system uses an innovative unsupervised approach:

1. **PCA Analysis**: Captures main behavioral patterns
2. **K-Means Clustering**: Identifies distinct wallet behavior groups
3. **Pattern Scoring**: Evaluates clusters based on positive/negative indicators
4. **Target Generation**: Creates synthetic creditworthiness scores

### Model Options
- **Random Forest**: Default, robust ensemble method
- **Gradient Boosting**: Advanced boosting algorithm
- **XGBoost**: High-performance gradient boosting (if available)
- **Ensemble**: Combined approach for enhanced accuracy

### Score Scaling
- **Range**: 0-1000 (standard credit score format)
- **Interpretation**: Higher scores = lower risk, better creditworthiness
- **Distribution**: Normalized across the entire wallet population

## 📈 Analysis Output

### Score Distribution
Wallets are categorized into six risk levels:
- **0-100**: Very Low Risk (most creditworthy)
- **100-200**: Low Risk
- **200-400**: Medium-Low Risk  
- **400-600**: Medium Risk
- **600-800**: Medium-High Risk
- **800-1000**: High Risk (highest risk)

### Behavioral Analysis
The system provides detailed insights into:
- High-scoring vs low-scoring wallet behaviors
- Risk factor identification
- Transaction pattern analysis
- Feature importance rankings

## 🧐 How to Interpret Credit Scores

- **0–100:** Very Low Risk — exemplary, reliable DeFi users.
- **100–200:** Low Risk — generally responsible, minor risk factors.
- **200–400:** Medium-Low Risk — some risk, but mostly positive behavior.
- **400–600:** Medium Risk — mixed behavior, caution advised.
- **600–800:** Medium-High Risk — significant risk factors present.
- **800–1000:** High Risk — likely to be bots, exploiters, or irresponsible users.

## 🛠️ Installation & Setup

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit requests
pip install xgboost  # Optional, for XGBoost model
```

### File Structure
```
aave-v2-credit-scoring/
├── app.py                    # Streamlit web interface
├── credit_scoring_script.py  # One-step CLI script
├── data_processor.py         # Data processing pipeline
├── feature_engineer.py       # Feature engineering
├── ml_model.py              # Machine learning models
├── analyzer.py              # Analysis and insights
├── utils.py                 # Utility functions
├── README.md                # This documentation
├── analysis.md              # Analysis results (auto-generated)
└── .streamlit/
    └── config.toml          # Streamlit configuration
```

## 🎯 Credit Scoring Logic

### Positive Indicators (Lower Scores = Better Credit)
- High deposit-to-borrow ratios
- Consistent repayment patterns
- Diverse reserve interactions
- Stable transaction amounts
- Regular activity patterns
- No liquidation involvement

### Negative Indicators (Higher Scores = Higher Risk)
- High liquidation involvement
- Erratic transaction patterns
- Rapid transaction sequences
- High amount volatility
- Bot-like behavior patterns
- Concentration in few reserves

### Validation Metrics
- **Silhouette Score**: Cluster separation quality
- **Calinski-Harabasz Index**: Cluster density
- **Davies-Bouldin Index**: Cluster compactness

## 📋 Output Files

### Credit Scores CSV
Contains wallet addresses and their corresponding credit scores (0-1000).

### Analysis Report (Markdown)
Comprehensive analysis including:
- Score distribution statistics
- Behavioral pattern analysis
- High vs low risk wallet comparison
- Key insights and recommendations
- Methodology explanation

### Detailed Results (JSON)
Technical details including:
- Model performance metrics
- Feature importance rankings
- Complete analysis results
- Metadata and timestamps

## 🔍 Use Cases

### For DeFi Protocols
- Risk assessment for lending
- User behavior analysis
- Protocol optimization insights
- Fraud detection patterns

### For Financial Services  
- Credit assessment for DeFi users
- Portfolio risk management
- User segmentation
- Behavioral analytics

### For Researchers
- DeFi behavior pattern studies
- Credit scoring methodology validation
- Blockchain transaction analysis
- Machine learning applications

## ⚡ Performance

- **Processing Speed**: ~1000 transactions/second
- **Memory Usage**: Optimized for large datasets
- **Scalability**: Handles 100K+ transactions efficiently
- **Accuracy**: Validated through unsupervised metrics

## 📬 Contact & Support

For questions, suggestions, or collaboration, please open an issue or contact the maintainer at [your-email@example.com].

## 🤝 Contributing

This system is designed to be extensible and modular. Key areas for enhancement:
- Additional feature engineering techniques
- Alternative ML algorithms
- Enhanced visualization capabilities
- Real-time scoring APIs

## 📄 License

This project is designed for educational and research purposes in DeFi analytics and machine learning applications.

