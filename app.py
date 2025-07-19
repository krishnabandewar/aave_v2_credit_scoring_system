import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
import os
from io import StringIO
import tempfile
import zipfile

from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
from ml_model import CreditScoreModel
from analyzer import WalletAnalyzer
from utils import download_file_from_google_drive

def main():
    st.set_page_config(
        page_title="Aave V2 Wallet Credit Scoring System",
        page_icon="💳",
        layout="wide"
    )
    
    st.title("💳 Aave V2 Wallet Credit Scoring System")
    st.markdown("A machine learning system that analyzes Aave V2 transaction data to generate wallet credit scores (0-1000)")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Loading & Processing", "Feature Engineering", "Model Training", "Credit Scoring", "Analysis & Insights"]
    )
    
    if page == "Data Loading & Processing":
        data_loading_page()
    elif page == "Feature Engineering":
        feature_engineering_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Credit Scoring":
        credit_scoring_page()
    elif page == "Analysis & Insights":
        analysis_page()

def data_loading_page():
    st.header("📊 Data Loading & Processing")
    
    # Option to upload file or use Google Drive link
    st.subheader("Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Upload JSON file", "Download from Google Drive"]
    )
    
    data = None
    
    if data_source == "Upload JSON file":
        uploaded_file = st.file_uploader(
            "Upload Aave V2 transaction data (JSON format)",
            type=['json'],
            help="Upload the user-transactions.json file"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading and processing data..."):
                    data = json.load(uploaded_file)
                    st.success(f"Successfully loaded {len(data)} transactions")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    else:  # Google Drive option
        st.markdown("**Google Drive File IDs:**")
        st.markdown("- Raw JSON file: `1ISFbAXxadMrt7Zl96rmzzZmEKZnyW7FS`")
        st.markdown("- Compressed ZIP file: `14ceBCLQ-BTcydDrFJauVA_PKAZ7VtDor`")
        
        file_type = st.selectbox("Choose file type:", ["JSON", "ZIP"])
        
        if st.button("Download and Load Data"):
            try:
                with st.spinner("Downloading data from Google Drive..."):
                    if file_type == "JSON":
                        file_id = "1ISFbAXxadMrt7Zl96rmzzZmEKZnyW7FS"
                        data = download_file_from_google_drive(file_id, file_type="json")
                    else:
                        file_id = "14ceBCLQ-BTcydDrFJauVA_PKAZ7VtDor"
                        data = download_file_from_google_drive(file_id, file_type="zip")
                    
                    if data:
                        st.success(f"Successfully loaded {len(data)} transactions")
                        st.session_state['raw_data'] = data
                    else:
                        st.error("Failed to load data")
            except Exception as e:
                st.error(f"Error downloading file: {str(e)}")
    
    if data:
        st.session_state['raw_data'] = data
        
        # Process the data
        with st.spinner("Processing transaction data..."):
            processor = DataProcessor()
            df = processor.process_transactions(data)
            st.session_state['processed_data'] = df
            st.session_state['processor'] = processor
        
        # Display data summary
        st.subheader("📈 Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            st.metric("Unique Wallets", df['user'].nunique())
        with col3:
            st.metric("Unique Reserves", df['reserve'].nunique())
        with col4:
            st.metric("Date Range", f"{df['timestamp'].dt.date.min()} to {df['timestamp'].dt.date.max()}")
        
        # Display sample data
        st.subheader("📋 Sample Data")
        st.dataframe(df.head(10))
        
        # Display data types and info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Data Types")
            data_types = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values
            })
            st.dataframe(data_types)
        
        with col2:
            st.subheader("📊 Action Distribution")
            action_counts = df['action'].value_counts()
            fig = px.pie(
                values=action_counts.values,
                names=action_counts.index,
                title="Distribution of Transaction Actions"
            )
            st.plotly_chart(fig, use_container_width=True)

def feature_engineering_page():
    st.header("🔧 Feature Engineering")
    
    if 'processed_data' not in st.session_state:
        st.warning("Please load and process data first from the 'Data Loading & Processing' page.")
        return
    
    df = st.session_state['processed_data']
    
    with st.spinner("Engineering features..."):
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.engineer_features(df)
        st.session_state['features'] = features_df
        st.session_state['feature_engineer'] = feature_engineer
    
    st.success(f"Generated {len(features_df.columns) - 1} features for {len(features_df)} wallets")
    
    # Display feature summary
    st.subheader("📊 Feature Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Features", len(features_df.columns) - 1)
        st.metric("Wallets with Features", len(features_df))
    
    with col2:
        # Feature categories
        feature_categories = {
            'Basic Stats': [col for col in features_df.columns if any(x in col for x in ['total_', 'avg_', 'count_'])],
            'Ratios': [col for col in features_df.columns if 'ratio' in col],
            'Time Features': [col for col in features_df.columns if any(x in col for x in ['days_', 'frequency'])],
            'Risk Features': [col for col in features_df.columns if any(x in col for x in ['volatility', 'concentration', 'liquidation'])]
        }
        
        for category, features in feature_categories.items():
            if features:
                st.write(f"**{category}:** {len(features)} features")
    
    # Display feature correlation heatmap
    st.subheader("🔥 Feature Correlation Matrix")
    
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = features_df[numeric_features].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display top features by variance
    st.subheader("📈 Feature Statistics")
    
    feature_stats = features_df[numeric_features].describe().T
    feature_stats['variance'] = features_df[numeric_features].var()
    feature_stats = feature_stats.sort_values('variance', ascending=False)
    
    st.dataframe(feature_stats.head(20))
    
    # Feature distribution plots
    st.subheader("📊 Feature Distributions")
    
    selected_features = st.multiselect(
        "Select features to visualize:",
        numeric_features.tolist(),
        default=numeric_features.tolist()[:4]
    )
    
    if selected_features:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=selected_features[:4],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, feature in enumerate(selected_features[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_histogram(
                x=features_df[feature],
                name=feature,
                row=row, col=col,
                showlegend=False
            )
        
        fig.update_layout(height=600, title="Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)

def model_training_page():
    st.header("🤖 Model Training")
    
    if 'features' not in st.session_state:
        st.warning("Please complete feature engineering first.")
        return
    
    features_df = st.session_state['features']
    
    st.subheader("⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select model type:",
            ["Random Forest", "Gradient Boosting", "XGBoost", "Ensemble"]
        )
        
        use_scaling = st.checkbox("Use feature scaling", value=True)
        
    with col2:
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, min_value=0)
    
    if st.button("Train Model"):
        with st.spinner("Training credit scoring model..."):
            model = CreditScoreModel(
                model_type=model_type.lower().replace(" ", "_"),
                use_scaling=use_scaling,
                random_state=random_state
            )
            
            scores, metrics, feature_importance = model.train_and_score(
                features_df, 
                test_size=test_size
            )
            
            st.session_state['model'] = model
            st.session_state['scores'] = scores
            st.session_state['model_metrics'] = metrics
            st.session_state['feature_importance'] = feature_importance
        
        st.success("Model training completed!")
        
        # Display model performance
        st.subheader("📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
        with col2:
            st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.2f}")
        with col3:
            st.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}")
        
        # Feature importance plot
        st.subheader("🎯 Feature Importance")
        
        importance_df = pd.DataFrame({
            'feature': feature_importance.index,
            'importance': feature_importance.values
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df.tail(20),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 20 Most Important Features"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Score distribution
        st.subheader("📈 Credit Score Distribution")
        
        fig = px.histogram(
            scores,
            x='credit_score',
            nbins=50,
            title="Distribution of Credit Scores"
        )
        fig.update_layout(xaxis_title="Credit Score", yaxis_title="Number of Wallets")
        st.plotly_chart(fig, use_container_width=True)

def credit_scoring_page():
    st.header("💳 Credit Scoring Results")
    
    if 'scores' not in st.session_state:
        st.warning("Please train the model first.")
        return
    
    scores = st.session_state['scores']
    
    # Score summary
    st.subheader("📊 Score Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Wallets", len(scores))
    with col2:
        st.metric("Average Score", f"{scores['credit_score'].mean():.1f}")
    with col3:
        st.metric("Median Score", f"{scores['credit_score'].median():.1f}")
    with col4:
        st.metric("Score Std Dev", f"{scores['credit_score'].std():.1f}")
    
    # Score ranges
    st.subheader("🎯 Score Ranges")
    
    ranges = [
        (0, 100, "Very Low Risk"),
        (100, 200, "Low Risk"),
        (200, 400, "Medium-Low Risk"),
        (400, 600, "Medium Risk"),
        (600, 800, "Medium-High Risk"),
        (800, 1000, "High Risk")
    ]
    
    range_data = []
    for min_score, max_score, label in ranges:
        count = len(scores[(scores['credit_score'] >= min_score) & (scores['credit_score'] < max_score)])
        percentage = count / len(scores) * 100
        range_data.append({
            'Range': f"{min_score}-{max_score}",
            'Label': label,
            'Count': count,
            'Percentage': f"{percentage:.1f}%"
        })
    
    range_df = pd.DataFrame(range_data)
    st.dataframe(range_df)
    
    # Score distribution by ranges
    fig = px.bar(
        range_df,
        x='Range',
        y='Count',
        title="Wallet Distribution by Credit Score Ranges",
        color='Count',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top and bottom wallets
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Wallets (Highest Scores)")
        top_wallets = scores.nlargest(10, 'credit_score')[['wallet', 'credit_score']]
        st.dataframe(top_wallets)
    
    with col2:
        st.subheader("⚠️ Bottom 10 Wallets (Lowest Scores)")
        bottom_wallets = scores.nsmallest(10, 'credit_score')[['wallet', 'credit_score']]
        st.dataframe(bottom_wallets)
    
    # Download scores
    st.subheader("💾 Download Results")
    
    csv = scores.to_csv(index=False)
    st.download_button(
        label="Download Credit Scores as CSV",
        data=csv,
        file_name="wallet_credit_scores.csv",
        mime="text/csv"
    )

def analysis_page():
    st.header("📈 Analysis & Insights")
    
    if 'scores' not in st.session_state or 'processed_data' not in st.session_state:
        st.warning("Please complete the full pipeline first.")
        return
    
    scores = st.session_state['scores']
    df = st.session_state['processed_data']
    
    with st.spinner("Generating comprehensive analysis..."):
        analyzer = WalletAnalyzer()
        analysis_results = analyzer.analyze_wallets(scores, df)
        st.session_state['analysis_results'] = analysis_results
    
    # Score distribution analysis
    st.subheader("📊 Score Distribution Analysis")
    
    # Score histogram with ranges
    fig = go.Figure()
    
    for i, (min_score, max_score, label, color) in enumerate([
        (0, 100, "Very Low (0-100)", "#d62728"),
        (100, 200, "Low (100-200)", "#ff7f0e"),
        (200, 400, "Med-Low (200-400)", "#ffbb78"),
        (400, 600, "Medium (400-600)", "#2ca02c"),
        (600, 800, "Med-High (600-800)", "#98df8a"),
        (800, 1000, "High (800-1000)", "#1f77b4")
    ]):
        range_scores = scores[
            (scores['credit_score'] >= min_score) & 
            (scores['credit_score'] < max_score)
        ]['credit_score']
        
        fig.add_histogram(
            x=range_scores,
            name=label,
            marker_color=color,
            opacity=0.7,
            nbinsx=20
        )
    
    fig.update_layout(
        title="Credit Score Distribution by Risk Categories",
        xaxis_title="Credit Score",
        yaxis_title="Number of Wallets",
        barmode='overlay'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Behavioral analysis
    st.subheader("🎭 Behavioral Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("High-Scoring Wallets (800-1000)")
        high_score_analysis = analysis_results['high_score_behavior']
        for key, value in high_score_analysis.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
    
    with col2:
        st.subheader("Low-Scoring Wallets (0-200)")
        low_score_analysis = analysis_results['low_score_behavior']
        for key, value in low_score_analysis.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
    
    # Feature comparison between high and low scoring wallets
    st.subheader("⚖️ Feature Comparison: High vs Low Scoring Wallets")
    
    if 'features' in st.session_state:
        features_df = st.session_state['features']
        merged_df = features_df.merge(scores, left_on='wallet', right_on='wallet')
        
        high_score_features = merged_df[merged_df['credit_score'] >= 800]
        low_score_features = merged_df[merged_df['credit_score'] <= 200]
        
        if len(high_score_features) > 0 and len(low_score_features) > 0:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            comparison_data = []
            
            for col in numeric_cols[:10]:  # Top 10 features
                high_mean = high_score_features[col].mean()
                low_mean = low_score_features[col].mean()
                comparison_data.append({
                    'Feature': col,
                    'High Score Avg': high_mean,
                    'Low Score Avg': low_mean,
                    'Ratio (High/Low)': high_mean / low_mean if low_mean != 0 else np.inf
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
    
    # Insights and recommendations
    st.subheader("💡 Key Insights")
    
    insights = analysis_results.get('insights', [])
    for insight in insights:
        st.write(f"• {insight}")
    
    # Generate analysis.md content
    if st.button("Generate Analysis Report"):
        analysis_content = analyzer.generate_analysis_report(analysis_results, scores)
        
        st.subheader("📄 Analysis Report")
        st.markdown(analysis_content)
        
        st.download_button(
            label="Download Analysis Report (Markdown)",
            data=analysis_content,
            file_name="analysis.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()
