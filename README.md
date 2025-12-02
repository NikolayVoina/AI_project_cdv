# Seismic Features & Tsunami Classification Dataset for Risk Assessment

## Team Members
- Olena Nesterets
- Mikalai Voina  
- Anastasia Leonova
- Victoria Pratkina

## Project Overview

**Problem Type**: Classification  
**Target Variable**: Risk_level  
**Evaluation Metric**: F1 Score  
**Baseline**: Dummy Classifier (most_frequent)  
**Data Split**: 80/20 (seed=42, stratified)  
**Target F1 Score**: ≥ 0.70

## Dataset

Source: [Global Earthquake Tsunami Risk Assessment Dataset](https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset)

## Project Structure

```
AI/
├── requirements.txt              # Python dependencies
├── seismic_risk_analysis.py     # Main analysis script
├── seismic_analysis.ipynb       # Jupyter notebook version
├── README.md                    # This file
├── histograms.png              # Generated EDA plots
└── scatter_plots.png           # Generated EDA plots
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**:
   - Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset)
   - Download the dataset
   - Place the CSV file as `earthquake_tsunami_dataset.csv` in this directory

3. **Run Analysis**:
   ```bash
   python seismic_risk_analysis.py
   ```
   
   Or use the Jupyter notebook:
   ```bash
   jupyter notebook seismic_analysis.ipynb
   ```

## Methodology

### Preprocessing Pipeline
- **Numerical Features**: StandardScaler normalization
- **Categorical Features**: One-hot encoding  
- **Missing Data**: SimpleImputer (mean for numerical, most_frequent for categorical)

### Models
1. **Baseline**: DummyClassifier (most_frequent strategy)
2. **Logistic Regression**: With balanced class weights
3. **Random Forest**: With balanced class weights

### Class Imbalance Handling
- Primary: `class_weight='balanced'` in models
- Backup: SMOTE oversampling if needed

### Evaluation
- Primary metric: F1 Score (weighted average)
- Cross-validation with stratification
- Classification reports and confusion matrices

## LLM Usage

**Task 1**: Asked LLM to generate comprehensive EDA code for:
- Histogram plots for numerical feature distributions
- Scatter plots to visualize feature relationships with risk levels
- Feature engineering suggestions for seismic data

**Task 2**: Verified LLM outputs by:
- Running generated code on data subsets
- Reviewing plot outputs for correctness and relevance
- Testing feature engineering suggestions on sample data

## Expectations & Risk Assessment

### Key Risks
- **Class imbalance**: May reduce model performance if not handled properly
- **Missing data**: Could impact feature quality and model accuracy
- **Data leakage**: Post-event features could artificially inflate performance

### Expected Best Model
Random Forest - due to ability to handle:
- Non-linear relationships in seismic data
- Mixed data types (numerical + categorical)
- Feature interactions without explicit engineering

### Performance Target
- **F1 Score ≥ 0.70** to meaningfully outperform dummy baseline
- Focus on balanced performance across all risk classes

## Presentation Questions Analysis

1. **Risk Level Definition**: Analysis of target variable distribution and class balance
2. **Data Leakage Prevention**: Identification of temporal features that could leak post-event information  
3. **Class Imbalance Strategy**: Comparison of class weighting vs. resampling approaches
4. **Feature Importance**: Top 2-3 most predictive features with domain intuition
5. **Generalization Testing**: Time/region-based validation strategies

## Model Generalization Considerations

- **Current Split**: Stratified random (may not capture temporal/geographic patterns)
- **Recommended**: Time-based or geographic cross-validation
- **Risk**: Models may not generalize across different regions or time periods

## Results Summary

Results will be displayed after running the analysis script, including:
- Baseline vs. model performance comparison
- Feature importance analysis
- Classification reports for all models
- Recommendations for improvement

## Next Steps (If Target Not Met)

1. **SMOTE Resampling**: Address class imbalance more aggressively
2. **Feature Engineering**: Domain-specific seismic feature creation
3. **Hyperparameter Tuning**: Grid search for optimal model parameters
4. **Advanced Models**: Gradient Boosting (XGBoost, LightGBM)
5. **Ensemble Methods**: Combine multiple model predictions
