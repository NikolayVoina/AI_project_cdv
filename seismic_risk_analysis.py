"""
AI Dataset: Seismic Features & Tsunami Classification Dataset for Risk Assessment

Team: Olena Nesterets, Mikalai Voina, Anastasia Leonova, Victoria Pratkina
Problem: Classification
Target: Risk_level
Metric: F1
Baseline: Dummy (most-freq)
Split: 80/20 (seed=42), Stratify
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class SeismicRiskAnalyzer:
    """Main class for seismic risk analysis and tsunami classification"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.models = {}
        self.results = {}
    
    def load_data(self, file_path):
        """Load the seismic features dataset"""
        print("Loading dataset...")
        try:
            self.data = pd.read_csv(file_path)
            print(f"Dataset loaded successfully: {self.data.shape}")
            return True
        except FileNotFoundError:
            print(f"Dataset file not found: {file_path}")
            print("Please download the dataset from Kaggle and place it in the project directory")
            return False
    
    def explore_data(self):
        """Initial data exploration"""
        if self.data is None:
            print("Please load data first!")
            return
        
        print("\n=== DATASET OVERVIEW ===")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        print("\n=== DATA TYPES ===")
        print(self.data.dtypes)
        
        print("\n=== MISSING VALUES ===")
        missing = self.data.isnull().sum()
        print(missing[missing > 0])
        
        print("\n=== TARGET DISTRIBUTION (Risk_level) ===")
        if 'Risk_level' in self.data.columns:
            risk_counts = self.data['Risk_level'].value_counts()
            print(risk_counts)
            print(f"\nClass proportions:")
            print(self.data['Risk_level'].value_counts(normalize=True))
        else:
            print("Risk_level column not found. Available columns:")
            print(list(self.data.columns))
    
    def generate_eda_plots(self):
        """Generate EDA plots - histograms and scatter plots"""
        if self.data is None:
            print("Please load data first!")
            return
        
        # Get numeric columns for histograms
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if 'Risk_level' in numeric_cols:
            numeric_cols.remove('Risk_level')
        
        # Create histograms
        print("\n=== GENERATING HISTOGRAMS ===")
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(numeric_cols[:n_rows * n_cols]):
            self.data[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create scatter plots with risk level
        print("\n=== GENERATING SCATTER PLOTS ===")
        if len(numeric_cols) >= 2 and 'Risk_level' in self.data.columns:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            # Select top 4 numeric features for scatter plots
            for i in range(min(4, len(numeric_cols)-1)):
                for j in range(i+1, min(4, len(numeric_cols))):
                    if i < len(axes):
                        # Convert categorical Risk_level to numeric codes for coloring
                        risk_codes = pd.Categorical(self.data['Risk_level']).codes
                        scatter = axes[i].scatter(
                            self.data[numeric_cols[i]], 
                            self.data[numeric_cols[j]], 
                            c=risk_codes, 
                            alpha=0.6, 
                            cmap='viridis'
                        )
                        axes[i].set_xlabel(numeric_cols[i])
                        axes[i].set_ylabel(numeric_cols[j])
                        axes[i].set_title(f'{numeric_cols[i]} vs {numeric_cols[j]}')
                        # Add colorbar with proper risk level labels
                        cbar = plt.colorbar(scatter, ax=axes[i])
                        cbar.set_label('Risk Level')
                        risk_levels = sorted(self.data['Risk_level'].unique())
                        cbar.set_ticks(range(len(risk_levels)))
                        cbar.set_ticklabels(risk_levels)
                        break
            
            plt.tight_layout()
            plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def prepare_features(self):
        """Prepare features and target variable"""
        if self.data is None or 'Risk_level' not in self.data.columns:
            print("Please load data with Risk_level column first!")
            return
        
        # Separate features and target
        X = self.data.drop('Risk_level', axis=1)
        y = self.data['Risk_level']
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nNumeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
            ('scaler', StandardScaler())  # Standardize numeric features
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # One-hot encode
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Split the data (80/20 with stratification)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        print(f"\nTrain set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Train class distribution:")
        print(self.y_train.value_counts(normalize=True))
    
    def create_baseline(self):
        """Create dummy classifier baseline"""
        print("\n=== CREATING BASELINE MODEL ===")
        
        # Create dummy classifier (most frequent strategy)
        dummy_clf = DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED)
        
        # Create pipeline with preprocessing
        baseline_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', dummy_clf)
        ])
        
        # Fit and predict
        baseline_pipeline.fit(self.X_train, self.y_train)
        y_pred_baseline = baseline_pipeline.predict(self.X_test)
        
        # Calculate F1 score
        f1_baseline = f1_score(self.y_test, y_pred_baseline, average='weighted')
        
        self.models['baseline'] = baseline_pipeline
        self.results['baseline'] = {
            'f1_score': f1_baseline,
            'predictions': y_pred_baseline
        }
        
        print(f"Baseline F1 Score: {f1_baseline:.4f}")
        return f1_baseline
    
    def train_models(self):
        """Train Logistic Regression and Random Forest models"""
        print("\n=== TRAINING MODELS ===")
        
        # Logistic Regression with balanced class weights
        print("Training Logistic Regression...")
        lr_clf = LogisticRegression(
            class_weight='balanced', 
            random_state=RANDOM_SEED, 
            max_iter=1000
        )
        
        lr_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', lr_clf)
        ])
        
        lr_pipeline.fit(self.X_train, self.y_train)
        y_pred_lr = lr_pipeline.predict(self.X_test)
        f1_lr = f1_score(self.y_test, y_pred_lr, average='weighted')
        
        self.models['logistic'] = lr_pipeline
        self.results['logistic'] = {
            'f1_score': f1_lr,
            'predictions': y_pred_lr
        }
        
        print(f"Logistic Regression F1 Score: {f1_lr:.4f}")
        
        # Random Forest
        print("Training Random Forest...")
        rf_clf = RandomForestClassifier(
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_estimators=100
        )
        
        rf_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', rf_clf)
        ])
        
        rf_pipeline.fit(self.X_train, self.y_train)
        y_pred_rf = rf_pipeline.predict(self.X_test)
        f1_rf = f1_score(self.y_test, y_pred_rf, average='weighted')
        
        self.models['random_forest'] = rf_pipeline
        self.results['random_forest'] = {
            'f1_score': f1_rf,
            'predictions': y_pred_rf
        }
        
        print(f"Random Forest F1 Score: {f1_rf:.4f}")
        
        return f1_lr, f1_rf
    
    def evaluate_models(self):
        """Evaluate all models and generate reports"""
        print("\n=== MODEL EVALUATION RESULTS ===")
        
        for model_name, results in self.results.items():
            print(f"\n--- {model_name.upper()} ---")
            print(f"F1 Score: {results['f1_score']:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, results['predictions']))
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        print(f"\n=== BEST MODEL ===")
        print(f"Model: {best_model[0]}")
        print(f"F1 Score: {best_model[1]['f1_score']:.4f}")
        
        return best_model
    
    def analyze_presentation_questions(self):
        """Analyze the 5 presentation questions"""
        print("\n=== PRESENTATION QUESTIONS ANALYSIS ===")
        
        print("1. What exactly does Risk_level represent, and how are the classes distributed?")
        if 'Risk_level' in self.data.columns:
            print(f"   Risk_level distribution: {dict(self.data['Risk_level'].value_counts())}")
            print(f"   Class balance ratio: {self.data['Risk_level'].value_counts(normalize=True).to_dict()}")
        
        print("\n2. Which columns could leak post-event information, and how will you prevent that?")
        print("   Need to review column descriptions to identify temporal features")
        print("   Recommendation: Use only pre-event seismic/geological features")
        
        print("\n3. How will you handle class imbalance (class weights vs. resampling), and why?")
        print("   Current approach: Using class_weight='balanced' in models")
        print("   Alternative: SMOTE resampling if needed for better performance")
        
        print("\n4. Which 2-3 features do you expect to be most predictive, and what's your intuition?")
        if hasattr(self, 'models') and 'random_forest' in self.models:
            try:
                # Get feature importance from Random Forest
                rf_model = self.models['random_forest']
                feature_names = (rf_model.named_steps['preprocessor']
                               .get_feature_names_out())
                importances = rf_model.named_steps['classifier'].feature_importances_
                
                # Get top 3 features
                top_features_idx = np.argsort(importances)[-3:][::-1]
                print("   Top 3 most important features:")
                for idx in top_features_idx:
                    print(f"   - {feature_names[idx]}: {importances[idx]:.4f}")
            except:
                print("   Feature importance analysis requires trained Random Forest model")
        
        print("\n5. How will you check that the model generalizes across time or regions?")
        print("   Recommendation: Cross-validation with temporal/geographical splits")
        print("   Current: Using stratified random split - may not capture regional/temporal patterns")

def main():
    """Main execution function"""
    analyzer = SeismicRiskAnalyzer()
    
    # Try to load the dataset (user needs to download it from Kaggle)
    dataset_loaded = analyzer.load_data('earthquake_tsunami_dataset.csv')
    
    if not dataset_loaded:
        print("\n=== DATASET DOWNLOAD INSTRUCTIONS ===")
        print("1. Go to: https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset")
        print("2. Download the dataset")
        print("3. Place the CSV file in this directory as 'earthquake_tsunami_dataset.csv'")
        print("4. Run this script again")
        
        # Create sample data for demonstration
        print("\n=== CREATING SAMPLE DATA FOR DEMONSTRATION ===")
        sample_data = create_sample_data()
        analyzer.data = sample_data
        print("Using sample data to demonstrate the pipeline...")
    
    # Run the complete analysis pipeline
    analyzer.explore_data()
    analyzer.generate_eda_plots()
    analyzer.prepare_features()
    analyzer.create_baseline()
    analyzer.train_models()
    best_model = analyzer.evaluate_models()
    analyzer.analyze_presentation_questions()
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Target F1 Score: ≥ 0.70")
    print(f"Best Model: {best_model[0]} (F1: {best_model[1]['f1_score']:.4f})")
    
    if best_model[1]['f1_score'] >= 0.70:
        print("✅ TARGET ACHIEVED!")
    else:
        print("❌ Target not met - consider SMOTE, feature engineering, or hyperparameter tuning")

def create_sample_data():
    """Create sample seismic data for demonstration purposes"""
    np.random.seed(RANDOM_SEED)
    n_samples = 1000
    
    # Generate synthetic seismic features
    data = {
        'magnitude': np.random.normal(5.5, 1.5, n_samples),
        'depth_km': np.random.exponential(50, n_samples),
        'latitude': np.random.uniform(-90, 90, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        'focal_mechanism': np.random.choice(['strike-slip', 'normal', 'reverse'], n_samples),
        'plate_boundary_distance': np.random.exponential(100, n_samples),
        'coastal_distance': np.random.exponential(200, n_samples),
        'population_density': np.random.lognormal(3, 1, n_samples),
    }
    
    # Create risk level based on magnitude and depth (simplified logic)
    risk_level = []
    for i in range(n_samples):
        mag = data['magnitude'][i]
        depth = data['depth_km'][i]
        coastal = data['coastal_distance'][i]
        
        if mag > 7.5 and depth < 70 and coastal < 100:
            risk_level.append('High')
        elif mag > 6.0 and depth < 100 and coastal < 200:
            risk_level.append('Medium')
        else:
            risk_level.append('Low')
    
    data['Risk_level'] = risk_level
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main()
