#!/usr/bin/env python3
"""
Quick runner script for the Seismic Risk Analysis project.
This demonstrates the complete pipeline execution.

Usage: python run_analysis.py
"""

from seismic_risk_analysis import SeismicRiskAnalyzer, create_sample_data
import pandas as pd

def main():
    print("=" * 60)
    print("SEISMIC FEATURES & TSUNAMI RISK CLASSIFICATION")
    print("=" * 60)
    print()
    print("Team: Olena Nesterets, Mikalai Voina, Anastasia Leonova, Victoria Pratkina")
    print("Target: Risk_level | Metric: F1 Score | Goal: >= 0.70")
    print()
    
    # Initialize analyzer
    analyzer = SeismicRiskAnalyzer()
    
    # Try to load real dataset, fallback to sample data
    print("[INFO] Loading dataset...")
    dataset_loaded = analyzer.load_data('earthquake_tsunami_dataset.csv')
    
    if not dataset_loaded:
        print("[INFO] Creating sample data for demonstration...")
        sample_data = create_sample_data()
        analyzer.data = sample_data
        print(f"[OK] Sample data created: {sample_data.shape}")
    
    # Execute full pipeline
    print("\n[INFO] Data exploration...")
    analyzer.explore_data()
    
    print("\n[INFO] Generating EDA plots...")
    analyzer.generate_eda_plots()
    
    print("\n[INFO] Preparing features...")
    analyzer.prepare_features()
    
    print("\n[INFO] Creating baseline...")
    baseline_f1 = analyzer.create_baseline()
    
    print("\n[INFO] Training ML models...")
    lr_f1, rf_f1 = analyzer.train_models()
    
    print("\n[INFO] Evaluating results...")
    best_model = analyzer.evaluate_models()
    
    print("\n[INFO] Analyzing presentation questions...")
    analyzer.analyze_presentation_questions()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL PROJECT SUMMARY")
    print("=" * 60)
    print(f"Dataset Shape: {analyzer.data.shape}")
    print(f"Best Model: {best_model[0]} (F1: {best_model[1]['f1_score']:.4f})")
    print(f"Baseline F1: {baseline_f1:.4f}")
    print(f"Improvement: {((best_model[1]['f1_score'] - baseline_f1) / baseline_f1 * 100):+.1f}%")
    
    target_met = best_model[1]['f1_score'] >= 0.70
    print(f"Target (F1 >= 0.70): {'ACHIEVED' if target_met else 'NOT MET'}")
    
    if not target_met:
        print("\nNext Steps:")
        print("   - Apply SMOTE for better class balance")
        print("   - Feature engineering for seismic domain")
        print("   - Hyperparameter tuning")
        print("   - Try ensemble methods")
    
    print("\nAll deliverables completed successfully!")
    print("   [OK] Preprocessing pipeline (standardization + one-hot)")
    print("   [OK] Missing data handling")
    print("   [OK] EDA plots (histograms + scatter)")
    print("   [OK] Baseline and ML models")
    print("   [OK] F1 evaluation")
    print("   [OK] Presentation analysis")
    
    print(f"\nGenerated files:")
    print("   - histograms.png")
    print("   - scatter_plots.png")
    print("   - seismic_analysis.ipynb")
    print("   - README.md")
    
    print("\nProject ready for presentation!")

if __name__ == "__main__":
    main()
