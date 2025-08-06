"""
Main training script for Customer Churn Prediction models.

This script trains multiple machine learning models for churn prediction,
performs hyperparameter tuning, and evaluates model performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.preprocessing import ChurnDataPreprocessor
from utils.model_trainer import ChurnModelTrainer
from utils.metrics import ModelEvaluator
from utils.visualization import ChurnVisualizer

def main():
    """Main training pipeline."""
    print("=== Customer Churn Prediction Model Training ===")
    print("🚀 Starting end-to-end machine learning pipeline...\n")
    
    # Configuration
    DATA_PATH = "data/churn_data.csv"
    MODEL_SAVE_DIR = "models"
    TARGET_COLUMN = "Exited"
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Initialize components
    preprocessor = ChurnDataPreprocessor()
    trainer = ChurnModelTrainer()
    evaluator = ModelEvaluator()
    visualizer = ChurnVisualizer()
    
    print("📊 Step 1: Data Loading and Preprocessing")
    print("-" * 50)
    
    # Full preprocessing pipeline
    result = preprocessor.full_preprocessing_pipeline(
        filepath=DATA_PATH,
        target_col=TARGET_COLUMN,
        test_size=0.3,
        handle_imbalance=True,
        random_state=42
    )
    
    if result is None:
        print("❌ Error in data preprocessing. Exiting.")
        return
    
    X_train, X_test, y_train, y_test = result
    
    print("\n🤖 Step 2: Model Training")
    print("-" * 50)
    
    # Train all models with default parameters
    models = trainer.train_all_models(X_train, y_train)
    
    print("\n📈 Step 3: Model Evaluation (Default Parameters)")
    print("-" * 50)
    
    # Evaluate all models
    model_results = []
    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        
        # Make predictions
        predictions = trainer.predict_with_model(model, X_test)
        if hasattr(model, 'predict_proba'):
            y_pred, y_pred_proba = trainer.predict_with_model(model, X_test, return_proba=True)
        else:
            y_pred = predictions
            y_pred_proba = None
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba, name)
        model_results.append(metrics)
        
        # Print detailed report
        evaluator.print_classification_report(y_test, y_pred, name)
    
    print("\n🎯 Step 4: Hyperparameter Tuning")
    print("-" * 50)
    
    # Perform hyperparameter tuning for best performing models
    tuned_models, best_params = trainer.tune_all_models(X_train, y_train, cv=5, n_iter=20)
    
    print("\n📊 Step 5: Tuned Model Evaluation")
    print("-" * 50)
    
    # Evaluate tuned models
    tuned_results = []
    for name, model in tuned_models.items():
        print(f"\n--- Evaluating Tuned {name} ---")
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_pred, y_pred_proba = trainer.predict_with_model(model, X_test, return_proba=True)
        else:
            y_pred = trainer.predict_with_model(model, X_test)
            y_pred_proba = None
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba, f"Tuned {name}")
        tuned_results.append(metrics)
        
        # Get feature importance
        importance_df = trainer.get_feature_importance(model, f"Tuned {name}")
        
        # Save model
        model_path = os.path.join(MODEL_SAVE_DIR, f"tuned_{name}_model.joblib")
        trainer.save_model(model, model_path, f"Tuned {name}")
    
    print("\n📈 Step 6: Model Comparison and Selection")
    print("-" * 50)
    
    # Compare all models
    evaluator.compare_models()
    
    # Get best model
    best_model = evaluator.get_best_model(metric='f1_score')
    
    print("\n💰 Step 7: Business Impact Analysis")
    print("-" * 50)
    
    # Cost-benefit analysis for the best model
    if best_model is not None:
        best_model_name = best_model['model_name']
        
        # Find the actual model object
        best_model_obj = None
        if 'Tuned' in best_model_name:
            clean_name = best_model_name.replace('Tuned ', '').lower().replace(' ', '_')
            best_model_obj = tuned_models.get(clean_name)
        else:
            clean_name = best_model_name.lower().replace(' ', '_')
            best_model_obj = models.get(clean_name)
        
        if best_model_obj is not None:
            # Make predictions with best model
            y_pred_best = trainer.predict_with_model(best_model_obj, X_test)
            
            # Cost-benefit analysis
            cost_benefit = evaluator.calculate_cost_benefit_analysis(
                y_test, y_pred_best,
                cost_fp=100,  # Cost of false positive (retention campaign cost)
                cost_fn=500,  # Cost of false negative (lost customer value)
                revenue_tp=200,  # Revenue from successfully retained customer
                model_name=best_model_name
            )
            
            # Save best model separately
            best_model_path = os.path.join(MODEL_SAVE_DIR, "best_churn_model.joblib")
            trainer.save_model(best_model_obj, best_model_path, "Best Model")
    
    print("\n✅ Step 8: Training Complete!")
    print("-" * 50)
    print("🎉 Model training pipeline completed successfully!")
    print(f"📁 Models saved in: {MODEL_SAVE_DIR}/")
    print(f"🏆 Best model: {best_model['model_name'] if best_model else 'N/A'}")
    print("\n💡 Key Insights:")
    print("- Retention is cheaper than acquisition")
    print("- XGBoost/Gradient Boosting typically perform best on churn prediction")
    print("- Feature importance helps identify key churn drivers")
    print("- Cost-benefit analysis guides business decision making")
    
    # Generate summary report
    print("\n📋 Generating Summary Report...")
    create_summary_report(model_results, tuned_results, best_model, MODEL_SAVE_DIR)
    
    print("\n🔍 Next Steps:")
    print("1. Review model performance in summary_report.txt")
    print("2. Run SHAP analysis for model explainability")
    print("3. Deploy best model using app/api.py")
    print("4. Create Streamlit dashboard with app/streamlit_app.py")

def create_summary_report(default_results, tuned_results, best_model, save_dir):
    """Create a summary report of model training results."""
    report_path = os.path.join(save_dir, "training_summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("CUSTOMER CHURN PREDICTION - TRAINING SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 DEFAULT MODEL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        for result in default_results:
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1-Score: {result['f1_score']:.4f}\n")
            if result['roc_auc']:
                f.write(f"  ROC-AUC: {result['roc_auc']:.4f}\n")
            f.write("\n")
        
        f.write("\n🎯 TUNED MODEL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        for result in tuned_results:
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1-Score: {result['f1_score']:.4f}\n")
            if result['roc_auc']:
                f.write(f"  ROC-AUC: {result['roc_auc']:.4f}\n")
            f.write("\n")
        
        f.write("\n🏆 BEST MODEL\n")
        f.write("-" * 30 + "\n")
        if best_model is not None:
            f.write(f"Best Model: {best_model['model_name']}\n")
            f.write(f"F1-Score: {best_model['f1_score']:.4f}\n")
            f.write(f"Accuracy: {best_model['accuracy']:.4f}\n")
            if best_model['roc_auc']:
                f.write(f"ROC-AUC: {best_model['roc_auc']:.4f}\n")
        else:
            f.write("No best model identified\n")
        
        f.write("\n💡 BUSINESS RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        f.write("1. Focus on high-risk customer segments identified by the model\n")
        f.write("2. Implement proactive retention campaigns\n")
        f.write("3. Monitor model performance regularly\n")
        f.write("4. Use SHAP values to understand individual predictions\n")
        f.write("5. A/B test retention strategies based on model insights\n")
    
    print(f"📄 Summary report saved: {report_path}")

if __name__ == "__main__":
    main()
