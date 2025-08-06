"""
Model evaluation metrics and utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'specificity': self._calculate_specificity(y_true, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def print_classification_report(self, y_true, y_pred, model_name="Model"):
        """Print detailed classification report."""
        print(f"\n=== {model_name} Classification Report ===")
        print(classification_report(y_true, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Additional metrics
        metrics = self.calculate_metrics(y_true, y_pred, model_name=model_name)
        print(f"\nAdditional Metrics:")
        for key, value in metrics.items():
            if key != 'model_name' and value is not None:
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", figsize=(8, 6)):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name="Model", figsize=(8, 6)):
        """Plot ROC curve."""
        if y_pred_proba is None:
            print("Predicted probabilities not available for ROC curve.")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name="Model", figsize=(8, 6)):
        """Plot Precision-Recall curve."""
        if y_pred_proba is None:
            print("Predicted probabilities not available for Precision-Recall curve.")
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} - Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, figsize=(12, 8)):
        """Compare multiple models using stored metrics."""
        if not self.metrics_history:
            print("No model metrics available for comparison.")
            return
        
        df_metrics = pd.DataFrame(self.metrics_history)
        
        # Remove None values for plotting
        df_plot = df_metrics.copy()
        for col in ['roc_auc', 'pr_auc']:
            if col in df_plot.columns:
                df_plot[col] = df_plot[col].fillna(0)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Accuracy comparison
        axes[0, 0].bar(df_plot['model_name'], df_plot['accuracy'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[0, 1].bar(df_plot['model_name'], df_plot['f1_score'])
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # ROC-AUC comparison (if available)
        if 'roc_auc' in df_plot.columns and df_plot['roc_auc'].sum() > 0:
            axes[1, 0].bar(df_plot['model_name'], df_plot['roc_auc'])
            axes[1, 0].set_title('Model ROC-AUC Comparison')
            axes[1, 0].set_ylabel('ROC-AUC')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'ROC-AUC not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('ROC-AUC Not Available')
        
        # Precision-Recall comparison
        precision_recall_data = df_plot[['model_name', 'precision', 'recall']].melt(
            id_vars='model_name', var_name='metric', value_name='value'
        )
        
        models = df_plot['model_name'].unique()
        x = np.arange(len(models))
        width = 0.35
        
        precision_values = df_plot['precision'].values
        recall_values = df_plot['recall'].values
        
        axes[1, 1].bar(x - width/2, precision_values, width, label='Precision')
        axes[1, 1].bar(x + width/2, recall_values, width, label='Recall')
        axes[1, 1].set_title('Precision vs Recall Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics table
        print("\n=== Model Comparison Table ===")
        display_cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        display_df = df_metrics[display_cols].copy()
        for col in display_cols[1:]:  # Skip model_name
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        print(display_df.to_string(index=False))
    
    def get_best_model(self, metric='f1_score'):
        """Get the best performing model based on specified metric."""
        if not self.metrics_history:
            print("No model metrics available.")
            return None
        
        df_metrics = pd.DataFrame(self.metrics_history)
        
        if metric not in df_metrics.columns:
            print(f"Metric '{metric}' not available. Available metrics: {df_metrics.columns.tolist()}")
            return None
        
        # Filter out None values
        valid_metrics = df_metrics[df_metrics[metric].notna()]
        
        if valid_metrics.empty:
            print(f"No valid {metric} values found.")
            return None
        
        best_model_idx = valid_metrics[metric].idxmax()
        best_model = df_metrics.iloc[best_model_idx]
        
        print(f"Best model based on {metric}: {best_model['model_name']} ({metric}={best_model[metric]:.4f})")
        return best_model
    
    def calculate_cost_benefit_analysis(self, y_true, y_pred, cost_fp=100, cost_fn=500, 
                                      revenue_tp=200, model_name="Model"):
        """
        Calculate cost-benefit analysis for churn prediction.
        
        Parameters:
        - cost_fp: Cost of false positive (incorrectly predicting churn)
        - cost_fn: Cost of false negative (missing actual churn)
        - revenue_tp: Revenue from correctly identifying and retaining churners
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        total_revenue = tp * revenue_tp
        net_benefit = total_revenue - total_cost
        
        cost_benefit = {
            'model_name': model_name,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'net_benefit': net_benefit,
            'cost_per_customer': total_cost / len(y_true),
            'revenue_per_customer': total_revenue / len(y_true)
        }
        
        print(f"\n=== Cost-Benefit Analysis for {model_name} ===")
        print(f"True Positives (Correct Churn Predictions): {tp}")
        print(f"False Positives (Incorrect Churn Predictions): {fp}")
        print(f"False Negatives (Missed Churns): {fn}")
        print(f"True Negatives (Correct Non-Churn Predictions): {tn}")
        print(f"Total Cost: ${total_cost:,.2f}")
        print(f"Total Revenue: ${total_revenue:,.2f}")
        print(f"Net Benefit: ${net_benefit:,.2f}")
        print(f"Cost per Customer: ${cost_benefit['cost_per_customer']:.2f}")
        print(f"Revenue per Customer: ${cost_benefit['revenue_per_customer']:.2f}")
        
        return cost_benefit
