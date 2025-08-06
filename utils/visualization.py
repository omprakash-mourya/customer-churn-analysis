"""
Visualization utilities for customer churn analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")


class ChurnVisualizer:
    """Comprehensive visualization utilities for churn analysis."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    def plot_target_distribution(self, y, title="Churn Distribution", figsize=None):
        """Plot target variable distribution."""
        if figsize is None:
            figsize = (8, 6)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Count plot
        y_counts = y.value_counts()
        ax1.pie(y_counts.values, labels=['Not Churned', 'Churned'], autopct='%1.1f%%', 
                colors=self.colors[:2], startangle=90)
        ax1.set_title('Churn Distribution (Pie Chart)')
        
        # Bar plot
        ax2.bar(['Not Churned', 'Churned'], y_counts.values, color=self.colors[:2])
        ax2.set_title('Churn Distribution (Bar Chart)')
        ax2.set_ylabel('Count')
        
        # Add count labels on bars
        for i, v in enumerate(y_counts.values):
            ax2.text(i, v + max(y_counts.values) * 0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        churn_rate = y.mean() if hasattr(y, 'mean') else np.mean(y)
        print(f"Total samples: {len(y)}")
        print(f"Churn rate: {churn_rate:.2%}")
        print(f"Class distribution: {dict(y_counts)}")
    
    def plot_numerical_features(self, df, target_col, figsize=None):
        """Plot distribution of numerical features by target variable."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        if not numerical_cols:
            print("No numerical features found.")
            return
        
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (15, 5 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numerical_cols):
            row = i // n_cols
            col_idx = i % n_cols
            
            # Box plot
            sns.boxplot(data=df, x=target_col, y=col, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'{col} by Churn Status')
        
        # Hide empty subplots
        for i in range(len(numerical_cols), n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_features(self, df, target_col, figsize=None):
        """Plot distribution of categorical features by target variable."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        if not categorical_cols:
            print("No categorical features found.")
            return
        
        n_cols = 2
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (15, 5 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(categorical_cols):
            row = i // n_cols
            col_idx = i % n_cols
            
            # Count plot
            sns.countplot(data=df, x=col, hue=target_col, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'{col} by Churn Status')
            axes[row, col_idx].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(categorical_cols), n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, df, figsize=None, annot=True):
        """Plot correlation matrix heatmap."""
        if figsize is None:
            figsize = (12, 10)
        
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            print("No numerical features found for correlation matrix.")
            return
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', center=0,
                   square=True, fmt='.2f' if annot else None)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df, title="Feature Importance", 
                               top_n=15, figsize=None):
        """Plot feature importance."""
        if importance_df is None or importance_df.empty:
            print("No feature importance data provided.")
            return
        
        if figsize is None:
            figsize = (10, max(6, top_n * 0.4))
        
        # Select top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=self.colors[0])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, metrics_df, figsize=None):
        """Plot model performance comparison."""
        if figsize is None:
            figsize = (15, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Accuracy
        axes[0, 0].bar(metrics_df['model_name'], metrics_df['accuracy'], 
                      color=self.colors)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[0, 1].bar(metrics_df['model_name'], metrics_df['f1_score'], 
                      color=self.colors)
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # ROC-AUC (if available)
        if 'roc_auc' in metrics_df.columns and metrics_df['roc_auc'].notna().any():
            valid_auc = metrics_df[metrics_df['roc_auc'].notna()]
            axes[1, 0].bar(valid_auc['model_name'], valid_auc['roc_auc'], 
                          color=self.colors[:len(valid_auc)])
            axes[1, 0].set_title('Model ROC-AUC Comparison')
            axes[1, 0].set_ylabel('ROC-AUC')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'ROC-AUC not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Precision vs Recall
        x = np.arange(len(metrics_df))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, metrics_df['precision'], width, 
                      label='Precision', color=self.colors[0])
        axes[1, 1].bar(x + width/2, metrics_df['recall'], width, 
                      label='Recall', color=self.colors[1])
        axes[1, 1].set_title('Precision vs Recall Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics_df['model_name'], rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_business_dashboard(self, df, target_col):
        """Create a comprehensive business dashboard."""
        # Calculate business metrics
        total_customers = len(df)
        churned_customers = df[target_col].sum()
        churn_rate = churned_customers / total_customers
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Churn Rate Overview', 'Customer Segments', 
                          'Churn by Feature', 'Revenue Impact'),
            specs=[[{"type": "indicator"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # KPI - Churn Rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=churn_rate * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Rate %"},
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgray"},
                        {'range': [20, 40], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ),
            row=1, col=1
        )
        
        # Pie chart - Churn distribution
        fig.add_trace(
            go.Pie(
                labels=['Retained', 'Churned'],
                values=[total_customers - churned_customers, churned_customers],
                hole=0.4
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Customer Churn Business Dashboard")
        fig.show()
    
    def plot_interactive_scatter(self, df, x_col, y_col, color_col, 
                                title="Interactive Scatter Plot"):
        """Create interactive scatter plot using Plotly."""
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col,
            title=title,
            labels={color_col: 'Churn Status'},
            hover_data=df.columns.tolist()
        )
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=600
        )
        
        fig.show()
    
    def save_plot(self, filename, dpi=300):
        """Save the current plot to file."""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Plot saved as {filename}")
