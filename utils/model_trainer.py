"""
Model training and management utilities.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class ChurnModelTrainer:
    """Train and manage customer churn prediction models."""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.feature_names = None
        
    def get_model_configs(self):
        """Get default model configurations."""
        return {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
    
    def train_model(self, model, X_train, y_train, model_name):
        """Train a single model."""
        print(f"Training {model_name}...")
        
        # Store feature names for later use
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        
        model.fit(X_train, y_train)
        self.models[model_name] = model
        
        print(f"{model_name} training completed.")
        return model
    
    def train_all_models(self, X_train, y_train):
        """Train all available models."""
        print("=== Training All Models ===")
        
        model_configs = self.get_model_configs()
        trained_models = {}
        
        for model_name, config in model_configs.items():
            model = config['model']
            trained_model = self.train_model(model, X_train, y_train, model_name)
            trained_models[model_name] = trained_model
        
        print("All models training completed.")
        return trained_models
    
    def hyperparameter_tuning(self, X_train, y_train, model_name, cv=5, n_iter=50, scoring='f1'):
        """Perform hyperparameter tuning for a specific model."""
        print(f"=== Hyperparameter Tuning for {model_name} ===")
        
        model_configs = self.get_model_configs()
        
        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not available. Available models: {list(model_configs.keys())}")
        
        config = model_configs[model_name]
        model = config['model']
        param_grid = config['params']
        
        # Create scorer
        scorer = make_scorer(f1_score) if scoring == 'f1' else scoring
        
        # Randomized search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scorer,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {random_search.best_params_}")
        print(f"Best cross-validation score: {random_search.best_score_:.4f}")
        
        # Store the best model
        self.best_models[model_name] = random_search.best_estimator_
        
        return random_search.best_estimator_, random_search.best_params_
    
    def tune_all_models(self, X_train, y_train, cv=5, n_iter=20):
        """Perform hyperparameter tuning for all models."""
        print("=== Hyperparameter Tuning for All Models ===")
        
        model_configs = self.get_model_configs()
        tuned_models = {}
        best_params = {}
        
        for model_name in model_configs.keys():
            try:
                best_model, params = self.hyperparameter_tuning(
                    X_train, y_train, model_name, cv=cv, n_iter=n_iter
                )
                tuned_models[model_name] = best_model
                best_params[model_name] = params
            except Exception as e:
                print(f"Error tuning {model_name}: {e}")
                continue
        
        return tuned_models, best_params
    
    def cross_validate_model(self, model, X_train, y_train, cv=5, scoring='f1'):
        """Perform cross-validation on a model."""
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        
        result = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
        
        print(f"Cross-validation {scoring} scores: {scores}")
        print(f"Mean {scoring}: {result['mean_score']:.4f} (+/- {result['std_score'] * 2:.4f})")
        
        return result
    
    def get_feature_importance(self, model, model_name=None, top_n=10):
        """Get feature importance from tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if self.feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            else:
                feature_names = self.feature_names
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n=== Top {top_n} Feature Importances ({model_name or 'Model'}) ===")
            print(feature_importance_df.head(top_n).to_string(index=False))
            
            return feature_importance_df
        else:
            print(f"Feature importance not available for {model_name or 'this model'}")
            return None
    
    def save_model(self, model, filepath, model_name=None):
        """Save trained model to disk."""
        try:
            joblib.dump(model, filepath)
            print(f"Model {model_name or ''} saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """Load trained model from disk."""
        try:
            model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict_with_model(self, model, X, return_proba=False):
        """Make predictions using a trained model."""
        if return_proba and hasattr(model, 'predict_proba'):
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]  # Probability of positive class
            return predictions, probabilities
        else:
            predictions = model.predict(X)
            return predictions
    
    def create_ensemble_prediction(self, models_dict, X, method='voting'):
        """Create ensemble predictions from multiple models."""
        predictions = {}
        probabilities = {}
        
        # Get predictions from all models
        for name, model in models_dict.items():
            pred = model.predict(X)
            predictions[name] = pred
            
            if hasattr(model, 'predict_proba'):
                probabilities[name] = model.predict_proba(X)[:, 1]
        
        if method == 'voting':
            # Simple majority voting
            pred_df = pd.DataFrame(predictions)
            ensemble_pred = pred_df.mode(axis=1)[0].astype(int)
            
        elif method == 'averaging' and probabilities:
            # Average probabilities
            prob_df = pd.DataFrame(probabilities)
            avg_proba = prob_df.mean(axis=1)
            ensemble_pred = (avg_proba >= 0.5).astype(int)
        
        else:
            raise ValueError("Invalid ensemble method or probabilities not available")
        
        return ensemble_pred.values if hasattr(ensemble_pred, 'values') else ensemble_pred
