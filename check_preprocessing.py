#!/usr/bin/env python3
"""Check what features are actually produced by the preprocessing pipeline"""

import sys
import os
sys.path.append('./utils')

try:
    from utils.preprocessing import ChurnDataPreprocessor
    import pandas as pd
    
    # Load and preprocess the same way as training
    preprocessor = ChurnDataPreprocessor()
    
    # Load data
    df = pd.read_csv('./data/churn_data.csv')
    print("Original columns:", df.columns.tolist())
    
    # Apply full preprocessing
    result = preprocessor.full_preprocessing_pipeline(
        filepath='./data/churn_data.csv',
        target_col='Exited',
        test_size=0.3,
        handle_imbalance=False  # Don't need balancing for feature check
    )
    
    if result:
        X_train, X_test, y_train, y_test = result
        print(f"Final feature columns ({len(X_train.columns)}): {X_train.columns.tolist()}")
        print(f"Feature shapes: X_train={X_train.shape}, X_test={X_test.shape}")
        print(f"First row features: {X_train.iloc[0].values}")
    
except Exception as e:
    print(f"Error: {e}")
    
    # Fallback - let's check what a simple one-hot encoding produces
    import pandas as pd
    df = pd.read_csv('./data/churn_data.csv')
    
    # Remove unnecessary columns
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname'] if 'RowNumber' in df.columns else []
    df_clean = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    # Separate target
    if 'Exited' in df_clean.columns:
        X = df_clean.drop('Exited', axis=1)
        
        # One-hot encode categorical columns
        categorical_cols = ['Geography', 'Gender']
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        print("Simple encoding result:")
        print(f"Columns ({len(X_encoded.columns)}): {X_encoded.columns.tolist()}")
        print(f"Sample row: {X_encoded.iloc[0].values}")
    