#!/usr/bin/env python3
"""
Quick model retraining to match current preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv('./data/churn_data.csv')
print(f"Original data shape: {df.shape}")

# Basic preprocessing to match our API
def preprocess_data(df):
    # Remove unnecessary columns
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname'] if 'RowNumber' in df.columns else []
    df_clean = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    # Feature engineering (same as API)
    df_clean['CreditUtilization'] = df_clean['Balance'] / df_clean['CreditScore']
    df_clean['InteractionScore'] = df_clean['NumOfProducts'] + df_clean['HasCrCard'] + df_clean['IsActiveMember']
    df_clean['BalanceToSalaryRatio'] = df_clean['Balance'] / df_clean['EstimatedSalary']
    df_clean['CreditScoreAgeInteraction'] = df_clean['CreditScore'] * df_clean['Age']
    
    # Credit Score Groups
    def get_credit_group(score):
        if score <= 669:
            return 0  # Low
        elif score <= 739:
            return 1  # Medium  
        else:
            return 2  # High
    
    df_clean['CreditScoreGroup'] = df_clean['CreditScore'].apply(get_credit_group)
    
    # Encode categorical variables
    # Geography: 0=France, 1=Germany, 2=Spain
    geography_map = {'France': 0, 'Germany': 1, 'Spain': 2}
    df_clean['Geography'] = df_clean['Geography'].map(geography_map)
    
    # Gender: 0=Female, 1=Male  
    df_clean['Gender'] = (df_clean['Gender'] == 'Male').astype(int)
    
    return df_clean

# Preprocess
df_processed = preprocess_data(df)

# Separate features and target
X = df_processed.drop('Exited', axis=1)
y = df_processed['Exited']

print(f"Features ({len(X.columns)}): {X.columns.tolist()}")
print(f"Processed data shape: X={X.shape}, y={y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.3f}")
print(f"Classification report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, './models/best_churn_model.joblib')
print("\n✅ Model saved to ./models/best_churn_model.joblib")

# Test a prediction
test_sample = X_test.iloc[0:1]
prediction = model.predict_proba(test_sample)
print(f"\nTest prediction: {prediction[0]}")
print(f"Churn probability: {prediction[0][1]:.3f}")
