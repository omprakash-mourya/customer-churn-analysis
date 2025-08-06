"""
Data preprocessing utilities for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE  # Disabled due to compatibility issues
import warnings
warnings.filterwarnings('ignore')


class ChurnDataPreprocessor:
    """Data preprocessing pipeline for customer churn data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load and return the dataset."""
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean the dataset by handling missing values and duplicates."""
        print("=== Data Cleaning ===")
        print(f"Initial shape: {df.shape}")
        
        # Remove unnecessary columns
        cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        df_cleaned = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        # Handle missing values
        missing_values = df_cleaned.isnull().sum()
        if missing_values.any():
            print(f"Missing values found: {missing_values[missing_values > 0]}")
            # For numerical columns, fill with median
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        
        # Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"Final shape after cleaning: {df_cleaned.shape}")
        
        return df_cleaned
    
    def feature_engineering(self, df):
        """Create new features based on existing ones."""
        print("=== Feature Engineering ===")
        df_fe = df.copy()
        
        # Credit Utilization (only if CreditScore and Balance exist)
        if 'CreditScore' in df_fe.columns and 'Balance' in df_fe.columns:
            df_fe['CreditUtilization'] = df_fe['Balance'] / df_fe['CreditScore']
        
        # Interaction Score (only if all components exist)
        interaction_cols = ['NumOfProducts', 'HasCrCard', 'IsActiveMember']
        if all(col in df_fe.columns for col in interaction_cols):
            df_fe['InteractionScore'] = df_fe['NumOfProducts'] + df_fe['HasCrCard'] + df_fe['IsActiveMember']
        
        # Balance to Salary Ratio (only if both exist)
        if 'Balance' in df_fe.columns and 'EstimatedSalary' in df_fe.columns:
            df_fe['BalanceToSalaryRatio'] = df_fe['Balance'] / df_fe['EstimatedSalary']
        
        # Credit Score Age Interaction (only if both exist)
        if 'CreditScore' in df_fe.columns and 'Age' in df_fe.columns:
            df_fe['CreditScoreAgeInteraction'] = df_fe['CreditScore'] * df_fe['Age']
        
        # Create Credit Score Groups (only if CreditScore exists)
        if 'CreditScore' in df_fe.columns:
            bins = [0, 669, 739, 850]
            labels = ['Low', 'Medium', 'High']
            df_fe['CreditScoreGroup'] = pd.cut(df_fe['CreditScore'], bins=bins, labels=labels, include_lowest=True)
        
        print(f"Features after engineering: {df_fe.shape[1]}")
        return df_fe
    
    def encode_categorical(self, df, target_col='Exited'):
        """Encode categorical variables."""
        print("=== Encoding Categorical Variables ===")
        df_encoded = df.copy()
        
        # Find categorical columns (excluding target)
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
            print(f"Encoded {col}")
        
        return df_encoded
    
    def prepare_features_target(self, df, target_col='Exited'):
        """Separate features and target variable."""
        if target_col not in df.columns:
            # Try alternative target column names
            target_alternatives = ['Churn', 'churn', 'target', 'Target']
            for alt in target_alternatives:
                if alt in df.columns:
                    target_col = alt
                    break
            else:
                raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Ensure target is binary (0/1)
        if y.dtype == 'object':
            if set(y.unique()) == {'Yes', 'No'}:
                y = y.map({'Yes': 1, 'No': 0})
            elif set(y.unique()) == {'True', 'False'}:
                y = y.map({'True': 1, 'False': 0})
        
        self.feature_names = X.columns.tolist()
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features."""
        print("=== Feature Scaling ===")
        
        # Identify numerical columns that need scaling
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude binary columns
        binary_cols = []
        for col in numerical_cols:
            if set(X_train[col].unique()).issubset({0, 1}):
                binary_cols.append(col)
        
        cols_to_scale = [col for col in numerical_cols if col not in binary_cols]
        
        if cols_to_scale:
            X_train_scaled = X_train.copy()
            X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
            
            if X_test is not None:
                X_test_scaled = X_test.copy()
                X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])
                return X_train_scaled, X_test_scaled
            
            return X_train_scaled
        
        return X_train, X_test if X_test is not None else X_train
    
    def handle_imbalance(self, X_train, y_train, strategy='SMOTE'):
        """Handle class imbalance using oversampling techniques."""
        print("=== Handling Class Imbalance ===")
        print(f"Original class distribution: {y_train.value_counts().to_dict()}")
        
        if strategy == 'SMOTE':
            # SMOTE disabled due to compatibility issues
            print("⚠️ SMOTE not available due to sklearn compatibility. Using original data.")
            return X_train, y_train
        
        return X_train, y_train
    
    def full_preprocessing_pipeline(self, filepath, target_col='Exited', test_size=0.3, 
                                   handle_imbalance=True, random_state=42):
        """Complete preprocessing pipeline."""
        print("=== Starting Full Preprocessing Pipeline ===")
        
        # Load data
        df = self.load_data(filepath)
        if df is None:
            return None
        
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df, target_col)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train, X_test = self.scale_features(X_train, X_test)
        
        # Handle imbalance
        if handle_imbalance:
            X_train, y_train = self.handle_imbalance(X_train, y_train)
        
        print("=== Preprocessing Complete ===")
        print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_test, y_train, y_test


def get_basic_data_info(df):
    """Get basic information about the dataset."""
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {col: df[col].value_counts().to_dict() 
                              for col in df.select_dtypes(include=['object', 'category']).columns}
    }
    return info
