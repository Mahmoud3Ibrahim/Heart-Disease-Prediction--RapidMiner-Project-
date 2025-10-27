"""
Data preprocessing utilities for heart disease prediction.
Handles data cleaning, feature engineering, and normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class HeartDiseasePreprocessor:
    """Preprocessor for heart disease dataset."""

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}

    def load_data(self, filepath):
        """Load heart disease dataset from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} features")
        return df

    def remove_duplicates(self, df):
        """Remove duplicate records from dataset."""
        original_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = original_count - len(df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate records")
        return df

    def handle_missing_values(self, df):
        """Replace missing values with column averages for numerical features."""
        # Numerical columns
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

        for col in numerical_cols:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                print(f"Filled {col} missing values with mean: {mean_val:.2f}")

        # Categorical columns - fill with mode
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled {col} missing values with mode: {mode_val}")

        return df

    def create_risk_categories(self, df):
        """Create categorical risk levels from numerical features."""

        # Age groups
        df['Age_Group'] = pd.cut(df['Age'],
                                 bins=[0, 19, 35, 50, 60, 70, 120],
                                 labels=['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior', 'Elderly'])

        # Cholesterol levels
        df['Cholesterol_Level'] = pd.cut(df['Cholesterol'],
                                         bins=[-1, 0, 200, 240, 600],
                                         labels=['Zero', 'Normal', 'Borderline', 'High'])

        # MaxHR levels
        df['MaxHR_Level'] = pd.cut(df['MaxHR'],
                                   bins=[0, 100, 150, 250],
                                   labels=['Low', 'Normal', 'High'])

        # Oldpeak risk
        df['Oldpeak_Risk'] = pd.cut(df['Oldpeak'],
                                    bins=[-1, 1, 2, 10],
                                    labels=['Normal', 'Moderate', 'Severe'])

        # RestingBP levels
        df['RestingBP_Level'] = pd.cut(df['RestingBP'],
                                       bins=[0, 120, 130, 300],
                                       labels=['Normal', 'Elevated', 'High'])

        return df

    def encode_categorical_features(self, df, fit=True):
        """Convert categorical features to numerical using label encoding."""
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])

        return df

    def normalize_features(self, df, columns=None, fit=True):
        """Normalize numerical features to 0-1 range."""
        if columns is None:
            columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

        if fit:
            df[columns] = self.scaler.fit_transform(df[columns])
        else:
            df[columns] = self.scaler.transform(df[columns])

        return df

    def prepare_for_clustering(self, df):
        """Prepare dataset for clustering analysis."""
        # Make a copy to avoid modifying original
        df_processed = df.copy()

        # Remove duplicates
        df_processed = self.remove_duplicates(df_processed)

        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)

        # Encode categorical features
        df_processed = self.encode_categorical_features(df_processed, fit=True)

        # Normalize numerical features
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        df_processed = self.normalize_features(df_processed, numerical_cols, fit=True)

        return df_processed

    def prepare_for_classification(self, df):
        """Prepare dataset for decision tree classification."""
        # Make a copy
        df_processed = df.copy()

        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)

        # Create risk categories
        df_processed = self.create_risk_categories(df_processed)

        # Convert categorical to numerical
        df_processed = self.encode_categorical_features(df_processed, fit=True)

        # Encode new categorical features
        new_categorical = ['Age_Group', 'Cholesterol_Level', 'MaxHR_Level',
                          'Oldpeak_Risk', 'RestingBP_Level']

        for col in new_categorical:
            self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])

        return df_processed
