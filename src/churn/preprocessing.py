from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

# Load data (I love DRY)
def load_data(path: str) -> pd.DataFrame:
    # index_col=None ensures we don't accidentally use a feature as an index
    df = pd.read_csv(path, index_col=None)
    return df

# Transformer for data cleaning
class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # Explicitly drop identification columns immediately
        if 'customerID' in X.columns:
            X = X.drop(['customerID'], axis=1)

        # Replace 'No internet service' with 'No'
        service_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        for col in service_cols:
            if col in X.columns:
                X[col] = X[col].replace('No internet service', 'No')

        # Encode binary features to 0/1
        binary_cols = [
            'Partner', 'Dependents', 'PhoneService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'PaperlessBilling'
        ]
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map({'Yes': 1, 'No': 0})

        # Convert TotalCharges to numeric format
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            
            # Remove rows with missing values in TotalCharges
            mask = ~X['TotalCharges'].isna()
            X = X[mask]
            
            # Sync target variable if provided
            if y is not None:
                y = y[mask] if hasattr(y, '__array__') else y
        
        # Remove non-informative features according to EDA
        cols_to_drop = ['gender', 'Partner']
        X = X.drop(cols_to_drop, axis=1, errors='ignore')
        
        if y is not None:
            return X, y
        return X

# Create preprocessing pipeline
def build_preprocessing_pipeline() -> Pipeline:
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']

    preprocessing_pipeline = Pipeline([
        ('feature_processing', ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough'))
    ])
    
    return preprocessing_pipeline