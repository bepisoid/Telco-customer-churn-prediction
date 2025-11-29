from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from churn.preprocessing import load_data, build_preprocessing_pipeline, DataCleaner
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import pandas as pd
import warnings
import joblib
import os

# Ignore warnings about feature names (does not affect model performance)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def main():
    # Use path relative to project root to ensure it runs from anywhere
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'data', 'Telco-Customer-Churn.csv')
    cleaned_data_path = os.path.join(base_dir, 'data', 'Telco-Customer-Churn-Cleaned.csv')
    models_dir = os.path.join(base_dir, 'models')

    # Load original dataset
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return
    
    # Split into features and target variable
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})  # Encoding only target variable

    # Clean data from NaN and transform binary features
    cleaner = DataCleaner()
    X_clean, y_clean = cleaner.transform(X, y)

    # SAVE FOR DEBUG (Fully Numeric)
    # We apply get_dummies ONLY for the CSV export. 
    # The training pipeline handles encoding internally, keeping the model flexible.
    print("Encoding data for CSV export...")
    X_for_csv = pd.get_dummies(X_clean, drop_first=True, dtype=int)
    
    full_clean_df = X_for_csv.copy()
    full_clean_df['Churn'] = y_clean
    full_clean_df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned (numeric) dataset saved to: {cleaned_data_path}")
    # ------------------------------------------

    # Split into train and holdout
    X_temp_clean, X_holdout_clean, y_temp_clean, y_holdout_clean = train_test_split(
        X_clean, y_clean, test_size=0.15, stratify=y_clean, random_state=42)

    # Additional split for validation
    X_train_clean, X_val_clean, y_train_clean, y_val_clean = train_test_split(
        X_temp_clean, y_temp_clean, test_size=0.18, stratify=y_temp_clean, random_state=42)

    # Create full pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', build_preprocessing_pipeline()),
        ('classifier', LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            verbose=-1
        ))
    ])

    # LightGBM parameter tuning
    param_dist = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    # RandomizedSearchCV for hyperparameter tuning
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    # Train model on cleaned data
    print("Starting training...")
    search.fit(X_train_clean, y_train_clean)
    best_pipeline = search.best_estimator_

    # Metrics on validation set
    y_pred = best_pipeline.predict(X_val_clean)
    y_pred_proba = best_pipeline.predict_proba(X_val_clean)[:, 1]

    print(f'Best params: {search.best_params_}')
    print(f'ROC AUC: {roc_auc_score(y_val_clean, y_pred_proba)}')
    print(f'Precision: {precision_score(y_val_clean, y_pred)}')
    print(f'Recall: {recall_score(y_val_clean, y_pred)}')
    print(f'F1: {f1_score(y_val_clean, y_pred)}')
    print(f'Confusion matrix: \n{confusion_matrix(y_val_clean, y_pred)}')

    # Final check on holdout set
    y_holdout_pred = best_pipeline.predict(X_holdout_clean)
    y_holdout_proba = best_pipeline.predict_proba(X_holdout_clean)[:, 1]

    print("Final holdout evaluation:")
    print(f"AUC: {roc_auc_score(y_holdout_clean, y_holdout_proba):.3f}")
    print(f"Precision: {precision_score(y_holdout_clean, y_holdout_pred):.3f}")
    print(f"Recall: {recall_score(y_holdout_clean, y_holdout_pred):.3f}")
    print(f"F1: {f1_score(y_holdout_clean, y_holdout_pred):.3f}")

    # Save to root models folder
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(best_pipeline, os.path.join(models_dir, 'lightgbm_pipeline.joblib'))
    joblib.dump(cleaner, os.path.join(models_dir, 'data_cleaner.joblib'))
    print(f"Pipeline saved to {os.path.join(models_dir, 'lightgbm_pipeline.joblib')}")
    print(f"Data cleaner saved to {os.path.join(models_dir, 'data_cleaner.joblib')}")

if __name__ == '__main__':
    main()