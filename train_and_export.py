import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier

def load_or_generate_data(sample_size=100000):
    dataset_path = "/Users/vishnuganugula/.cache/kagglehub/datasets/ealaxi/paysim1/versions/2/PS_20174392719_1491204439457_log.csv"
    if os.path.exists(dataset_path):
        print(f"Loading dataset from cached path: {dataset_path}")
        df = pd.read_csv(dataset_path)
        if sample_size and len(df) > sample_size:
            print(f"Sampling {sample_size} rows for fast export...")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        print("Dataset not found locally. Generating synthetic PaySim transactions for pipeline serialization...")
        np.random.seed(42)
        n_samples = sample_size or 10000
        types = np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], size=n_samples, p=[0.35, 0.25, 0.25, 0.05, 0.10])
        amounts = np.random.exponential(scale=50000, size=n_samples) + 10
        oldbal_org = np.random.exponential(scale=100000, size=n_samples)
        newbal_org = np.maximum(0, oldbal_org - amounts)
        oldbal_dest = np.random.exponential(scale=200000, size=n_samples)
        newbal_dest = oldbal_dest + amounts
        steps = np.random.randint(1, 744, size=n_samples)
        is_fraud = np.where((types == 'TRANSFER') & (amounts > 200000), np.random.choice([0, 1], p=[0.1, 0.9]), 0)
        
        df = pd.DataFrame({
            'step': steps,
            'type': types,
            'amount': amounts,
            'nameOrig': [f'C{np.random.randint(100000, 999999)}' for _ in range(n_samples)],
            'oldbalanceOrg': oldbal_org,
            'newbalanceOrig': newbal_org,
            'nameDest': [f'M{np.random.randint(100000, 999999)}' for _ in range(n_samples)],
            'oldbalanceDest': oldbal_dest,
            'newbalanceDest': newbal_dest,
            'isFraud': is_fraud,
            'isFlaggedFraud': 0
        })
    return df

def feature_engineering(df):
    df = df.copy()
    df['hour_of_day'] = df['step'] % 24
    df = df.sort_values(by=['nameOrig', 'step']).reset_index(drop=True)
    
    df['txn_count_rolling'] = df.groupby('nameOrig').cumcount() + 1
    df['txn_amount_rolling'] = df.groupby('nameOrig')['amount'].cumsum()
    
    df['orig_balance_inconsistency'] = df['newbalanceOrig'] - (df['oldbalanceOrg'] - df['amount'])
    df['dest_balance_inconsistency'] = df['newbalanceDest'] - (df['oldbalanceDest'] + df['amount'])
    return df

def train_and_export():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    df = load_or_generate_data(sample_size=150000)
    df_engineered = feature_engineering(df)
    
    numeric_cols = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'txn_count_rolling', 'txn_amount_rolling', 'orig_balance_inconsistency', 'dest_balance_inconsistency'
    ]
    categorical_cols = ['type']
    
    print("Fitting ColumnTransformer...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'
    )
    
    y = df_engineered['isFraud'].values
    X_base = preprocessor.fit_transform(df_engineered)
    
    print("Fitting Gaussian Mixture Model (GMM)...")
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    cluster_probs = gmm.fit_predict(X_base)
    cluster_probabilities = gmm.predict_proba(X_base)
    
    print("Fitting Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_forest.fit(X_base)
    anomaly_scores = iso_forest.decision_function(X_base).reshape(-1, 1)
    
    X_enriched = np.hstack((X_base, cluster_probabilities, anomaly_scores))
    print(f"Enriched feature matrix shape: {X_enriched.shape}")
    
    print("Initializing Ensemble Learners & Stacking Classifier...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    svm_approx = SGDClassifier(loss='modified_huber', class_weight='balanced', random_state=42)
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42, replacement=True)
    meta_learner = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=[
            ('lr', lr),
            ('svm_approx', svm_approx),
            ('brf', brf)
        ],
        final_estimator=meta_learner,
        cv=3,
        n_jobs=-1
    )
    
    print("Fitting Calibrated Classifier...")
    calibrated_fraud_engine = CalibratedClassifierCV(
        estimator=stacking_clf,
        method='isotonic',
        cv=3
    )
    calibrated_fraud_engine.fit(X_enriched, y)
    
    print("Saving serialized model artifacts to models/...")
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(gmm, os.path.join(models_dir, "gmm.joblib"))
    joblib.dump(iso_forest, os.path.join(models_dir, "iso_forest.joblib"))
    joblib.dump(calibrated_fraud_engine, os.path.join(models_dir, "calibrated_fraud_engine.joblib"))
    
    metadata = {
        "model_name": "High-Frequency Transaction Fraud Engine",
        "optimal_threshold": 0.09,
        "n_enriched_features": X_enriched.shape[1],
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "unsupervised_components": 5,
        "contamination": 0.01
    }
    with open(os.path.join(models_dir, "pipeline_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Model serialization complete! All artifacts successfully saved.")

if __name__ == "__main__":
    train_and_export()
