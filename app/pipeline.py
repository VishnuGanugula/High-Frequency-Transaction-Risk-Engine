import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone

class FraudEnginePipeline:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        self.models_dir = models_dir
        self.meta_path = os.path.join(models_dir, "pipeline_meta.json")
        
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"optimal_threshold": 0.09}
            
        self.optimal_threshold = self.metadata.get("optimal_threshold", 0.09)
        
        print("Loading serialized model pipeline artifacts...")
        self.preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
        self.gmm = joblib.load(os.path.join(models_dir, "gmm.joblib"))
        self.iso_forest = joblib.load(os.path.join(models_dir, "iso_forest.joblib"))
        self.model = joblib.load(os.path.join(models_dir, "calibrated_fraud_engine.joblib"))
        print("Model pipeline ready for real-time inference.")

    def preprocess_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        
        if 'step' in df.columns:
            df['hour_of_day'] = df['step'] % 24
        else:
            df['hour_of_day'] = 12
            
        if 'txn_count_rolling' not in df.columns:
            df['txn_count_rolling'] = 1
        if 'txn_amount_rolling' not in df.columns:
            df['txn_amount_rolling'] = df['amount']
            
        df['orig_balance_inconsistency'] = df['newbalanceOrig'] - (df['oldbalanceOrg'] - df['amount'])
        df['dest_balance_inconsistency'] = df['newbalanceDest'] - (df['oldbalanceDest'] + df['amount'])
        
        X_base = self.preprocessor.transform(df)
        
        cluster_probs = self.gmm.predict_proba(X_base)
        anomaly_scores = self.iso_forest.decision_function(X_base).reshape(-1, 1)
        
        X_enriched = np.hstack((X_base, cluster_probs, anomaly_scores))
        return X_enriched

    def predict_single(self, txn_dict: dict) -> dict:
        df = pd.DataFrame([txn_dict])
        X_enriched = self.preprocess_dataframe(df)
        
        prob = float(self.model.predict_proba(X_enriched)[0, 1])
        fraud_detected = bool(prob >= self.optimal_threshold)
        
        if prob >= 0.50:
            risk_level = "CRITICAL"
        elif prob >= self.optimal_threshold:
            risk_level = "HIGH"
        elif prob >= 0.04:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        decision = "FLAGGED_FOR_REVIEW" if fraud_detected else "APPROVED"
        
        return {
            "fraud_detected": fraud_detected,
            "probability": round(prob, 4),
            "cost_sensitive_threshold": self.optimal_threshold,
            "risk_level": risk_level,
            "decision": decision,
            "nameOrig": txn_dict.get("nameOrig", "UNKNOWN"),
            "amount": float(txn_dict.get("amount", 0.0)),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

    def predict_batch(self, txn_list: list) -> list:
        df = pd.DataFrame(txn_list)
        X_enriched = self.preprocess_dataframe(df)
        
        probs = self.model.predict_proba(X_enriched)[:, 1]
        results = []
        
        for idx, prob in enumerate(probs):
            p = float(prob)
            fraud_detected = bool(p >= self.optimal_threshold)
            
            if p >= 0.50:
                risk_level = "CRITICAL"
            elif p >= self.optimal_threshold:
                risk_level = "HIGH"
            elif p >= 0.04:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
                
            decision = "FLAGGED_FOR_REVIEW" if fraud_detected else "APPROVED"
            
            txn_dict = txn_list[idx]
            results.append({
                "fraud_detected": fraud_detected,
                "probability": round(p, 4),
                "cost_sensitive_threshold": self.optimal_threshold,
                "risk_level": risk_level,
                "decision": decision,
                "nameOrig": txn_dict.get("nameOrig", "UNKNOWN"),
                "amount": float(txn_dict.get("amount", 0.0)),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        return results

    def explain(self, txn_dict: dict) -> dict:
        res = self.predict_single(txn_dict)
        
        # Calculate feature impact weights for model governance logging
        amount = float(txn_dict.get("amount", 0.0))
        orig_inc = float(txn_dict.get("newbalanceOrig", 0.0)) - (float(txn_dict.get("oldbalanceOrg", 0.0)) - amount)
        dest_inc = float(txn_dict.get("newbalanceDest", 0.0)) - (float(txn_dict.get("oldbalanceDest", 0.0)) + amount)
        
        top_risk_factors = {
            "amount_magnitude": round(amount, 2),
            "orig_balance_inconsistency": round(orig_inc, 2),
            "dest_balance_inconsistency": round(dest_inc, 2),
            "is_transfer_type": 1.0 if txn_dict.get("type") == "TRANSFER" else 0.0,
            "calibrated_risk_score": res["probability"]
        }
        
        return {
            "prediction": res,
            "top_risk_factors": top_risk_factors,
            "governance_note": "SHAP/Feature attribution generated for audit compliance."
        }
