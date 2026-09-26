from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import json
from app.schemas import TransactionInput, BatchTransactionInput, PredictionResult, HealthStatus
from app.pipeline import FraudEnginePipeline

pipeline: FraudEnginePipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        try:
            pipeline = FraudEnginePipeline()
        except Exception as e:
            print(f"Error loading pipeline: {e}")
    return pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    yield

app = FastAPI(
    title="High-Frequency Transaction Fraud Engine API",
    description="Containerized MLOps Microservice for Real-Time Fraud Detection using scikit-learn & LightGBM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
def root():
    return {
        "service": "High-Frequency Transaction Fraud Engine",
        "status": "ONLINE",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthStatus, tags=["Monitoring"])
def health_check():
    p = get_pipeline()
    is_loaded = p is not None
    threshold = p.optimal_threshold if is_loaded else 0.09
    return HealthStatus(
        status="HEALTHY" if is_loaded else "UNHEALTHY",
        model_loaded=is_loaded,
        threshold=threshold,
        version="1.0.0"
    )

@app.post("/predict_fraud", response_model=PredictionResult, tags=["Inference"])
def predict_fraud(transaction: TransactionInput):
    p = get_pipeline()
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud engine model pipeline is not initialized."
        )
    try:
        result = p.predict_single(transaction.model_dump())
        return PredictionResult(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/predict_batch", tags=["Inference"])
def predict_batch(batch: BatchTransactionInput):
    p = get_pipeline()
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud engine model pipeline is not initialized."
        )
    try:
        txns = [t.model_dump() for t in batch.transactions]
        results = p.predict_batch(txns)
        return {"total_transactions": len(results), "results": results}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )

@app.post("/explain_fraud", tags=["Governance & Explainability"])
def explain_fraud(transaction: TransactionInput):
    p = get_pipeline()
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud engine model pipeline is not initialized."
        )
    try:
        explanation = p.explain(transaction.model_dump())
        return explanation
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability error: {str(e)}"
        )
