"""
FastAPI Real-Time Fraud Detection Service.

Production-grade inference API with sub-50ms latency target.
Integrates with Redis Feature Store for real-time feature injection.
"""

import json
import logging
import time
from pathlib import Path
import pandas as pd
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.logger import log_shadow_prediction
from src.api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.features.store import RedisFeatureStore
from src.explainability import FraudExplainer


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Real-time fraud detection API with Redis feature store integration",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global resources (loaded on startup)
pipeline = None
threshold = None
feature_store: Optional[RedisFeatureStore] = None
explainer: Optional[FraudExplainer] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def load_resources():
    """
    Load model and initialize Redis on startup.

    This runs once when the API starts, avoiding per-request overhead.
    """
    global pipeline, threshold, feature_store, explainer

    logger.info("Loading model and resources...")

    # Load trained pipeline
    model_path = Path(settings.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    pipeline = joblib.load(model_path)
    logger.info(f"✓ Loaded model from {model_path}")

    # Load optimal threshold
    threshold_path = Path(settings.threshold_path)
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)
        threshold = threshold_data["optimal_threshold"]

    logger.info(f"✓ Loaded threshold: {threshold:.4f}")

    # Initialize Redis Feature Store
    try:
        feature_store = RedisFeatureStore(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
        )
        logger.info("✓ Connected to Redis Feature Store")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Feature store disabled.")
        feature_store = None

    # Initialize SHAP Explainer
    try:
        explainer = FraudExplainer(str(model_path))
        logger.info("✓ Initialized SHAP Explainer")
    except Exception as e:
        logger.warning(f"SHAP initialization failed: {e}. Explainability disabled.")
        explainer = None

    logger.info("=" * 60)
    logger.info("API Ready!")
    logger.info(f"Shadow Mode: {settings.shadow_mode}")
    logger.info(f"Max Latency Target: {settings.max_latency_ms}ms")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_resources():
    """Clean up resources on shutdown."""
    global feature_store

    if feature_store:
        feature_store.close()
        logger.info("✓ Closed Redis connection")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns service status and resource availability.
    """
    redis_connected = False
    if feature_store:
        try:
            health = feature_store.health_check()
            redis_connected = health["status"] == "healthy"
        except Exception:
            redis_connected = False

    status = "healthy" if (pipeline is not None and threshold is not None) else "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=pipeline is not None,
        redis_connected=redis_connected,
        version=settings.api_version,
    )


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Real-time fraud detection endpoint.

    Workflow:
    1. Parse & validate request
    2. Query Redis for real-time features (trans_count_24h, avg_spend_24h)
    3. Combine features + run inference
    4. Apply decision threshold
    5. Shadow mode override (if enabled)
    6. Return decision with latency tracking

    Args:
        request: Transaction data

    Returns:
        Fraud decision with probability and latency

    Raises:
        HTTPException: If model not loaded or validation fails
    """
    start_time = time.time()

    # Verify resources are loaded
    if pipeline is None or threshold is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Model not loaded")

    try:
        # Step 1: Convert request to dict
        request_data = request.dict()

        # Step 2: Query Redis/Use Overrides
        # Priority: Override > Redis > Default
        trans_count_24h = request.trans_count_24h
        avg_spend_24h = request.avg_spend_24h
        amt_to_avg_ratio_24h = request.amt_to_avg_ratio_24h
        user_avg_amt_all_time = request.user_avg_amt_all_time

        # If any real-time feature is missing from overrides, try Redis
        if (
            trans_count_24h is None or avg_spend_24h is None or user_avg_amt_all_time is None
        ) and feature_store:
            try:
                # Uses transaction timestamp for time-based lookup
                trans_time = pd.to_datetime(request.trans_date_trans_time)
                timestamp = int(trans_time.timestamp())

                features = feature_store.get_features(request.user_id, timestamp)

                if trans_count_24h is None:
                    trans_count_24h = features.get("trans_count_24h", 0)

                if avg_spend_24h is None:
                    avg_spend_24h = features.get("avg_spend_24h", request.amt)

                # Note: Redis Feature Store doesn't currently track all-time average
                # This would need to be added to the Feature Store implementation
                # For now, we'll use avg_spend_24h as a proxy if not overridden
                if user_avg_amt_all_time is None:
                    user_avg_amt_all_time = features.get("user_avg_amt_all_time", avg_spend_24h)

            except Exception as e:
                logger.warning(
                    f"Redis feature lookup failed: {e}. Using defaults for missing values."
                )

        # Fill remaining defaults
        if trans_count_24h is None:
            trans_count_24h = 0
        if avg_spend_24h is None:
            avg_spend_24h = request.amt
        if user_avg_amt_all_time is None:
            user_avg_amt_all_time = avg_spend_24h  # Use 24h avg as proxy

        # Calculate derived ratio if not overridden
        if amt_to_avg_ratio_24h is None:
            amt_to_avg_ratio_24h = request.amt / avg_spend_24h if avg_spend_24h > 0 else 1.0

        # Inject into request data
        request_data["trans_count_24h"] = trans_count_24h
        request_data["avg_spend_24h"] = avg_spend_24h
        request_data["amt_to_avg_ratio_24h"] = amt_to_avg_ratio_24h
        request_data["amt_relative_to_all_time"] = 1.0  # Default if not computed

        # Step 3: Convert to DataFrame for pipeline
        df = pd.DataFrame([request_data])

        # Step 4: Inference
        prob = pipeline.predict_proba(df)[:, 1][0]

        # Step 5: Apply threshold
        real_decision = "BLOCK" if prob >= threshold else "APPROVE"

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Step 6: Shadow mode override
        final_decision = real_decision
        if settings.shadow_mode:
            log_shadow_prediction(
                request_data=request_data,
                probability=prob,
                real_decision=real_decision,
                latency_ms=latency_ms,
            )
            # But always approve in shadow mode
            final_decision = "APPROVE"

        # Log performance warning if latency exceeds target
        if latency_ms > settings.max_latency_ms:
            logger.warning(
                f"Latency exceeded target: {latency_ms:.2f}ms > {settings.max_latency_ms}ms"
            )

        # Capture features used for response
        features_used = {
            "trans_count_24h": trans_count_24h,
            "avg_spend_24h": avg_spend_24h,
            "amt_to_avg_ratio_24h": amt_to_avg_ratio_24h,
            "user_avg_amt_all_time": user_avg_amt_all_time,  # Now uses real/override value
        }

        # Calculate SHAP values if explainer is available
        shap_contributions = {}
        if explainer is not None and settings.enable_explainability:
            try:
                explanation = explainer.explain_prediction(df, threshold=threshold)
                # Get top 5 features by absolute impact
                shap_contributions = {
                    item["feature"]: item["impact"] for item in explanation["top_features"]
                }
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")

        # Persist transaction to Redis (if no overrides were used and not in shadow mode)
        # This ensures velocity features accumulate for future predictions
        if feature_store and not settings.shadow_mode:
            # Only persist if user didn't override features (to avoid polluting real data)
            no_overrides = (
                request.trans_count_24h is None
                and request.avg_spend_24h is None
                and request.user_avg_amt_all_time is None
            )
            if no_overrides:
                try:
                    trans_time = pd.to_datetime(request.trans_date_trans_time)
                    timestamp = int(trans_time.timestamp())
                    feature_store.add_transaction(
                        user_id=request.user_id, amount=request.amt, timestamp=timestamp
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist transaction to Redis: {e}")

        return PredictionResponse(
            decision=final_decision,
            probability=float(prob),
            risk_score=float(prob * 100),
            latency_ms=latency_ms,
            shadow_mode=settings.shadow_mode,
            features=features_used,
            shap_values=shap_contributions,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "status": "running",
        "endpoints": {
            "predict": "/v1/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)",
        },
    }


__all__ = ["app"]
