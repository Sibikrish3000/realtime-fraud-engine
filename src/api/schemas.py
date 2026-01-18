"""
API Request/Response Schemas.

Pydantic models for API contract validation.
"""

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request schema for fraud prediction endpoint.

    Contains transaction details needed for real-time fraud detection.
    Matches the InferenceTransactionSchema from data ingestion.
    """

    user_id: str = Field(..., description="Unique user identifier (replaces cc_num for privacy)")
    trans_date_trans_time: str = Field(
        ..., description="Transaction timestamp (YYYY-MM-DD HH:MM:SS)"
    )
    amt: float = Field(..., gt=0, description="Transaction amount")
    lat: float = Field(..., ge=-90, le=90, description="User latitude")
    long: float = Field(..., ge=-180, le=180, description="User longitude")
    merch_lat: float = Field(..., ge=-90, le=90, description="Merchant latitude")
    merch_long: float = Field(..., ge=-180, le=180, description="Merchant longitude")
    job: str = Field(..., description="User occupation")
    category: str = Field(..., description="Merchant category")
    gender: str = Field(..., description="User gender (M/F)")
    dob: str = Field(..., description="User date of birth (YYYY-MM-DD)")

    # Optional Feature Overrides for Analysis
    trans_count_24h: Optional[float] = None
    avg_spend_24h: Optional[float] = None
    amt_to_avg_ratio_24h: Optional[float] = None
    user_avg_amt_all_time: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "u12345",
                "trans_date_trans_time": "2020-06-15 14:30:00",
                "amt": 150.00,
                "lat": 40.7128,
                "long": -74.0060,
                "merch_lat": 40.7200,
                "merch_long": -74.0100,
                "job": "Engineer, biomedical",
                "category": "grocery_pos",
                "gender": "M",
                "dob": "1985-03-20",
                "trans_count_24h": 5,  # Optional override
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for fraud prediction endpoint."""

    decision: Literal["BLOCK", "APPROVE"] = Field(..., description="Final decision")
    probability: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    shadow_mode: bool = Field(default=False, description="Whether shadow mode is active")
    features: Dict[str, Any] = Field(
        default_factory=dict, description="Features used for inference"
    )
    shap_values: Dict[str, float] = Field(
        default_factory=dict, description="SHAP feature contributions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "decision": "BLOCK",
                "probability": 0.923,
                "risk_score": 92.3,
                "latency_ms": 12.5,
                "shadow_mode": False,
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    redis_connected: bool
    version: str


__all__ = ["PredictionRequest", "PredictionResponse", "HealthResponse"]
