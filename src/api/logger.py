"""
Shadow Mode Logger.

Structured JSON logging for shadow predictions (production testing).
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any


# Configure shadow prediction logger
shadow_logger = logging.getLogger("shadow_predictions")
shadow_logger.setLevel(logging.INFO)

# File handler for shadow predictions
handler = logging.FileHandler("logs/shadow_predictions.jsonl")
handler.setFormatter(logging.Formatter("%(message)s"))
shadow_logger.addHandler(handler)


def log_shadow_prediction(
    request_data: Dict[str, Any], probability: float, real_decision: str, latency_ms: float
) -> None:
    """
    Log a shadow prediction for comparison with production.

    Shadow mode allows testing new models in production without
    affecting real transactions. All decisions are logged but
    the API always returns APPROVE to the user.

    Args:
        request_data: Original request payload
        probability: Model's fraud probability
        real_decision: What the model would have decided (BLOCK/APPROVE)
        latency_ms: Processing time

    Example log entry:
        {
            "timestamp": "2020-06-15T14:30:00Z",
            "user_id": "u12345",
            "amt": 150.0,
            "real_decision": "BLOCK",
            "probability": 0.923,
            "latency_ms": 12.5,
            "shadow_mode": true
        }
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": request_data.get("user_id"),
        "amt": float(request_data.get("amt")) if request_data.get("amt") is not None else None,
        "category": request_data.get("category"),
        "real_decision": real_decision,
        "probability": float(probability),  # Convert numpy float32 to Python float
        "latency_ms": float(latency_ms),
        "shadow_mode": True,
    }

    shadow_logger.info(json.dumps(log_entry))


__all__ = ["log_shadow_prediction", "shadow_logger"]
