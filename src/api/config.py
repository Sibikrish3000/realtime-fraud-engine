"""
API Configuration.

Environment-based configuration using Pydantic settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Usage:
        >>> from src.api.config import settings
        >>> print(settings.model_path)
    """

    # Model paths
    model_path: str = "models/fraud_model.pkl"
    threshold_path: str = "models/threshold.json"

    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # Feature flags
    shadow_mode: bool = False
    enable_explainability: bool = False

    # Performance
    max_latency_ms: float = 50.0

    # API metadata
    api_version: str = "1.0.0"
    api_title: str = "PayShield Fraud Detection API"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra fields in .env
    }


# Global settings instance
settings = Settings()


__all__ = ["settings", "Settings"]
