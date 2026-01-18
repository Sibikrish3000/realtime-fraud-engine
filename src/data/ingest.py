"""
Data Ingestion Module

This module handles loading and validating credit card transaction data.
It uses Pydantic for schema validation to ensure data quality before processing.

Author: PayShield-ML Team
"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator, model_validator

from src.features.constants import category_names, job_names


class TransactionSchema(BaseModel):
    """
    Pydantic model for validating individual transaction records.

    Enforces strict business rules:
    - Transaction amounts must be positive
    - Coordinates must be valid (lat: [-90, 90], long: [-180, 180])
    - Category and job must be from known sets
    - Timestamps must be valid

    Attributes:
        trans_date_trans_time: Transaction timestamp
        cc_num: Credit card number (PII - handle with care)
        merchant: Merchant name
        category: Transaction category (e.g., 'grocery_pos', 'gas_transport')
        amt: Transaction amount in USD
        first: Customer first name
        last: Customer last name
        gender: Customer gender
        street: Street address
        city: City name
        state: State code (2 letters)
        zip: ZIP code
        lat: Customer latitude (-90 to 90)
        long: Customer longitude (-180 to 180)
        city_pop: City population
        job: Customer job title
        dob: Date of birth
        trans_num: Unique transaction identifier
        unix_time: Unix timestamp
        merch_lat: Merchant latitude
        merch_long: Merchant longitude
        is_fraud: Fraud label (0 or 1)
    """

    # Transaction Details
    trans_date_trans_time: str = Field(
        ..., description="Transaction timestamp in format 'YYYY-MM-DD HH:MM:SS'"
    )
    cc_num: int = Field(..., description="Credit card number", gt=0)
    merchant: str = Field(..., min_length=1, description="Merchant name")
    category: str = Field(..., description="Transaction category")
    amt: float = Field(..., gt=0.0, description="Transaction amount (must be positive)")

    # Customer Information
    first: str = Field(..., min_length=1, description="First name")
    last: str = Field(..., min_length=1, description="Last name")
    gender: Literal["M", "F"] = Field(..., description="Gender")
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City")
    state: str = Field(..., min_length=2, max_length=2, description="State code")
    zip: int = Field(..., ge=1000, le=99999, description="ZIP code")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Customer latitude")
    long: float = Field(..., ge=-180.0, le=180.0, description="Customer longitude")
    city_pop: int = Field(..., ge=0, description="City population")
    job: str = Field(..., description="Job title")
    dob: str = Field(..., description="Date of birth in format 'YYYY-MM-DD'")

    # Transaction Metadata
    trans_num: str = Field(..., description="Unique transaction ID (hex string)")
    unix_time: int = Field(..., gt=0, description="Unix timestamp")
    merch_lat: float = Field(..., ge=-90.0, le=90.0, description="Merchant latitude")
    merch_long: float = Field(..., ge=-180.0, le=180.0, description="Merchant longitude")
    is_fraud: Literal[0, 1] = Field(..., description="Fraud indicator")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Ensure category is from known set."""
        if v not in category_names:
            raise ValueError(
                f"Invalid category '{v}'. Must be one of: {', '.join(category_names[:5])}..."
            )
        return v

    @field_validator("job")
    @classmethod
    def validate_job(cls, v: str) -> str:
        """Ensure job is from known set."""
        if v not in job_names:
            raise ValueError(
                f"Invalid job '{v}'. Must be one of the {len(job_names)} known job titles"
            )
        return v

    @field_validator("trans_date_trans_time")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Ensure timestamp is valid."""
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            raise ValueError(
                f"Invalid timestamp format '{v}'. Expected 'YYYY-MM-DD HH:MM:SS'"
            ) from e
        return v

    @field_validator("dob")
    @classmethod
    def validate_dob(cls, v: str) -> str:
        """Ensure date of birth is valid."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date of birth format '{v}'. Expected 'YYYY-MM-DD'") from e
        return v

    @model_validator(mode="after")
    def validate_distance_sanity(self) -> "TransactionSchema":
        """
        Sanity check: Ensure customer and merchant coordinates are reasonable.
        This catches data corruption where lat/long might be swapped.
        """
        # Check if coordinates are swapped (common data error)
        if abs(self.lat) > 50 and abs(self.long) < 50:
            # Likely US-based dataset, this pattern suggests swap
            raise ValueError(
                f"Suspicious coordinates: lat={self.lat}, long={self.long}. "
                f"Check if latitude and longitude are swapped."
            )
        return self


class InferenceTransactionSchema(BaseModel):
    """
    Simplified schema for real-time inference requests.

    Only includes features needed for prediction (no PII like names/addresses).
    This is what the API endpoint expects.

    Attributes:
        user_id: Internal user identifier (replaces cc_num for privacy)
        amt: Transaction amount
        lat: User's last known latitude
        long: User's last known longitude
        category: Transaction category
        job: User's job (from profile)
        merch_lat: Merchant latitude
        merch_long: Merchant longitude
        unix_time: Transaction timestamp (Unix epoch)
    """

    user_id: str = Field(..., min_length=1, description="User identifier")
    amt: float = Field(..., gt=0.0, description="Transaction amount")
    lat: float = Field(..., ge=-90.0, le=90.0, description="User latitude")
    long: float = Field(..., ge=-180.0, le=180.0, description="User longitude")
    category: str = Field(..., description="Transaction category")
    job: str = Field(..., description="User job title")
    merch_lat: float = Field(..., ge=-90.0, le=90.0, description="Merchant latitude")
    merch_long: float = Field(..., ge=-180.0, le=180.0, description="Merchant longitude")
    unix_time: int = Field(..., gt=0, description="Transaction timestamp")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Ensure category is from known set."""
        if v not in category_names:
            raise ValueError(f"Invalid category '{v}'. Must be one of: {', '.join(category_names)}")
        return v

    @field_validator("job")
    @classmethod
    def validate_job(cls, v: str) -> str:
        """Ensure job is from known set."""
        if v not in job_names:
            raise ValueError(f"Invalid job '{v}'. Not in approved job list")
        return v


def load_dataset(
    file_path: Union[str, Path], validate: bool = True, sample_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Load credit card fraud dataset from CSV or Parquet with optional validation.

    This function handles both training data loads (with validation) and
    production loads (validation optional for speed).

    Args:
        file_path: Path to CSV or Parquet file
        validate: If True, validate each row against TransactionSchema.
                 Set to False for faster loading in production.
        sample_n: If specified, return only N randomly sampled rows (for testing)

    Returns:
        DataFrame with validated transaction data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails for any row

    Example:
        >>> # Load and validate training data
        >>> df = load_dataset("fraudTrain.csv", validate=True)
        >>>
        >>> # Fast load for inference (skip validation)
        >>> df = load_dataset("fraudTrain.parquet", validate=False)
        >>>
        >>> # Load sample for testing
        >>> df_sample = load_dataset("fraudTrain.csv", sample_n=1000)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Load based on file extension
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .csv or .parquet")

    # Sample if requested
    if sample_n is not None:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)

    # Validate if requested
    if validate:
        print(f"Validating {len(df):,} transactions...")
        errors = []

        for idx, row in df.iterrows():
            try:
                TransactionSchema(**row.to_dict())
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                if len(errors) >= 10:  # Stop after 10 errors to avoid spam
                    errors.append("... (stopped after 10 errors)")
                    break

        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Validation failed:\n{error_msg}")

        print(f"✓ All {len(df):,} transactions validated successfully")

    return df


__all__ = [
    "TransactionSchema",
    "InferenceTransactionSchema",
    "load_dataset",
]
