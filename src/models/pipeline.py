"""
Feature Engineering Pipeline.

Constructs a robust Scikit-Learn pipeline for fraud detection.
Includes custom transformers for feature extraction and standard transformers
for scaling and encoding.

Derived from notebook analysis:
- Categorical: WOE Encoding (job, category)
- Numerical: Robust Scaling (amt, distance)
- Time: Cyclical encoding (sin/cos)
- Geo: Haversine distance
"""

from typing import Dict

import numpy as np
import pandas as pd
from category_encoders import WOEEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


class FraudFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to compute derived features for fraud detection.

    Implements feature engineering logic from research notebook:
    1. Distance calculation (Haversine)
    2. Cyclical time features (hour/day sin/cos)
    3. Log transformations (amount, time diff)
    4. Age calculation
    5. Ratio features (if not already computed)
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Compute derived features.

        Args:
            X: DataFrame with raw columns

        Returns:
            DataFrame with additional feature columns
        """
        # Avoid modifying original dataframe
        X = X.copy()

        # 1. Date/Time Features
        if "trans_date_trans_time" in X.columns:
            # Convert to datetime if string
            if X["trans_date_trans_time"].dtype == "object":
                X["trans_date_trans_time"] = pd.to_datetime(X["trans_date_trans_time"])

            dt = X["trans_date_trans_time"].dt

            # Cyclical encoding for hour (0-23)
            X["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
            X["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)

            # Cyclical encoding for day of week (0-6)
            X["day_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
            X["day_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

            # Calculate Age from DOB
            if "dob" in X.columns:
                if X["dob"].dtype == "object":
                    X["dob"] = pd.to_datetime(X["dob"])
                # Approximation: (Dataset Year - DOB Year)
                # Using transaction year
                X["age"] = dt.year - X["dob"].dt.year

        # 2. Geolocation Features (Haversine Distance)
        if all(c in X.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
            X["distance_km"] = self._haversine_distance(
                X["lat"], X["long"], X["merch_lat"], X["merch_long"]
            )

        # 3. Log Transformations
        if "amt" in X.columns:
            X["amt_log"] = np.log1p(X["amt"])

        # 4. Gender Mapping (M=1, F=0)
        if "gender" in X.columns:
            X["gender"] = X["gender"].map({"M": 1, "F": 0}).astype(int)

        return X

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees).
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r


def create_fraud_pipeline(params: Dict[str, any]) -> Pipeline:
    """
    Create a complete training pipeline.

    Args:
        params: Dictionary of hyperparameters for XGBoost and encoders.

    Returns:
        Sklearn Pipeline: FeatureExtraction -> ColumnTransformer -> XGBClassifier
    """

    # Define feature groups
    categorical_features = ["job", "category"]

    # Numerical features to scale (continuous, unbounded)
    numerical_features = [
        "amt_log",
        "age",
        "distance_km",
        "trans_count_24h",
        "amt_to_avg_ratio_24h",
        "amt_relative_to_all_time",
    ]

    # Binary features (0/1, no processing needed)
    binary_features = ["gender"]

    # Cyclical features (already normalized to -1 to 1, no processing needed)
    cyclical_features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", WOEEncoder(sigma=0.05, regularization=1.0), categorical_features),
            ("num", RobustScaler(), numerical_features),
            ("binary", "passthrough", binary_features),
            ("cyclical", "passthrough", cyclical_features),
        ],
        remainder="drop",  # Drop unused columns (like raw lat/long/timestamps)
        verbose_feature_names_out=False,
    )

    # Full Pipeline
    pipeline = Pipeline(
        [
            ("features", FraudFeatureExtractor()),
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    tree_method="hist",
                    max_depth=params.get("max_depth", 6),
                    learning_rate=params.get("learning_rate", 0.1),
                    n_estimators=params.get("n_estimators", 100),
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=params.get("scale_pos_weight", 100),  # Handle class imbalance
                ),
            ),
        ]
    )

    return pipeline
