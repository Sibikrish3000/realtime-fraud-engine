"""
SHAP Explainability Engine.

Implements regulatory-compliant explainability using SHAP (SHapley Additive exPlanations).
Provides both local (per-transaction) and global (model-wide) explanations.

Based on research notebook SHAP implementation:
- TreeExplainer for XGBoost models
- Waterfall plots for local explanations
- Summary plots for global feature importance
"""

import base64
import io
from pathlib import Path
from typing import Dict, Tuple, Union

import joblib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


class FraudExplainer:
    """
    SHAP-based explainability engine for fraud detection model.

    Provides transparent, auditable explanations for fraud predictions:
    - **Local Explanations**: Why a specific transaction was flagged (waterfall)
    - **Global Explanations**: Overall feature importance (summary plot)

    Example:
        >>> explainer = FraudExplainer("models/fraud_model.pkl")
        >>>
        >>> # Explain a single transaction
        >>> transaction = pd.DataFrame([{...}])
        >>> waterfall_b64 = explainer.generate_waterfall(transaction)
        >>>
        >>> # Global feature importance
        >>> summary_b64 = explainer.generate_summary(X_test_sample)
    """

    def __init__(self, pipeline_path: str):
        """
        Initialize SHAP explainer with trained pipeline.

        Args:
            pipeline_path: Path to saved pipeline (.pkl file)

        Raises:
            FileNotFoundError: If pipeline file doesn't exist
            ValueError: If pipeline structure is invalid
        """
        pipeline_path = Path(pipeline_path)
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")

        # Load trained pipeline
        self.pipeline: Pipeline = joblib.load(pipeline_path)

        # Extract components
        if "model" not in self.pipeline.named_steps:
            raise ValueError("Pipeline must contain 'model' step")
        if "preprocessor" not in self.pipeline.named_steps:
            raise ValueError("Pipeline must contain 'preprocessor' step")

        self.model = self.pipeline.named_steps["model"]
        self.preprocessor = self.pipeline.named_steps["preprocessor"]

        # Initialize SHAP TreeExplainer
        # TreeExplainer is optimized for tree-based models (XGBoost, RandomForest)
        self.explainer = shap.TreeExplainer(self.model)

        # Get feature names after transformation
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self) -> list:
        """
        Extract feature names from preprocessor.

        Returns:
            List of feature names after ColumnTransformer
        """
        try:
            # Try sklearn 1.0+ method
            return list(self.preprocessor.get_feature_names_out())
        except AttributeError:
            # Fallback: Manually construct from transformer configuration
            # This matches our pipeline structure:
            # cat: ['job', 'category']
            # num: ['amt_log', 'age', 'distance_km', 'trans_count_24h', ...]
            # binary: ['gender']
            # cyclical: ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            categorical = ["job", "category"]
            numerical = [
                "amt_log",
                "age",
                "distance_km",
                "trans_count_24h",
                "amt_to_avg_ratio_24h",
                "amt_relative_to_all_time",
            ]
            binary = ["gender"]
            cyclical = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
            return categorical + numerical + binary + cyclical

    def _transform_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform raw transaction data through pipeline preprocessor.

        This is the crucial step mentioned in the notebook to resolve
        "You have categorical data..." errors.

        Args:
            X: Raw transaction DataFrame

        Returns:
            Transformed numerical array ready for SHAP
        """
        # Apply feature extraction (if 'features' step exists)
        if "features" in self.pipeline.named_steps:
            X = self.pipeline.named_steps["features"].transform(X)

        # Apply preprocessing (WOE, scaling, passthrough)
        X_transformed = self.preprocessor.transform(X)

        return X_transformed

    def calculate_shap_values(
        self, X: pd.DataFrame, transformed: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate SHAP values for input data.

        Args:
            X: Transaction data (raw or transformed)
            transformed: If True, X is already transformed. If False, transform it.

        Returns:
            Tuple of (shap_values, transformed_X)
        """
        if not transformed:
            X_transformed = self._transform_data(X)
        else:
            X_transformed = X

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_transformed)

        return shap_values, X_transformed

    def generate_waterfall(
        self, transaction: pd.DataFrame, return_base64: bool = True, max_display: int = 10
    ) -> Union[str, matplotlib.figure.Figure]:
        """
        Generate SHAP waterfall plot for a single transaction.

        Shows how each feature contributed to pushing the prediction
        from the base value (average) to the final prediction.

        Args:
            transaction: Single transaction DataFrame (1 row)
            return_base64: If True, return base64 PNG. If False, return Figure.
            max_display: Maximum features to display

        Returns:
            Base64-encoded PNG string or matplotlib Figure

        Example:
            >>> waterfall_img = explainer.generate_waterfall(transaction_df)
            >>> # Save to file
            >>> with open('waterfall.png', 'wb') as f:
            ...     f.write(base64.b64decode(waterfall_img))
        """
        if len(transaction) != 1:
            raise ValueError(f"Expected 1 transaction, got {len(transaction)}")

        # Transform and calculate SHAP
        X_transformed = self._transform_data(transaction)

        # Create DataFrame with feature names for plotting
        X_df = pd.DataFrame(X_transformed, columns=self.feature_names)

        # Generate SHAP explanation object
        explanation = self.explainer(X_df)

        # Create waterfall plot
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
        plt.tight_layout()

        if return_base64:
            img_base64 = self._plot_to_base64(fig)
            return img_base64
        else:
            return fig

    def generate_summary(
        self, X_sample: pd.DataFrame, return_base64: bool = True, max_display: int = 20
    ) -> Union[str, matplotlib.figure.Figure]:
        """
        Generate SHAP summary plot for global feature importance.

        Shows which features are most important across all predictions.
        Each dot represents a transaction, color indicates feature value.

        Args:
            X_sample: Sample of transactions (typically 100-1000 rows)
            return_base64: If True, return base64 PNG. If False, return Figure.
            max_display: Maximum features to display

        Returns:
            Base64-encoded PNG string or matplotlib Figure

        Example:
            >>> # Analyze 500 test transactions
            >>> summary_img = explainer.generate_summary(X_test[:500])
        """
        # Transform data
        X_transformed = self._transform_data(X_sample)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_transformed)

        # Create summary plot
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()

        if return_base64:
            img_base64 = self._plot_to_base64(fig)
            return img_base64
        else:
            return fig

    def explain_prediction(
        self, transaction: pd.DataFrame, threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Get comprehensive explanation for a single prediction.

        Args:
            transaction: Single transaction DataFrame
            threshold: Decision threshold

        Returns:
            Dictionary with:
            - prediction: fraud probability
            - decision: "BLOCK" or "APPROVE"
            - shap_values: feature contributions
            - top_features: top 5 features sorted by impact
            - base_value: model's base prediction (average)

        Example:
            >>> explanation = explainer.explain_prediction(transaction_df, threshold=0.895)
            >>> print(explanation['decision'])  # "BLOCK"
            >>> print(explanation['top_features'])
            [{'feature': 'amt_log', 'impact': 0.32}, ...]
        """
        # Get prediction probability
        y_prob = self.pipeline.predict_proba(transaction)[0, 1]

        # Transform for SHAP
        X_transformed = self._transform_data(transaction)
        shap_values = self.explainer.shap_values(X_transformed)

        # Get base value (expected value)
        base_value = self.explainer.expected_value

        # Sort features by absolute impact
        feature_impacts = [
            {"feature": feat, "impact": float(shap_val), "abs_impact": abs(float(shap_val))}
            for feat, shap_val in zip(self.feature_names, shap_values[0])
        ]
        feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)

        return {
            "prediction": float(y_prob),
            "decision": "BLOCK" if y_prob >= threshold else "APPROVE",
            "threshold": threshold,
            "shap_values": {
                feat: float(val) for feat, val in zip(self.feature_names, shap_values[0])
            },
            "top_features": feature_impacts[:5],
            "base_value": float(base_value),
        }

    def _plot_to_base64(self, fig: matplotlib.figure.Figure) -> str:
        """
        Convert matplotlib figure to base64-encoded PNG.

        Args:
            fig: Matplotlib figure

        Returns:
            Base64-encoded PNG string
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_base64


__all__ = ["FraudExplainer"]
