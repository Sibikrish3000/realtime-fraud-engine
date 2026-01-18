"""
Model Training Script with MLflow Tracking.

CLI script to train the fraud detection model with comprehensive experiment tracking.
Logs hyperparameters, metrics, and artifacts to MLflow.

Usage:
    python src/models/train.py --data_path data/fraudTrain.csv
    python src/models/train.py --data_path data/fraudTrain.csv --experiment_name fraud_v2
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.ingest import load_dataset
from src.models.metrics import calculate_metrics, find_optimal_threshold
from src.models.pipeline import create_fraud_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train fraud detection model")

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to input CSV/Parquet file"
    )

    parser.add_argument(
        "--params_path",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration YAML",
    )

    parser.add_argument(
        "--experiment_name", type=str, default="fraud_detection", help="MLflow experiment name"
    )

    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion (0-1)")

    parser.add_argument(
        "--min_recall",
        type=float,
        default=0.80,
        help="Minimum recall target for threshold optimization (Notebook: 0.80)",
    )

    parser.add_argument(
        "--output_dir", type=str, default="models", help="Directory to save model artifacts"
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load model configuration from YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from raw dataframe.

    Args:
        df: Raw transaction data
        df contains Training set for Credit Card Transactions
        index - Unique Identifier for each row
        trans_date_trans_time - Transaction DateTime
        cc_num - Credit Card Number of Customer
        merchant - Merchant Name
        category - Category of Merchant
        amt - Amount of Transaction
        first - First Name of Credit Card Holder
        last - Last Name of Credit Card Holder
        gender - Gender of Credit Card Holder
        street - Street Address of Credit Card Holder
        city - City of Credit Card Holder
        state - State of Credit Card Holder
        zip - Zip of Credit Card Holder
        lat - Latitude Location of Credit Card Holder
        long - Longitude Location of Credit Card Holder
        city_pop - Credit Card Holder's City Population
        job - Job of Credit Card Holder
        dob - Date of Birth of Credit Card Holder
        trans_num - Transaction Number
        unix_time - UNIX Time of transaction
        merch_lat - Latitude Location of Merchant
        merch_long - Longitude Location of Merchant
        is_fraud - Fraud Flag <--- Target Class

    Returns:
        Tuple of (X, y)
    """
    # Required columns for training
    required_cols = [
        "trans_date_trans_time",
        "amt",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "job",
        "category",
        "gender",
        "dob",
        "is_fraud",
    ]

    # Compute feature store features from raw data
    # Sort by user and timestamp for rolling window calculations
    print("  → Computing rolling window features (trans_count_24h, avg_amt_24h)...")

    # CRITICAL: Convert to datetime BEFORE using as index for time-based rolling windows
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df = df.sort_values(["cc_num", "trans_date_trans_time"])
    df = df.set_index("trans_date_trans_time")

    # 1. Transaction Velocity (Rolling Count)
    # Identifies sudden bursts in card usage
    df["trans_count_24h"] = (
        df.groupby("cc_num")["amt"]
        .rolling("24h")
        .count()
        .shift(1)
        .reset_index(0, drop=True)
        .fillna(0)
    )

    # 2. Recent Spending Baseline (Rolling Mean)
    # Needed for the 24h ratio calculation
    df["avg_amt_24h"] = (
        df.groupby("cc_num")["amt"]
        .rolling("24h")
        .mean()
        .shift(1)
        .reset_index(0, drop=True)
        .fillna(df["amt"])
    )

    # 3. All-time Spending Profile (Expanding Mean)
    # Captures long-term user behavior
    df["user_avg_amt_all_time"] = (
        df.groupby("cc_num")["amt"]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(df["amt"])
    )

    # Reset index to restore dataframe structure
    df = df.reset_index()

    # 4. Derived Ratio Features
    # Identifies spikes relative to recent 24-hour activity (Burst Detection)
    df["amt_to_avg_ratio_24h"] = df["amt"] / df["avg_amt_24h"]

    # Identifies spikes relative to long-term behavior (Anomaly Detection)
    df["amt_relative_to_all_time"] = df["amt"] / df["user_avg_amt_all_time"]

    # Extract target
    y = df["is_fraud"]

    # Features (pipeline will extract derived features)
    feature_cols = [c for c in df.columns if c != "is_fraud"]
    X = df[feature_cols]

    return X, y


def train_model(args):
    """Main training workflow."""

    print("=" * 70)
    print("PayShield-ML: Fraud Detection Training Pipeline")
    print("=" * 70)

    # 1. Load Configuration
    print(f"\n[1/7] Loading configuration from {args.params_path}")
    config = load_config(args.params_path)
    model_params = config.get("model", {})

    # 2. Load Data
    print(f"\n[2/7] Loading data from {args.data_path}")
    df = load_dataset(args.data_path, validate=False)  # Skip validation for speed
    print(f"  → Loaded {len(df):,} transactions")
    print(f"  → Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")

    # 3. Prepare Features
    print(f"\n[3/7] Preparing features and target")
    X, y = prepare_data(df)
    print(f"  → Features shape: {X.shape}")
    print(f"  → Target shape: {y.shape}")

    # 4. Train/Test Split (TEMPORAL - No Data Leakage)
    print(f"\n[4/7] Splitting data temporally (test_size={args.test_size})")

    # A. Data is already sorted from prepare_data (for rolling window calculations)
    # But let's ensure it's sorted and reset index
    df_combined = pd.concat([X, y], axis=1)
    df_combined = df_combined.sort_values("trans_date_trans_time").reset_index(drop=True)

    # B. Calculate split index (strictly temporal)
    split_index = int(len(df_combined) * (1 - args.test_size))

    # C. Split strictly by index (No shuffling)
    train_df = df_combined.iloc[:split_index]
    test_df = df_combined.iloc[split_index:]

    # D. Separate Features and Target
    X_train = train_df.drop("is_fraud", axis=1)
    y_train = train_df["is_fraud"]
    X_test = test_df.drop("is_fraud", axis=1)
    y_test = test_df["is_fraud"]

    # E. Report temporal boundaries and fraud rates
    print(f"  → Train: {len(X_train):,} samples")
    print(f"    • Earliest: {train_df['trans_date_trans_time'].min()}")
    print(f"    • Latest:   {train_df['trans_date_trans_time'].max()}")
    print(f"    • Fraud Rate: {y_train.mean():.4%}")

    print(f"  → Test:  {len(X_test):,} samples")
    print(f"    • Earliest: {test_df['trans_date_trans_time'].min()}")
    print(f"    • Latest:   {test_df['trans_date_trans_time'].max()}")
    print(f"    • Fraud Rate: {y_test.mean():.4%}")

    # F. Sanity check: Ensure test is strictly after train
    if train_df["trans_date_trans_time"].max() >= test_df["trans_date_trans_time"].min():
        print("  ⚠ WARNING: Temporal overlap detected between train and test sets!")

    # 5. Initialize MLflow
    print(f"\n[5/7] Initializing MLflow experiment: {args.experiment_name}")
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        # Calculate class imbalance ratio from actual training data
        imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"\n  → Class Imbalance Ratio: {imbalance_ratio:.2f}:1 (negative:positive)")

        # Override scale_pos_weight with calculated ratio
        model_params["scale_pos_weight"] = imbalance_ratio

        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("min_recall_target", args.min_recall)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))

        # 6. Train Pipeline
        print(f"\n[6/7] Training pipeline")
        pipeline = create_fraud_pipeline(model_params)

        print("  → Fitting model...")
        pipeline.fit(X_train, y_train)
        print("  ✓ Training complete")

        # Predict probabilities
        y_train_prob = pipeline.predict_proba(X_train)[:, 1]
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]

        # 7. Optimize Threshold
        print(f"\n[7/7] Optimizing decision threshold (target recall >= {args.min_recall:.2%})")
        optimal_threshold, threshold_metrics = find_optimal_threshold(
            y_test, y_test_prob, min_recall=args.min_recall
        )
        print(f"  → Optimal threshold: {optimal_threshold:.4f}")
        print(f"  → Precision: {threshold_metrics['precision']:.4f}")
        print(f"  → Recall:    {threshold_metrics['recall']:.4f}")
        print(f"  → F1 Score:  {threshold_metrics['f1']:.4f}")
        print(f"  → PR-AUC:    {threshold_metrics['pr_auc']:.4f}")

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "train_pr_auc": float(calculate_metrics(y_train, y_train_prob, 0.5)["pr_auc"]),
                "test_precision": threshold_metrics["precision"],
                "test_recall": threshold_metrics["recall"],
                "test_f1": threshold_metrics["f1"],
                "test_pr_auc": threshold_metrics["pr_auc"],
                "optimal_threshold": optimal_threshold,
            }
        )

        # Save artifacts locally
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / "fraud_model.pkl"
        joblib.dump(pipeline, model_path)
        print(f"\n✓ Model saved to {model_path}")

        # Save threshold
        threshold_path = output_dir / "threshold.json"
        with open(threshold_path, "w") as f:
            json.dump(
                {"optimal_threshold": optimal_threshold, "metrics": threshold_metrics}, f, indent=2
            )
        print(f"✓ Threshold saved to {threshold_path}")

        # Log artifacts to MLflow
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_artifact(str(threshold_path))

        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
