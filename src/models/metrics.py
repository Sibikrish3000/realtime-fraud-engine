"""
Evaluation Metrics.

Utilities for calculating custom performance metrics and optimizing thresholds.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc, recall_score, precision_score


def calculate_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive set of evaluation metrics.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        threshold: Decision threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(pr_auc),
        "threshold_used": float(threshold),
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.80
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold based on 'Recall Constraint' strategy (Notebook Method).

    Strategy:
    1. Filter for thresholds where Recall >= min_recall (e.g., 0.80)
    2. From that subset, choose the threshold that yields the HIGHEST Precision

    This ensures we catch at least 80% of fraud (primary goal) while minimizing
    false alarms (customer friction) as much as possible.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        min_recall: Minimum required recall (default 0.80 from Notebook)

    Returns:
        Tuple: (best_threshold, metrics_at_threshold)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Remove last 1 to match dimensions (sklearn quirk)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # 1. Filter for Recall Requirement (Catching enough fraud)
    valid_indices = np.where(recalls >= min_recall)[0]

    if len(valid_indices) > 0:
        # 2. Maximize Precision among those valid points
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]
        best_thresh = thresholds[best_idx]
        print(f"Target met: Recall >= {min_recall:.2%}")
    else:
        # Fallback: If model is too weak to hit target, maximize F1
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        print(
            f"Target missed (Recall < {min_recall:.2%}). Maximizing F1. Best Recall: {recalls[best_idx]:.4f}"
        )

    # Calculate final metrics for the chosen threshold
    metrics = calculate_metrics(y_true, y_prob, best_thresh)
    return float(best_thresh), metrics


def save_threshold(threshold: float, path: str = "models/threshold.json"):
    """Save optimized threshold to JSON"""
    with open(path, "w") as f:
        json.dump({"optimal_threshold": threshold}, f)
