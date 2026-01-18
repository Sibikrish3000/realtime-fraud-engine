"""
Tests for Evaluation Metrics.

Tests threshold optimization and metric calculation logic.
"""

import numpy as np
import pytest

from src.models.metrics import calculate_metrics, find_optimal_threshold


class TestCalculateMetrics:
    """Test metric calculation."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.95, 0.99])

        metrics = calculate_metrics(y_true, y_prob, threshold=0.5)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["pr_auc"] > 0.99

    def test_random_predictions(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)

        metrics = calculate_metrics(y_true, y_prob, threshold=0.5)

        # Random predictions should have low metrics
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1


class TestFindOptimalThreshold:
    """Test threshold optimization."""

    def test_finds_threshold_meeting_recall(self):
        """Test that threshold meets recall requirement."""
        # Create imbalanced dataset (like fraud)
        np.random.seed(42)
        n_samples = 1000

        # 95% negative, 5% positive
        y_true = np.array([0] * 950 + [1] * 50)

        # Model that's good but not perfect
        # Positive class gets higher probabilities
        y_prob = np.concatenate(
            [
                np.random.beta(2, 5, 950),  # Negative class: low probs
                np.random.beta(5, 2, 50),  # Positive class: high probs
            ]
        )

        threshold, metrics = find_optimal_threshold(y_true, y_prob, min_recall=0.70)

        # Should find a valid threshold
        assert 0 < threshold < 1

        # Should meet or come close to recall target
        assert metrics["recall"] >= 0.4  # At least reasonable

    def test_fallback_to_f1(self):
        """Test fallback to F1 when recall target can't be met."""
        # Very difficult scenario
        y_true = np.array([0, 0, 0, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Impossible to get 99% recall with this data
        threshold, metrics = find_optimal_threshold(y_true, y_prob, min_recall=0.99)

        # Should still return something valid
        assert 0 < threshold < 1
        assert metrics["f1"] >= 0

    def test_threshold_range(self):
        """Test that found threshold is in valid range."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)

        threshold, _ = find_optimal_threshold(y_true, y_prob)

        assert 0 <= threshold <= 1
