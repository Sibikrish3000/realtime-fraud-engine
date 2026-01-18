"""
Tests for Redis Feature Store

Tests sliding window logic, EMA computation, and Redis operations.
"""

import time

import pytest

from src.features.store import RedisFeatureStore


class TestRedisFeatureStore:
    """Test suite for RedisFeatureStore."""

    def test_connection(self, feature_store):
        """Test that Redis connection is established."""
        health = feature_store.health_check()
        assert health["status"] == "healthy"
        assert "ping_ms" in health

    def test_add_transaction(self, feature_store):
        """Test adding a transaction."""
        feature_store.add_transaction(user_id="test_user_1", amount=100.00, timestamp=1000000)

        # Verify transaction was added
        features = feature_store.get_features("test_user_1", current_timestamp=1000000)
        assert features["trans_count_24h"] == 1.0
        assert features["avg_spend_24h"] == 100.00

    def test_sliding_window_count(self, feature_store):
        """Test that transaction count respects 24-hour window."""
        base_time = 1000000

        # Add 3 transactions within 24 hours
        for i in range(3):
            feature_store.add_transaction(
                user_id="test_user_2",
                amount=50.00,
                timestamp=base_time + (i * 3600),  # 1 hour apart
            )

        # Check count
        features = feature_store.get_features(
            "test_user_2",
            current_timestamp=base_time + 10800,  # 3 hours later
        )
        assert features["trans_count_24h"] == 3.0

        # Add transaction 25 hours later - old ones should be excluded
        future_time = base_time + (25 * 3600)
        feature_store.add_transaction(user_id="test_user_2", amount=50.00, timestamp=future_time)

        features = feature_store.get_features("test_user_2", current_timestamp=future_time)
        # Should only count the new transaction
        assert features["trans_count_24h"] == 1.0

    def test_exponential_moving_average(self, feature_store):
        """Test EMA computation."""
        base_time = 1000000

        # First transaction: EMA = amount
        feature_store.add_transaction(user_id="test_user_3", amount=100.00, timestamp=base_time)

        features = feature_store.get_features("test_user_3", current_timestamp=base_time)
        assert features["avg_spend_24h"] == 100.00

        # Second transaction: EMA updates
        feature_store.add_transaction(
            user_id="test_user_3", amount=200.00, timestamp=base_time + 3600
        )

        features = feature_store.get_features("test_user_3", current_timestamp=base_time + 3600)
        # EMA = alpha * 200 + (1-alpha) * 100
        # With alpha = 0.08: 0.08 * 200 + 0.92 * 100 = 16 + 92 = 108
        assert 107 < features["avg_spend_24h"] < 109  # Allow small floating point error

    def test_get_transaction_history(self, feature_store):
        """Test retrieving transaction history."""
        base_time = 1000000

        # Add multiple transactions
        amounts = [100.00, 150.00, 200.00]
        for i, amt in enumerate(amounts):
            feature_store.add_transaction(
                user_id="test_user_4", amount=amt, timestamp=base_time + (i * 3600)
            )

        history = feature_store.get_transaction_history("test_user_4", lookback_hours=24)

        assert len(history) == 3
        # Should be sorted newest first
        assert history[0][1] == 200.00
        assert history[1][1] == 150.00
        assert history[2][1] == 100.00

    def test_delete_user_data(self, feature_store):
        """Test GDPR-compliant data deletion."""
        feature_store.add_transaction(user_id="test_user_5", amount=100.00, timestamp=1000000)

        # Verify data exists
        features = feature_store.get_features("test_user_5", current_timestamp=1000000)
        assert features["trans_count_24h"] > 0

        # Delete user data
        deleted_count = feature_store.delete_user_data("test_user_5")
        assert deleted_count == 2  # tx_history + avg_spend

        # Verify data is gone
        features = feature_store.get_features("test_user_5", current_timestamp=1000000)
        assert features["trans_count_24h"] == 0.0
        assert features["avg_spend_24h"] == 0.0

    def test_concurrent_transactions(self, feature_store):
        """Test that multiple users can be tracked concurrently."""
        base_time = 1000000

        # Add transactions for different users
        for user_id in ["user_a", "user_b", "user_c"]:
            feature_store.add_transaction(user_id=user_id, amount=100.00, timestamp=base_time)

        # Verify each user has independent state
        for user_id in ["user_a", "user_b", "user_c"]:
            features = feature_store.get_features(user_id, current_timestamp=base_time)
            assert features["trans_count_24h"] == 1.0

    def test_empty_user(self, feature_store):
        """Test getting features for user with no history."""
        features = feature_store.get_features("nonexistent_user", current_timestamp=1000000)
        assert features["trans_count_24h"] == 0.0
        assert features["avg_spend_24h"] == 0.0
