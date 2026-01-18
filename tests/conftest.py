"""
Pytest Configuration and Fixtures

Provides shared fixtures for testing data ingestion and feature store.
"""

import os
from typing import Generator

import pytest
import redis
from redis import Redis

from src.features.store import RedisFeatureStore


@pytest.fixture(scope="session")
def redis_client() -> Generator[Redis, None, None]:
    """
    Provide a Redis client for testing.

    Uses a separate test database (db=15) to avoid polluting production data.
    Requires Redis to be running on localhost:6379.
    """
    # Check if Redis is available
    client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=15,  # Use separate DB for tests
        decode_responses=True,
    )

    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis not available. Start with: docker run -d -p 6379:6379 redis:7-alpine")

    yield client

    # Cleanup: flush test database
    client.flushdb()
    client.close()


@pytest.fixture
def feature_store(redis_client: Redis) -> RedisFeatureStore:
    """
    Provide a RedisFeatureStore instance for testing.

    Automatically cleans up after each test.
    """
    store = RedisFeatureStore(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=15,  # Test database
    )

    yield store

    # Cleanup is handled by redis_client fixture


@pytest.fixture
def sample_transaction() -> dict:
    """Provide a valid sample transaction for testing."""
    return {
        "trans_date_trans_time": "2019-01-01 00:00:18",
        "cc_num": 2703186189652095,
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "misc_net",
        "amt": 4.97,
        "first": "Jennifer",
        "last": "Banks",
        "gender": "F",
        "street": "561 Perry Cove",
        "city": "Moravian Falls",
        "state": "NC",
        "zip": 28654,
        "lat": 36.0788,
        "long": -81.1781,
        "city_pop": 3495,
        "job": "Psychologist, counselling",
        "dob": "1988-03-09",
        "trans_num": "0b242abb623afc578575680df30655b9",
        "unix_time": 1325376018,
        "merch_lat": 36.011293,
        "merch_long": -82.048315,
        "is_fraud": 0,
    }


@pytest.fixture
def invalid_transaction() -> dict:
    """Provide an invalid transaction for validation testing."""
    return {
        "trans_date_trans_time": "2019-01-01 00:00:18",
        "cc_num": 2703186189652095,
        "merchant": "Test Merchant",
        "category": "invalid_category",  # Invalid!
        "amt": -50.00,  # Negative amount - invalid!
        "first": "John",
        "last": "Doe",
        "gender": "M",
        "street": "123 Main St",
        "city": "Springfield",
        "state": "IL",
        "zip": 62701,
        "lat": 200.0,  # Invalid latitude!
        "long": -81.1781,
        "city_pop": 100000,
        "job": "Engineer",
        "dob": "1990-01-01",
        "trans_num": "abc123",
        "unix_time": 1325376018,
        "merch_lat": 36.011293,
        "merch_long": -82.048315,
        "is_fraud": 0,
    }
