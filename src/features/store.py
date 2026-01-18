"""
Redis Feature Store

Real-time feature storage and computation for fraud detection.
Implements stateful features that require historical context:
- Sliding window transaction counts (O(log N))
- Exponential moving averages for spending (O(1))

Architecture:
- Uses Redis Sorted Sets (ZSET) for time-based sliding windows
- Uses Redis Strings for EMA computation with atomic operations
- Connection pooling for low-latency concurrent requests

Author: PayShield-ML Team
"""

import time
from typing import Dict, List, Optional, Tuple

import redis
from redis.client import Pipeline
from redis.connection import ConnectionPool


class RedisFeatureStore:
    """
    Redis-backed feature store for real-time fraud detection.

    This class manages stateful features that cannot be computed from a single
    transaction alone. It solves the "Stateful Feature Problem" by maintaining
    rolling windows of user behavior.

    Features Computed:
    1. **trans_count_24h**: Number of transactions in last 24 hours
       - Data Structure: Redis Sorted Set (ZSET)
       - Complexity: O(log N) insert, O(log N + M) range query
       - Key Format: user:{user_id}:tx_history

    2. **avg_spend_24h**: Exponential moving average of spending
       - Data Structure: Redis String (float)
       - Complexity: O(1) update
       - Key Format: user:{user_id}:avg_spend
       - Formula: EMA_new = α * amt_current + (1-α) * EMA_old
       - α = 2/(n+1) where n=24 (for 24-hour window)

    Connection Management:
    - Uses connection pooling to avoid TCP overhead
    - Thread-safe for concurrent API requests
    - Automatic reconnection on failure

    Example:
        >>> store = RedisFeatureStore(host="localhost", port=6379)
        >>>
        >>> # Record a new transaction
        >>> store.add_transaction(
        ...     user_id="u12345",
        ...     amount=150.00,
        ...     timestamp=1234567890
        ... )
        >>>
        >>> # Get features for inference
        >>> features = store.get_features(user_id="u12345")
        >>> print(features)
        {'trans_count_24h': 5, 'avg_spend_24h': 120.50}
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        decode_responses: bool = True,
        ema_alpha: Optional[float] = None,
    ) -> None:
        """
        Initialize Redis Feature Store with connection pooling.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number (0-15)
            password: Redis password (if authentication enabled)
            max_connections: Maximum connections in pool
            decode_responses: If True, decode bytes to strings
            ema_alpha: Exponential moving average smoothing factor.
                      Default is 2/(24+1) ≈ 0.08 for 24-hour window.
        """
        # Create connection pool for thread-safe access
        self.pool: ConnectionPool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=decode_responses,
            socket_connect_timeout=2,  # 2s connection timeout
            socket_timeout=1,  # 1s operation timeout
        )

        self.client: redis.Redis = redis.Redis(connection_pool=self.pool)

        # EMA configuration: α = 2/(n+1) for n=24 hours
        self.ema_alpha: float = ema_alpha if ema_alpha is not None else 2.0 / (24 + 1)

        # TTL for keys (7 days = 604800 seconds)
        # This prevents unbounded memory growth
        self.key_ttl: int = 604800

        # Test connection
        try:
            self.client.ping()
        except redis.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to Redis at {host}:{port}. Ensure Redis is running. Error: {e}"
            ) from e

    def _get_tx_history_key(self, user_id: str) -> str:
        """Generate Redis key for transaction history ZSET."""
        return f"user:{user_id}:tx_history"

    def _get_avg_spend_key(self, user_id: str) -> str:
        """Generate Redis key for average spend EMA."""
        return f"user:{user_id}:avg_spend"

    def add_transaction(self, user_id: str, amount: float, timestamp: Optional[int] = None) -> None:
        """
        Record a new transaction and update features atomically.

        This method performs three atomic operations:
        1. Add transaction to sliding window (ZSET)
        2. Remove expired transactions (older than 24h)
        3. Update exponential moving average

        All operations are pipelined for performance (single round trip).

        Args:
            user_id: User identifier
            amount: Transaction amount in USD
            timestamp: Unix timestamp. If None, uses current time.

        Raises:
            redis.exceptions.RedisError: If Redis operation fails

        Example:
            >>> store.add_transaction("u12345", 150.00, 1234567890)
        """
        if timestamp is None:
            timestamp = int(time.time())

        # Calculate window boundaries
        window_start = timestamp - 86400  # 24 hours = 86400 seconds

        tx_key = self._get_tx_history_key(user_id)
        avg_key = self._get_avg_spend_key(user_id)

        # Use pipeline for atomic multi-operation
        pipe: Pipeline = self.client.pipeline()

        # 1. Add transaction to sorted set (score = timestamp)
        # Using transaction hash as member to allow duplicate amounts
        tx_member = f"{timestamp}:{amount}"
        pipe.zadd(tx_key, {tx_member: timestamp})

        # 2. Remove transactions older than 24 hours
        pipe.zremrangebyscore(tx_key, "-inf", window_start)

        # 3. Set TTL to prevent unbounded growth (reset on each transaction)
        pipe.expire(tx_key, self.key_ttl)

        # 4. Update exponential moving average
        # Get current EMA (default to amount if first transaction)
        current_ema = self.client.get(avg_key)
        if current_ema is None:
            new_ema = amount
        else:
            current_ema = float(current_ema)
            # EMA formula: α * x_new + (1-α) * EMA_old
            new_ema = self.ema_alpha * amount + (1 - self.ema_alpha) * current_ema

        pipe.set(avg_key, new_ema)
        pipe.expire(avg_key, self.key_ttl)

        # Execute all operations atomically
        pipe.execute()

    def get_features(
        self, user_id: str, current_timestamp: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Retrieve real-time features for a user.

        This is the primary method called during inference. It returns
        the stateful features needed by the fraud detection model.

        Args:
            user_id: User identifier
            current_timestamp: Current Unix timestamp. If None, uses system time.

        Returns:
            Dictionary containing:
            - trans_count_24h: Number of transactions in last 24 hours
            - avg_spend_24h: Exponential moving average of spending

        Example:
            >>> features = store.get_features("u12345")
            >>> print(features)
            {'trans_count_24h': 5.0, 'avg_spend_24h': 120.50}
        """
        if current_timestamp is None:
            current_timestamp = int(time.time())

        window_start = current_timestamp - 86400

        tx_key = self._get_tx_history_key(user_id)
        avg_key = self._get_avg_spend_key(user_id)

        # Use pipeline for efficiency
        pipe: Pipeline = self.client.pipeline()

        # Count transactions in window (ZCOUNT is O(log N))
        pipe.zcount(tx_key, window_start, current_timestamp)

        # Get average spend
        pipe.get(avg_key)

        results = pipe.execute()

        trans_count = float(results[0])
        avg_spend = float(results[1]) if results[1] is not None else 0.0

        return {
            "trans_count_24h": trans_count,
            "avg_spend_24h": avg_spend,
        }

    def get_transaction_history(
        self, user_id: str, lookback_hours: int = 24, current_timestamp: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve raw transaction history for a user.

        Useful for debugging and analytics. Not typically used in inference.

        Args:
            user_id: User identifier
            lookback_hours: How many hours of history to retrieve
            current_timestamp: Reference timestamp. If None, uses system time.

        Returns:
            List of tuples: [(timestamp, amount), ...]
            Sorted by timestamp (newest first)

        Example:
            >>> history = store.get_transaction_history("u12345", lookback_hours=48)
            >>> for ts, amt in history:
            ...     print(f"{ts}: ${amt:.2f}")
        """
        if current_timestamp is None:
            current_timestamp = int(time.time())

        window_start = current_timestamp - (lookback_hours * 3600)

        tx_key = self._get_tx_history_key(user_id)

        # Get all transactions in window with scores (timestamps)
        # ZRANGEBYSCORE with WITHSCORES
        raw_results = self.client.zrangebyscore(
            tx_key, window_start, current_timestamp, withscores=True
        )

        # Parse results: member format is "timestamp:amount"
        transactions = []
        for member, score in raw_results:
            timestamp_str, amount_str = member.split(":")
            transactions.append((int(timestamp_str), float(amount_str)))

        # Sort by timestamp descending (newest first)
        transactions.sort(reverse=True, key=lambda x: x[0])

        return transactions

    def delete_user_data(self, user_id: str) -> int:
        """
        Delete all feature data for a user.

        Used for GDPR compliance / right to be forgotten.

        Args:
            user_id: User identifier

        Returns:
            Number of keys deleted (should be 2)

        Example:
            >>> deleted = store.delete_user_data("u12345")
            >>> print(f"Deleted {deleted} keys")
        """
        tx_key = self._get_tx_history_key(user_id)
        avg_key = self._get_avg_spend_key(user_id)

        return self.client.delete(tx_key, avg_key)

    def health_check(self) -> Dict[str, any]:
        """
        Check Redis connection health and get statistics.

        Returns:
            Dictionary with health metrics

        Example:
            >>> health = store.health_check()
            >>> print(health)
            {'status': 'healthy', 'ping_ms': 0.5, 'connected_clients': 10}
        """
        try:
            start = time.time()
            self.client.ping()
            ping_ms = (time.time() - start) * 1000

            # Get Redis info
            info = self.client.info("stats")

            return {
                "status": "healthy",
                "ping_ms": round(ping_ms, 2),
                "connected_clients": info.get("connected_clients", -1),
                "total_commands_processed": info.get("total_commands_processed", -1),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def close(self) -> None:
        """
        Close the connection pool.

        Call this when shutting down the application.
        """
        self.pool.disconnect()


__all__ = ["RedisFeatureStore"]
