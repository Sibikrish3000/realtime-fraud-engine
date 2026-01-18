"""
Example script demonstrating the data ingestion and feature store usage.

This shows how to use the modules we just built.
"""

from src.data.ingest import InferenceTransactionSchema
from src.features.store import RedisFeatureStore


def main():
    """Demonstrate basic usage of Phase 1 modules."""

    print("=== PayShield-ML Phase 1 Demo ===\n")

    # 1. Data Validation Example
    print("1. Testing Data Validation...")
    try:
        valid_transaction = InferenceTransactionSchema(
            user_id="u12345",
            amt=150.00,
            lat=40.7128,
            long=-74.0060,
            category="grocery_pos",
            job="Engineer, biomedical",
            merch_lat=40.7200,
            merch_long=-74.0100,
            unix_time=1234567890,
        )
        print(f"   ✓ Valid transaction: ${valid_transaction.amt:.2f}")
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")

    # 2. Feature Store Example
    print("\n2. Testing Feature Store...")
    try:
        store = RedisFeatureStore(host="localhost", port=6379, db=0)

        # Check health
        health = store.health_check()
        print(f"   ✓ Redis connection: {health['status']} (ping: {health['ping_ms']}ms)")

        # Add some transactions
        user_id = "demo_user_001"
        import time

        base_time = int(time.time())

        print(f"\n   Adding 3 transactions for {user_id}...")
        for i, amount in enumerate([50.00, 75.00, 100.00]):
            store.add_transaction(
                user_id=user_id,
                amount=amount,
                timestamp=base_time + (i * 3600),  # 1 hour apart
            )
            print(f"      Transaction {i + 1}: ${amount:.2f}")

        # Get features
        features = store.get_features(user_id, current_timestamp=base_time + 7200)
        print("\n   Features computed:")
        print(f"      - Transaction count (24h): {features['trans_count_24h']:.0f}")
        print(f"      - Average spend (24h): ${features['avg_spend_24h']:.2f}")

        # Get history
        history = store.get_transaction_history(user_id, lookback_hours=24)
        print(f"\n   Transaction history ({len(history)} transactions):")
        for ts, amt in history[:3]:
            print(f"      - Timestamp {ts}: ${amt:.2f}")

        # Cleanup
        deleted = store.delete_user_data(user_id)
        print(f"\n   ✓ Cleaned up {deleted} keys")

    except Exception as e:
        print(f"   ✗ Feature store error: {e}")
        print(
            "   Make sure Redis is running: docker-compose -f docker/docker-compose.yml up -d redis"
        )

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
