#!/usr/bin/env python3
"""
Quick verification script for Phase 1 implementation.
Tests imports and basic module functionality without requiring Redis.
"""


def test_imports():
    """Test that all Phase 1 modules can be imported."""
    print("Testing imports...")

    try:
        from src.data.ingest import TransactionSchema, InferenceTransactionSchema, load_dataset  # noqa: F401

        print("  ✓ Data ingestion module imports successfully")
    except ImportError as e:
        print(f"  ✗ Data ingestion import failed: {e}")
        return False

    try:
        from src.features.store import RedisFeatureStore  # noqa: F401

        print("  ✓ Feature store module imports successfully")
    except ImportError as e:
        print(f"  ✗ Feature store import failed: {e}")
        return False

    try:
        from src.features.constants import category_names, job_names

        print(
            f"  ✓ Constants module loaded ({len(category_names)} categories, {len(job_names)} jobs)"
        )
    except ImportError as e:
        print(f"  ✗ Constants import failed: {e}")
        return False

    return True


def test_pydantic_validation():
    """Test Pydantic validation logic."""
    print("\nTesting Pydantic validation...")

    from src.data.ingest import TransactionSchema, InferenceTransactionSchema
    from pydantic import ValidationError

    # Test valid transaction
    try:
        valid_data = {
            "trans_date_trans_time": "2019-01-01 00:00:18",
            "cc_num": 2703186189652095,
            "merchant": "Test Merchant",
            "category": "misc_net",
            "amt": 100.50,
            "first": "John",
            "last": "Doe",
            "gender": "M",
            "street": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zip": 62701,
            "lat": 39.7817,
            "long": -89.6501,
            "city_pop": 100000,
            "job": "Engineer, biomedical",
            "dob": "1990-01-01",
            "trans_num": "abc123",
            "unix_time": 1325376018,
            "merch_lat": 39.7900,
            "merch_long": -89.6600,
            "is_fraud": 0,
        }
        schema = TransactionSchema(**valid_data)
        print(f"  ✓ Valid transaction accepted (amt: ${schema.amt:.2f})")
    except ValidationError as e:
        print(f"  ✗ Valid transaction rejected: {e}")
        return False

    # Test invalid amount
    try:
        invalid_data = valid_data.copy()
        invalid_data["amt"] = -50.00
        schema = TransactionSchema(**invalid_data)
        print("  ✗ Negative amount not rejected!")
        return False
    except ValidationError:
        print("  ✓ Negative amount correctly rejected")

    # Test invalid coordinates
    try:
        invalid_data = valid_data.copy()
        invalid_data["lat"] = 200.0
        schema = TransactionSchema(**invalid_data)
        print("  ✗ Invalid latitude not rejected!")
        return False
    except ValidationError:
        print("  ✓ Invalid latitude correctly rejected")

    # Test inference schema
    try:
        inference_data = {
            "user_id": "u12345",
            "amt": 150.00,
            "lat": 40.7128,
            "long": -74.0060,
            "category": "grocery_pos",
            "job": "Engineer, biomedical",
            "merch_lat": 40.7200,
            "merch_long": -74.0100,
            "unix_time": 1234567890,
        }
        schema = InferenceTransactionSchema(**inference_data)
        print(f"  ✓ Inference schema works (user: {schema.user_id})")
    except ValidationError as e:
        print(f"  ✗ Inference schema failed: {e}")
        return False

    return True


def test_constants():
    """Test that constants are properly loaded."""
    print("\nTesting constants...")

    from src.features.constants import category_names, job_names

    # Check categories
    expected_categories = ["misc_net", "gas_transport", "grocery_pos", "entertainment"]
    for cat in expected_categories:
        if cat not in category_names:
            print(f"  ✗ Missing category: {cat}")
            return False
    print(f"  ✓ All expected categories present ({len(category_names)} total)")

    # Check jobs
    expected_jobs = ["Engineer, biomedical", "Data scientist", "Mechanical engineer"]
    found_jobs = [j for j in expected_jobs if j in job_names]
    print(
        f"  ✓ Job list loaded ({len(job_names)} jobs, {len(found_jobs)}/{len(expected_jobs)} test jobs found)"
    )

    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Phase 1 Verification Script")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Pydantic Validation", test_pydantic_validation),
        ("Constants", test_constants),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ {name} test crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:>10} - {name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All verification tests passed!")
        print("\nPhase 1 implementation is working correctly.")
        print("\nNext steps:")
        print("  1. Install dependencies: uv sync")
        print("  2. Start Redis: docker-compose -f docker/docker-compose.yml up -d redis")
        print("  3. Run full test suite: pytest tests/ -v")
        print("  4. Try demo: python scripts/demo_phase1.py")
    else:
        print("❌ Some tests failed. Check the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
