#!/usr/bin/env python3
"""Test the Priority-0 production safety fixes."""

import asyncio
import tempfile
from pathlib import Path
import json

async def test_regime_separation():
    """Test that cost vs latency fitting is properly separated."""
    from open_memory_suite.core.pricebook import fit_coeffs
    
    print("Testing regime separation...")
    
    # Mixed samples: some with cost data (API calls), some without (local ops)
    samples = [
        # Local FAISS operations (no cost data)
        {"observed_cents": None, "observed_ms": 45.0, "tokens": 100, "k": 5, "item_count": 1000},
        {"observed_cents": None, "observed_ms": 52.0, "tokens": 200, "k": 5, "item_count": 1000},
        {"observed_cents": None, "observed_ms": 48.0, "tokens": 150, "k": 5, "item_count": 1000},
        
        # API operations (with real cost data)  
        {"observed_cents": 15, "observed_ms": 1200.0, "tokens": 100, "k": 1, "item_count": 0},
        {"observed_cents": 28, "observed_ms": 2100.0, "tokens": 200, "k": 1, "item_count": 0},
        {"observed_cents": 22, "observed_ms": 1800.0, "tokens": 150, "k": 1, "item_count": 0},
    ]
    
    coeffs = fit_coeffs(samples)
    
    # Cost coefficients should be based only on API samples with real costs
    print(f"  Base cost: {coeffs.base_cents} cents (should be >0 from API data)")
    print(f"  Per-token: {coeffs.per_token_micros} microcents (should be >0)")
    
    # Latency should use all samples
    print(f"  p50 latency: {coeffs.p50_ms:.1f}ms (should blend local + API)")
    print(f"  p95 latency: {coeffs.p95_ms:.1f}ms")
    
    # Verify cost coefficients are not zero (would indicate contamination)
    assert coeffs.base_cents > 0, "Cost coefficients collapsed - regime separation failed!"
    print("✓ Regime separation working correctly")

async def test_atomic_operations():
    """Test that pricebook operations are atomic and thread-safe."""
    from open_memory_suite.benchmark.cost_model import CostModel, OperationType
    
    print("Testing atomic operations...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        pb_path = Path(tmp_dir) / "test_atomic.json"
        cm = CostModel(pricebook_path=pb_path)
        
        # Simulate concurrent reconcile operations
        tasks = []
        for i in range(50):
            task = cm.reconcile(
                op=OperationType.STORE,
                adapter="test_adapter",
                predicted_cents=10,
                predicted_ms=50.0,
                observed_cents=12 + i,  # Varying costs
                observed_ms=48.0 + i * 0.5,
                tokens=100 + i,
                item_count=1000
            )
            tasks.append(task)
        
        # Run concurrently
        await asyncio.gather(*tasks)
        
        # Force a refit
        await cm._refit_locked()
        
        # Verify pricebook file is valid JSON
        assert pb_path.exists(), "Pricebook not saved"
        with open(pb_path) as f:
            data = json.load(f)
        assert "entries" in data, "Pricebook structure invalid"
        
        print("✓ Atomic operations working correctly")

async def test_memory_guards():
    """Test that sample memory is bounded per key."""
    from open_memory_suite.benchmark.cost_model import CostModel, OperationType
    
    print("Testing memory guards...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        pb_path = Path(tmp_dir) / "test_memory.json"
        cm = CostModel(pricebook_path=pb_path)
        cm._max_samples_per_key = 100  # Low limit for testing
        
        key = "test_adapter|store"
        
        # Add many samples beyond the limit
        for i in range(250):
            await cm.reconcile(
                op=OperationType.STORE,
                adapter="test_adapter", 
                predicted_cents=10,
                predicted_ms=50.0,
                observed_cents=None,
                observed_ms=45.0 + i * 0.1,
                tokens=100,
                item_count=1000
            )
        
        # Check memory is bounded
        sample_count = len(cm._samples_per_key.get(key, []))
        print(f"  Sample count: {sample_count} (max: {cm._max_samples_per_key})")
        assert sample_count <= cm._max_samples_per_key, "Memory guard failed!"
        
        print("✓ Memory guards working correctly")

async def test_config_safety():
    """Test that missing YAML doesn't crash the system."""
    from open_memory_suite.benchmark.cost_model import CostModel
    
    print("Testing config safety...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Point to non-existent YAML file
        missing_yaml = Path(tmp_dir) / "missing.yaml"
        pb_path = Path(tmp_dir) / "test_safe.json"
        
        # This should not crash
        cm = CostModel(config_path=missing_yaml, pricebook_path=pb_path)
        
        # Should have empty config
        assert cm._config == {}, "Config not defaulted to empty dict"
        
        # Should still be able to make predictions (using pricebook defaults)
        cents, ms = cm.predict(
            op=OperationType.STORE,
            adapter="test_adapter",
            tokens=100
        )
        
        print(f"  Prediction with no YAML: {cents} cents, {ms:.1f}ms")
        assert cents >= 0 and ms >= 0, "Predictions failed with missing config"
        
        print("✓ Config safety working correctly")

async def test_single_source_truth():
    """Test that estimate_*_cost routes through predict()."""
    from open_memory_suite.benchmark.cost_model import CostModel
    
    print("Testing single source of truth...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        pb_path = Path(tmp_dir) / "test_truth.json"
        cm = CostModel(pricebook_path=pb_path)
        
        # Both methods should give same underlying prediction
        direct_cents, direct_ms = cm.predict(
            op=OperationType.STORE,
            adapter="test_adapter",
            tokens=100,
            item_count=1000
        )
        
        legacy_estimate = cm.estimate_storage_cost(
            adapter_name="test_adapter",
            content="This is a test with about one hundred tokens to verify consistency between methods",
            item_count=1000
        )
        
        # Should be consistent (within token counting differences)
        predicted_cents = legacy_estimate.metadata.get("predicted_cents", 0)
        print(f"  Direct prediction: {direct_cents} cents")  
        print(f"  Legacy estimate: {predicted_cents} cents")
        
        # They should be exactly equal since they use the same predict() path
        assert predicted_cents == direct_cents, "Single source of truth violation!"
        
        print("✓ Single source of truth working correctly")

async def main():
    """Run all Priority-0 safety tests."""
    print("=== Testing Priority-0 Production Safety Fixes ===\n")
    
    tests = [
        ("Regime Separation", test_regime_separation),
        ("Atomic Operations", test_atomic_operations), 
        ("Memory Guards", test_memory_guards),
        ("Config Safety", test_config_safety),
        ("Single Source Truth", test_single_source_truth),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            await test_func()
            results.append((test_name, True))
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=== Results Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL" 
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} critical fixes verified")
    
    if passed == total:
        print("🎉 All Priority-0 fixes are production-safe!")
    else:
        print("⚠ Some critical issues detected - review failures above")

if __name__ == "__main__":
    asyncio.run(main())