#!/usr/bin/env python3
"""Validation script for telemetry system implementation."""

import asyncio
import time
from pathlib import Path

def test_core_imports():
    """Test that all core telemetry components can be imported."""
    try:
        from open_memory_suite.core.telemetry import probe, log_event
        from open_memory_suite.core.tokens import TokenCounter  
        from open_memory_suite.core.pricebook import Pricebook, AdapterCoeffs, fit_coeffs
        from open_memory_suite.benchmark.cost_model import OperationType, CostModel
        print("✓ All telemetry imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_token_counting():
    """Test token counting functionality."""
    try:
        from open_memory_suite.core.tokens import TokenCounter
        
        tc = TokenCounter()
        test_texts = [
            "Hello world",
            "This is a longer test message with more words to count",
            "",
            "Short"
        ]
        
        for text in test_texts:
            tokens = tc.count(text)
            print(f"✓ Token count for '{text[:30]}': {tokens}")
            assert tokens >= 0, "Token count should be non-negative"
        
        # Test monotonicity
        short_tokens = tc.count("Hi")
        long_tokens = tc.count("This is a much longer message with many more words")
        assert long_tokens > short_tokens, "Longer text should have more tokens"
        print("✓ Token counting monotonicity verified")
        return True
    except Exception as e:
        print(f"✗ Token counting error: {e}")
        return False

def test_pricebook_operations():
    """Test pricebook save/load and coefficient operations."""
    try:
        from open_memory_suite.core.pricebook import Pricebook, AdapterCoeffs, fit_coeffs
        
        # Test pricebook creation
        test_path = Path("./test_pricebook.json")
        pb = Pricebook.load(test_path)
        print("✓ Pricebook creation successful")
        
        # Test coefficient defaults
        pb.entries["test|store"] = AdapterCoeffs(
            base_cents=10,
            per_token_micros=1000,
            per_k_cents=0.1,
            per_logN_cents=0.5,
            p50_ms=50.0,
            p95_ms=150.0
        )
        
        # Test save/load cycle
        pb.save(test_path)
        pb2 = Pricebook.load(test_path)
        assert "test|store" in pb2.entries
        print("✓ Pricebook save/load cycle successful")
        
        # Clean up
        test_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"✗ Pricebook error: {e}")
        return False

def test_cost_model_predictions():
    """Test cost model prediction functionality."""
    try:
        from open_memory_suite.benchmark.cost_model import CostModel, OperationType, ConcurrencyLevel
        
        cm = CostModel(pricebook_path="./test_pricebook_validate.json")
        
        # Test basic predictions
        cents, ms = cm.predict(
            op=OperationType.STORE,
            adapter="test_adapter",
            tokens=100,
            item_count=1000
        )
        print(f"✓ Cost prediction: {cents} cents, {ms:.1f} ms")
        assert cents >= 0, "Cost should be non-negative"
        assert ms >= 0, "Latency should be non-negative"
        
        # Test monotonicity
        cents1, _ = cm.predict(op=OperationType.STORE, adapter="test", tokens=10)
        cents2, _ = cm.predict(op=OperationType.STORE, adapter="test", tokens=100)
        assert cents2 >= cents1, "Cost should be monotonic in tokens"
        print("✓ Cost monotonicity verified")
        
        # Clean up
        Path("./test_pricebook_validate.json").unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"✗ Cost model error: {e}")
        return False

async def test_telemetry_probe():
    """Test telemetry probe context manager."""
    try:
        from open_memory_suite.core.telemetry import probe
        
        # Test probe usage
        with probe("test_op", "test_adapter", 5, 100.0, meta={"tokens": 50}):
            await asyncio.sleep(0.01)  # Simulate work
        
        print("✓ Telemetry probe execution successful")
        
        # Check if log file was created
        log_file = Path("./telemetry.jsonl")
        if log_file.exists():
            content = log_file.read_text()
            if content.strip():
                print("✓ Telemetry logging verified")
            else:
                print("⚠ Telemetry file created but empty")
            # Clean up
            log_file.unlink()
        else:
            print("⚠ Telemetry log file not created")
        
        return True
    except Exception as e:
        print(f"✗ Telemetry probe error: {e}")
        return False

async def main():
    """Run all validation tests."""
    print("=== Telemetry System Validation ===\n")
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Token Counting", test_token_counting),
        ("Pricebook Operations", test_pricebook_operations),
        ("Cost Model Predictions", test_cost_model_predictions),
        ("Telemetry Probe", test_telemetry_probe),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=== Results Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All telemetry components are working correctly!")
    else:
        print("⚠ Some issues detected - check errors above")

if __name__ == "__main__":
    asyncio.run(main())