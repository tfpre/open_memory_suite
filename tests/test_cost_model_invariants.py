"""Tests for cost model invariants and monotonicity properties."""

import asyncio
import math
from pathlib import Path
import tempfile
import pytest

from open_memory_suite.benchmark.cost_model import (
    CostModel, 
    OperationType, 
    ConcurrencyLevel, 
    BudgetType
)
from open_memory_suite.core.pricebook import Pricebook, AdapterCoeffs


@pytest.fixture
def cost_model():
    """Create a cost model with temporary pricebook for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pricebook_path = Path(tmp_dir) / "test_pricebook.json"
        # Use existing config if available, otherwise skip YAML-dependent tests
        yield CostModel(pricebook_path=pricebook_path)


class TestMonotonicityProperties:
    """Test that cost/latency predictions are monotonic in input parameters."""
    
    def test_monotonic_tokens(self, cost_model):
        """Increasing tokens should never reduce predicted costs."""
        base_cents, _ = cost_model.predict(
            op=OperationType.RETRIEVE,
            adapter="test_adapter",
            tokens=10
        )
        
        high_cents, _ = cost_model.predict(
            op=OperationType.RETRIEVE,
            adapter="test_adapter", 
            tokens=100
        )
        
        assert high_cents >= base_cents, f"Cost decreased with more tokens: {base_cents} -> {high_cents}"
    
    def test_monotonic_k(self, cost_model):
        """Increasing k (retrieve count) should never reduce predicted costs."""
        base_cents, _ = cost_model.predict(
            op=OperationType.RETRIEVE,
            adapter="test_adapter",
            k=5
        )
        
        high_cents, _ = cost_model.predict(
            op=OperationType.RETRIEVE, 
            adapter="test_adapter",
            k=20
        )
        
        assert high_cents >= base_cents, f"Cost decreased with higher k: {base_cents} -> {high_cents}"
    
    def test_monotonic_item_count(self, cost_model):
        """Increasing item count should never reduce predicted costs."""
        base_cents, _ = cost_model.predict(
            op=OperationType.RETRIEVE,
            adapter="test_adapter",
            item_count=100
        )
        
        high_cents, _ = cost_model.predict(
            op=OperationType.RETRIEVE,
            adapter="test_adapter", 
            item_count=10000
        )
        
        assert high_cents >= base_cents, f"Cost decreased with more items: {base_cents} -> {high_cents}"


class TestUnitsAndBounds:
    """Test that predictions have correct units and bounds."""
    
    def test_non_negative_costs(self, cost_model):
        """All cost predictions should be non-negative."""
        cents, ms = cost_model.predict(
            op=OperationType.STORE,
            adapter="test_adapter",
            tokens=0
        )
        
        assert cents >= 0, f"Negative cost prediction: {cents}"
        assert ms >= 0.0, f"Negative latency prediction: {ms}"
    
    def test_integer_cents(self, cost_model):
        """Cost predictions should be integers (cents)."""
        cents, ms = cost_model.predict(
            op=OperationType.STORE,
            adapter="test_adapter",
            tokens=50
        )
        
        assert isinstance(cents, int), f"Cost should be int, got {type(cents)}: {cents}"
        assert isinstance(ms, float), f"Latency should be float, got {type(ms)}: {ms}"
    
    def test_no_nans_or_infs(self, cost_model):
        """Predictions should never be NaN or infinite.""" 
        cents, ms = cost_model.predict(
            op=OperationType.RETRIEVE,
            adapter="test_adapter",
            tokens=1000,
            k=50,
            item_count=100000
        )
        
        assert not math.isnan(cents), f"Cost prediction is NaN: {cents}"
        assert not math.isinf(cents), f"Cost prediction is infinite: {cents}"
        assert not math.isnan(ms), f"Latency prediction is NaN: {ms}"
        assert not math.isinf(ms), f"Latency prediction is infinite: {ms}"


class TestCalibrationRefit:
    """Test that the pricebook calibration system works correctly."""
    
    @pytest.mark.asyncio
    async def test_calibration_refit(self, cost_model):
        """Test that reconcile() updates coefficients based on observations."""
        # Inject synthetic samples with known linear relationship
        # cents = 2 + 0.001 * tokens
        for tokens in range(10, 1000, 50):
            await cost_model.reconcile(
                op=OperationType.RETRIEVE,
                adapter="test_adapter",
                predicted_cents=0,  # Start with poor prediction
                predicted_ms=0.0,
                observed_cents=2 + int(0.001 * tokens),
                observed_ms=50.0 + 0.1 * tokens,
                tokens=tokens,
                k=5,
                item_count=1000
            )
        
        # Force refit
        await cost_model._refit_locked()
        
        # Test that predictions improved
        cents, ms = cost_model.predict(
            op=OperationType.RETRIEVE,
            adapter="test_adapter",
            tokens=500
        )
        
        expected_cents = 2 + int(0.001 * 500)
        assert abs(cents - expected_cents) <= 10, f"Calibration failed: got {cents}, expected ~{expected_cents}"
    
    def test_coeffs_defaults(self, cost_model):
        """Test that unknown adapters get sensible default coefficients.""" 
        coeffs = cost_model._coeffs("unknown_adapter", OperationType.STORE)
        
        assert coeffs.base_cents >= 0
        assert coeffs.per_token_micros >= 0
        assert coeffs.p50_ms > 0
        assert coeffs.p95_ms >= coeffs.p50_ms


class TestBudgetCompliance:
    """Test budget enforcement invariants."""
    
    def test_idempotent_budget_checks(self, cost_model):
        """Budget compliance should be idempotent for same inputs."""
        # This test requires the legacy cost estimation methods
        # Skip if config file not available
        try:
            cost_estimate = cost_model.estimate_storage_cost(
                "memory_store", 
                "test content",
                item_count=100
            )
            
            # Create a tracker
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            tracker = loop.run_until_complete(cost_model.create_tracker("test_session"))
            
            # Check compliance twice
            result1 = cost_model.check_budget_compliance(cost_estimate, tracker, BudgetType.STANDARD)
            result2 = cost_model.check_budget_compliance(cost_estimate, tracker, BudgetType.STANDARD)
            
            assert result1 == result2, "Budget compliance check is not idempotent"
            
            loop.close()
            
        except (FileNotFoundError, ValueError):
            pytest.skip("Config file not available for budget testing")


class TestPropertyInvariance:
    """Test mathematical properties that must always hold."""
    
    def test_log_scaling_property(self, cost_model):
        """Test that log(N) scaling behaves correctly."""
        # Test that doubling item count increases cost by at most log(2) factor
        cents_1k, _ = cost_model.predict(
            op=OperationType.RETRIEVE,
            adapter="test_adapter",
            item_count=1000
        )
        
        cents_2k, _ = cost_model.predict(
            op=OperationType.RETRIEVE, 
            adapter="test_adapter",
            item_count=2000
        )
        
        if cents_1k > 0:  # Only test if there's base cost
            ratio = cents_2k / cents_1k
            log2_factor = math.log(2) + 1  # Allow for base cost
            assert ratio <= log2_factor, f"Cost scaling violated log property: {ratio} > {log2_factor}"
    
    def test_concurrency_scaling_bounds(self, cost_model):
        """Test that concurrency never causes extreme latency scaling."""
        base_cents, base_ms = cost_model.predict(
            op=OperationType.STORE,
            adapter="test_adapter", 
            concurrency=ConcurrencyLevel.SINGLE
        )
        
        heavy_cents, heavy_ms = cost_model.predict(
            op=OperationType.STORE,
            adapter="test_adapter",
            concurrency=ConcurrencyLevel.HEAVY
        )
        
        if base_ms > 0:
            latency_ratio = heavy_ms / base_ms
            assert latency_ratio <= 10, f"Concurrency scaling too extreme: {latency_ratio}x"


if __name__ == "__main__":
    pytest.main([__file__])