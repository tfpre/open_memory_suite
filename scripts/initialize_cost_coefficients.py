#!/usr/bin/env python3
"""
Initialize cost model with real coefficients from YAML configuration.

This script populates the pricebook with realistic 5-component cost coefficients
based on the 2025 market pricing in cost_model.yaml, implementing the friend's recommendations.
"""

import asyncio
import time
import yaml
from pathlib import Path
from typing import Dict, Any

from open_memory_suite.benchmark.cost_model import CostModel, OperationType  
from open_memory_suite.core.pricebook import AdapterCoeffs, Pricebook


def load_yaml_costs(config_path: Path) -> Dict[str, Any]:
    """Load cost configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_adapter_coefficients(adapter_name: str, operation: str, yaml_config: Dict[str, Any]) -> AdapterCoeffs:
    """
    Create AdapterCoeffs from YAML configuration using friend's 5-component model.
    
    Maps YAML costs to the 5 components:
    - write: Cost to store new data 
    - read: Cost to retrieve data
    - index_maint: Cost for indexing and upkeep
    - gc_maint: Cost for garbage collection
    - storage_month: Storage cost per MB per month
    """
    
    # Base cost lookup from YAML
    storage_config = yaml_config.get("storage", {})
    retrieval_config = yaml_config.get("retrieval", {})
    processing_config = yaml_config.get("processing", {})
    
    # Get adapter-specific costs
    adapter_storage = storage_config.get(adapter_name, {})
    adapter_retrieval = retrieval_config.get(adapter_name, {})
    maintenance_config = processing_config.get("maintenance", {})
    
    # Map to 5-component model based on operation type
    if operation == "store":
        # For storage operations, focus on write costs
        write_cost = (
            adapter_storage.get("store_item", 0.0) +
            adapter_storage.get("embedding_generation", 0.0) + 
            adapter_storage.get("vector_store", 0.0) +
            adapter_storage.get("file_write", 0.0) +
            adapter_storage.get("entity_extraction", 0.0)
        )
        
        read_cost = 0.01  # Minimal read cost for storage ops
        index_maint_cost = adapter_storage.get("index_maintenance", 0.001)
        gc_maint_cost = maintenance_config.get("archival_migration", 0.0005)
        storage_month_cost = 0.001  # Base monthly storage cost
        
        # Latency targets
        latency_config = yaml_config.get("latency_targets", {}).get("storage", {})
        p50_ms = latency_config.get(adapter_name, 50.0)
        p95_ms = p50_ms * 2.5  # Estimate p95 as 2.5x p50
        
    elif operation == "retrieve":
        # For retrieval operations, focus on read costs
        write_cost = 0.001  # Minimal write cost for retrieval ops
        
        read_cost = (
            adapter_retrieval.get("tfidf_search", 0.0) +
            adapter_retrieval.get("query_embedding", 0.0) +
            adapter_retrieval.get("vector_search", 0.0) +
            adapter_retrieval.get("linear_scan", 0.0) +
            adapter_retrieval.get("semantic_query", 0.0)
        )
        
        index_maint_cost = 0.0005  # Lower maintenance for reads
        gc_maint_cost = 0.0001     # Minimal GC cost
        storage_month_cost = 0.0005  # Lower storage access cost
        
        # Latency targets
        latency_config = yaml_config.get("latency_targets", {}).get("retrieval", {})
        p50_ms = latency_config.get(adapter_name, 25.0)
        p95_ms = p50_ms * 2.0  # Estimate p95 as 2x p50
        
    else:
        # Default fallback for other operations
        write_cost = 0.01
        read_cost = 0.005
        index_maint_cost = 0.001
        gc_maint_cost = 0.0005
        storage_month_cost = 0.001
        p50_ms = 100.0
        p95_ms = 250.0
    
    # Create legacy coefficients for backward compatibility
    base_cents = int(max(1, (write_cost + read_cost) * 100))  # Convert to cents
    per_token_micros = int(write_cost * 1000)  # Micro-cents per token
    per_k_cents = read_cost * 10  # Scale with retrieval count
    per_logN_cents = index_maint_cost * 100  # Scale with index size
    
    return AdapterCoeffs(
        # Friend's 5-component model (in cents)
        write=float(write_cost),
        read=float(read_cost),
        index_maint=float(index_maint_cost),
        gc_maint=float(gc_maint_cost),
        storage_month=float(storage_month_cost),
        
        # Legacy compatibility
        base_cents=base_cents,
        per_token_micros=per_token_micros,
        per_k_cents=float(per_k_cents),
        per_logN_cents=float(per_logN_cents),
        p50_ms=float(p50_ms),
        p95_ms=float(p95_ms)
    )


def initialize_pricebook_with_yaml_costs(pricebook_path: Path, yaml_config: Dict[str, Any]) -> Pricebook:
    """Initialize pricebook with realistic coefficients from YAML."""
    
    # Define adapter/operation combinations based on our system
    adapter_operations = [
        ("memory_store", "store"),
        ("memory_store", "retrieve"), 
        ("faiss_store", "store"),
        ("faiss_store", "retrieve"),
        ("file_store", "store"),
        ("file_store", "retrieve"),
        ("zep_store", "store"),
        ("zep_store", "retrieve"),
        
        # Graph and summary adapters (friend's Graph-Lite + Summary-Lite recommendations)
        ("graph_lite", "store"),
        ("graph_lite", "retrieve"),
        ("summary_lite", "store"),
        ("summary_lite", "retrieve"),
        
        # Maintenance operations
        ("faiss_store", "maintain"),
        ("file_store", "maintain"),
        ("graph_lite", "maintain"),
        ("summary_lite", "maintain"),
    ]
    
    # Create pricebook entries
    entries = {}
    for adapter_name, operation in adapter_operations:
        key = f"{adapter_name}|{operation}"
        
        try:
            coeffs = create_adapter_coefficients(adapter_name, operation, yaml_config)
            entries[key] = coeffs
            print(f"✓ Created coefficients for {key}: "
                  f"write={coeffs.write:.4f}¢, read={coeffs.read:.4f}¢, "
                  f"latency={coeffs.p50_ms:.1f}ms")
        except Exception as e:
            print(f"⚠ Warning: Could not create coefficients for {key}: {e}")
            # Use safe defaults
            entries[key] = AdapterCoeffs()
    
    # Create and save pricebook
    pricebook = Pricebook(
        entries=entries,
        version=1,
        updated_at=time.time()
    )
    
    pricebook.save(pricebook_path)
    print(f"\n✅ Saved pricebook with {len(entries)} coefficient sets to {pricebook_path}")
    
    return pricebook


def validate_cost_predictions(cost_model: CostModel):
    """Validate that cost predictions are working with new coefficients."""
    print("\n=== Validating Cost Predictions ===")
    
    test_cases = [
        # (operation, adapter, tokens, k, item_count, expected_range)
        (OperationType.STORE, "memory_store", 100, 0, 1000, (0, 0.1)),   # Should be very cheap/free
        (OperationType.STORE, "faiss_store", 100, 0, 1000, (0, 0.5)),    # Local embedding cost
        (OperationType.STORE, "zep_store", 100, 0, 1000, (2, 10)),       # API costs higher
        
        (OperationType.RETRIEVE, "memory_store", 50, 5, 1000, (0, 0.1)), # Free local search  
        (OperationType.RETRIEVE, "faiss_store", 50, 5, 1000, (0, 0.5)),  # Local vector search
        (OperationType.RETRIEVE, "zep_store", 50, 5, 1000, (1, 8)),      # API graph query
    ]
    
    for op, adapter, tokens, k, item_count, (min_cents, max_cents) in test_cases:
        try:
            cents, ms = cost_model.predict(
                op=op, 
                adapter=adapter,
                tokens=tokens,
                k=k,
                item_count=item_count
            )
            
            cents_float = cents / 100.0  # Convert to dollars for display
            
            status = "✓" if min_cents <= cents_float <= max_cents else "⚠"
            print(f"{status} {adapter}|{op.value}: ${cents_float:.4f}, {ms:.1f}ms "
                  f"(expected ${min_cents:.4f}-${max_cents:.4f})")
                  
        except Exception as e:
            print(f"✗ {adapter}|{op.value}: Error - {e}")
    
    print("✅ Cost prediction validation complete")


async def main():
    """Initialize cost model with real coefficients from YAML configuration."""
    import time
    
    print("=== Initializing Cost Model with Real Coefficients ===")
    print("Implementing friend's 5-component cost model recommendations\n")
    
    # Paths
    repo_root = Path(__file__).parent.parent
    yaml_config_path = repo_root / "open_memory_suite" / "benchmark" / "cost_model.yaml"
    pricebook_path = repo_root / "cost_model_pricebook.json"
    
    # Load YAML configuration
    print(f"Loading cost configuration from {yaml_config_path}")
    yaml_config = load_yaml_costs(yaml_config_path)
    
    # Initialize pricebook with realistic coefficients
    pricebook = initialize_pricebook_with_yaml_costs(pricebook_path, yaml_config)
    
    # Create cost model and validate predictions
    cost_model = CostModel(
        config_path=yaml_config_path,
        pricebook_path=pricebook_path
    )
    
    validate_cost_predictions(cost_model)
    
    # Display cost analysis summary
    print(f"\n=== Cost Model Foundation Summary ===")
    print(f"📊 Total adapter configurations: {len(pricebook.entries)}")
    print(f"💰 5-component model: write, read, index_maint, gc_maint, storage_month")
    print(f"🎯 Market pricing: Based on 2025 API costs and local compute estimates")
    print(f"🔧 Backward compatibility: Legacy coefficients maintained for existing code")
    print(f"📈 Ready for ML router training with realistic cost foundations")
    
    print(f"\n🎉 Cost model foundation complete!")
    print(f"Next step: Create 3-class router with calibrated abstention")


if __name__ == "__main__":
    asyncio.run(main())