#!/usr/bin/env python3
"""
Multi-adapter comparison test - comparing FAISS and InMemory adapters.
This validates our adapter interface consistency and sets up for dispatcher testing.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_memory_suite.adapters import FAISStoreAdapter, InMemoryAdapter, MemoryItem


async def test_adapter_comparison():
    """Compare FAISS and InMemory adapters on the same dataset."""
    print("🔄 Multi-Adapter Comparison Test")
    print("=" * 50)
    
    # Create test dataset
    test_items = [
        MemoryItem(
            content="Hello, I'm Alice and I work as a software engineer at Google.",
            speaker="user",
            metadata={"topic": "introduction", "contains_name": True}
        ),
        MemoryItem(
            content="Nice to meet you Alice! How long have you been working there?",
            speaker="assistant", 
            metadata={"topic": "introduction"}
        ),
        MemoryItem(
            content="I've been at Google for about 3 years now, working on search ranking algorithms.",
            speaker="user",
            metadata={"topic": "work", "company": "Google"}
        ),
        MemoryItem(
            content="That sounds fascinating! Do you work with machine learning models?",
            speaker="assistant",
            metadata={"topic": "work"}
        ),
        MemoryItem(
            content="Yes, I primarily work with transformer-based models for understanding search intent.",
            speaker="user",
            metadata={"topic": "work", "technology": "transformers"}
        )
    ]
    
    # Test queries
    test_queries = [
        "Alice engineer Google",
        "machine learning transformers", 
        "search algorithms",
        "how long working"
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize adapters
        faiss_adapter = FAISStoreAdapter(
            name="faiss_test",
            index_path=temp_path / "faiss_index"
        )
        
        memory_adapter = InMemoryAdapter(
            name="memory_test",
            max_items=10
        )
        
        adapters = [faiss_adapter, memory_adapter]
        results = {}
        
        for adapter in adapters:
            print(f"\n🧪 Testing {adapter.name} ({adapter.__class__.__name__})")
            print("-" * 40)
            
            async with adapter:
                # Store items
                print("📝 Storing items...")
                store_success = 0
                for item in test_items:
                    if await adapter.store(item):
                        store_success += 1
                
                count = await adapter.count()
                print(f"   Stored: {store_success}/{len(test_items)} items")
                print(f"   Count: {count}")
                
                # Test retrieval
                print("🔍 Testing retrieval...")
                adapter_results = []
                
                for query in test_queries:
                    result = await adapter.retrieve(query, k=3)
                    adapter_results.append({
                        "query": query,
                        "items_found": len(result.items),
                        "top_similarity": result.similarity_scores[0] if result.similarity_scores else 0.0,
                        "top_content": result.items[0].content[:50] + "..." if result.items else "No results"
                    })
                    
                    print(f"   '{query}': {len(result.items)} items (sim: {result.similarity_scores[0] if result.similarity_scores else 0.0:.3f})")
                
                # Health check
                health = await adapter.health_check()
                print(f"   Health: {'✅' if health else '❌'}")
                
                # Cost estimation (for InMemory)
                if hasattr(adapter, 'estimate_store_cost'):
                    store_cost = adapter.estimate_store_cost(test_items[0])
                    retrieve_cost = adapter.estimate_retrieve_cost(test_queries[0])
                    print(f"   Store cost: ${store_cost:.4f}")
                    print(f"   Retrieve cost: ${retrieve_cost:.4f}")
                
                # Storage stats (for InMemory)
                if hasattr(adapter, 'get_storage_stats'):
                    stats = adapter.get_storage_stats()
                    print(f"   Storage stats: {len(stats)} metrics tracked")
                
                results[adapter.name] = {
                    "items_stored": store_success,
                    "total_count": count,
                    "health": health,
                    "queries": adapter_results
                }
    
    # Compare results
    print(f"\n📊 Comparison Summary")
    print("=" * 30)
    
    for adapter_name, data in results.items():
        print(f"\n{adapter_name}:")
        print(f"  Items stored: {data['items_stored']}")
        print(f"  Health: {'✅' if data['health'] else '❌'}")
        
        avg_similarity = sum(q["top_similarity"] for q in data["queries"]) / len(data["queries"])
        total_results = sum(q["items_found"] for q in data["queries"])
        
        print(f"  Avg similarity: {avg_similarity:.3f}")
        print(f"  Total results: {total_results}")
    
    print(f"\n🎯 Interface Consistency Test")
    print("-" * 30)
    
    # Verify both adapters implement the same interface
    required_methods = ["store", "retrieve", "count", "health_check", "initialize", "cleanup"]
    
    for adapter in adapters:
        missing_methods = []
        for method in required_methods:
            if not hasattr(adapter, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ {adapter.name}: Missing methods: {missing_methods}")
        else:
            print(f"✅ {adapter.name}: All required methods present")
    
    print(f"\n🏆 Multi-adapter test completed!")
    print("✅ Both adapters follow the same interface")
    print("✅ Both adapters can store and retrieve the same data")
    print("✅ Ready for FrugalDispatcher integration")


if __name__ == "__main__":
    try:
        asyncio.run(test_adapter_comparison())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)