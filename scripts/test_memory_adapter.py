#!/usr/bin/env python3
"""
Quick test of the InMemoryAdapter implementation.
"""

import asyncio
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_memory_suite.adapters import InMemoryAdapter, MemoryItem


async def test_memory_adapter():
    """Test InMemoryAdapter basic functionality."""
    print("🧠 Testing InMemoryAdapter Implementation")
    print("=" * 50)
    
    adapter = InMemoryAdapter(name="test_memory", max_items=5)
    
    async with adapter:
        print(f"✅ Adapter initialized: {adapter.name}")
        
        # Create test items
        items = [
            MemoryItem(content="Hello, I'm Alice and I work as a software engineer at Google."),
            MemoryItem(content="Nice to meet you Alice! How long have you been there?"),
            MemoryItem(content="I've been at Google for about 3 years now, working on search algorithms."),
            MemoryItem(content="That sounds fascinating! Do you work with machine learning?"),
            MemoryItem(content="Yes, I primarily work with transformer models for search.")
        ]
        
        # Store items
        print(f"\n📝 Storing {len(items)} memory items...")
        for i, item in enumerate(items):
            success = await adapter.store(item)
            print(f"   Item {i+1}: {'✅' if success else '❌'}")
        
        # Check count
        count = await adapter.count()
        print(f"\n📊 Total items stored: {count}")
        
        # Test retrieval
        print(f"\n🔍 Testing retrieval...")
        queries = [
            ("Alice engineer", "Should find Alice's introduction"),
            ("Google search algorithms", "Should find work-related content"),
            ("transformer models", "Should find ML-related content"),
            ("nonexistent query", "Should return empty results")
        ]
        
        for query, description in queries:
            result = await adapter.retrieve(query, k=3)
            print(f"   Query: '{query}'")
            print(f"   Description: {description}")
            print(f"   Results: {len(result.items)} items found")
            if result.items:
                print(f"   Top match: '{result.items[0].content[:50]}...'")
                print(f"   Similarity: {result.similarity_scores[0]:.3f}")
            print()
        
        # Test storage stats
        stats = adapter.get_storage_stats()
        print(f"📈 Storage Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test cost estimation
        print(f"\n💰 Cost Estimation:")
        test_item = MemoryItem(content="Test content")
        store_cost = adapter.estimate_store_cost(test_item)
        retrieve_cost = adapter.estimate_retrieve_cost("test query")
        print(f"   Store cost: ${store_cost:.4f}")
        print(f"   Retrieve cost: ${retrieve_cost:.4f}")
        
        print(f"\n🎯 Health check: {'✅ Healthy' if await adapter.health_check() else '❌ Unhealthy'}")
    
    print(f"\n✅ InMemoryAdapter test completed successfully!")
    return True


async def test_performance():
    """Test InMemoryAdapter performance characteristics."""
    print(f"\n⚡ Performance Test")
    print("=" * 30)
    
    adapter = InMemoryAdapter(name="perf_test")
    
    async with adapter:
        # Measure store performance
        import time
        
        items = [MemoryItem(content=f"Performance test item {i} with some content") for i in range(100)]
        
        start_time = time.perf_counter()
        for item in items:
            await adapter.store(item)
        store_time = time.perf_counter() - start_time
        
        print(f"Stored 100 items in {store_time:.3f}s ({100/store_time:.0f} items/sec)")
        
        # Measure retrieve performance
        start_time = time.perf_counter()
        result = await adapter.retrieve("performance test", k=10)
        retrieve_time = time.perf_counter() - start_time
        
        print(f"Retrieved {len(result.items)} items in {retrieve_time:.3f}s ({len(result.items)/retrieve_time:.0f} items/sec)")
        print(f"Average similarity: {sum(result.similarity_scores)/len(result.similarity_scores):.3f}")


if __name__ == "__main__":
    try:
        asyncio.run(test_memory_adapter())
        asyncio.run(test_performance())
        print(f"\n🏆 All tests passed! InMemoryAdapter is ready for integration.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)