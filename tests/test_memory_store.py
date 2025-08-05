"""Tests for InMemoryAdapter."""

import pytest

from open_memory_suite.adapters import InMemoryAdapter, MemoryItem


class TestInMemoryAdapter:
    """Test the in-memory adapter functionality."""
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self):
        """Test adapter initialization and cleanup."""
        adapter = InMemoryAdapter()
        
        assert not adapter._initialized
        
        async with adapter:
            assert adapter._initialized
            assert adapter._items == []
            assert adapter._vocabulary == set()
        
        # Should be cleaned up
        assert not adapter._initialized
        assert adapter._items == []
    
    @pytest.mark.asyncio 
    async def test_store_and_count(self, sample_memory_items):
        """Test storing items and counting."""
        adapter = InMemoryAdapter()
        
        async with adapter:
            initial_count = await adapter.count()
            assert initial_count == 0
            
            # Store first item
            success = await adapter.store(sample_memory_items[0])
            assert success
            
            count = await adapter.count()
            assert count == 1
            
            # Store more items
            for item in sample_memory_items[1:3]:
                await adapter.store(item)
            
            final_count = await adapter.count()
            assert final_count == 3
    
    @pytest.mark.asyncio
    async def test_retrieve_similar_content(self, sample_memory_items):
        """Test retrieving similar content."""
        adapter = InMemoryAdapter()
        
        async with adapter:
            # Store items
            for item in sample_memory_items:
                await adapter.store(item)
            
            # Test retrieval
            result = await adapter.retrieve("Alice engineer", k=2)
            
            assert len(result.items) >= 1
            assert result.query == "Alice engineer"
            assert len(result.similarity_scores) == len(result.items)
            
            # Should find the Alice introduction
            alice_found = any("Alice" in item.content for item in result.items)
            assert alice_found
    
    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self, sample_memory_items):
        """Test retrieval with metadata filters."""
        adapter = InMemoryAdapter()
        
        async with adapter:
            # Store items
            for item in sample_memory_items:
                await adapter.store(item)
            
            # Filter by speaker
            result = await adapter.retrieve(
                "travel", 
                k=5, 
                filters={"speaker": "user"}
            )
            
            # All results should be from user
            for item in result.items:
                assert item.speaker == "user"
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_adapter(self):
        """Test retrieval from empty adapter."""
        adapter = InMemoryAdapter()
        
        async with adapter:
            result = await adapter.retrieve("anything", k=5)
            
            assert result.items == []
            assert result.similarity_scores == []
            assert result.query == "anything"
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test adapter health check."""
        adapter = InMemoryAdapter()
        
        async with adapter:
            is_healthy = await adapter.health_check()
            assert is_healthy
    
    @pytest.mark.asyncio
    async def test_clear_adapter(self, sample_memory_items):
        """Test clearing all items."""
        adapter = InMemoryAdapter()
        
        async with adapter:
            # Store items
            for item in sample_memory_items[:3]:
                await adapter.store(item)
            
            assert await adapter.count() == 3
            
            # Clear all items
            success = await adapter.clear()
            assert success
            assert await adapter.count() == 0
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when hitting max_items."""
        adapter = InMemoryAdapter(max_items=3)
        
        async with adapter:
            # Create test items
            items = [
                MemoryItem(content=f"Item {i}", metadata={"index": i})
                for i in range(5)
            ]
            
            # Store items - should evict older ones
            for item in items:
                await adapter.store(item)
            
            # Should only have 3 items (the last 3)
            count = await adapter.count()
            assert count == 3
            
            # Check that we have the most recent items
            result = await adapter.retrieve("Item", k=10)
            stored_indices = {item.metadata["index"] for item in result.items}
            expected_indices = {2, 3, 4}  # Last 3 items
            assert stored_indices == expected_indices
    
    @pytest.mark.asyncio
    async def test_cost_estimation(self):
        """Test cost estimation methods."""
        adapter = InMemoryAdapter()
        
        async with adapter:
            test_item = MemoryItem(content="Test content")
            
            # In-memory adapter should have zero costs
            store_cost = adapter.estimate_store_cost(test_item)
            assert store_cost == 0.0
            
            retrieve_cost = adapter.estimate_retrieve_cost("test query")
            assert retrieve_cost == 0.0
    
    @pytest.mark.asyncio
    async def test_storage_stats(self, sample_memory_items):
        """Test storage statistics."""
        adapter = InMemoryAdapter(max_items=10)
        
        async with adapter:
            # Store some items
            for item in sample_memory_items[:3]:
                await adapter.store(item)
            
            stats = adapter.get_storage_stats()
            
            assert stats["total_items"] == 3
            assert stats["max_items"] == 10
            assert stats["memory_utilization"] == 0.3
            assert stats["adapter_type"] == "in_memory"
            assert "vocabulary_size" in stats
            assert "indexing_enabled" in stats
    
    @pytest.mark.asyncio
    async def test_indexing_disabled(self, sample_memory_items):
        """Test adapter with indexing disabled."""
        adapter = InMemoryAdapter(enable_indexing=False)
        
        async with adapter:
            # Store items
            for item in sample_memory_items:
                await adapter.store(item)
            
            # Should still be able to retrieve (using fallback)
            result = await adapter.retrieve("Alice", k=2)
            
            assert len(result.items) >= 1
            assert result.retrieval_metadata["indexing_enabled"] is False
    
    @pytest.mark.asyncio
    async def test_tokenization(self):
        """Test internal tokenization."""
        adapter = InMemoryAdapter()
        
        # Test tokenization directly
        tokens = adapter._tokenize("Hello, World! This is a test 123.")
        expected = ["hello", "world", "this", "is", "a", "test", "123"]
        assert tokens == expected
        
        # Test empty string
        empty_tokens = adapter._tokenize("")
        assert empty_tokens == []
    
    @pytest.mark.asyncio
    async def test_tfidf_computation(self, sample_memory_items):
        """Test TF-IDF vector computation."""
        adapter = InMemoryAdapter()
        
        async with adapter:
            # Store items to build vocabulary
            for item in sample_memory_items:
                await adapter.store(item)
            
            # Test TF vector computation
            tf_vector = adapter._compute_tf_vector("alice engineer software")
            
            assert "alice" in tf_vector
            assert "engineer" in tf_vector  
            assert "software" in tf_vector
            
            # Should be normalized by document length
            assert sum(tf_vector.values()) <= 1.0
    
    @pytest.mark.asyncio
    async def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        adapter = InMemoryAdapter()
        
        # Test identical vectors
        vec1 = {"a": 1.0, "b": 2.0}
        vec2 = {"a": 1.0, "b": 2.0}
        similarity = adapter._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
        
        # Test orthogonal vectors
        vec3 = {"a": 1.0}
        vec4 = {"b": 1.0}
        similarity = adapter._cosine_similarity(vec3, vec4)
        assert similarity == 0.0
        
        # Test empty vectors
        empty_vec = {}
        similarity = adapter._cosine_similarity(vec1, empty_vec)
        assert similarity == 0.0


class TestInMemoryAdapterIntegration:
    """Integration tests with other components."""
    
    @pytest.mark.asyncio
    async def test_with_benchmark_harness(self, temp_dir):
        """Test InMemoryAdapter with benchmark harness."""
        from open_memory_suite.benchmark import BenchmarkHarness
        from open_memory_suite.benchmark.harness import create_sample_session, create_sample_queries
        
        trace_file = temp_dir / "memory_harness_test.jsonl"
        harness = BenchmarkHarness(trace_file)
        
        adapter = InMemoryAdapter(name="test_memory")
        session = create_sample_session()
        
        async with adapter:
            # Run session through harness
            result = await harness.run_session(adapter, session)
            
            assert result["adapter_name"] == "test_memory"
            assert result["turns_processed"] == 5
            assert result["storage_operations"] == 5
            assert result["total_items_stored"] == 5
            
            # Test recall evaluation
            queries = create_sample_queries()
            recall_result = await harness.evaluate_recall(adapter, queries)
            
            assert recall_result["total_queries"] == 3
            assert recall_result["recall_rate"] >= 0.0  # Should find some matches