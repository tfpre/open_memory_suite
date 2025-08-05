"""Tests for memory adapters."""

import pytest
from uuid import uuid4

from open_memory_suite.adapters import FAISStoreAdapter, MemoryItem, RetrievalResult


class TestMemoryItem:
    """Test MemoryItem data model."""
    
    def test_memory_item_creation(self):
        """Test basic MemoryItem creation."""
        item = MemoryItem(content="Test content")
        
        assert item.content == "Test content"
        assert item.id is not None
        assert item.timestamp is not None
        assert item.metadata == {}
        assert item.speaker is None
        assert item.turn_id is None
    
    def test_memory_item_with_metadata(self):
        """Test MemoryItem with full metadata."""
        item = MemoryItem(
            content="Hello world",
            speaker="user",
            turn_id="turn_1",
            metadata={"topic": "greeting", "sentiment": "positive"}
        )
        
        assert item.content == "Hello world"
        assert item.speaker == "user"
        assert item.turn_id == "turn_1"
        assert item.metadata["topic"] == "greeting"
        assert item.metadata["sentiment"] == "positive"


class TestRetrievalResult:
    """Test RetrievalResult data model."""
    
    def test_empty_retrieval_result(self):
        """Test empty retrieval result."""
        result = RetrievalResult(query="test query")
        
        assert result.query == "test query"
        assert result.items == []
        assert result.content_only == []
        assert result.similarity_scores is None
    
    def test_retrieval_result_with_items(self, sample_memory_items):
        """Test retrieval result with items."""
        result = RetrievalResult(
            query="test query",
            items=sample_memory_items[:2],
            similarity_scores=[0.9, 0.8]
        )
        
        assert len(result.items) == 2
        assert len(result.content_only) == 2
        assert result.content_only[0] == "Hello, I'm Alice and I work as a software engineer."
        assert result.similarity_scores == [0.9, 0.8]


class TestFAISStoreAdapter:
    """Test FAISS store adapter."""
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, temp_dir):
        """Test adapter initialization and cleanup."""
        adapter = FAISStoreAdapter(index_path=temp_dir / "test_index")
        
        assert not adapter._initialized
        
        async with adapter:
            assert adapter._initialized
            assert adapter._embedding_model is not None
            assert adapter._index is not None
        
        # Should be cleaned up
        assert not adapter._initialized
    
    @pytest.mark.asyncio
    async def test_store_and_count(self, faiss_adapter, sample_memory_items):
        """Test storing items and counting."""
        initial_count = await faiss_adapter.count()
        assert initial_count == 0
        
        # Store first item
        success = await faiss_adapter.store(sample_memory_items[0])
        assert success
        
        count = await faiss_adapter.count()
        assert count == 1
        
        # Store more items
        for item in sample_memory_items[1:3]:
            await faiss_adapter.store(item)
        
        final_count = await faiss_adapter.count()
        assert final_count == 3
    
    @pytest.mark.asyncio
    async def test_retrieve_similar_content(self, faiss_adapter, sample_memory_items):
        """Test retrieving similar content."""
        # Store items
        for item in sample_memory_items:
            await faiss_adapter.store(item)
        
        # Query for travel-related content
        result = await faiss_adapter.retrieve("travel to Japan", k=3)
        
        assert isinstance(result, RetrievalResult)
        assert result.query == "travel to Japan"
        assert len(result.items) <= 3
        assert len(result.similarity_scores) == len(result.items)
        
        # Should find travel-related content
        content_texts = result.content_only
        japan_mentions = sum(1 for text in content_texts if "Japan" in text)
        assert japan_mentions > 0
    
    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self, faiss_adapter, sample_memory_items):
        """Test retrieval with metadata filters."""
        # Store items  
        for item in sample_memory_items:
            await faiss_adapter.store(item)
        
        # Query with speaker filter
        result = await faiss_adapter.retrieve(
            "software engineer",
            k=5,
            filters={"speaker": "user"}
        )
        
        # All returned items should be from user
        for item in result.items:
            assert item.speaker == "user"
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_index(self, faiss_adapter):
        """Test retrieval from empty index."""
        result = await faiss_adapter.retrieve("anything", k=5)
        
        assert result.items == []
        assert result.similarity_scores == []
        assert result.content_only == []
    
    @pytest.mark.asyncio
    async def test_health_check(self, faiss_adapter):
        """Test adapter health check."""
        # Should be healthy when initialized
        is_healthy = await faiss_adapter.health_check()
        assert is_healthy
    
    @pytest.mark.asyncio
    async def test_clear_index(self, faiss_adapter, sample_memory_items):
        """Test clearing the index."""
        # Store some items
        for item in sample_memory_items[:3]:
            await faiss_adapter.store(item)
        
        assert await faiss_adapter.count() == 3
        
        # Clear the index
        await faiss_adapter.clear()
        
        assert await faiss_adapter.count() == 0
    
    @pytest.mark.asyncio
    async def test_persistence(self, temp_dir, sample_memory_items):
        """Test saving and loading index from disk."""
        index_path = temp_dir / "persistence_test"
        
        # Create adapter and store items
        adapter1 = FAISStoreAdapter(index_path=index_path)
        async with adapter1:
            for item in sample_memory_items[:3]:
                await adapter1.store(item)
            assert await adapter1.count() == 3
        
        # Create new adapter with same path - should load existing data
        adapter2 = FAISStoreAdapter(index_path=index_path)
        async with adapter2:
            assert await adapter2.count() == 3
            
            # Should be able to retrieve previously stored items
            result = await adapter2.retrieve("software engineer", k=1)
            assert len(result.items) > 0