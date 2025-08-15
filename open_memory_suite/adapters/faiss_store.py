"""FAISS-based vector memory adapter."""

import asyncio
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import MemoryAdapter, MemoryItem, RetrievalResult
from .registry import AdapterRegistry
from ..core.telemetry import probe
from ..core.tokens import TokenCounter
from ..benchmark.cost_model import OperationType


@AdapterRegistry.register(capabilities={
    AdapterRegistry.CAPABILITY_VECTOR,
    AdapterRegistry.CAPABILITY_SEMANTIC,
    AdapterRegistry.CAPABILITY_FAST,
    AdapterRegistry.CAPABILITY_SCALABLE,
    AdapterRegistry.CAPABILITY_PERSISTENT
})
class FAISStoreAdapter(MemoryAdapter):
    """Memory adapter using FAISS for vector similarity search."""
    
    def __init__(
        self,
        name: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        index_path: Optional[Path] = None,
        dimension: int = 384
    ):
        """
        Initialize FAISS adapter.
        
        Args:
            name: Adapter name
            embedding_model: SentenceTransformer model name
            index_path: Path to save/load index (None for in-memory only)
            dimension: Vector dimension for the embedding model
        """
        super().__init__(name)
        self.embedding_model_name = embedding_model
        self.index_path = index_path
        self.dimension = dimension
        
        # Will be initialized in _initialize_impl
        self._embedding_model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._items: List[MemoryItem] = []
        self._token_counter = TokenCounter()
        
        # For cost prediction (will be set by dispatcher)
        self.cost_model = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the embedding model and FAISS index."""
        # Initialize embedding model in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        self._embedding_model = await loop.run_in_executor(
            None, 
            lambda: SentenceTransformer(self.embedding_model_name)
        )
        
        # Create or load FAISS index
        if (self.index_path and 
            self.index_path.with_suffix('.index').exists() and 
            self.index_path.with_suffix('.items').exists()):
            await self._load_index()
        else:
            self._index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self._items = []
    
    async def _cleanup_impl(self) -> None:
        """Save index if path is specified."""
        if self.index_path and self._index:
            await self._save_index()
    
    async def _save_index(self) -> None:
        """Save FAISS index and items to disk."""
        if not self.index_path:
            return
            
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_index_sync)
    
    def _save_index_sync(self) -> None:
        """Synchronous index saving."""
        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path.with_suffix('.index')))
        
        # Save items metadata
        with open(self.index_path.with_suffix('.items'), 'wb') as f:
            pickle.dump(self._items, f)
    
    async def _load_index(self) -> None:
        """Load FAISS index and items from disk."""
        loop = asyncio.get_event_loop()
        self._index, self._items = await loop.run_in_executor(None, self._load_index_sync)
    
    def _load_index_sync(self) -> tuple[faiss.Index, List[MemoryItem]]:
        """Synchronous index loading."""
        # Load FAISS index
        index = faiss.read_index(str(self.index_path.with_suffix('.index')))
        
        # Load items metadata
        with open(self.index_path.with_suffix('.items'), 'rb') as f:
            items = pickle.load(f)
        
        return index, items
    
    async def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using the sentence transformer."""
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode([text], normalize_embeddings=True)
        )
        return embedding[0]
    
    async def store(self, item: MemoryItem) -> bool:
        """Store a memory item by adding it to the FAISS index."""
        # Calculate telemetry metadata
        tokens = self._token_counter.count(item.content)
        item_count = len(self._items)
        
        # Get cost prediction if available
        predicted_cents, predicted_ms = 0, 0.0
        if self.cost_model:
            predicted_cents, predicted_ms = self.cost_model.predict(
                op=OperationType.STORE,
                adapter=self.name,
                tokens=tokens,
                item_count=item_count
            )
        
        with probe("store", self.name, predicted_cents, predicted_ms, meta={
            "tokens": tokens,
            "item_count": item_count,
            "observed_cents": None  # Local operation, no direct cost
        }):
            try:
                # Generate embedding
                embedding = await self._embed_text(item.content)
                
                # Add to FAISS index
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self._index.add(embedding.reshape(1, -1))
                )
                
                # Store item metadata
                self._items.append(item)
                
                return True
            except Exception:
                return False
    
    async def retrieve(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve relevant items using vector similarity."""
        # Calculate telemetry metadata
        tokens = self._token_counter.count(query)
        item_count = len(self._items)
        
        # Get cost prediction if available
        predicted_cents, predicted_ms = 0, 0.0
        if self.cost_model:
            predicted_cents, predicted_ms = self.cost_model.predict(
                op=OperationType.RETRIEVE,
                adapter=self.name,
                tokens=tokens,
                k=k,
                item_count=item_count
            )
        
        with probe("retrieve", self.name, predicted_cents, predicted_ms, meta={
            "tokens": tokens,
            "k": k,
            "item_count": item_count,
            "observed_cents": None  # Local operation, no direct cost
        }):
            # Return empty result if no items in index
            if len(self._items) == 0:
                return RetrievalResult(query=query, items=[], similarity_scores=[])
            
            # Generate query embedding
            query_embedding = await self._embed_text(query)
            
            # Search FAISS index
            loop = asyncio.get_event_loop()
            scores, indices = await loop.run_in_executor(
                None,
                lambda: self._index.search(query_embedding.reshape(1, -1), min(k, len(self._items)))
            )
            
            # Extract results
            retrieved_items = []
            similarity_scores = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    break
                    
                item = self._items[idx]
                
                # Apply filters if specified
                if filters and not self._matches_filters(item, filters):
                    continue
                    
                retrieved_items.append(item)
                similarity_scores.append(float(score))
            
            return RetrievalResult(
                items=retrieved_items,
                query=query,
                similarity_scores=similarity_scores,
                retrieval_metadata={
                    "index_size": len(self._items),
                    "embedding_model": self.embedding_model_name
                }
            )
    
    def _matches_filters(self, item: MemoryItem, filters: Dict[str, Any]) -> bool:
        """Check if an item matches the given filters."""
        for key, value in filters.items():
            if key == "speaker" and item.speaker != value:
                return False
            elif key in item.metadata and item.metadata[key] != value:
                return False
        return True
    
    async def count(self) -> int:
        """Return the total number of stored items."""
        return len(self._items)
    
    async def clear(self) -> None:
        """Clear all stored items (useful for testing)."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._items = []
    
    def estimate_store_cost(self, item: MemoryItem) -> float:
        """Estimate cost to store an item in FAISS."""
        # Cost breakdown:
        # 1. Embedding generation (SentenceTransformer inference)
        # 2. Vector storage in FAISS index (minimal)
        # 3. Persistence save (if index_path is set)
        
        base_embedding_cost = 0.0001  # $0.0001 per embedding
        
        # Scale by content size (longer text = more compute)
        content_size_factor = min(len(item.content) / 500.0, 3.0)  # Cap at 3x
        
        # Add persistence cost if saving to disk
        persistence_cost = 0.0001 if self.index_path else 0.0
        
        total_cost = (base_embedding_cost * content_size_factor) + persistence_cost
        return total_cost
    
    def estimate_retrieve_cost(self, query: str, k: int = 5) -> float:
        """Estimate cost to retrieve items from FAISS."""
        # Cost breakdown:
        # 1. Query embedding generation
        # 2. FAISS similarity search
        # 3. Result ranking and filtering
        
        base_query_cost = 0.0001  # Query embedding
        base_search_cost = 0.0001  # FAISS search
        
        # Scale by query length
        query_size_factor = min(len(query) / 200.0, 2.0)  # Cap at 2x
        
        # Scale by number of items requested (minimal impact)
        k_factor = min(k / 5.0, 1.5)  # Cap at 1.5x
        
        # Scale by index size (larger index = slightly more expensive)
        index_size_factor = min(len(self._items) / 1000.0, 2.0)  # Cap at 2x
        
        total_cost = (base_query_cost * query_size_factor) + (base_search_cost * k_factor * index_size_factor)
        return total_cost
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for monitoring."""
        base_stats = super().get_storage_stats()
        
        # FAISS-specific metrics
        index_size = len(self._items)
        embedding_model = self.embedding_model_name
        has_persistence = self.index_path is not None
        
        base_stats.update({
            "total_items": index_size,
            "embedding_model": embedding_model,
            "vector_dimension": self.dimension,
            "persistence_enabled": has_persistence,
            "index_path": str(self.index_path) if self.index_path else None,
            "adapter_type": "faiss_vector_store"
        })
        
        return base_stats