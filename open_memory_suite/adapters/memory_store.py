"""In-memory storage adapter for fast, ephemeral memory operations."""

import asyncio
import math
import re
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Set

from .base import MemoryAdapter, MemoryItem, RetrievalResult


class InMemoryAdapter(MemoryAdapter):
    """
    Ultra-fast in-memory storage adapter.
    
    Features:
    - Zero storage costs, sub-millisecond latency
    - TF-IDF based text similarity for retrieval  
    - Optional memory limits with LRU eviction
    - Perfect for session-scoped or temporary storage
    """
    
    def __init__(
        self,
        name: str = "memory_store",
        max_items: Optional[int] = None,
        enable_indexing: bool = True
    ):
        """
        Initialize in-memory adapter.
        
        Args:
            name: Adapter name for identification
            max_items: Maximum items to store (None = unlimited)
            enable_indexing: Whether to build TF-IDF index for fast retrieval
        """
        super().__init__(name)
        self.max_items = max_items
        self.enable_indexing = enable_indexing
        
        # Storage
        self._items: List[MemoryItem] = []
        self._access_order: OrderedDict[str, int] = OrderedDict()  # id -> index for LRU
        
        # TF-IDF indexing for fast retrieval
        self._tfidf_vectors: List[Dict[str, float]] = []
        self._vocabulary: Set[str] = set()
        self._idf_cache: Dict[str, float] = {}
        self._doc_freq: Dict[str, int] = {}  # Document frequency cache
        
    async def _initialize_impl(self) -> None:
        """Initialize the in-memory storage (no-op for memory store)."""
        # In-memory adapter requires no initialization
        pass
    
    async def _cleanup_impl(self) -> None:
        """Clear all stored data."""
        self._items.clear()
        self._access_order.clear()
        self._tfidf_vectors.clear()
        self._vocabulary.clear()
        self._idf_cache.clear()
        self._doc_freq.clear()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for TF-IDF."""
        # Convert to lowercase, split on non-alphanumeric, filter empty
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return tokens
    
    def _compute_tf_vector(self, text: str) -> Dict[str, float]:
        """Compute term frequency vector for text."""
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        
        # Count terms
        term_counts = Counter(tokens)
        total_terms = len(tokens)
        
        # Compute TF (normalized by document length)
        tf_vector = {}
        for term, count in term_counts.items():
            tf_vector[term] = count / total_terms
            
        return tf_vector
    
    def _add_to_doc_freq(self, tokens: Set[str]) -> None:
        """Incrementally add tokens to document frequency counter."""
        for token in tokens:
            self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
    
    def _remove_from_doc_freq(self, tokens: Set[str]) -> None:
        """Incrementally remove tokens from document frequency counter."""
        for token in tokens:
            if token in self._doc_freq:
                self._doc_freq[token] -= 1
                if self._doc_freq[token] <= 0:
                    del self._doc_freq[token]
    
    def _compute_idf_value(self, term: str) -> float:
        """Compute IDF value for a single term."""
        if not self._items or term not in self._doc_freq:
            return 0.0
        
        total_docs = len(self._items)
        doc_freq = self._doc_freq[term]
        return math.log(total_docs / doc_freq) if doc_freq > 0 else 0.0
    
    def _compute_tfidf_vector(self, tf_vector: Dict[str, float]) -> Dict[str, float]:
        """Convert TF vector to TF-IDF using computed IDF values."""
        tfidf_vector = {}
        for term, tf in tf_vector.items():
            idf = self._compute_idf_value(term)
            tfidf_vector[term] = tf * idf
        return tfidf_vector
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0
        
        # Compute dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(val * val for val in vec1.values()))
        mag2 = math.sqrt(sum(val * val for val in vec2.values()))
        
        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0
            
        return dot_product / (mag1 * mag2)
    
    def _evict_lru_if_needed(self) -> None:
        """Evict least recently used item if at capacity."""
        if self.max_items is None or len(self._items) < self.max_items:
            return
            
        # Find LRU item
        lru_id, lru_index = next(iter(self._access_order.items()))
        
        # Remove from storage
        del self._items[lru_index]
        if self.enable_indexing and lru_index < len(self._tfidf_vectors):
            del self._tfidf_vectors[lru_index]
        
        # Update access order indices
        del self._access_order[lru_id]
        for item_id, index in self._access_order.items():
            if index > lru_index:
                self._access_order[item_id] = index - 1
    
    def _update_access_order(self, item_id: str, index: int) -> None:
        """Update LRU access order for an item."""
        # Remove if already exists
        if item_id in self._access_order:
            del self._access_order[item_id]
        # Add to end (most recent)
        self._access_order[item_id] = index
    
    async def store(self, item: MemoryItem) -> bool:
        """
        Store a memory item.
        
        Args:
            item: The memory item to store
            
        Returns:
            True (in-memory storage always succeeds)
        """
        try:
            # Evict LRU item if at capacity
            self._evict_lru_if_needed()
            
            # Add to storage
            index = len(self._items)
            self._items.append(item)
            self._update_access_order(item.id, index)
            
            # Update TF-IDF index if enabled
            if self.enable_indexing:
                tf_vector = self._compute_tf_vector(item.content)
                self._tfidf_vectors.append(tf_vector)
                
                # Update vocabulary and document frequency incrementally
                tokens = set(tf_vector.keys())
                self._vocabulary.update(tokens)
                self._add_to_doc_freq(tokens)
            
            return True
            
        except Exception:
            return False
    
    async def retrieve(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant items using TF-IDF similarity.
        
        Args:
            query: The search query
            k: Number of items to retrieve
            filters: Optional metadata filters
            
        Returns:
            RetrievalResult with ranked items
        """
        if not self._items:
            return RetrievalResult(query=query, items=[], similarity_scores=[])
        
        # Compute query TF-IDF vector
        query_tf = self._compute_tf_vector(query)
        if not query_tf:
            return RetrievalResult(query=query, items=[], similarity_scores=[])
        
        query_tfidf = self._compute_tfidf_vector(query_tf) if self.enable_indexing else query_tf
        
        # Compute similarities
        similarities = []
        for i, item in enumerate(self._items):
            # Apply filters if specified
            if filters and not self._matches_filters(item, filters):
                continue
                
            # Update access order for LRU
            self._update_access_order(item.id, i)
            
            # Compute similarity
            if self.enable_indexing and i < len(self._tfidf_vectors):
                item_tfidf = self._compute_tfidf_vector(self._tfidf_vectors[i])
                similarity = self._cosine_similarity(query_tfidf, item_tfidf)
            else:
                # Fallback to simple keyword matching
                query_tokens = set(self._tokenize(query))
                item_tokens = set(self._tokenize(item.content))
                if query_tokens and item_tokens:
                    similarity = len(query_tokens & item_tokens) / len(query_tokens | item_tokens)
                else:
                    similarity = 0.0
            
            similarities.append((similarity, i, item))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:k]
        
        # Extract items and scores
        retrieved_items = [item for _, _, item in top_results]
        similarity_scores = [score for score, _, _ in top_results]
        
        return RetrievalResult(
            query=query,
            items=retrieved_items,
            similarity_scores=similarity_scores,
            retrieval_metadata={
                "adapter_type": "in_memory",
                "total_items": len(self._items),
                "indexing_enabled": self.enable_indexing
            }
        )
    
    def _matches_filters(self, item: MemoryItem, filters: Dict[str, Any]) -> bool:
        """Check if item matches metadata filters."""
        for key, value in filters.items():
            if key == "speaker":
                if item.speaker != value:
                    return False
            elif key in item.metadata:
                if item.metadata[key] != value:
                    return False
            else:
                return False
        return True
    
    async def count(self) -> int:
        """Return the total number of items stored."""
        return len(self._items)
    
    async def clear(self) -> bool:
        """Clear all stored items."""
        try:
            await self._cleanup_impl()
            return True
        except Exception:
            return False
    
    # Cost model hooks for dispatcher integration
    def estimate_store_cost(self, item: MemoryItem) -> float:
        """Estimate cost to store an item (always 0.0 for in-memory)."""
        # In-memory storage has zero cost (just RAM allocation)
        return 0.0
    
    def estimate_retrieve_cost(self, query: str, k: int = 5) -> float:
        """Estimate cost to retrieve items (always 0.0 for in-memory).""" 
        # TF-IDF computation is free (local CPU)
        return 0.0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for monitoring."""
        return {
            "total_items": len(self._items),
            "max_items": self.max_items,
            "memory_utilization": len(self._items) / self.max_items if self.max_items else 0.0,
            "vocabulary_size": len(self._vocabulary),
            "indexing_enabled": self.enable_indexing,
            "adapter_type": "in_memory"
        }