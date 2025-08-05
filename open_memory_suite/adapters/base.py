"""Base classes and interfaces for memory adapters."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """A single piece of information stored in memory."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="The text content to store")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    speaker: Optional[str] = Field(None, description="Who said this (user/assistant)")
    turn_id: Optional[str] = Field(None, description="Conversation turn identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RetrievalResult(BaseModel):
    """Result of a memory retrieval operation."""
    
    items: List[MemoryItem] = Field(default_factory=list)
    query: str = Field(..., description="The query that was used")
    similarity_scores: Optional[List[float]] = Field(None, description="Similarity scores if available")
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def content_only(self) -> List[str]:
        """Extract just the content strings from retrieved items."""
        return [item.content for item in self.items]


class MemoryAdapter(ABC):
    """Abstract base class for all memory storage adapters."""
    
    def __init__(self, name: str):
        """Initialize the adapter with a unique name."""
        self.name = name
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the adapter (create indices, connections, etc.)."""
        if self._initialized:
            return
        await self._initialize_impl()
        self._initialized = True
    
    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization logic."""
        pass
    
    async def cleanup(self) -> None:
        """Clean up resources (close connections, etc.)."""
        if not self._initialized:
            return
        await self._cleanup_impl()
        self._initialized = False
    
    async def _cleanup_impl(self) -> None:
        """Implementation-specific cleanup logic. Override if needed."""
        pass
    
    @abstractmethod
    async def store(self, item: MemoryItem) -> bool:
        """
        Store a memory item.
        
        Args:
            item: The memory item to store
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def retrieve(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant memory items for a query.
        
        Args:
            query: The search query
            k: Number of items to retrieve
            filters: Optional filters to apply
            
        Returns:
            RetrievalResult containing matched items
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Return the total number of items stored."""
        pass
    
    async def health_check(self) -> bool:
        """Check if the adapter is healthy and operational."""
        try:
            await self.count()
            return True
        except Exception:
            return False
    
    def estimate_store_cost(self, item: MemoryItem) -> float:
        """
        Estimate the cost to store an item (in cents).
        
        Args:
            item: The memory item to store
            
        Returns:
            Estimated cost in cents
        """
        return 0.0  # Default implementation for backwards compatibility
    
    def estimate_retrieve_cost(self, query: str, k: int = 5) -> float:
        """
        Estimate the cost to retrieve items (in cents).
        
        Args:
            query: The search query
            k: Number of items to retrieve
            
        Returns:
            Estimated cost in cents
        """
        return 0.0  # Default implementation for backwards compatibility
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for monitoring and cost analysis.
        
        Returns:
            Dictionary with storage metrics
        """
        return {
            "adapter_name": self.name,
            "adapter_type": self.__class__.__name__,
            "initialized": self._initialized
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()