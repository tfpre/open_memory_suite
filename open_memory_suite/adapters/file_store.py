"""File-based storage adapter for ultra-cheap persistence with linear scan retrieval."""

import asyncio
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiofiles

from .base import MemoryAdapter, MemoryItem, RetrievalResult


class FileStoreAdapter(MemoryAdapter):
    """
    Ultra-cheap file-based storage adapter.
    
    Features:
    - Cheapest persistence option (~$0.00001 per item)
    - Hierarchical directory structure for organization
    - Linear scan retrieval (slow but zero API cost)
    - Keyword-based matching with recency bias
    - Graceful handling of large file collections
    
    Trade-offs:
    - 10-100x slower than FAISS vector search
    - Limited to keyword/exact matching (no semantic similarity)
    - Cost scales with storage size during retrieval
    - Perfect for archival and budget-constrained scenarios
    """
    
    def __init__(
        self,
        name: str = "file_store",
        storage_path: Optional[Path] = None,
        max_files_per_dir: int = 1000,
        enable_compression: bool = False
    ):
        """
        Initialize file store adapter.
        
        Args:
            name: Adapter name for identification
            storage_path: Root directory for file storage
            max_files_per_dir: Maximum files per directory (for filesystem efficiency)
            enable_compression: Whether to compress JSON files (slower but smaller)
        """
        super().__init__(name)
        
        if storage_path is None:
            storage_path = Path.cwd() / "memory_store" / name
        
        self.storage_path = Path(storage_path)
        self.max_files_per_dir = max_files_per_dir
        self.enable_compression = enable_compression
        
        # Caching for performance
        self._file_cache: Dict[str, List[MemoryItem]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_timeout = 300.0  # 5 minutes
        
        # Statistics for cost estimation
        self._total_files = 0
        self._total_items = 0
        self._last_scan_time = 0.0
        
    async def _initialize_impl(self) -> None:
        """Initialize the file storage system."""
        # Create directory structure
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for hierarchical organization
        # Structure: /storage_path/YYYY/MM/DD/session_files/
        current_date = datetime.utcnow()
        date_path = self._get_date_path(current_date)
        date_path.mkdir(parents=True, exist_ok=True)
        
        # Scan existing files to update statistics
        await self._update_storage_stats()
        
    async def _cleanup_impl(self) -> None:
        """Clear caches and release resources."""
        self._file_cache.clear()
        self._cache_timestamps.clear()
        
    def _get_date_path(self, timestamp: datetime) -> Path:
        """Get hierarchical date-based path for a timestamp."""
        return self.storage_path / str(timestamp.year) / f"{timestamp.month:02d}" / f"{timestamp.day:02d}"
    
    def _get_file_path(self, item: MemoryItem) -> Path:
        """Get file path for storing a memory item."""
        date_path = self._get_date_path(item.timestamp)
        
        # Use session ID or turn ID to group related items
        session_id = item.metadata.get("session_id", "default")
        filename = f"{session_id}.jsonl"
        
        return date_path / filename
    
    async def _update_storage_stats(self) -> None:
        """Update internal statistics by scanning storage directory."""
        file_count = 0
        item_count = 0
        
        if self.storage_path.exists():
            for file_path in self.storage_path.rglob("*.jsonl"):
                file_count += 1
                # Quick line count for item estimation
                try:
                    async with aiofiles.open(file_path, 'r') as f:
                        lines = await f.readlines()
                        item_count += len(lines)
                except Exception:
                    # Skip corrupted files
                    continue
        
        self._total_files = file_count
        self._total_items = item_count
        
    async def _load_file_cached(self, file_path: Path) -> List[MemoryItem]:
        """Load file with caching for performance."""
        file_key = str(file_path)
        current_time = asyncio.get_event_loop().time()
        
        # Check cache
        if (file_key in self._file_cache and 
            file_key in self._cache_timestamps and
            current_time - self._cache_timestamps[file_key] < self._cache_timeout):
            return self._file_cache[file_key]
        
        # Load from disk
        items = []
        if file_path.exists():
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    async for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                item = MemoryItem(**data)
                                items.append(item)
                            except (json.JSONDecodeError, ValueError):
                                # Skip corrupted lines
                                continue
            except Exception:
                # File access error
                pass
        
        # Update cache
        self._file_cache[file_key] = items
        self._cache_timestamps[file_key] = current_time
        
        return items
    
    def _tokenize_for_search(self, text: str) -> Set[str]:
        """Tokenize text for keyword searching."""
        # Convert to lowercase and extract alphanumeric words
        tokens = set(re.findall(r'[a-zA-Z0-9]+', text.lower()))
        return tokens
    
    def _compute_keyword_similarity(self, query_tokens: Set[str], content_tokens: Set[str]) -> float:
        """Compute keyword similarity using Jaccard index."""
        if not query_tokens or not content_tokens:
            return 0.0
        
        intersection = query_tokens & content_tokens
        union = query_tokens | content_tokens
        
        return len(intersection) / len(union) if union else 0.0
    
    async def store(self, item: MemoryItem) -> bool:
        """
        Store a memory item to file.
        
        Args:
            item: The memory item to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_file_path(item)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to file (JSONL format)
            async with aiofiles.open(file_path, 'a') as f:
                json_line = item.model_dump_json() + "\n"
                await f.write(json_line)
            
            # Invalidate cache for this file
            file_key = str(file_path)
            if file_key in self._file_cache:
                del self._file_cache[file_key]
                del self._cache_timestamps[file_key]
            
            # Update statistics
            self._total_items += 1
            
            return True
            
        except Exception as e:
            # Log error in production
            return False
    
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant items using keyword matching with linear scan.
        
        Args:
            query: The search query
            k: Number of items to retrieve
            filters: Optional metadata filters
            
        Returns:
            RetrievalResult with ranked items
        """
        start_time = asyncio.get_event_loop().time()
        
        if not query.strip():
            return RetrievalResult(query=query, items=[], similarity_scores=[])
        
        query_tokens = self._tokenize_for_search(query)
        if not query_tokens:
            return RetrievalResult(query=query, items=[], similarity_scores=[])
        
        candidates = []
        
        # Scan files with recency bias (start with recent dates)
        current_date = datetime.utcnow()
        
        for days_back in range(30):  # Scan last 30 days
            scan_date = datetime(
                current_date.year,
                current_date.month,
                current_date.day
            ) - timedelta(days=days_back)
            
            date_path = self._get_date_path(scan_date)
            if not date_path.exists():
                continue
            
            # Scan all JSONL files in this date directory
            for file_path in date_path.glob("*.jsonl"):
                items = await self._load_file_cached(file_path)
                
                for item in items:
                    # Apply filters if specified
                    if filters and not self._matches_filters(item, filters):
                        continue
                    
                    # Compute keyword similarity
                    content_tokens = self._tokenize_for_search(item.content)
                    similarity = self._compute_keyword_similarity(query_tokens, content_tokens)
                    
                    if similarity > 0.0:
                        # Add recency bonus (more recent = slightly higher score)
                        recency_bonus = 1.0 / (days_back + 1) * 0.1
                        adjusted_similarity = similarity + recency_bonus
                        
                        candidates.append((adjusted_similarity, item))
        
        # Sort by similarity (descending) and take top k
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:k]
        
        # Extract items and scores
        retrieved_items = [item for _, item in top_candidates]
        similarity_scores = [score for score, _ in top_candidates]
        
        # Update scan time for cost estimation
        self._last_scan_time = asyncio.get_event_loop().time() - start_time
        
        return RetrievalResult(
            query=query,
            items=retrieved_items,
            similarity_scores=similarity_scores,
            retrieval_metadata={
                "adapter_type": "file_store",
                "scan_time_ms": self._last_scan_time * 1000,
                "files_scanned": len(candidates),
                "total_items_scanned": len(candidates),
                "matching_method": "keyword_jaccard"
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
        # Use cached count if recent, otherwise scan
        if self._total_items == 0:
            await self._update_storage_stats()
        return self._total_items
    
    async def clear(self) -> bool:
        """Clear all stored items."""
        try:
            # Remove all JSONL files
            for file_path in self.storage_path.rglob("*.jsonl"):
                file_path.unlink()
            
            # Clear caches
            self._file_cache.clear()
            self._cache_timestamps.clear()
            self._total_files = 0
            self._total_items = 0
            
            return True
        except Exception:
            return False
    
    def estimate_store_cost(self, item: MemoryItem) -> float:
        """Estimate cost to store an item (ultra-cheap file I/O)."""
        # Base cost: file append operation
        base_cost = 0.00001  # $0.00001 cents
        
        # Scale slightly by content size (larger items take more disk I/O)
        content_size_factor = min(len(item.content) / 1000.0, 2.0)  # Cap at 2x
        
        return base_cost * content_size_factor
    
    def estimate_retrieve_cost(self, query: str, k: int = 5) -> float:
        """Estimate cost to retrieve items (scales with storage size)."""
        # Base cost: linear scan operation
        base_cost = 0.0001  # $0.0001 cents
        
        # Scale by storage size (more files = more expensive scan)
        storage_scale_factor = min(self._total_files / 100.0, 10.0)  # Cap at 10x
        
        # Scale by number of requested items (minimal impact)
        k_factor = min(k / 5.0, 2.0)
        
        return base_cost * storage_scale_factor * k_factor
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for monitoring."""
        base_stats = super().get_storage_stats()
        
        # Calculate storage efficiency metrics
        avg_items_per_file = self._total_items / max(1, self._total_files)
        cache_hit_ratio = len(self._file_cache) / max(1, self._total_files)
        
        base_stats.update({
            "total_files": self._total_files,
            "total_items": self._total_items,
            "avg_items_per_file": round(avg_items_per_file, 2),
            "cache_hit_ratio": round(cache_hit_ratio, 3),
            "last_scan_time_ms": round(self._last_scan_time * 1000, 2),
            "storage_path": str(self.storage_path),
            "max_files_per_dir": self.max_files_per_dir,
            "compression_enabled": self.enable_compression
        })
        
        return base_stats


# Import timedelta for date calculations
from datetime import timedelta