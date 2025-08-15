"""Summarization service for memory content compression."""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import logging
from pydantic import BaseModel
import tiktoken


logger = logging.getLogger(__name__)


class SummaryResult(BaseModel):
    """Result from summarization operation."""
    original_content: str
    summary: str
    original_token_count: int
    summary_token_count: int
    compression_ratio: float
    cost_cents: float = 0.0
    processing_time_ms: float = 0.0


class SummarizerConfig(BaseModel):
    """Configuration for summarization service."""
    model: str = "local_chunking"  # Default to simple chunking
    max_input_tokens: int = 4000
    target_compression_ratio: float = 0.3  # Target 30% of original length
    chunk_overlap: int = 100
    openai_api_key: Optional[str] = None
    enable_openai: bool = False


class BaseSummarizer(ABC):
    """Abstract base class for summarization services."""
    
    def __init__(self, config: SummarizerConfig):
        self.config = config
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    @abstractmethod
    async def summarize(self, content: str, context: Optional[str] = None) -> SummaryResult:
        """Summarize the given content."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, max_tokens: int) -> list[str]:
        """Break text into chunks that fit within token limits."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens - self.config.chunk_overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks


class LocalChunkingSummarizer(BaseSummarizer):
    """Simple local summarizer using text chunking and truncation."""
    
    async def summarize(self, content: str, context: Optional[str] = None) -> SummaryResult:
        """Summarize content using simple chunking approach."""
        import time
        start_time = time.time()
        
        original_tokens = self.count_tokens(content)
        
        # If content is already short enough, return as-is
        if original_tokens <= self.config.max_input_tokens * self.config.target_compression_ratio:
            return SummaryResult(
                original_content=content,
                summary=content,
                original_token_count=original_tokens,
                summary_token_count=original_tokens,
                compression_ratio=1.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Simple summarization: extract key sentences
        sentences = content.split('.')
        target_sentences = max(1, int(len(sentences) * self.config.target_compression_ratio))
        
        # Take first, middle, and last sentences for basic coverage
        if len(sentences) <= 3:
            summary_sentences = sentences
        else:
            indices = [0]  # First sentence
            if target_sentences > 2:
                # Add middle sentences
                mid_indices = list(range(1, len(sentences) - 1))
                step = max(1, len(mid_indices) // (target_sentences - 2))
                indices.extend(mid_indices[::step][:target_sentences - 2])
            if target_sentences > 1:
                indices.append(len(sentences) - 1)  # Last sentence
        
        summary_sentences = [sentences[i].strip() for i in sorted(set(indices)) if i < len(sentences)]
        summary = '. '.join(summary_sentences)
        
        if not summary.endswith('.'):
            summary += '.'
        
        summary_tokens = self.count_tokens(summary)
        compression_ratio = summary_tokens / original_tokens if original_tokens > 0 else 1.0
        
        return SummaryResult(
            original_content=content,
            summary=summary,
            original_token_count=original_tokens,
            summary_token_count=summary_tokens,
            compression_ratio=compression_ratio,
            cost_cents=0.0,  # Local processing is free
            processing_time_ms=(time.time() - start_time) * 1000
        )


class OpenAISummarizer(BaseSummarizer):
    """OpenAI-powered summarizer using GPT models."""
    
    def __init__(self, config: SummarizerConfig):
        super().__init__(config)
        if not config.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI summarizer")
        
        # Mock OpenAI client for now - would use real openai library in production
        self.mock_openai = True
    
    async def summarize(self, content: str, context: Optional[str] = None) -> SummaryResult:
        """Summarize content using OpenAI API."""
        import time
        start_time = time.time()
        
        original_tokens = self.count_tokens(content)
        
        # Mock OpenAI response for now - replace with real API call
        if self.mock_openai:
            # Simulate OpenAI processing time
            await asyncio.sleep(0.1)
            
            # Create a mock summary
            lines = content.split('\n')
            summary_lines = lines[:max(1, len(lines) // 3)]  # Take first third
            summary = '\n'.join(summary_lines)
            
            if len(summary) < 50:  # If too short, add more context
                summary = content[:200] + "..." if len(content) > 200 else content
        else:
            # Real OpenAI API call would go here
            # import openai
            # response = await openai.ChatCompletion.acreate(...)
            summary = content  # Fallback
        
        summary_tokens = self.count_tokens(summary)
        compression_ratio = summary_tokens / original_tokens if original_tokens > 0 else 1.0
        
        # Estimate cost (GPT-4 pricing: ~$0.03/1K tokens input, $0.06/1K tokens output)
        input_cost = (original_tokens / 1000) * 0.03
        output_cost = (summary_tokens / 1000) * 0.06
        total_cost_cents = (input_cost + output_cost) * 100
        
        return SummaryResult(
            original_content=content,
            summary=summary,
            original_token_count=original_tokens,
            summary_token_count=summary_tokens,
            compression_ratio=compression_ratio,
            cost_cents=total_cost_cents,
            processing_time_ms=(time.time() - start_time) * 1000
        )


class Summarizer:
    """Main summarization service with multiple backend support."""
    
    def __init__(self, config: Optional[SummarizerConfig] = None):
        """Initialize summarizer with configuration."""
        self.config = config or SummarizerConfig()
        self._backend = self._create_backend()
        self._stats = {
            "summaries_created": 0,
            "total_tokens_processed": 0,
            "total_cost_cents": 0.0,
            "avg_compression_ratio": 0.0
        }
    
    def _create_backend(self) -> BaseSummarizer:
        """Create appropriate summarizer backend."""
        if self.config.enable_openai and self.config.openai_api_key:
            logger.info("Using OpenAI summarizer backend")
            return OpenAISummarizer(self.config)
        else:
            logger.info("Using local chunking summarizer backend")
            return LocalChunkingSummarizer(self.config)
    
    async def summarize(
        self, 
        content: str, 
        context: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> SummaryResult:
        """
        Summarize content with optional context.
        
        Args:
            content: Text content to summarize
            context: Optional context to help with summarization
            session_id: Optional session ID for tracking
            
        Returns:
            SummaryResult with summary and metadata
        """
        try:
            result = await self._backend.summarize(content, context)
            
            # Update statistics
            self._stats["summaries_created"] += 1
            self._stats["total_tokens_processed"] += result.original_token_count
            self._stats["total_cost_cents"] += result.cost_cents
            
            # Update average compression ratio
            current_avg = self._stats["avg_compression_ratio"]
            count = self._stats["summaries_created"]
            self._stats["avg_compression_ratio"] = (
                (current_avg * (count - 1) + result.compression_ratio) / count
            )
            
            logger.info(
                f"Summarized content: {result.original_token_count} -> "
                f"{result.summary_token_count} tokens "
                f"({result.compression_ratio:.2%} compression)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Return original content as fallback
            return SummaryResult(
                original_content=content,
                summary=content,
                original_token_count=self._backend.count_tokens(content),
                summary_token_count=self._backend.count_tokens(content),
                compression_ratio=1.0,
                cost_cents=0.0
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get summarization statistics."""
        return self._stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of summarization service."""
        try:
            # Test with small content
            test_content = "This is a test message to verify summarization service health."
            result = await self.summarize(test_content)
            
            return {
                "status": "healthy",
                "backend": type(self._backend).__name__,
                "model": self.config.model,
                "test_compression_ratio": result.compression_ratio,
                "stats": self._stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend": type(self._backend).__name__ if hasattr(self, '_backend') else "unknown"
            }