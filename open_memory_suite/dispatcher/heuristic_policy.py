"""Rule-based heuristic policy for intelligent memory routing."""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ..adapters.base import MemoryAdapter, MemoryItem
from ..adapters.registry import AdapterRegistry
from ..benchmark.cost_model import BudgetType
from .core import (
    ContentType,
    ConversationContext,
    MemoryAction,
    MemoryPolicy,
    Priority,
)


class HeuristicPolicy(MemoryPolicy):
    """
    Rule-based memory routing policy implementing intelligent heuristics.
    
    This policy implements the core routing logic for M2 milestone:
    - 40% cost reduction vs "store everything in FAISS" baseline
    - ≥90% recall retention through intelligent content prioritization
    - Human-interpretable decision reasoning
    
    Content Analysis Rules:
    1. Names/dates/numbers → Store (high value for recall)
    2. User questions → Always store (highly queryable)
    3. Assistant acknowledgments → Drop (low information value)
    4. Long responses (>500 tokens) → Summarize (cost optimization)
    
    Adapter Selection Rules:
    1. Factual content → FAISS (semantic retrieval)
    2. Recent conversation → InMemory (fast access)
    3. Archival content → FileStore (cheap persistence)
    4. Structured entities → Zep (when available)
    
    Cost-Aware Routing:
    1. Budget exceeded → prefer cheaper adapters
    2. Recall-critical queries → use best adapter regardless of cost
    """
    
    def __init__(self, name: str = "heuristic_v1", version: str = "1.0"):
        """Initialize the heuristic policy with routing rules."""
        super().__init__(name, version)
        
        # Compile regex patterns for efficiency
        self._name_patterns = [
            re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # "John Smith"
            re.compile(r'\bmy name is ([A-Z][a-z]+)', re.IGNORECASE),  # "my name is John"
            re.compile(r'\bi\'?m ([A-Z][a-z]+)', re.IGNORECASE),  # "I'm John"
            re.compile(r'\bcalled ([A-Z][a-z]+)', re.IGNORECASE),  # "I'm called John"
        ]
        
        self._date_patterns = [
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),  # "12/25/2023"
            re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),  # "January 15, 2023"
            re.compile(r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b', re.IGNORECASE),  # "15 Jan 2023"
        ]
        
        self._number_patterns = [
            re.compile(r'\b\d+\.\d+\b'),  # Decimal numbers
            re.compile(r'\b\d{4,}\b'),    # Large numbers (phone, ID, etc.)
            re.compile(r'\$\d+(?:\.\d{2})?\b'),  # Money amounts
        ]
        
        self._acknowledgment_patterns = [
            re.compile(r'^\s*(ok|okay|yes|no|thanks?|got it|understood|alright|sure)\s*[.!]*\s*$', re.IGNORECASE),
            re.compile(r'^\s*(👍|👌|✓|✅|❤️|🙏)\s*$'),  # Emoji responses
        ]
        
        self._question_patterns = [
            re.compile(r'\?'),  # Contains question mark
            re.compile(r'^(what|when|where|who|why|how|can|could|would|should|do|does|did|is|are|was|were)', re.IGNORECASE),
        ]
        
        # Keywords indicating factual content
        self._factual_keywords = {
            'name', 'called', 'age', 'born', 'date', 'time', 'location', 'address',
            'phone', 'email', 'company', 'job', 'title', 'position', 'salary',
            'price', 'cost', 'amount', 'number', 'count', 'quantity',
            'fact', 'information', 'data', 'statistics', 'measurement'
        }
        
        # Keywords indicating procedural content
        self._procedural_keywords = {
            'how', 'step', 'process', 'method', 'way', 'procedure', 'instruction',
            'tutorial', 'guide', 'manual', 'recipe', 'algorithm', 'workflow'
        }
        
        # Low-value phrases that can be dropped
        self._drop_phrases = {
            'got it', 'ok', 'okay', 'yes', 'no', 'thanks', 'thank you',
            'alright', 'sure', 'right', 'i see', 'understood', 'makes sense',
            'sounds good', 'perfect', 'great', 'awesome', 'cool'
        }
    
    async def decide_action(
        self, 
        item: MemoryItem, 
        context: ConversationContext
    ) -> MemoryAction:
        """
        Decide what action to take with a memory item using heuristic rules.
        
        Args:
            item: Memory item to analyze
            context: Conversation context for decision making
            
        Returns:
            The action to take (store/summarize/drop/defer)
        """
        content = item.content.strip()
        content_lower = content.lower()
        
        # Rule 1: Empty or very short content → Drop
        if len(content) < 5:
            return MemoryAction.DROP
        
        # Rule 2: Simple acknowledgments → Drop (cost optimization)
        if self._is_acknowledgment(content):
            return MemoryAction.DROP
        
        # Rule 3: User questions → Always store (high recall value)
        if item.speaker == "user" and self._contains_question(content):
            return MemoryAction.STORE
        
        # Rule 4: Contains names, dates, or numbers → Store (factual information)
        if self._contains_factual_info(content):
            return MemoryAction.STORE
        
        # Rule 5: Very long content → Summarize (cost optimization)
        word_count = len(content.split())
        if word_count > 125:  # ~500 tokens (4 chars/token estimate)
            return MemoryAction.SUMMARIZE
        
        # Rule 6: Budget critical → Be more selective
        if context.is_budget_critical():
            # Only store high-priority content when budget is tight
            if (item.speaker == "user" or 
                self._contains_factual_info(content) or
                self._is_procedural_content(content)):
                return MemoryAction.STORE
            else:
                return MemoryAction.DROP
        
        # Rule 7: Recent conversation turn → Store in fast access
        if context.turn_count <= 10:
            return MemoryAction.STORE
        
        # Rule 8: Contains important keywords → Store
        if self._has_important_keywords(content_lower):
            return MemoryAction.STORE
        
        # Rule 9: Assistant responses with substantial content → Store
        if (item.speaker == "assistant" and 
            word_count > 20 and 
            not self._is_generic_response(content_lower)):
            return MemoryAction.STORE
        
        # Rule 10: Default for conversational content → Store (but low priority)
        # This maintains recall while allowing adapter selection to optimize cost
        return MemoryAction.STORE
    
    async def choose_adapter(
        self,
        item: MemoryItem,
        available_adapters: List[MemoryAdapter],
        context: ConversationContext
    ) -> Optional[MemoryAdapter]:
        """
        Choose the best adapter for storing an item using heuristic rules.
        
        Args:
            item: Memory item to store
            available_adapters: List of healthy adapters
            context: Conversation context
            
        Returns:
            Selected adapter or None if no suitable adapter
        """
        if not available_adapters:
            return None
        
        # Create adapter lookup
        adapter_map = {adapter.name: adapter for adapter in available_adapters}
        
        # Content analysis for adapter selection
        content_analysis = await self.analyze_content(item)
        content_type = ContentType(content_analysis.get("content_type", ContentType.CONVERSATIONAL))
        priority = Priority(content_analysis.get("priority", Priority.MEDIUM))
        
        # Rule 1: Budget critical → Always choose cheapest option
        if context.is_budget_critical():
            return self._choose_adapter_by_capability({AdapterRegistry.CAPABILITY_CHEAP}, available_adapters)
        
        # Rule 2: Critical priority → Use best adapter regardless of cost
        if priority == Priority.CRITICAL:
            # Prefer vector + semantic for critical content
            best_adapter = self._choose_adapter_by_capability(
                {AdapterRegistry.CAPABILITY_VECTOR, AdapterRegistry.CAPABILITY_SEMANTIC}, 
                available_adapters
            )
            if best_adapter:
                return best_adapter
            return self._choose_best_adapter(available_adapters)
        
        # Rule 3: Recent conversation (last 10 turns) → Fast adapters for quick access
        if context.turn_count <= 10:
            fast_adapter = self._choose_adapter_by_capability(
                {AdapterRegistry.CAPABILITY_FAST}, available_adapters
            )
            if fast_adapter:
                return fast_adapter
        
        # Rule 4: Factual content → Vector-capable adapters for semantic retrieval
        if (content_type == ContentType.FACTUAL or 
            self._contains_factual_info(item.content)):
            semantic_adapter = self._choose_adapter_by_capability(
                {AdapterRegistry.CAPABILITY_VECTOR, AdapterRegistry.CAPABILITY_SEMANTIC},
                available_adapters
            )
            if semantic_adapter:
                return semantic_adapter
        
        # Rule 5: User queries → Semantic search capability
        if (item.speaker == "user" and 
            self._contains_question(item.content)):
            query_adapter = self._choose_adapter_by_capability(
                {AdapterRegistry.CAPABILITY_SEMANTIC, AdapterRegistry.CAPABILITY_SEARCHABLE},
                available_adapters
            )
            if query_adapter:
                return query_adapter
        
        # Rule 6: Procedural content → Vector search for semantic retrieval
        if content_type == ContentType.PROCEDURAL:
            procedural_adapter = self._choose_adapter_by_capability(
                {AdapterRegistry.CAPABILITY_VECTOR, AdapterRegistry.CAPABILITY_SEMANTIC},
                available_adapters
            )
            if procedural_adapter:
                return procedural_adapter
        
        # Rule 7: Long-term archival → Persistent + cheap storage
        session_duration = context.get_session_duration_minutes()
        if (session_duration > 30 and  # Session has been going for a while
            priority in [Priority.LOW, Priority.MEDIUM]):
            archival_adapter = self._choose_adapter_by_capability(
                {AdapterRegistry.CAPABILITY_PERSISTENT, AdapterRegistry.CAPABILITY_CHEAP},
                available_adapters
            )
            if archival_adapter:
                return archival_adapter
        
        # Rule 8: Premium budget → Use vector-capable adapters for quality
        if context.budget_type == BudgetType.PREMIUM:
            premium_adapter = self._choose_adapter_by_capability(
                {AdapterRegistry.CAPABILITY_VECTOR, AdapterRegistry.CAPABILITY_SEMANTIC},
                available_adapters
            )
            if premium_adapter:
                return premium_adapter
        
        # Rule 9: Minimal budget → Use cheapest available adapter
        if context.budget_type == BudgetType.MINIMAL:
            minimal_adapter = self._choose_adapter_by_capability(
                {AdapterRegistry.CAPABILITY_CHEAP}, available_adapters
            )
            if minimal_adapter:
                return minimal_adapter
        
        # Rule 11: Default selection based on content length and adapter availability
        word_count = len(item.content.split())
        
        if word_count < 50:  # Short content
            # Prefer InMemory → FAISS → FileStore
            for adapter_name in ["memory_store", "faiss_store", "file_store"]:
                if adapter_name in adapter_map:
                    return adapter_map[adapter_name]
        else:  # Longer content
            # Prefer FAISS → FileStore → InMemory
            for adapter_name in ["faiss_store", "file_store", "memory_store"]:
                if adapter_name in adapter_map:
                    return adapter_map[adapter_name]
        
        # Fallback: Return first available adapter
        return available_adapters[0] if available_adapters else None
    
    async def analyze_content(self, item: MemoryItem) -> Dict[str, Any]:
        """Enhanced content analysis for routing decisions."""
        base_analysis = await super().analyze_content(item)
        content = item.content.strip()
        content_lower = content.lower()
        
        # Enhanced analysis
        enhanced_features = {
            **base_analysis,
            
            # Content type classification
            "is_factual": self._contains_factual_info(content),
            "is_procedural": self._is_procedural_content(content),
            "is_query": self._contains_question(content),
            "is_acknowledgment": self._is_acknowledgment(content),
            "is_emotional": self._contains_emotional_content(content_lower),
            
            # Importance indicators
            "has_names": self._contains_names(content),
            "has_dates": self._contains_dates(content),
            "has_numbers": self._contains_numbers(content),
            "has_structured_entities": self._has_structured_entities(content),
            
            # Content characteristics
            "word_count": len(content.split()),
            "sentence_count": len([s for s in content.split('.') if s.strip()]),
            "avg_word_length": sum(len(word) for word in content.split()) / max(1, len(content.split())),
            "has_technical_terms": self._has_technical_terms(content_lower),
            
            # Conversation context
            "speaker": item.speaker or "unknown",
            "timestamp": item.timestamp.isoformat() if item.timestamp else None,
        }
        
        # Determine priority based on enhanced analysis
        if enhanced_features["is_query"] and item.speaker == "user":
            enhanced_features["priority"] = Priority.CRITICAL
        elif enhanced_features["has_names"] or enhanced_features["has_dates"]:
            enhanced_features["priority"] = Priority.HIGH
        elif enhanced_features["is_acknowledgment"]:
            enhanced_features["priority"] = Priority.LOW
        elif enhanced_features["is_factual"] or enhanced_features["is_procedural"]:
            enhanced_features["priority"] = Priority.HIGH
        else:
            enhanced_features["priority"] = Priority.MEDIUM
        
        # Determine content type
        if enhanced_features["is_factual"]:
            enhanced_features["content_type"] = ContentType.FACTUAL
        elif enhanced_features["is_procedural"]:
            enhanced_features["content_type"] = ContentType.PROCEDURAL
        elif enhanced_features["is_query"]:
            enhanced_features["content_type"] = ContentType.QUERY
        elif enhanced_features["is_emotional"]:
            enhanced_features["content_type"] = ContentType.EMOTIONAL
        elif enhanced_features["is_acknowledgment"]:
            enhanced_features["content_type"] = ContentType.META
        else:
            enhanced_features["content_type"] = ContentType.CONVERSATIONAL
        
        return enhanced_features
    
    # Helper methods for content analysis
    
    def _is_acknowledgment(self, content: str) -> bool:
        """Check if content is a simple acknowledgment."""
        content_clean = content.strip().lower()
        
        # Check against acknowledgment patterns
        for pattern in self._acknowledgment_patterns:
            if pattern.match(content):
                return True
        
        # Check against drop phrases
        return content_clean in self._drop_phrases
    
    def _contains_question(self, content: str) -> bool:
        """Check if content contains a question."""
        for pattern in self._question_patterns:
            if pattern.search(content):
                return True
        return False
    
    def _contains_factual_info(self, content: str) -> bool:
        """Check if content contains factual information."""
        return (self._contains_names(content) or 
                self._contains_dates(content) or 
                self._contains_numbers(content) or
                self._has_factual_keywords(content.lower()))
    
    def _contains_names(self, content: str) -> bool:
        """Check if content contains person names."""
        for pattern in self._name_patterns:
            if pattern.search(content):
                return True
        return False
    
    def _contains_dates(self, content: str) -> bool:
        """Check if content contains dates."""
        for pattern in self._date_patterns:
            if pattern.search(content):
                return True
        return False
    
    def _contains_numbers(self, content: str) -> bool:
        """Check if content contains significant numbers."""
        for pattern in self._number_patterns:
            if pattern.search(content):
                return True
        return False
    
    def _has_factual_keywords(self, content_lower: str) -> bool:
        """Check if content has factual keywords."""
        words = set(content_lower.split())
        return bool(words & self._factual_keywords)
    
    def _is_procedural_content(self, content: str) -> bool:
        """Check if content is procedural (how-to, instructions)."""
        content_lower = content.lower()
        words = set(content_lower.split())
        return bool(words & self._procedural_keywords)
    
    def _contains_emotional_content(self, content_lower: str) -> bool:
        """Check if content contains emotional indicators."""
        emotional_keywords = {
            'feel', 'feeling', 'emotion', 'happy', 'sad', 'angry', 'excited',
            'frustrated', 'love', 'hate', 'like', 'dislike', 'prefer',
            'enjoy', 'worry', 'concern', 'hope', 'wish', 'dream'
        }
        words = set(content_lower.split())
        return bool(words & emotional_keywords)
    
    def _has_structured_entities(self, content: str) -> bool:
        """Check if content has structured entities suitable for graph storage."""
        # Simple heuristics for structured data
        return (
            ' -> ' in content or  # Relationship indicators
            ' relates to ' in content.lower() or
            ' is a ' in content.lower() or
            ' works at ' in content.lower() or
            ' lives in ' in content.lower() or
            re.search(r'\b\w+ of \w+\b', content)  # "CEO of Company"
        )
    
    def _has_important_keywords(self, content_lower: str) -> bool:
        """Check if content has generally important keywords."""
        important_keywords = {
            'important', 'critical', 'urgent', 'remember', 'note', 'key',
            'essential', 'vital', 'crucial', 'significant', 'main', 'primary'
        }
        words = set(content_lower.split())
        return bool(words & important_keywords)
    
    def _has_technical_terms(self, content_lower: str) -> bool:
        """Check if content has technical terminology."""
        technical_keywords = {
            'api', 'database', 'server', 'client', 'function', 'method',
            'algorithm', 'code', 'programming', 'software', 'hardware',
            'system', 'network', 'protocol', 'framework', 'library'
        }
        words = set(content_lower.split())
        return bool(words & technical_keywords)
    
    def _is_generic_response(self, content_lower: str) -> bool:
        """Check if assistant response is generic/template-like."""
        generic_phrases = {
            'i understand', 'i see', 'that makes sense', 'good question',
            'let me help', 'i can help', 'happy to help', 'of course',
            'certainly', 'absolutely', 'definitely', 'sure thing'
        }
        
        # Check if content starts with generic phrases
        for phrase in generic_phrases:
            if content_lower.startswith(phrase):
                return True
        
        return False
    
    def _choose_cheapest_adapter(self, adapters: List[MemoryAdapter]) -> MemoryAdapter:
        """Choose the cheapest adapter from available options."""
        # Cost hierarchy: FileStore < InMemory < FAISS < Zep
        cost_order = ["file_store", "memory_store", "faiss_store", "zep_store"]
        
        for adapter_name in cost_order:
            for adapter in adapters:
                if adapter.name == adapter_name:
                    return adapter
        
        return adapters[0]  # Fallback
    
    def _choose_best_adapter(self, adapters: List[MemoryAdapter]) -> MemoryAdapter:
        """Choose the best quality adapter from available options."""
        # Quality hierarchy: FAISS > Zep > InMemory > FileStore
        quality_order = ["faiss_store", "zep_store", "memory_store", "file_store"]
        
        for adapter_name in quality_order:
            for adapter in adapters:
                if adapter.name == adapter_name:
                    return adapter
        
        return adapters[0]  # Fallback
    
    def _choose_adapter_by_capability(
        self, 
        required_capabilities: Set[str], 
        available_adapters: List[MemoryAdapter]
    ) -> Optional[MemoryAdapter]:
        """
        Choose an adapter that has all required capabilities.
        
        Args:
            required_capabilities: Set of capabilities that must all be present
            available_adapters: List of available adapters to choose from
            
        Returns:
            First adapter that matches all capabilities, or None if no match
        """
        for adapter in available_adapters:
            adapter_capabilities = AdapterRegistry.get_capabilities(adapter.__class__.__name__)
            if required_capabilities.issubset(adapter_capabilities):
                return adapter
        return None