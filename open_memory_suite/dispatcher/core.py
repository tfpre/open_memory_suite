"""Core interfaces and data structures for the frugal memory dispatcher."""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from ..adapters.base import MemoryAdapter, MemoryItem
from ..benchmark.cost_model import BudgetType, CostEstimate, CostTracker


class MemoryAction(str, Enum):
    """Actions the dispatcher can take with a memory item."""
    STORE = "store"           # Store in memory adapter
    SUMMARIZE = "summarize"   # Compress and store summary
    DROP = "drop"             # Discard (low value content)
    DEFER = "defer"           # Delay decision (needs more context)


class Priority(str, Enum):
    """Priority levels for memory operations."""
    CRITICAL = "critical"     # Must store (names, important facts)
    HIGH = "high"            # Should store (user questions, key info)
    MEDIUM = "medium"        # May store (general conversation)
    LOW = "low"              # Consider dropping (acknowledgments)


class ContentType(str, Enum):
    """Semantic categories of memory content."""
    FACTUAL = "factual"           # Facts, names, dates, numbers
    CONVERSATIONAL = "conversational"  # General dialogue
    PROCEDURAL = "procedural"     # Instructions, how-to information
    EMOTIONAL = "emotional"       # Sentiment, preferences, reactions
    META = "meta"                # System messages, acknowledgments
    QUERY = "query"              # User questions (highly valuable)


class ConversationContext(BaseModel):
    """Rich context about the conversation state for informed routing decisions."""
    
    session_id: str = Field(..., description="Unique session identifier")
    turn_count: int = Field(default=0, description="Number of turns in conversation")
    recent_turns: List[MemoryItem] = Field(default_factory=list, description="Last N turns for context")
    
    # Budget and cost tracking
    cost_tracker: Optional[CostTracker] = Field(None, description="Current session cost tracking")
    budget_type: BudgetType = Field(default=BudgetType.STANDARD, description="Budget constraint level")
    budget_exhausted: bool = Field(default=False, description="Whether budget limits are hit")
    
    # Memory pressure indicators
    total_stored_items: int = Field(default=0, description="Total items across all adapters")
    storage_by_adapter: Dict[str, int] = Field(default_factory=dict, description="Items per adapter")
    
    # Performance requirements
    latency_target_ms: float = Field(default=200.0, description="Target response latency")
    quality_threshold: float = Field(default=0.90, description="Minimum acceptable recall")
    
    # User preferences and session characteristics
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User-specific settings")
    conversation_type: str = Field(default="general", description="Type of conversation")
    criticality_level: Priority = Field(default=Priority.MEDIUM, description="Overall conversation importance")
    
    # Temporal context
    session_start_time: datetime = Field(default_factory=datetime.utcnow, description="When session began")
    last_activity_time: datetime = Field(default_factory=datetime.utcnow, description="Last user interaction")
    
    def get_session_duration_minutes(self) -> float:
        """Calculate session duration in minutes."""
        return (datetime.utcnow() - self.session_start_time).total_seconds() / 60.0
    
    def get_recent_content_summary(self, max_turns: int = 5) -> str:
        """Get summary of recent conversation content."""
        recent = self.recent_turns[-max_turns:] if self.recent_turns else []
        return " | ".join([item.content[:100] + "..." if len(item.content) > 100 else item.content 
                          for item in recent])
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_time = datetime.utcnow()
    
    def is_budget_critical(self) -> bool:
        """Check if we're approaching budget limits."""
        if not self.cost_tracker:
            return False
        
        # Simple heuristic: >80% of session budget used
        if self.budget_type == BudgetType.MINIMAL:
            return self.cost_tracker.total_spent > 0.8
        elif self.budget_type == BudgetType.STANDARD:
            return self.cost_tracker.total_spent > 4.0
        elif self.budget_type == BudgetType.PREMIUM:
            return self.cost_tracker.total_spent > 16.0
        
        return False


class RoutingDecision(BaseModel):
    """Represents a routing decision made by the memory dispatcher."""
    
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    action: MemoryAction = Field(..., description="What action to take")
    selected_adapter: Optional[str] = Field(None, description="Which adapter to use (if storing)")
    
    # Cost analysis
    estimated_cost: Optional[CostEstimate] = Field(None, description="Cost estimate for this decision")
    cost_benefit_ratio: float = Field(default=1.0, description="Expected value / cost ratio")
    
    # Decision reasoning
    reasoning: str = Field(..., description="Human-readable explanation of the decision")
    confidence: float = Field(default=0.8, description="Confidence in decision (0.0-1.0)")
    
    # Content analysis
    detected_priority: Priority = Field(default=Priority.MEDIUM, description="Detected content priority")
    detected_content_type: ContentType = Field(default=ContentType.CONVERSATIONAL, description="Content category")
    key_features: List[str] = Field(default_factory=list, description="Important features detected")
    
    # Context used
    context_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Context state at decision time")
    alternatives_considered: List[str] = Field(default_factory=list, description="Other options considered")
    
    # Timing
    decision_timestamp: datetime = Field(default_factory=datetime.utcnow)
    decision_latency_ms: float = Field(default=0.0, description="Time taken to make decision")
    
    # Metadata for analysis
    policy_version: str = Field(default="unknown", description="Version of policy that made decision")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional decision metadata")
    
    def to_trace_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for trace logging."""
        return {
            "decision_id": self.decision_id,
            "action": self.action.value,
            "adapter": self.selected_adapter,
            "cost_cents": self.estimated_cost.total_cost if self.estimated_cost else 0.0,
            "confidence": self.confidence,
            "priority": self.detected_priority.value,
            "content_type": self.detected_content_type.value,
            "reasoning": self.reasoning,
            "latency_ms": self.decision_latency_ms,
            "timestamp": self.decision_timestamp.isoformat()
        }


class MemoryPolicy(ABC):
    """
    Abstract base class for memory routing policies.
    
    Policies make intelligent decisions about how to handle memory items based on:
    - Content analysis (what type of information is this?)
    - Cost constraints (what can we afford?)
    - Quality requirements (what recall do we need?)
    - Context awareness (what's the conversation state?)
    
    This interface supports evolution from rule-based → ML → RL policies.
    """
    
    def __init__(self, name: str, version: str = "1.0"):
        """Initialize policy with identifying information."""
        self.name = name
        self.version = version
        self._stats = {
            "decisions_made": 0,
            "items_stored": 0,
            "items_dropped": 0,
            "items_summarized": 0,
            "total_cost_saved": 0.0
        }
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def decide_action(
        self, 
        item: MemoryItem, 
        context: ConversationContext
    ) -> MemoryAction:
        """
        Decide what action to take with a memory item.
        
        Args:
            item: The memory item to route
            context: Current conversation context
            
        Returns:
            The action to take (store/summarize/drop/defer)
        """
        pass
    
    @abstractmethod
    async def choose_adapter(
        self,
        item: MemoryItem,
        available_adapters: List[MemoryAdapter],
        context: ConversationContext
    ) -> Optional[MemoryAdapter]:
        """
        Choose which adapter to use for storing an item.
        
        Args:
            item: The memory item to store
            available_adapters: List of available storage adapters
            context: Current conversation context
            
        Returns:
            The selected adapter, or None if unable to choose
        """
        pass
    
    async def analyze_content(self, item: MemoryItem) -> Dict[str, Any]:
        """
        Analyze memory item content to extract features for routing decisions.
        
        Args:
            item: Memory item to analyze
            
        Returns:
            Dictionary of extracted features and metadata
        """
        # Default implementation - subclasses can override for more sophisticated analysis
        content = item.content.lower().strip()
        
        # Basic content analysis
        features = {
            "length": len(item.content),
            "word_count": len(item.content.split()),
            "has_numbers": any(c.isdigit() for c in content),
            "has_questions": "?" in content,
            "has_names": any(word.istitle() for word in item.content.split()),
            "is_short_response": len(content) < 20,
            "speaker": item.speaker or "unknown"
        }
        
        # Simple heuristics for content type
        if "?" in content and item.speaker == "user":
            features["content_type"] = ContentType.QUERY
            features["priority"] = Priority.HIGH
        elif any(word in content for word in ["name", "called", "i am", "my name"]):
            features["content_type"] = ContentType.FACTUAL
            features["priority"] = Priority.CRITICAL
        elif any(word in content for word in ["ok", "yes", "no", "thanks", "got it"]):
            features["content_type"] = ContentType.META
            features["priority"] = Priority.LOW
        elif len(content) > 500:
            features["content_type"] = ContentType.CONVERSATIONAL
            features["priority"] = Priority.MEDIUM
        else:
            features["content_type"] = ContentType.CONVERSATIONAL
            features["priority"] = Priority.MEDIUM
            
        return features
    
    async def record_decision(self, decision: RoutingDecision) -> None:
        """Record decision statistics for monitoring and optimization."""
        async with self._lock:
            self._stats["decisions_made"] += 1
            
            if decision.action == MemoryAction.STORE:
                self._stats["items_stored"] += 1
            elif decision.action == MemoryAction.DROP:
                self._stats["items_dropped"] += 1
            elif decision.action == MemoryAction.SUMMARIZE:
                self._stats["items_summarized"] += 1
            
            if decision.estimated_cost:
                # Estimate cost savings vs. "store everything in FAISS"
                baseline_cost = 0.0002  # Rough FAISS storage cost
                if decision.action == MemoryAction.DROP:
                    self._stats["total_cost_saved"] += baseline_cost
                elif decision.estimated_cost.total_cost < baseline_cost:
                    self._stats["total_cost_saved"] += (baseline_cost - decision.estimated_cost.total_cost)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get policy performance statistics."""
        return {
            "policy_name": self.name,
            "policy_version": self.version,
            **self._stats.copy()
        }


class PolicyRegistry:
    """Registry for managing different memory policies."""
    
    def __init__(self):
        self._policies: Dict[str, MemoryPolicy] = {}
        self._default_policy: Optional[str] = None
    
    def register(self, policy: MemoryPolicy, set_as_default: bool = False) -> None:
        """Register a new policy."""
        self._policies[policy.name] = policy
        if set_as_default or not self._default_policy:
            self._default_policy = policy.name
    
    def get_policy(self, name: Optional[str] = None) -> Optional[MemoryPolicy]:
        """Get a policy by name, or the default policy."""
        if name is None:
            name = self._default_policy
        return self._policies.get(name) if name else None
    
    def list_policies(self) -> List[str]:
        """List all registered policy names."""
        return list(self._policies.keys())
    
    def get_stats_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance stats for all registered policies."""
        return {name: policy.get_stats() for name, policy in self._policies.items()}