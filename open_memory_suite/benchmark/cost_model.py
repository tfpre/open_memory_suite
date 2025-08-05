"""Cost modeling system for memory operations and dispatcher decisions."""

import asyncio
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class ContentSize(str, Enum):
    """Content size categories for cost scaling."""
    SMALL = "small"      # <100 tokens
    MEDIUM = "medium"    # 100-500 tokens
    LARGE = "large"      # 500-1000 tokens
    XLARGE = "xlarge"    # >1000 tokens


class MemoryPressure(str, Enum):
    """Memory pressure levels for cost scaling."""
    LOW = "low"          # <1000 items
    MEDIUM = "medium"    # 1000-10000 items  
    HIGH = "high"        # 10000-100000 items
    CRITICAL = "critical" # >100000 items


class ConcurrencyLevel(str, Enum):
    """Concurrency levels for cost scaling."""
    SINGLE = "single"    # Single user
    LIGHT = "light"      # <10 concurrent users
    MODERATE = "moderate" # 10-100 concurrent users
    HEAVY = "heavy"      # >100 concurrent users


class BudgetType(str, Enum):
    """Budget constraint types."""
    MINIMAL = "minimal"      # Ultra-frugal mode
    STANDARD = "standard"    # Balanced mode
    PREMIUM = "premium"      # High-quality mode
    UNLIMITED = "unlimited"  # Research/development mode


class OperationType(str, Enum):
    """Types of memory operations."""
    STORE = "store"
    RETRIEVE = "retrieve"
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    MAINTAIN = "maintain"


class CostEstimate(BaseModel):
    """Detailed cost breakdown for an operation."""
    
    base_cost: float = Field(..., description="Base operation cost in cents")
    content_multiplier: float = Field(default=1.0, description="Content size scaling")
    pressure_multiplier: float = Field(default=1.0, description="Memory pressure scaling")
    concurrency_multiplier: float = Field(default=1.0, description="Concurrency scaling")
    total_cost: float = Field(..., description="Final computed cost")
    
    adapter_name: str = Field(..., description="Target adapter")
    operation_type: OperationType = Field(..., description="Operation category")
    estimated_latency_ms: float = Field(..., description="Expected latency")
    
    reasoning: str = Field(default="", description="Human-readable explanation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    @validator('total_cost')
    def validate_total_cost(cls, v, values):
        """Ensure total cost matches computed value."""
        base = values.get('base_cost', 0)
        content_mult = values.get('content_multiplier', 1.0)
        pressure_mult = values.get('pressure_multiplier', 1.0)
        concurrency_mult = values.get('concurrency_multiplier', 1.0)
        
        expected = base * content_mult * pressure_mult * concurrency_mult
        if abs(v - expected) > 0.0001:  # Allow small floating point errors
            raise ValueError(f"Total cost {v} doesn't match computed {expected}")
        return v


class BudgetConstraints(BaseModel):
    """Budget limits for cost control."""
    
    conversation_limit: float = Field(..., description="Per-conversation limit (cents)")
    operation_limit: float = Field(..., description="Per-operation limit (cents)")
    session_limit: float = Field(..., description="Per-session limit (cents)")
    daily_limit: float = Field(..., description="Daily limit (cents)")
    burst_limit: float = Field(..., description="Burst limit (cents)")


class CostTracker(BaseModel):
    """Tracks actual costs during operation."""
    
    session_id: str = Field(..., description="Session identifier")
    total_spent: float = Field(default=0.0, description="Total cost so far (cents)")
    operation_count: int = Field(default=0, description="Number of operations")
    start_time: float = Field(default_factory=time.time, description="Session start timestamp")
    
    # Per-adapter breakdown
    adapter_costs: Dict[str, float] = Field(default_factory=dict, description="Cost by adapter")
    operation_costs: Dict[str, float] = Field(default_factory=dict, description="Cost by operation type")
    
    def add_cost(self, cost: CostEstimate) -> None:
        """Record a cost expenditure."""
        self.total_spent += cost.total_cost
        self.operation_count += 1
        
        # Track by adapter
        if cost.adapter_name not in self.adapter_costs:
            self.adapter_costs[cost.adapter_name] = 0.0
        self.adapter_costs[cost.adapter_name] += cost.total_cost
        
        # Track by operation type
        op_type = cost.operation_type.value
        if op_type not in self.operation_costs:
            self.operation_costs[op_type] = 0.0
        self.operation_costs[op_type] += cost.total_cost
    
    def get_session_duration_hours(self) -> float:
        """Get session duration in hours."""
        return (time.time() - self.start_time) / 3600.0
    
    def get_cost_per_operation(self) -> float:
        """Get average cost per operation."""
        return self.total_spent / max(1, self.operation_count)


class CostModel:
    """
    Production-grade cost modeling system for memory operations.
    
    Features:
    - YAML-based configuration for easy updates
    - Dynamic cost scaling based on context
    - Budget enforcement and tracking
    - Latency estimation for SLA compliance
    - Thread-safe operation tracking
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize cost model.
        
        Args:
            config_path: Path to cost_model.yaml, defaults to package location
        """
        if config_path is None:
            config_path = Path(__file__).parent / "cost_model.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._trackers: Dict[str, CostTracker] = {}
        self._lock = asyncio.Lock()
        
        self.reload_config()
    
    def reload_config(self) -> None:
        """Reload cost configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def _get_content_size(self, content: str) -> ContentSize:
        """Classify content size based on token count (rough estimate)."""
        # Rough token estimation: ~4 chars per token
        token_count = len(content) // 4
        
        if token_count < 100:
            return ContentSize.SMALL
        elif token_count < 500:
            return ContentSize.MEDIUM
        elif token_count < 1000:
            return ContentSize.LARGE
        else:
            return ContentSize.XLARGE
    
    def _get_memory_pressure(self, item_count: int) -> MemoryPressure:
        """Determine memory pressure based on stored item count."""
        if item_count < 1000:
            return MemoryPressure.LOW
        elif item_count < 10000:
            return MemoryPressure.MEDIUM
        elif item_count < 100000:
            return MemoryPressure.HIGH
        else:
            return MemoryPressure.CRITICAL
    
    def estimate_storage_cost(
        self,
        adapter_name: str,
        content: str,
        item_count: int = 0,
        concurrency: ConcurrencyLevel = ConcurrencyLevel.SINGLE
    ) -> CostEstimate:
        """
        Estimate cost to store content in an adapter.
        
        Args:
            adapter_name: Target memory adapter (e.g., "faiss_store")
            content: Content to store
            item_count: Current number of stored items
            concurrency: Concurrent operation level
            
        Returns:
            Detailed cost estimate
        """
        # Get base costs from config
        storage_config = self._config.get("storage", {})
        adapter_config = storage_config.get(adapter_name, {})
        
        if not adapter_config:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        
        # Primary storage operation cost (choose most relevant)
        if "store_item" in adapter_config:
            base_cost = adapter_config["store_item"]
        elif "file_write" in adapter_config:
            base_cost = adapter_config["file_write"]
        elif "graph_write" in adapter_config:
            base_cost = adapter_config["graph_write"]
        else:
            # Use first available cost
            base_cost = next(iter(adapter_config.values()))
        
        # Apply scaling multipliers
        content_size = self._get_content_size(content)
        memory_pressure = self._get_memory_pressure(item_count)
        
        multipliers = self._config.get("multipliers", {})
        content_mult = multipliers.get("content_size", {}).get(content_size.value, 1.0)
        pressure_mult = multipliers.get("memory_pressure", {}).get(memory_pressure.value, 1.0)
        concurrency_mult = multipliers.get("concurrency", {}).get(concurrency.value, 1.0)
        
        total_cost = base_cost * content_mult * pressure_mult * concurrency_mult
        
        # Get latency estimate
        latency_config = self._config.get("latency_targets", {}).get("storage", {})
        estimated_latency = latency_config.get(adapter_name, 100.0)
        
        # Build reasoning
        reasoning = (
            f"Storage in {adapter_name}: base=${base_cost:.4f}, "
            f"content_size={content_size.value}({content_mult}x), "
            f"pressure={memory_pressure.value}({pressure_mult}x), "
            f"concurrency={concurrency.value}({concurrency_mult}x)"
        )
        
        return CostEstimate(
            base_cost=base_cost,
            content_multiplier=content_mult,
            pressure_multiplier=pressure_mult,
            concurrency_multiplier=concurrency_mult,
            total_cost=total_cost,
            adapter_name=adapter_name,
            operation_type=OperationType.STORE,
            estimated_latency_ms=estimated_latency,
            reasoning=reasoning,
            metadata={
                "content_length": len(content),
                "content_size": content_size.value,
                "memory_pressure": memory_pressure.value,
                "item_count": item_count
            }
        )
    
    def estimate_retrieval_cost(
        self,
        adapter_name: str,
        query: str,
        k: int = 5,
        item_count: int = 0,
        concurrency: ConcurrencyLevel = ConcurrencyLevel.SINGLE
    ) -> CostEstimate:
        """
        Estimate cost to retrieve from an adapter.
        
        Args:
            adapter_name: Target memory adapter
            query: Search query
            k: Number of items to retrieve
            item_count: Current number of stored items
            concurrency: Concurrent operation level
            
        Returns:
            Detailed cost estimate
        """
        # Get base costs from config
        retrieval_config = self._config.get("retrieval", {})
        adapter_config = retrieval_config.get(adapter_name, {})
        
        if not adapter_config:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        
        # Primary retrieval operation cost (choose most relevant)
        if "tfidf_search" in adapter_config:
            base_cost = adapter_config["tfidf_search"]
        elif "vector_search" in adapter_config:
            base_cost = adapter_config["vector_search"]
        elif "linear_scan" in adapter_config:
            base_cost = adapter_config["linear_scan"]
        elif "semantic_query" in adapter_config:
            base_cost = adapter_config["semantic_query"]
        else:
            base_cost = next(iter(adapter_config.values()))
        
        # Apply scaling multipliers
        content_size = self._get_content_size(query)
        memory_pressure = self._get_memory_pressure(item_count)
        
        multipliers = self._config.get("multipliers", {})
        content_mult = multipliers.get("content_size", {}).get(content_size.value, 1.0)
        pressure_mult = multipliers.get("memory_pressure", {}).get(memory_pressure.value, 1.0)
        concurrency_mult = multipliers.get("concurrency", {}).get(concurrency.value, 1.0)
        
        # Scale by number of items requested
        k_multiplier = min(k / 5.0, 2.0)  # Cap at 2x for very large k
        
        total_cost = base_cost * content_mult * pressure_mult * concurrency_mult * k_multiplier
        
        # Get latency estimate
        latency_config = self._config.get("latency_targets", {}).get("retrieval", {})
        estimated_latency = latency_config.get(adapter_name, 50.0)
        
        reasoning = (
            f"Retrieval from {adapter_name}: base=${base_cost:.4f}, "
            f"k={k}({k_multiplier:.1f}x), "
            f"pressure={memory_pressure.value}({pressure_mult}x), "
            f"concurrency={concurrency.value}({concurrency_mult}x)"
        )
        
        return CostEstimate(
            base_cost=base_cost,
            content_multiplier=content_mult,
            pressure_multiplier=pressure_mult,
            concurrency_multiplier=concurrency_mult,
            total_cost=total_cost,
            adapter_name=adapter_name,
            operation_type=OperationType.RETRIEVE,
            estimated_latency_ms=estimated_latency,
            reasoning=reasoning,
            metadata={
                "query_length": len(query),
                "k": k,
                "item_count": item_count
            }
        )
    
    def estimate_summarization_cost(
        self,
        content: str,
        model: str = "local_llama_7b"
    ) -> CostEstimate:
        """
        Estimate cost to summarize content.
        
        Args:
            content: Content to summarize
            model: Summarization model to use
            
        Returns:
            Detailed cost estimate
        """
        processing_config = self._config.get("processing", {})
        summarization_config = processing_config.get("summarization", {})
        
        if model not in summarization_config:
            raise ValueError(f"Unknown summarization model: {model}")
        
        base_cost = summarization_config[model]
        
        # Apply content size scaling
        content_size = self._get_content_size(content)
        multipliers = self._config.get("multipliers", {})
        content_mult = multipliers.get("content_size", {}).get(content_size.value, 1.0)
        
        total_cost = base_cost * content_mult
        
        # Estimate latency based on model
        if "gpt" in model:
            estimated_latency = 2000.0  # API call latency
        else:
            estimated_latency = 500.0   # Local inference
        
        reasoning = f"Summarization with {model}: base=${base_cost:.4f}, size={content_size.value}({content_mult}x)"
        
        return CostEstimate(
            base_cost=base_cost,
            content_multiplier=content_mult,
            pressure_multiplier=1.0,
            concurrency_multiplier=1.0,
            total_cost=total_cost,
            adapter_name=f"summarizer_{model}",
            operation_type=OperationType.SUMMARIZE,
            estimated_latency_ms=estimated_latency,
            reasoning=reasoning,
            metadata={
                "model": model,
                "content_length": len(content)
            }
        )
    
    def get_budget_constraints(self, budget_type: BudgetType) -> BudgetConstraints:
        """Get budget constraints for a given budget type."""
        budget_config = self._config.get("budgets", {})
        conversation_limits = budget_config.get("conversation", {})
        operation_limits = budget_config.get("operation", {})
        session_limits = budget_config.get("session", {})
        
        return BudgetConstraints(
            conversation_limit=conversation_limits.get(budget_type.value, 5.0),
            operation_limit=operation_limits.get("storage_max", 0.1),
            session_limit=session_limits.get("session_limit", 10.0),
            daily_limit=session_limits.get("daily_limit", 50.0),
            burst_limit=session_limits.get("burst_limit", 2.0)
        )
    
    async def create_tracker(self, session_id: str) -> CostTracker:
        """Create a new cost tracker for a session."""
        async with self._lock:
            tracker = CostTracker(session_id=session_id)
            self._trackers[session_id] = tracker
            return tracker
    
    async def get_tracker(self, session_id: str) -> Optional[CostTracker]:
        """Get existing cost tracker for a session."""
        async with self._lock:
            return self._trackers.get(session_id)
    
    async def record_cost(self, session_id: str, cost_estimate: CostEstimate) -> None:
        """Record actual cost expenditure."""
        async with self._lock:
            if session_id in self._trackers:
                self._trackers[session_id].add_cost(cost_estimate)
    
    def check_budget_compliance(
        self,
        cost_estimate: CostEstimate,
        tracker: CostTracker,
        budget_type: BudgetType = BudgetType.STANDARD
    ) -> bool:
        """
        Check if a proposed operation fits within budget constraints.
        
        Args:
            cost_estimate: Proposed operation cost
            tracker: Current cost tracker
            budget_type: Budget constraint level
            
        Returns:
            True if operation is within budget
        """
        constraints = self.get_budget_constraints(budget_type)
        
        # Check per-operation limit
        if cost_estimate.total_cost > constraints.operation_limit:
            return False
        
        # Check session limit
        projected_session_cost = tracker.total_spent + cost_estimate.total_cost
        if projected_session_cost > constraints.session_limit:
            return False
        
        # Check burst limit (cost per hour)
        session_hours = max(tracker.get_session_duration_hours(), 0.1)  # Minimum 0.1h
        hourly_rate = projected_session_cost / session_hours
        if hourly_rate > constraints.burst_limit:
            return False
        
        return True
    
    def get_cost_summary(self, tracker: CostTracker) -> Dict[str, Any]:
        """Generate a human-readable cost summary."""
        return {
            "session_id": tracker.session_id,
            "total_cost_cents": round(tracker.total_spent, 4),
            "total_cost_usd": round(tracker.total_spent / 100, 6),
            "operation_count": tracker.operation_count,
            "avg_cost_per_operation": round(tracker.get_cost_per_operation(), 4),
            "session_duration_hours": round(tracker.get_session_duration_hours(), 2),
            "cost_by_adapter": {k: round(v, 4) for k, v in tracker.adapter_costs.items()},
            "cost_by_operation": {k: round(v, 4) for k, v in tracker.operation_costs.items()}
        }