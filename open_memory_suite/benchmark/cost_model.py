"""Cost modeling system for memory operations and dispatcher decisions."""

import asyncio
import math
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator

from ..core.tokens import TokenCounter
from ..core.pricebook import Pricebook, AdapterCoeffs, fit_coeffs


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
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, pricebook_path: Optional[Union[str, Path]] = None):
        """
        Initialize cost model.
        
        Args:
            config_path: Path to cost_model.yaml, defaults to package location
            pricebook_path: Path to pricebook.json for learned coefficients
        """
        if config_path is None:
            config_path = Path(__file__).parent / "cost_model.yaml"
        if pricebook_path is None:
            pricebook_path = Path("./pricebook.json")
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._trackers: Dict[str, CostTracker] = {}
        self._lock = asyncio.Lock()
        
        # New telemetry-based cost prediction
        self._tokens = TokenCounter()
        self._pb_path = Path(pricebook_path)
        self._pb = Pricebook.load(self._pb_path)
        # CRITICAL FIX: Per-key sample buffers with memory guards
        self._samples_per_key: Dict[str, List[dict]] = {}   # key -> samples
        self._max_samples_per_key = 10000  # Prevent memory leaks
        self._refit_threshold_per_key = 200  # Refit when key has enough samples
        
        self.reload_config()
    
    def reload_config(self) -> None:
        """Reload cost configuration from YAML file."""
        p = self.config_path
        if not p.exists():
            self._config = {}   # sensible empty defaults
            return
        with p.open("r") as f:
            self._config = yaml.safe_load(f) or {}
    
    def _get_content_size(self, content: str) -> ContentSize:
        """Classify content size based on accurate token count."""
        token_count = self._tokens.count(content)
        
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
    
    def _coeffs(self, adapter: str, op: OperationType, model: str | None = None) -> AdapterCoeffs:
        """Get learned coefficients for adapter/operation, with sane defaults."""
        return self._coeffs_from_pb(self._pb, adapter, op, model)
    
    def _coeffs_from_pb(self, pb: Pricebook, adapter: str, op: OperationType, model: str | None = None) -> AdapterCoeffs:
        """Get learned coefficients from a specific pricebook snapshot."""
        key = f"{adapter}|{op.value}" + (f"|model={model}" if model else "")
        if key not in pb.entries:
            # Sane defaults: 0 cost for local, conservative latency
            default_coeffs = AdapterCoeffs(
                base_cents=0, 
                per_token_micros=0,
                per_k_cents=0.0, 
                per_logN_cents=0.0,
                p50_ms=50.0, 
                p95_ms=120.0
            )
            # Add to current pricebook for future use (not the snapshot)
            if key not in self._pb.entries:
                self._pb.entries[key] = default_coeffs
            return default_coeffs
        return pb.entries[key]

    def predict(
        self, 
        *, 
        op: OperationType, 
        adapter: str, 
        tokens: int = 0, 
        k: int = 0, 
        item_count: int = 0, 
        concurrency: ConcurrencyLevel = ConcurrencyLevel.SINGLE,
        mb_stored: float = 1.0
    ) -> tuple[int, float]:
        """
        Enhanced prediction using 5-component cost model from friend's recommendations.
        
        Args:
            op: Operation type (STORE, RETRIEVE, SUMMARIZE)
            adapter: Adapter name (e.g., "faiss_store")
            tokens: Token count for content
            k: Number of items to retrieve (for RETRIEVE ops)
            item_count: Current number of items in adapter
            concurrency: Concurrency level
            mb_stored: Storage size in MB (for storage cost calculation)
            
        Returns:
            (predicted_cost_cents, predicted_latency_ms)
        """
        # Take snapshot to avoid torn reads during concurrent refit
        pb_snapshot = self._pb
        c = self._coeffs_from_pb(pb_snapshot, adapter, op)
        
        # Enhanced cost prediction using friend's 5-component model
        if op == OperationType.STORE:
            # Write operation: base write cost + token scaling + amortized maintenance
            cents_float = c.get_write_cost_cents(tokens)
            
        elif op == OperationType.RETRIEVE:
            # Read operation: base read cost + retrieval complexity
            cents_float = c.get_read_cost_cents(k, item_count)
            
        elif op == OperationType.MAINTAIN:
            # Maintenance operation: index + GC costs
            cents_float = c.get_maintenance_cost_cents(item_count)
            
        else:
            # Fallback to legacy model for other operations (SUMMARIZE, ANALYZE)
            logN = math.log(max(1, item_count), 10)
            cents_float = (
                c.base_cents + 
                (c.per_token_micros * tokens) / 1_000_000 + 
                c.per_k_cents * max(1, k) + 
                c.per_logN_cents * logN
            )
        
        # Convert to integer cents
        cents = max(0, int(round(cents_float)))
        
        # Latency prediction: blend p50→p95 with concurrency scaling
        concurrency_factor = 0.25 if concurrency in (ConcurrencyLevel.SINGLE, ConcurrencyLevel.LIGHT) else 0.8
        ms = c.p50_ms + (c.p95_ms - c.p50_ms) * concurrency_factor
        ms = max(0.0, ms)
        
        return cents, ms

    async def reconcile(
        self, 
        *, 
        op: OperationType, 
        adapter: str, 
        predicted_cents: int, 
        predicted_ms: float, 
        observed_cents: Optional[int], 
        observed_ms: float,
        tokens: int = 0,
        k: int = 0, 
        item_count: int = 0
    ) -> None:
        """
        Record observed vs predicted performance for model improvement.
        
        Args:
            op: Operation type
            adapter: Adapter name
            predicted_cents: What the model predicted for cost
            predicted_ms: What the model predicted for latency
            observed_cents: Actual cost (None for local ops)
            observed_ms: Actual latency measured
            tokens: Token count (for refit analysis)
            k: Items retrieved (for refit analysis) 
            item_count: Index size (for refit analysis)
        """
        async with self._lock:
            # CRITICAL FIX: Per-key sample collection with memory guards
            key = f"{adapter}|{op.value}"
            
            # Initialize key if needed
            if key not in self._samples_per_key:
                self._samples_per_key[key] = []
            
            # Add sample to key-specific buffer
            sample = {
                "op": op.value, 
                "adapter": adapter,
                "predicted_cents": predicted_cents, 
                "predicted_ms": predicted_ms,
                "observed_cents": observed_cents, 
                "observed_ms": observed_ms,
                "tokens": tokens, 
                "k": k, 
                "item_count": item_count
            }
            
            samples_for_key = self._samples_per_key[key]
            samples_for_key.append(sample)
            
            # Memory guard: keep only last N samples (reservoir sampling)
            if len(samples_for_key) > self._max_samples_per_key:
                # Keep recent samples (simple truncation)
                self._samples_per_key[key] = samples_for_key[-self._max_samples_per_key:]
            
            # Per-key refit threshold: refit when this key has enough samples
            if len(samples_for_key) >= self._refit_threshold_per_key:
                await self._refit_key_locked(key)

    async def _refit_key_locked(self, key: str) -> None:
        """Refit coefficients for a specific key using copy-on-write."""
        if key not in self._samples_per_key:
            return
        
        samples = self._samples_per_key[key]
        if len(samples) < 50:  # Need minimum signal for stable fit
            return
        
        # Create new pricebook with updated coefficients (copy-on-write)
        from ..core.pricebook import Pricebook
        new_pb = Pricebook(
            entries=self._pb.entries.copy(), 
            version=self._pb.version, 
            updated_at=time.time()
        )
        
        # Refit coefficients for this key
        new_pb.entries[key] = fit_coeffs(samples)
        
        # Atomically save and swap pricebook
        new_pb.save(self._pb_path)
        self._pb = new_pb  # Atomic reference swap
        
        # Clear samples for this key after successful refit
        self._samples_per_key[key] = []

    async def _refit_locked(self):
        """Refit pricebook coefficients from all collected samples using copy-on-write."""
        if not self._samples_per_key:
            return
        
        # Create new pricebook with updated coefficients (copy-on-write)
        from ..core.pricebook import Pricebook
        new_pb = Pricebook(
            entries=self._pb.entries.copy(), 
            version=self._pb.version, 
            updated_at=time.time()
        )
        
        # Refit coefficients for all keys with sufficient data
        keys_to_clear = []
        for key, samples in self._samples_per_key.items():
            if len(samples) >= 50:  # Need minimum signal
                new_pb.entries[key] = fit_coeffs(samples)
                keys_to_clear.append(key)
        
        if keys_to_clear:  # Only save if we actually updated something
            # Atomically save and swap pricebook
            new_pb.save(self._pb_path)
            self._pb = new_pb  # Atomic reference swap
            
            # Clear samples for refitted keys
            for key in keys_to_clear:
                self._samples_per_key[key] = []
    
    def estimate_storage_cost(
        self,
        adapter_name: str,
        content: str,
        item_count: int = 0,
        concurrency: ConcurrencyLevel = ConcurrencyLevel.SINGLE
    ) -> CostEstimate:
        """
        Estimate cost to store content in an adapter using learned coefficients.
        
        Args:
            adapter_name: Target memory adapter (e.g., "faiss_store")
            content: Content to store
            item_count: Current number of stored items
            concurrency: Concurrent operation level
            
        Returns:
            Detailed cost estimate
        """
        # CRITICAL FIX: Route through predict() for single source of truth
        tokens = self._tokens.count(content)
        pred_cents, pred_ms = self.predict(
            op=OperationType.STORE, 
            adapter=adapter_name,
            tokens=tokens, 
            item_count=item_count, 
            concurrency=concurrency
        )
        
        # Convert to legacy CostEstimate format for compatibility
        base_cost = float(pred_cents)   # keep cents
        estimated_latency = pred_ms
        
        # For backwards compatibility, analyze content/pressure for metadata
        content_size = self._get_content_size(content)
        memory_pressure = self._get_memory_pressure(item_count)
        
        reasoning = (
            f"Storage in {adapter_name} via pricebook: {pred_cents} cents, "
            f"tokens={tokens}, items={item_count}, "
            f"concurrency={concurrency.value}"
        )
        
        return CostEstimate(
            base_cost=base_cost,
            content_multiplier=1.0,  # Multipliers now embedded in predict()
            pressure_multiplier=1.0,
            concurrency_multiplier=1.0,
            total_cost=base_cost,
            adapter_name=adapter_name,
            operation_type=OperationType.STORE,
            estimated_latency_ms=estimated_latency,
            reasoning=reasoning,
            metadata={
                "content_length": len(content),
                "tokens": tokens,
                "content_size": content_size.value,
                "memory_pressure": memory_pressure.value,
                "item_count": item_count,
                "predicted_cents": pred_cents  # Keep raw prediction
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
        Estimate cost to retrieve from an adapter using learned coefficients.
        
        Args:
            adapter_name: Target memory adapter
            query: Search query
            k: Number of items to retrieve
            item_count: Current number of stored items
            concurrency: Concurrent operation level
            
        Returns:
            Detailed cost estimate
        """
        # CRITICAL FIX: Route through predict() for single source of truth
        tokens = self._tokens.count(query)
        pred_cents, pred_ms = self.predict(
            op=OperationType.RETRIEVE,
            adapter=adapter_name,
            tokens=tokens,
            k=k,
            item_count=item_count,
            concurrency=concurrency
        )
        
        # Convert to legacy CostEstimate format for compatibility  
        base_cost = float(pred_cents)   # cents
        estimated_latency = pred_ms
        
        # For backwards compatibility, analyze content/pressure for metadata
        content_size = self._get_content_size(query)
        memory_pressure = self._get_memory_pressure(item_count)
        
        reasoning = (
            f"Retrieval from {adapter_name} via pricebook: {pred_cents} cents, "
            f"tokens={tokens}, k={k}, items={item_count}, "
            f"concurrency={concurrency.value}"
        )
        
        return CostEstimate(
            base_cost=base_cost,
            content_multiplier=1.0,  # Multipliers now embedded in predict()
            pressure_multiplier=1.0,
            concurrency_multiplier=1.0,
            total_cost=base_cost,
            adapter_name=adapter_name,
            operation_type=OperationType.RETRIEVE,
            estimated_latency_ms=estimated_latency,
            reasoning=reasoning,
            metadata={
                "query_length": len(query),
                "tokens": tokens,
                "k": k,
                "item_count": item_count,
                "predicted_cents": pred_cents  # Keep raw prediction
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
        
        base_cost = float(summarization_config[model])  # cents
        
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
        
        reasoning = f"Summarization with {model}: base={base_cost:.4f}¢, size={content_size.value}({content_mult}x)"
        
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