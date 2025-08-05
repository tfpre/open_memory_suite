"""FrugalDispatcher - Intelligent cost-aware memory routing system."""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from ..adapters.base import MemoryAdapter, MemoryItem, RetrievalResult
from ..benchmark.cost_model import BudgetType, CostModel, CostTracker
from ..benchmark.trace import TraceEvent, TraceLogger
from .core import (
    ConversationContext,
    MemoryAction,
    MemoryPolicy,
    PolicyRegistry,
    Priority,
    RoutingDecision,
)


class FrugalDispatcher:
    """
    Intelligent cost-aware memory routing system.
    
    The FrugalDispatcher is the core intelligence that makes routing decisions
    about memory items based on:
    - Content analysis and importance
    - Cost constraints and budgets
    - Quality requirements and recall targets
    - Conversation context and user preferences
    
    Key Features:
    - Pluggable policy architecture (rules → ML → RL evolution)
    - Rich conversation context tracking
    - Comprehensive cost modeling and budget enforcement
    - Real-time decision explanation and logging
    - Thread-safe concurrent operation support
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        adapters: List[MemoryAdapter],
        cost_model: Optional[CostModel] = None,
        policy_registry: Optional[PolicyRegistry] = None,
        trace_logger: Optional[TraceLogger] = None,
        default_budget: BudgetType = BudgetType.STANDARD
    ):
        """
        Initialize the FrugalDispatcher.
        
        Args:
            adapters: List of available memory storage adapters
            cost_model: Cost modeling system for budget tracking
            policy_registry: Registry of available routing policies
            trace_logger: Logger for decision tracing and analysis
            default_budget: Default budget type for new sessions
        """
        self.adapters = {adapter.name: adapter for adapter in adapters}
        self.cost_model = cost_model or CostModel()
        self.policy_registry = policy_registry or PolicyRegistry()
        
        # Create default trace logger if none provided
        if trace_logger is None:
            from pathlib import Path
            trace_file = Path("./memory_traces.jsonl")
            self.trace_logger = TraceLogger(trace_file)
        else:
            self.trace_logger = trace_logger
            
        self.default_budget = default_budget
        
        # Session management
        self._contexts: Dict[str, ConversationContext] = {}
        self._context_lock = asyncio.Lock()
        
        # Performance monitoring
        self._stats = {
            "total_routing_decisions": 0,
            "items_stored": 0,
            "items_dropped": 0,
            "items_summarized": 0,
            "total_cost_saved": 0.0,
            "avg_decision_time_ms": 0.0,
            "active_sessions": 0
        }
        self._stats_lock = asyncio.Lock()
        
        # Adapter health monitoring
        self._adapter_health: Dict[str, bool] = {}
        self._last_health_check = 0.0
        self._health_check_interval = 300.0  # 5 minutes
    
    async def initialize(self) -> None:
        """Initialize the dispatcher and all adapters."""
        # Initialize all adapters
        for adapter in self.adapters.values():
            await adapter.initialize()
        
        # Initial health check
        await self._check_adapter_health()
        
        # Initialize cost model if needed
        if hasattr(self.cost_model, 'initialize'):
            await self.cost_model.initialize()
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        # Clean up all adapters
        for adapter in self.adapters.values():
            await adapter.cleanup()
        
        # Clear contexts
        async with self._context_lock:
            self._contexts.clear()
    
    async def get_or_create_context(
        self,
        session_id: str,
        budget_type: Optional[BudgetType] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> ConversationContext:
        """
        Get existing conversation context or create a new one.
        
        Args:
            session_id: Unique session identifier
            budget_type: Budget constraint level for this session
            user_preferences: User-specific settings and preferences
            
        Returns:
            ConversationContext for the session
        """
        async with self._context_lock:
            if session_id not in self._contexts:
                # Create new context
                cost_tracker = await self.cost_model.create_tracker(session_id)
                
                context = ConversationContext(
                    session_id=session_id,
                    cost_tracker=cost_tracker,
                    budget_type=budget_type or self.default_budget,
                    user_preferences=user_preferences or {},
                    total_stored_items=await self._get_total_stored_items()
                )
                
                self._contexts[session_id] = context
                
                async with self._stats_lock:
                    self._stats["active_sessions"] += 1
            
            return self._contexts[session_id]
    
    async def route_memory(
        self,
        item: MemoryItem,
        session_id: str,
        policy_name: Optional[str] = None,
        force_action: Optional[MemoryAction] = None
    ) -> RoutingDecision:
        """
        Route a memory item through the intelligent dispatch system.
        
        Args:
            item: The memory item to route
            session_id: Session identifier for context
            policy_name: Specific policy to use (defaults to registry default)
            force_action: Override policy decision with specific action
            
        Returns:
            RoutingDecision with full reasoning and cost analysis
        """
        start_time = time.time()
        
        # Get conversation context
        context = await self.get_or_create_context(session_id)
        context.update_activity()
        
        # Get routing policy
        policy = self.policy_registry.get_policy(policy_name)
        if not policy:
            raise ValueError(f"No policy available: {policy_name}")
        
        try:
            # Update context with current storage state
            await self._update_context_storage_info(context)
            
            # Make routing decision
            if force_action:
                action = force_action
                reasoning = f"Forced action: {force_action.value}"
            else:
                action = await policy.decide_action(item, context)
                reasoning = f"Policy {policy.name} decided: {action.value}"
            
            # Choose adapter if storing
            selected_adapter = None
            estimated_cost = None
            
            if action == MemoryAction.STORE:
                available_adapters = await self._get_healthy_adapters()
                selected_adapter = await policy.choose_adapter(item, available_adapters, context)
                
                if selected_adapter:
                    # Estimate cost for the operation
                    estimated_cost = self.cost_model.estimate_storage_cost(
                        adapter_name=selected_adapter.name,
                        content=item.content,
                        item_count=context.total_stored_items
                    )
                    
                    # Check budget constraints
                    if context.cost_tracker and not self.cost_model.check_budget_compliance(
                        estimated_cost, context.cost_tracker, context.budget_type
                    ):
                        # Budget exceeded - override to cheaper option or drop
                        action = MemoryAction.DROP
                        selected_adapter = None
                        reasoning += " | Budget exceeded, dropping item"
                else:
                    # No adapter available
                    action = MemoryAction.DROP
                    reasoning += " | No healthy adapter available"
            
            elif action == MemoryAction.SUMMARIZE:
                # Estimate summarization cost
                estimated_cost = self.cost_model.estimate_summarization_cost(item.content)
            
            # Analyze content for decision metadata
            content_analysis = await policy.analyze_content(item)
            
            # Create routing decision
            decision = RoutingDecision(
                action=action,
                selected_adapter=selected_adapter.name if selected_adapter else None,
                estimated_cost=estimated_cost,
                reasoning=reasoning,
                detected_priority=Priority(content_analysis.get("priority", Priority.MEDIUM)),
                context_snapshot={
                    "session_id": session_id,
                    "turn_count": context.turn_count,
                    "budget_type": context.budget_type.value,
                    "total_stored": context.total_stored_items,
                    "budget_critical": context.is_budget_critical()
                },
                policy_version=f"{policy.name}:{policy.version}",
                decision_latency_ms=(time.time() - start_time) * 1000
            )
            
            # Record decision for monitoring
            await policy.record_decision(decision)
            await self._update_stats(decision)
            
            # Update context
            context.turn_count += 1
            context.recent_turns.append(item)
            if len(context.recent_turns) > 10:  # Keep last 10 turns
                context.recent_turns = context.recent_turns[-10:]
            
            # Log decision for analysis
            await self._log_decision(decision.to_trace_dict())
            
            return decision
            
        except Exception as e:
            # Create error decision
            error_decision = RoutingDecision(
                action=MemoryAction.DROP,
                reasoning=f"Error in routing: {str(e)}",
                confidence=0.0,
                decision_latency_ms=(time.time() - start_time) * 1000
            )
            
            await self._log_error(f"Routing error for {session_id}: {e}")
            return error_decision
    
    async def execute_decision(
        self,
        decision: RoutingDecision,
        item: MemoryItem,
        session_id: str
    ) -> bool:
        """
        Execute a routing decision by performing the specified action.
        
        Args:
            decision: The routing decision to execute
            item: The memory item to act on
            session_id: Session identifier for cost tracking
            
        Returns:
            True if execution was successful
        """
        try:
            if decision.action == MemoryAction.STORE and decision.selected_adapter:
                # Store in selected adapter
                adapter = self.adapters.get(decision.selected_adapter)
                if adapter:
                    success = await adapter.store(item)
                    if success and decision.estimated_cost:
                        # Record actual cost
                        await self.cost_model.record_cost(session_id, decision.estimated_cost)
                    return success
                else:
                    await self._log_error(f"Adapter not found: {decision.selected_adapter}")
                    return False
            
            elif decision.action == MemoryAction.SUMMARIZE:
                # TODO: Implement summarization logic
                # For now, just log that we would summarize
                await self._log_info(f"Would summarize: {item.content[:100]}...")
                return True
            
            elif decision.action == MemoryAction.DROP:
                # Nothing to do - item is dropped
                await self._log_info(f"Dropped item: {item.content[:50]}...")
                return True
            
            elif decision.action == MemoryAction.DEFER:
                # Defer decision - could implement queuing logic here
                await self._log_info(f"Deferred decision for: {item.content[:50]}...")
                return True
            
            return False
            
        except Exception as e:
            await self._log_error(f"Error executing decision: {e}")
            return False
    
    async def route_and_execute(
        self,
        item: MemoryItem,
        session_id: str,
        policy_name: Optional[str] = None
    ) -> Tuple[RoutingDecision, bool]:
        """
        Route an item and immediately execute the decision.
        
        Args:
            item: Memory item to route and execute
            session_id: Session identifier
            policy_name: Specific policy to use
            
        Returns:
            Tuple of (RoutingDecision, execution_success)
        """
        decision = await self.route_memory(item, session_id, policy_name)
        success = await self.execute_decision(decision, item, session_id)
        return decision, success
    
    async def batch_route_and_execute(
        self,
        items: List[MemoryItem],
        session_id: str,
        policy_name: Optional[str] = None
    ) -> List[Tuple[RoutingDecision, bool]]:
        """
        Route and execute multiple items efficiently.
        
        Args:
            items: List of memory items to process
            session_id: Session identifier
            policy_name: Specific policy to use
            
        Returns:
            List of (RoutingDecision, execution_success) tuples
        """
        results = []
        
        # Process items concurrently for efficiency
        tasks = [
            self.route_and_execute(item, session_id, policy_name)
            for item in items
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error decision
                error_decision = RoutingDecision(
                    action=MemoryAction.DROP,
                    reasoning=f"Batch processing error: {str(result)}",
                    confidence=0.0
                )
                processed_results.append((error_decision, False))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def retrieve_memories(
        self,
        query: str,
        session_id: str,
        k: int = 5,
        adapter_preferences: Optional[List[str]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant memories across adapters with cost optimization.
        
        Args:
            query: Search query
            session_id: Session identifier for context
            k: Number of items to retrieve
            adapter_preferences: Preferred adapters (in order)
            
        Returns:
            Combined retrieval results with cost tracking
        """
        context = await self.get_or_create_context(session_id)
        
        # Determine which adapters to query based on preferences and costs
        adapters_to_query = adapter_preferences or list(self.adapters.keys())
        
        # Query adapters in parallel
        retrieval_tasks = []
        for adapter_name in adapters_to_query:
            if adapter_name in self.adapters:
                adapter = self.adapters[adapter_name]
                retrieval_tasks.append(adapter.retrieve(query, k))
        
        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        
        # Combine and rank results
        all_items = []
        all_scores = []
        
        for result in results:
            if isinstance(result, RetrievalResult):
                all_items.extend(result.items)
                if result.similarity_scores:
                    all_scores.extend(result.similarity_scores)
                else:
                    all_scores.extend([0.5] * len(result.items))  # Default score
        
        # Sort by score and take top k
        if all_scores:
            scored_items = list(zip(all_scores, all_items))
            scored_items.sort(key=lambda x: x[0], reverse=True)
            final_items = [item for _, item in scored_items[:k]]
            final_scores = [score for score, _ in scored_items[:k]]
        else:
            final_items = all_items[:k]
            final_scores = [0.5] * len(final_items)
        
        return RetrievalResult(
            query=query,
            items=final_items,
            similarity_scores=final_scores,
            retrieval_metadata={
                "adapters_queried": adapters_to_query,
                "session_id": session_id,
                "total_candidates": len(all_items)
            }
        )
    
    # Private helper methods
    
    async def _get_total_stored_items(self) -> int:
        """Get total number of items stored across all adapters."""
        total = 0
        for adapter in self.adapters.values():
            try:
                total += await adapter.count()
            except Exception:
                pass  # Skip failing adapters
        return total
    
    async def _update_context_storage_info(self, context: ConversationContext) -> None:
        """Update storage information in conversation context."""
        context.total_stored_items = await self._get_total_stored_items()
        
        # Update per-adapter counts
        for adapter_name, adapter in self.adapters.items():
            try:
                count = await adapter.count()
                context.storage_by_adapter[adapter_name] = count
            except Exception:
                context.storage_by_adapter[adapter_name] = 0
    
    async def _check_adapter_health(self) -> None:
        """Check health of all adapters."""
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return
        
        for adapter_name, adapter in self.adapters.items():
            try:
                self._adapter_health[adapter_name] = await adapter.health_check()
            except Exception:
                self._adapter_health[adapter_name] = False
        
        self._last_health_check = current_time
    
    async def _get_healthy_adapters(self) -> List[MemoryAdapter]:
        """Get list of healthy adapters."""
        await self._check_adapter_health()
        
        healthy = []
        for adapter_name, adapter in self.adapters.items():
            if self._adapter_health.get(adapter_name, False):
                healthy.append(adapter)
        
        return healthy
    
    async def _update_stats(self, decision: RoutingDecision) -> None:
        """Update dispatcher performance statistics."""
        async with self._stats_lock:
            self._stats["total_routing_decisions"] += 1
            
            if decision.action == MemoryAction.STORE:
                self._stats["items_stored"] += 1
            elif decision.action == MemoryAction.DROP:
                self._stats["items_dropped"] += 1
            elif decision.action == MemoryAction.SUMMARIZE:
                self._stats["items_summarized"] += 1
            
            # Update average decision time
            current_avg = self._stats["avg_decision_time_ms"]
            new_latency = decision.decision_latency_ms
            total_decisions = self._stats["total_routing_decisions"]
            
            self._stats["avg_decision_time_ms"] = (
                (current_avg * (total_decisions - 1) + new_latency) / total_decisions
            )
    
    # Helper methods for logging
    
    async def _log_decision(self, decision_data: Dict[str, Any]) -> None:
        """Log a routing decision."""
        try:
            event = TraceEvent(
                event_type="routing_decision",
                adapter_name=decision_data.get("adapter", "unknown"),
                operation_id=decision_data.get("decision_id", "unknown"),
                success=True,
                latency_ms=decision_data.get("latency_ms", 0.0),
                operation_details=decision_data
            )
            async with self.trace_logger:
                await self.trace_logger.log_event(event)
        except Exception:
            pass  # Don't let logging errors break the system
    
    async def _log_error(self, message: str, adapter_name: str = "dispatcher") -> None:
        """Log an error message."""
        try:
            event = TraceEvent(
                event_type="error",
                adapter_name=adapter_name,
                operation_id=f"error_{time.time()}",
                success=False,
                error_message=message,
                latency_ms=0.0
            )
            async with self.trace_logger:
                await self.trace_logger.log_event(event)
        except Exception:
            pass
    
    async def _log_info(self, message: str, adapter_name: str = "dispatcher") -> None:
        """Log an info message."""
        try:
            event = TraceEvent(
                event_type="info",
                adapter_name=adapter_name,
                operation_id=f"info_{time.time()}",
                success=True,
                latency_ms=0.0,
                operation_details={"message": message}
            )
            async with self.trace_logger:
                await self.trace_logger.log_event(event)
        except Exception:
            pass
    
    # Public API methods
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive dispatcher statistics."""
        return {
            "dispatcher_stats": self._stats.copy(),
            "adapter_health": self._adapter_health.copy(),
            "active_sessions": len(self._contexts),
            "policy_stats": self.policy_registry.get_stats_summary()
        }
    
    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific session."""
        context = self._contexts.get(session_id)
        if not context:
            return None
        
        return {
            "session_id": session_id,
            "turn_count": context.turn_count,
            "session_duration_minutes": context.get_session_duration_minutes(),
            "budget_type": context.budget_type.value,
            "total_stored_items": context.total_stored_items,
            "recent_content": context.get_recent_content_summary(),
            "cost_summary": (
                self.cost_model.get_cost_summary(context.cost_tracker)
                if context.cost_tracker else None
            )
        }