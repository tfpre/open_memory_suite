#!/usr/bin/env python3
"""
Benchmark Harness: Unified evaluation engine for memory systems.

This harness integrates the research integration layer, cost model, and 3-class router
to provide comprehensive cost-aware evaluation of memory adapters.

Key Components:
- BenchmarkHarness: Main evaluation engine
- ConversationSession: Structured conversation data
- EvaluationQuery: Memory recall evaluation queries
- Integration with ResearchIntegrationRunner for existing frameworks
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
import logging

from open_memory_suite.core.telemetry import probe
from open_memory_suite.core.tokens import count_tokens
from open_memory_suite.benchmark.cost_model import CostModel, OperationType
from .research_integration import ResearchIntegrationRunner, BenchmarkResult
from open_memory_suite.adapters.registry import MemoryAdapter

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Individual turn in a conversation."""
    
    turn_id: str
    speaker: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass  
class ConversationSession:
    """Complete conversation session for memory evaluation."""
    
    session_id: str
    turns: List[ConversationTurn]
    session_metadata: Dict[str, Any]
    start_time: float
    end_time: float
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def turn_count(self) -> int:
        return len(self.turns)
    
    @property
    def total_content_length(self) -> int:
        return sum(len(turn.content) for turn in self.turns)


@dataclass
class EvaluationQuery:
    """Memory recall evaluation query."""
    
    query_id: str
    query_text: str
    query_type: str  # 'factual', 'semantic', 'temporal', 'relational'
    expected_content: List[str]  # Expected memory content that should be recalled
    context: Dict[str, Any]
    target_session_ids: List[str]  # Sessions this query should find memories from
    
    
@dataclass
class RecallEvaluationResult:
    """Result of memory recall evaluation."""
    
    query_id: str
    memories_retrieved: List[Dict[str, Any]]
    recall_score: float  # 0.0 - 1.0, based on expected content retrieval
    precision_score: float
    f1_score: float
    retrieval_latency_ms: float
    cost_cents: int


@dataclass
class SessionEvaluationResult:
    """Result of evaluating a single conversation session."""
    
    session_id: str
    storage_operations: int
    total_storage_cost_cents: int
    routing_decisions: List[str]  # Track 3-class router decisions
    processing_latency_ms: float
    memory_efficiency_score: float  # Cost per stored memory item


@dataclass
class BenchmarkEvaluationResult:
    """Complete benchmark evaluation result."""
    
    # Core metrics
    sessions_processed: int
    total_turns: int
    recall_evaluation: Dict[str, Any]
    
    # Session-level results
    session_results: List[SessionEvaluationResult]
    
    # Cost analysis
    total_cost_cents: int
    cost_per_turn: float
    cost_per_session: float
    
    # Performance metrics
    avg_recall_score: float
    avg_precision_score: float  
    avg_f1_score: float
    avg_latency_ms: float
    
    # Metadata
    adapter_name: str
    evaluation_timestamp: float
    benchmark_metadata: Dict[str, Any]


class BenchmarkHarness:
    """
    Main benchmark evaluation engine.
    
    Integrates cost model, research frameworks, and 3-class router for comprehensive
    memory system evaluation with cost awareness.
    """
    
    def __init__(self, trace_file: Optional[Path] = None):
        """
        Initialize benchmark harness.
        
        Args:
            trace_file: Optional path for detailed execution tracing
        """
        self.trace_file = trace_file
        self.cost_model = CostModel()
        self._trace_entries: List[Dict[str, Any]] = []
        
        # Initialize research integration if available
        try:
            from open_memory_suite.dispatcher.frugal_dispatcher import FrugalDispatcher
            self.dispatcher = FrugalDispatcher()
            self.research_runner = ResearchIntegrationRunner(self.cost_model, self.dispatcher)
            self.research_integration_available = True
        except ImportError:
            logger.warning("Research integration not available - some features will be limited")
            self.research_integration_available = False
    
    async def run_full_evaluation(
        self,
        adapter: MemoryAdapter,
        sessions: List[ConversationSession], 
        queries: List[EvaluationQuery],
        enable_tracing: bool = True
    ) -> BenchmarkEvaluationResult:
        """
        Run complete benchmark evaluation with cost tracking.
        
        Args:
            adapter: Memory adapter to evaluate
            sessions: Conversation sessions to store and recall from
            queries: Evaluation queries for recall testing
            enable_tracing: Whether to capture detailed execution traces
            
        Returns:
            Comprehensive evaluation results with cost analysis
        """
        logger.info(f"Starting benchmark evaluation for {adapter.name}")
        evaluation_start_time = time.time()
        
        # Phase 1: Storage evaluation (process conversation sessions)
        logger.info("Phase 1: Processing conversation sessions...")
        session_results = []
        total_storage_cost = 0
        
        for session in sessions:
            session_result = await self._evaluate_session_storage(
                adapter, session, enable_tracing
            )
            session_results.append(session_result)
            total_storage_cost += session_result.total_storage_cost_cents
        
        # Phase 2: Recall evaluation  
        logger.info("Phase 2: Evaluating memory recall...")
        recall_results = []
        total_retrieval_cost = 0
        
        for query in queries:
            recall_result = await self._evaluate_query_recall(
                adapter, query, enable_tracing
            )
            recall_results.append(recall_result)
            total_retrieval_cost += recall_result.cost_cents
        
        # Phase 3: Performance analysis
        total_turns = sum(session.turn_count for session in sessions)
        total_sessions = len(sessions)
        total_cost_cents = total_storage_cost + total_retrieval_cost
        
        # Calculate aggregate recall metrics
        recall_scores = [r.recall_score for r in recall_results]
        precision_scores = [r.precision_score for r in recall_results] 
        f1_scores = [r.f1_score for r in recall_results]
        latencies = [r.retrieval_latency_ms for r in recall_results]
        
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        # Compile final results
        result = BenchmarkEvaluationResult(
            sessions_processed=total_sessions,
            total_turns=total_turns,
            recall_evaluation={
                'queries_tested': len(queries),
                'recall_rate': avg_recall,
                'precision_rate': avg_precision,
                'f1_score': avg_f1,
                'successful_retrievals': sum(1 for r in recall_results if r.recall_score > 0.5),
                'avg_retrieval_latency_ms': avg_latency
            },
            session_results=session_results,
            total_cost_cents=total_cost_cents,
            cost_per_turn=total_cost_cents / max(1, total_turns),
            cost_per_session=total_cost_cents / max(1, total_sessions),
            avg_recall_score=avg_recall,
            avg_precision_score=avg_precision,
            avg_f1_score=avg_f1,
            avg_latency_ms=avg_latency,
            adapter_name=adapter.name,
            evaluation_timestamp=evaluation_start_time,
            benchmark_metadata={
                'evaluation_duration_seconds': time.time() - evaluation_start_time,
                'tracing_enabled': enable_tracing,
                'research_integration_available': self.research_integration_available,
                'adapter_capabilities': getattr(adapter, 'capabilities', []),
                'total_trace_entries': len(self._trace_entries)
            }
        )
        
        # Write traces if configured
        if enable_tracing and self.trace_file:
            await self._write_traces()
        
        logger.info(f"Benchmark evaluation completed: {avg_recall:.3f} recall, "
                   f"{total_cost_cents}¢ total cost")
        
        return result
    
    async def _evaluate_session_storage(
        self, 
        adapter: MemoryAdapter,
        session: ConversationSession,
        enable_tracing: bool
    ) -> SessionEvaluationResult:
        """Evaluate storage phase for a conversation session."""
        session_start_time = time.perf_counter()
        
        storage_operations = 0
        total_cost_cents = 0
        routing_decisions = []
        
        for turn in session.turns:
            # Simulate 3-class router decision (would use actual router in production)
            routing_decision = await self._simulate_3class_routing(turn.content, turn.metadata)
            routing_decisions.append(routing_decision)
            
            # Store memory based on routing decision
            if routing_decision != 'discard':
                try:
                    # Store the memory
                    memory_item = {
                        'content': turn.content,
                        'metadata': {
                            **turn.metadata,
                            'session_id': session.session_id,
                            'turn_id': turn.turn_id,
                            'speaker': turn.speaker,
                            'timestamp': turn.timestamp,
                            'routing_decision': routing_decision
                        }
                    }
                    
                    await adapter.store(turn.turn_id, memory_item)
                    storage_operations += 1
                    
                    # Calculate storage cost
                    with probe(self.cost_model) as cost_probe:
                        cost_estimate = self.cost_model.predict(
                            op=OperationType.STORE,
                            adapter=adapter.name,
                            tokens=count_tokens(turn.content),
                            k=0,
                            item_count=storage_operations
                        )
                        storage_cost = cost_estimate[0]  # cents
                        total_cost_cents += storage_cost
                    
                    if enable_tracing:
                        self._add_trace_entry({
                            'event': 'memory_stored',
                            'session_id': session.session_id,
                            'turn_id': turn.turn_id,
                            'routing_decision': routing_decision,
                            'cost_cents': storage_cost,
                            'content_length': len(turn.content)
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to store memory for turn {turn.turn_id}: {e}")
                    if enable_tracing:
                        self._add_trace_entry({
                            'event': 'storage_error',
                            'turn_id': turn.turn_id,
                            'error': str(e)
                        })
        
        processing_latency = (time.perf_counter() - session_start_time) * 1000
        efficiency_score = total_cost_cents / max(1, storage_operations)  # Cost per stored item
        
        return SessionEvaluationResult(
            session_id=session.session_id,
            storage_operations=storage_operations,
            total_storage_cost_cents=total_cost_cents,
            routing_decisions=routing_decisions,
            processing_latency_ms=processing_latency,
            memory_efficiency_score=efficiency_score
        )
    
    async def _evaluate_query_recall(
        self,
        adapter: MemoryAdapter,
        query: EvaluationQuery,
        enable_tracing: bool
    ) -> RecallEvaluationResult:
        """Evaluate memory recall for a single query."""
        query_start_time = time.perf_counter()
        
        try:
            # Retrieve memories
            retrieved_memories = await adapter.retrieve(query.query_text, k=5)
            
            # Calculate retrieval cost
            with probe(self.cost_model) as cost_probe:
                cost_estimate = self.cost_model.predict(
                    op=OperationType.RETRIEVE,
                    adapter=adapter.name,
                    tokens=count_tokens(query.query_text),
                    k=5,
                    item_count=len(retrieved_memories)
                )
                retrieval_cost = cost_estimate[0]  # cents
            
            # Evaluate recall quality
            recall_score, precision_score, f1_score = self._calculate_recall_metrics(
                query, retrieved_memories
            )
            
            retrieval_latency = (time.perf_counter() - query_start_time) * 1000
            
            if enable_tracing:
                self._add_trace_entry({
                    'event': 'memory_retrieved',
                    'query_id': query.query_id,
                    'query_type': query.query_type,
                    'memories_found': len(retrieved_memories),
                    'recall_score': recall_score,
                    'cost_cents': retrieval_cost,
                    'latency_ms': retrieval_latency
                })
            
            return RecallEvaluationResult(
                query_id=query.query_id,
                memories_retrieved=retrieved_memories,
                recall_score=recall_score,
                precision_score=precision_score,
                f1_score=f1_score,
                retrieval_latency_ms=retrieval_latency,
                cost_cents=retrieval_cost
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate recall for query {query.query_id}: {e}")
            if enable_tracing:
                self._add_trace_entry({
                    'event': 'recall_error',
                    'query_id': query.query_id,
                    'error': str(e)
                })
            
            return RecallEvaluationResult(
                query_id=query.query_id,
                memories_retrieved=[],
                recall_score=0.0,
                precision_score=0.0,
                f1_score=0.0,
                retrieval_latency_ms=0.0,
                cost_cents=0
            )
    
    async def _simulate_3class_routing(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Simulate 3-class router decision.
        
        In production, this would use the actual trained XGBoost router.
        For now, use heuristics that match the 3-class schema.
        """
        content_length = len(content)
        word_count = len(content.split())
        
        # Class 0: Discard (chit-chat, acknowledgments, very short)
        if (content_length < 10 or 
            word_count < 3 or
            content.lower().strip() in ['ok', 'thanks', 'yes', 'no', 'hi', 'hello', 'bye']):
            return 'discard'
        
        # Class 2: Compress (long content requiring summarization) 
        elif content_length > 500 or word_count > 80:
            return 'compress'
        
        # Class 1: Store (factual content worth keeping)
        else:
            return 'store'
    
    def _calculate_recall_metrics(
        self, 
        query: EvaluationQuery, 
        retrieved_memories: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """Calculate recall, precision, and F1 scores for memory retrieval."""
        
        if not query.expected_content:
            return 1.0, 1.0, 1.0  # Perfect if no expectations
        
        if not retrieved_memories:
            return 0.0, 0.0, 0.0  # Zero if nothing retrieved
        
        # Extract content from retrieved memories
        retrieved_content = []
        for memory in retrieved_memories:
            if isinstance(memory, dict):
                content = memory.get('content', '')
                if isinstance(content, str):
                    retrieved_content.append(content.lower())
        
        if not retrieved_content:
            return 0.0, 0.0, 0.0
        
        # Calculate matches against expected content
        expected_terms = set()
        for expected in query.expected_content:
            expected_terms.update(expected.lower().split())
        
        retrieved_terms = set()
        for content in retrieved_content:
            retrieved_terms.update(content.split())
        
        # Calculate metrics
        true_positives = len(expected_terms & retrieved_terms)
        false_positives = len(retrieved_terms - expected_terms)
        false_negatives = len(expected_terms - retrieved_terms)
        
        if true_positives == 0:
            return 0.0, 0.0, 0.0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return recall, precision, f1
    
    def _add_trace_entry(self, entry: Dict[str, Any]) -> None:
        """Add entry to execution trace."""
        entry['timestamp'] = time.time()
        entry['trace_id'] = str(uuid.uuid4())[:8]
        self._trace_entries.append(entry)
    
    async def _write_traces(self) -> None:
        """Write execution traces to file."""
        if not self.trace_file or not self._trace_entries:
            return
            
        try:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.trace_file, 'w') as f:
                for entry in self._trace_entries:
                    f.write(json.dumps(entry) + '\n')
                    
            logger.info(f"Wrote {len(self._trace_entries)} trace entries to {self.trace_file}")
            
        except Exception as e:
            logger.error(f"Failed to write traces: {e}")

    async def run_research_framework_evaluation(
        self,
        framework: str,
        dataset_path: Path,
        adapter: MemoryAdapter,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation using research integration layer.
        
        This provides cost-aware enhancements to epmembench and longmemeval frameworks.
        """
        if not self.research_integration_available:
            raise RuntimeError("Research integration not available")
            
        logger.info(f"Running {framework} evaluation with cost awareness")
        
        # Use research integration runner
        results = await self.research_runner.run_comparative_evaluation(
            framework=framework,
            dataset_path=dataset_path,
            output_path=output_path
        )
        
        return results


# Utility functions for sample data creation
def create_sample_session() -> ConversationSession:
    """Create a sample conversation session for testing."""
    
    turns = [
        ConversationTurn(
            turn_id="turn_1",
            speaker="user", 
            content="Hi, I need to schedule a meeting with the engineering team for next Tuesday at 3pm",
            timestamp=time.time(),
            metadata={"intent": "scheduling", "entities": ["engineering team", "Tuesday", "3pm"]}
        ),
        ConversationTurn(
            turn_id="turn_2",
            speaker="assistant",
            content="I'll help you schedule a meeting with the engineering team for Tuesday at 3pm. Let me check availability and send out calendar invites.",
            timestamp=time.time() + 1,
            metadata={"intent": "confirmation", "action": "schedule_meeting"}
        ),
        ConversationTurn(
            turn_id="turn_3", 
            speaker="user",
            content="Great! Also, can you remind me to prepare the quarterly budget review?",
            timestamp=time.time() + 10,
            metadata={"intent": "reminder", "task": "quarterly budget review"}
        ),
        ConversationTurn(
            turn_id="turn_4",
            speaker="assistant",
            content="I've added a reminder for you to prepare the quarterly budget review. Is there a specific deadline for this?",
            timestamp=time.time() + 11,
            metadata={"intent": "clarification", "task_created": True}
        )
    ]
    
    return ConversationSession(
        session_id="sample_session_1",
        turns=turns,
        session_metadata={"topic": "work planning", "priority": "high"},
        start_time=time.time(),
        end_time=time.time() + 20
    )


def create_sample_queries() -> List[EvaluationQuery]:
    """Create sample evaluation queries for testing."""
    
    return [
        EvaluationQuery(
            query_id="query_1",
            query_text="When is the engineering team meeting?", 
            query_type="factual",
            expected_content=["Tuesday", "3pm", "engineering team", "meeting"],
            context={"expected_accuracy": "high"},
            target_session_ids=["sample_session_1"]
        ),
        EvaluationQuery(
            query_id="query_2",
            query_text="What tasks do I need to prepare?",
            query_type="semantic", 
            expected_content=["quarterly budget review", "prepare"],
            context={"expected_accuracy": "medium"},
            target_session_ids=["sample_session_1"]
        ),
        EvaluationQuery(
            query_id="query_3",
            query_text="What did we discuss about work planning?",
            query_type="semantic",
            expected_content=["meeting", "engineering team", "budget review"],
            context={"expected_accuracy": "medium"},
            target_session_ids=["sample_session_1"]
        )
    ]