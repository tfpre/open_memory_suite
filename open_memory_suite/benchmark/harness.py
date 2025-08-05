"""Test harness for running memory benchmark evaluations."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel

from ..adapters import MemoryAdapter, MemoryItem
from .trace import TraceLogger, trace_operation


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    turn_id: str
    speaker: str  # "user" or "assistant"
    content: str
    timestamp: str
    metadata: Dict = {}


class ConversationSession(BaseModel):
    """A complete conversation session."""
    session_id: str
    turns: List[ConversationTurn]
    metadata: Dict = {}


class EvaluationQuery(BaseModel):
    """A query to test memory recall."""
    query_id: str
    session_id: str
    query_text: str
    expected_answer: str
    turn_context: Optional[str] = None  # Which turn this refers to


class BenchmarkHarness:
    """Harness for running memory benchmark evaluations."""
    
    def __init__(self, trace_file: Path):
        """Initialize with a trace file path."""
        self.trace_file = trace_file
        self._results: List[Dict] = []
    
    async def run_session(
        self,
        adapter: MemoryAdapter,
        session: ConversationSession,
        store_all: bool = True
    ) -> Dict:
        """
        Run a conversation session through a memory adapter.
        
        Args:
            adapter: The memory adapter to test
            session: The conversation session to run
            store_all: Whether to store all turns (if False, need dispatcher)
            
        Returns:
            Dictionary with session results
        """
        async with TraceLogger(self.trace_file) as logger:
            session_result = {
                "session_id": session.session_id,
                "adapter_name": adapter.name,
                "turns_processed": 0,
                "storage_operations": 0,
                "total_items_stored": 0,
                "errors": []
            }
            
            for turn in session.turns:
                operation_id = str(uuid4())
                
                if store_all:
                    # Convert turn to MemoryItem
                    memory_item = MemoryItem(
                        content=turn.content,
                        speaker=turn.speaker,
                        turn_id=turn.turn_id,
                        metadata=turn.metadata
                    )
                    
                    # Store the item with tracing
                    try:
                        async with trace_operation(
                            logger,
                            "store",
                            adapter.name,
                            operation_id,
                            content=turn.content,
                            tokens_processed=len(turn.content.split())
                        ) as event:
                            success = await adapter.store(memory_item)
                            event.success = success
                            event.items_count = 1 if success else 0
                            
                            if success:
                                session_result["storage_operations"] += 1
                            else:
                                session_result["errors"].append(f"Failed to store turn {turn.turn_id}")
                    
                    except Exception as e:
                        session_result["errors"].append(f"Error storing turn {turn.turn_id}: {str(e)}")
                
                session_result["turns_processed"] += 1
            
            # Get final count
            session_result["total_items_stored"] = await adapter.count()
            
            return session_result
    
    async def evaluate_recall(
        self,
        adapter: MemoryAdapter,
        queries: List[EvaluationQuery],
        k: int = 5
    ) -> Dict:
        """
        Evaluate memory recall using test queries.
        
        Args:
            adapter: The memory adapter to test
            queries: List of evaluation queries
            k: Number of items to retrieve for each query
            
        Returns:
            Dictionary with recall evaluation results
        """
        async with TraceLogger(self.trace_file) as logger:
            recall_results = {
                "total_queries": len(queries),
                "successful_retrievals": 0,
                "query_results": []
            }
            
            for query in queries:
                operation_id = str(uuid4())
                
                try:
                    async with trace_operation(
                        logger,
                        "retrieve",
                        adapter.name,
                        operation_id,
                        query=query.query_text
                    ) as event:
                        # Retrieve relevant items
                        result = await adapter.retrieve(query.query_text, k=k)
                        
                        event.items_count = len(result.items)
                        event.tokens_processed = sum(len(item.content.split()) for item in result.items)
                        
                        # Simple relevance check (can be enhanced later)
                        relevant_content = "\n".join(result.content_only)
                        
                        query_result = {
                            "query_id": query.query_id,
                            "query_text": query.query_text,
                            "expected_answer": query.expected_answer,
                            "retrieved_items": len(result.items),
                            "retrieved_content": relevant_content,
                            "similarity_scores": result.similarity_scores,
                            # Placeholder for actual evaluation - would use LLM judge
                            "recall_success": len(result.items) > 0  # Simple placeholder
                        }
                        
                        recall_results["query_results"].append(query_result)
                        
                        if query_result["recall_success"]:
                            recall_results["successful_retrievals"] += 1
                
                except Exception as e:
                    query_result = {
                        "query_id": query.query_id,
                        "query_text": query.query_text,
                        "error": str(e),
                        "recall_success": False
                    }
                    recall_results["query_results"].append(query_result)
            
            # Calculate overall recall rate
            recall_results["recall_rate"] = (
                recall_results["successful_retrievals"] / recall_results["total_queries"]
                if recall_results["total_queries"] > 0 else 0.0
            )
            
            return recall_results
    
    async def run_full_evaluation(
        self,
        adapter: MemoryAdapter,
        sessions: List[ConversationSession],
        queries: List[EvaluationQuery]
    ) -> Dict:
        """
        Run a complete evaluation: store sessions then test recall.
        
        Args:
            adapter: Memory adapter to evaluate
            sessions: Conversation sessions to store
            queries: Queries to test recall
            
        Returns:
            Complete evaluation results
        """
        # Initialize adapter
        async with adapter:
            evaluation_results = {
                "adapter_name": adapter.name,
                "sessions_processed": len(sessions),
                "total_turns": sum(len(session.turns) for session in sessions),
                "session_results": [],
                "recall_evaluation": None
            }
            
            # Process all sessions
            for session in sessions:
                session_result = await self.run_session(adapter, session)
                evaluation_results["session_results"].append(session_result)
            
            # Evaluate recall
            recall_result = await self.evaluate_recall(adapter, queries)
            evaluation_results["recall_evaluation"] = recall_result
            
            return evaluation_results


def create_sample_session() -> ConversationSession:
    """Create a sample conversation session for testing."""
    turns = [
        ConversationTurn(
            turn_id="turn_1",
            speaker="user",
            content="Hello, I'm Alice and I work as a software engineer at Google.",
            timestamp="2024-01-15T10:00:00Z",
            metadata={"topic": "introduction"}
        ),
        ConversationTurn(
            turn_id="turn_2", 
            speaker="assistant",
            content="Nice to meet you Alice! It's great to meet someone from Google. How long have you been working there?",
            timestamp="2024-01-15T10:00:05Z",
            metadata={"topic": "introduction"}
        ),
        ConversationTurn(
            turn_id="turn_3",
            speaker="user",
            content="I've been at Google for about 3 years now. I work on the search ranking algorithms.",
            timestamp="2024-01-15T10:00:15Z",
            metadata={"topic": "work", "company": "Google"}
        ),
        ConversationTurn(
            turn_id="turn_4",
            speaker="assistant",
            content="That sounds fascinating! Search ranking is such a complex and important problem. Do you work with machine learning models?",
            timestamp="2024-01-15T10:00:25Z",
            metadata={"topic": "work"}
        ),
        ConversationTurn(
            turn_id="turn_5",
            speaker="user",
            content="Yes, I primarily work with transformer-based models for understanding search intent.",
            timestamp="2024-01-15T10:00:40Z", 
            metadata={"topic": "work", "technology": "transformers"}
        )
    ]
    
    return ConversationSession(
        session_id="sample_session_1",
        turns=turns,
        metadata={"domain": "casual_chat", "length": "short"}
    )


def create_sample_queries() -> List[EvaluationQuery]:
    """Create sample evaluation queries."""
    return [
        EvaluationQuery(
            query_id="query_1",
            session_id="sample_session_1",
            query_text="What is Alice's job?",
            expected_answer="Alice is a software engineer at Google.",
            turn_context="turn_1"
        ),
        EvaluationQuery(
            query_id="query_2",
            session_id="sample_session_1", 
            query_text="How long has Alice worked at Google?",
            expected_answer="3 years",
            turn_context="turn_3"
        ),
        EvaluationQuery(
            query_id="query_3",
            session_id="sample_session_1",
            query_text="What kind of models does Alice work with?",
            expected_answer="Transformer-based models for understanding search intent",
            turn_context="turn_5"
        )
    ]