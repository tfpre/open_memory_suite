"""Tests for the benchmark harness."""

import pytest

from open_memory_suite.adapters import FAISStoreAdapter
from open_memory_suite.benchmark import (
    BenchmarkHarness,
    ConversationSession, 
    ConversationTurn,
    EvaluationQuery
)
from open_memory_suite.benchmark.harness import create_sample_session, create_sample_queries


class TestBenchmarkHarness:
    """Test the benchmark harness functionality."""
    
    @pytest.mark.asyncio
    async def test_run_session_basic(self, temp_dir):
        """Test running a basic conversation session."""
        trace_file = temp_dir / "harness_test.jsonl"
        harness = BenchmarkHarness(trace_file)
        
        # Create adapter and session
        adapter = FAISStoreAdapter(name="test_faiss")
        session = create_sample_session()
        
        async with adapter:
            result = await harness.run_session(adapter, session)
        
        # Check results
        assert result["session_id"] == "sample_session_1"
        assert result["adapter_name"] == "test_faiss"
        assert result["turns_processed"] == 5  # 5 turns in sample session
        assert result["storage_operations"] == 5  # All turns stored
        assert result["total_items_stored"] == 5
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_recall_basic(self, temp_dir):
        """Test basic recall evaluation."""
        trace_file = temp_dir / "recall_test.jsonl"
        harness = BenchmarkHarness(trace_file)
        
        # Setup adapter with some data
        adapter = FAISStoreAdapter(name="test_recall")
        session = create_sample_session()
        queries = create_sample_queries()
        
        async with adapter:
            # First store the session
            await harness.run_session(adapter, session)
            
            # Then evaluate recall
            recall_result = await harness.evaluate_recall(adapter, queries)
        
        # Check recall results
        assert recall_result["total_queries"] == 3
        assert len(recall_result["query_results"]) == 3
        assert "recall_rate" in recall_result
        assert 0.0 <= recall_result["recall_rate"] <= 1.0
        
        # Each query should have results
        for query_result in recall_result["query_results"]:
            assert "query_id" in query_result
            assert "query_text" in query_result
            assert "retrieved_items" in query_result
            assert "recall_success" in query_result
    
    @pytest.mark.asyncio
    async def test_full_evaluation(self, temp_dir):
        """Test complete evaluation workflow."""
        trace_file = temp_dir / "full_eval.jsonl"
        harness = BenchmarkHarness(trace_file)
        
        adapter = FAISStoreAdapter(name="test_full")
        sessions = [create_sample_session()]
        queries = create_sample_queries()
        
        result = await harness.run_full_evaluation(adapter, sessions, queries)
        
        # Check complete results structure
        assert result["adapter_name"] == "test_full"
        assert result["sessions_processed"] == 1
        assert result["total_turns"] == 5
        assert len(result["session_results"]) == 1
        assert result["recall_evaluation"] is not None
        
        # Session results should be present
        session_result = result["session_results"][0]
        assert session_result["session_id"] == "sample_session_1"
        assert session_result["turns_processed"] == 5
        
        # Recall evaluation should be present
        recall_eval = result["recall_evaluation"]
        assert recall_eval["total_queries"] == 3
        assert "recall_rate" in recall_eval


class TestDataModels:
    """Test the data models for conversations and queries."""
    
    def test_conversation_turn_creation(self):
        """Test ConversationTurn model."""
        turn = ConversationTurn(
            turn_id="test_turn",
            speaker="user",
            content="Hello world",
            timestamp="2024-01-15T10:00:00Z"
        )
        
        assert turn.turn_id == "test_turn"
        assert turn.speaker == "user"
        assert turn.content == "Hello world"
        assert turn.metadata == {}
    
    def test_conversation_session_creation(self):
        """Test ConversationSession model."""
        turns = [
            ConversationTurn(
                turn_id="turn_1",
                speaker="user", 
                content="Hi",
                timestamp="2024-01-15T10:00:00Z"
            ),
            ConversationTurn(
                turn_id="turn_2",
                speaker="assistant",
                content="Hello!",
                timestamp="2024-01-15T10:00:05Z"
            )
        ]
        
        session = ConversationSession(
            session_id="test_session",
            turns=turns
        )
        
        assert session.session_id == "test_session"
        assert len(session.turns) == 2
        assert session.turns[0].speaker == "user"
        assert session.turns[1].speaker == "assistant"
    
    def test_evaluation_query_creation(self):
        """Test EvaluationQuery model."""
        query = EvaluationQuery(
            query_id="q1",
            session_id="s1",
            query_text="What is X?",
            expected_answer="X is Y",
            turn_context="turn_3"
        )
        
        assert query.query_id == "q1"
        assert query.session_id == "s1"
        assert query.query_text == "What is X?"
        assert query.expected_answer == "X is Y"
        assert query.turn_context == "turn_3"
    
    def test_sample_data_creation(self):
        """Test the sample data creation functions."""
        session = create_sample_session()
        queries = create_sample_queries()
        
        assert session.session_id == "sample_session_1"
        assert len(session.turns) == 5
        assert session.turns[0].speaker == "user"
        assert "Alice" in session.turns[0].content
        
        assert len(queries) == 3
        assert all(q.session_id == "sample_session_1" for q in queries)
        assert "Alice" in queries[0].query_text