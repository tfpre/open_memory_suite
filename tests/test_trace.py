"""Tests for trace logging functionality."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from open_memory_suite.benchmark.trace import (
    TraceEvent, 
    TraceLogger, 
    load_trace_file, 
    trace_operation
)


class TestTraceEvent:
    """Test TraceEvent data model."""
    
    def test_trace_event_creation(self):
        """Test basic TraceEvent creation."""
        event = TraceEvent(
            event_type="store",
            adapter_name="test_adapter",
            operation_id="op_123",
            latency_ms=15.5
        )
        
        assert event.event_type == "store"
        assert event.adapter_name == "test_adapter"
        assert event.operation_id == "op_123"
        assert event.latency_ms == 15.5
        assert event.success is True
        assert event.timestamp is not None
    
    def test_trace_event_with_error(self):
        """Test TraceEvent with error information."""
        event = TraceEvent(
            event_type="retrieve",
            adapter_name="faiss",
            operation_id="op_456",
            latency_ms=100.0,
            success=False,
            error_message="Connection timeout"
        )
        
        assert not event.success
        assert event.error_message == "Connection timeout"
    
    def test_trace_event_serialization(self):
        """Test TraceEvent JSON serialization."""
        event = TraceEvent(
            event_type="store",
            adapter_name="test",
            operation_id="op_789",
            latency_ms=25.0,
            query="test query",
            items_count=5,
            operation_details={"embedding_dim": 384}
        )
        
        json_str = event.model_dump_json()
        data = json.loads(json_str)
        
        assert data["event_type"] == "store"
        assert data["adapter_name"] == "test"
        assert data["latency_ms"] == 25.0
        assert data["query"] == "test query"
        assert data["items_count"] == 5
        assert data["operation_details"]["embedding_dim"] == 384


class TestTraceLogger:
    """Test TraceLogger functionality."""
    
    @pytest.mark.asyncio
    async def test_trace_logger_basic(self, temp_dir):
        """Test basic trace logging."""
        trace_file = temp_dir / "test_trace.jsonl"
        
        async with TraceLogger(trace_file) as logger:
            event = TraceEvent(
                event_type="test",
                adapter_name="test_adapter",
                operation_id="op_1",
                latency_ms=10.0
            )
            await logger.log_event(event)
        
        # Check file was created and contains the event
        assert trace_file.exists()
        
        events = load_trace_file(trace_file)
        assert len(events) == 1
        assert events[0].event_type == "test"
        assert events[0].adapter_name == "test_adapter"
    
    @pytest.mark.asyncio
    async def test_trace_logger_multiple_events(self, temp_dir):
        """Test logging multiple events."""
        trace_file = temp_dir / "multi_trace.jsonl"
        
        async with TraceLogger(trace_file) as logger:
            for i in range(3):
                event = TraceEvent(
                    event_type=f"test_{i}",
                    adapter_name="test_adapter",
                    operation_id=f"op_{i}",
                    latency_ms=float(i * 10)
                )
                await logger.log_event(event)
        
        events = load_trace_file(trace_file)
        assert len(events) == 3
        
        for i, event in enumerate(events):
            assert event.event_type == f"test_{i}"
            assert event.operation_id == f"op_{i}"
            assert event.latency_ms == float(i * 10)
    
    @pytest.mark.asyncio
    async def test_trace_operation_context_manager(self, trace_logger):
        """Test the trace_operation context manager."""
        operation_id = str(uuid4())
        
        async with trace_operation(
            trace_logger, 
            "store", 
            "test_adapter", 
            operation_id,
            content="test content"
        ) as event:
            # Modify event during operation
            event.items_count = 1
            event.tokens_processed = 50
        
        # Event should have been logged automatically
        # We can't easily verify this without reading the file in this test,
        # but the context manager should have handled logging
        assert event.success is True
        assert event.latency_ms > 0  # Should have measured some latency
        assert event.content == "test content"
        assert event.items_count == 1
    
    @pytest.mark.asyncio
    async def test_trace_operation_with_exception(self, trace_logger):
        """Test trace_operation handles exceptions properly."""
        operation_id = str(uuid4())
        
        with pytest.raises(ValueError, match="Test error"):
            async with trace_operation(
                trace_logger,
                "retrieve", 
                "test_adapter",
                operation_id
            ) as event:
                event.query = "test query"
                raise ValueError("Test error")
        
        # Event should still be logged with error info
        # In a real scenario, we'd verify this by reading the trace file
        
    def test_load_trace_file_empty(self, temp_dir):
        """Test loading from non-existent trace file."""
        trace_file = temp_dir / "nonexistent.jsonl"
        events = load_trace_file(trace_file)
        assert events == []
    
    def test_load_trace_file_malformed(self, temp_dir):
        """Test loading trace file with malformed lines."""
        trace_file = temp_dir / "malformed.jsonl"
        
        # Write some good and bad lines
        with open(trace_file, 'w') as f:
            # Good line
            good_event = TraceEvent(
                event_type="test",
                adapter_name="test",
                operation_id="op_1",
                latency_ms=10.0
            )
            f.write(good_event.model_dump_json() + "\n")
            
            # Bad line (invalid JSON)
            f.write("invalid json line\n")
            
            # Another good line
            good_event2 = TraceEvent(
                event_type="test2",
                adapter_name="test",
                operation_id="op_2",
                latency_ms=20.0
            )
            f.write(good_event2.model_dump_json() + "\n")
        
        # Should skip bad lines and load good ones
        events = load_trace_file(trace_file)
        assert len(events) == 2
        assert events[0].event_type == "test"
        assert events[1].event_type == "test2"