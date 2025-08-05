"""Trace logging for memory operations."""

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import aiofiles
from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """A single trace event in the memory benchmark."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str = Field(..., description="Type of event (store, retrieve, etc.)")
    adapter_name: str = Field(..., description="Name of the memory adapter")
    operation_id: str = Field(..., description="Unique ID for this operation")
    
    # Input data
    query: Optional[str] = Field(None, description="Query text for retrieval")
    content: Optional[str] = Field(None, description="Content being stored")
    
    # Results
    success: bool = Field(True, description="Whether operation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    items_count: Optional[int] = Field(None, description="Number of items retrieved/stored")
    
    # Performance metrics
    latency_ms: float = Field(..., description="Operation latency in milliseconds")
    tokens_processed: Optional[int] = Field(None, description="Number of tokens processed")
    
    # Cost tracking data
    operation_details: Dict[str, Any] = Field(default_factory=dict, description="Details for cost calculation")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TraceLogger:
    """Async logger for memory operation traces."""
    
    def __init__(self, trace_file: Path):
        """Initialize with a trace file path."""
        self.trace_file = trace_file
        self._file_handle: Optional[aiofiles.threadpool.text.AsyncTextIOWrapper] = None
    
    async def __aenter__(self):
        """Open the trace file for writing."""
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = await aiofiles.open(self.trace_file, 'a', encoding='utf-8')
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the trace file."""
        if self._file_handle:
            await self._file_handle.close()
            self._file_handle = None
    
    async def log_event(self, event: TraceEvent) -> None:
        """Log a trace event to the file."""
        if not self._file_handle:
            raise RuntimeError("TraceLogger not properly initialized")
        
        json_line = event.model_dump_json() + "\n"
        await self._file_handle.write(json_line)
        await self._file_handle.flush()


@asynccontextmanager
async def trace_operation(
    logger: TraceLogger,
    event_type: str,
    adapter_name: str,
    operation_id: str,
    **kwargs
) -> AsyncGenerator[TraceEvent, None]:
    """
    Context manager for tracing memory operations.
    
    Args:
        logger: The trace logger to use
        event_type: Type of operation being traced
        adapter_name: Name of the memory adapter
        operation_id: Unique operation identifier
        **kwargs: Additional fields for the trace event
    """
    start_time = time.perf_counter()
    
    # Create initial event
    event = TraceEvent(
        event_type=event_type,
        adapter_name=adapter_name,
        operation_id=operation_id,
        latency_ms=0.0,  # Will be updated
        **kwargs
    )
    
    try:
        yield event
        event.success = True
    except Exception as e:
        event.success = False
        event.error_message = str(e)
        raise
    finally:
        # Calculate latency
        end_time = time.perf_counter()
        event.latency_ms = (end_time - start_time) * 1000
        
        # Log the event
        await logger.log_event(event)


def load_trace_file(trace_file: Path) -> list[TraceEvent]:
    """Load and parse a trace file into TraceEvent objects."""
    events = []
    
    if not trace_file.exists():
        return events
    
    with open(trace_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                event = TraceEvent(**data)
                events.append(event)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not parse line {line_num} in {trace_file}: {e}")
    
    return events