"""Pytest configuration and fixtures for the test suite."""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest

from open_memory_suite.adapters import FAISStoreAdapter, MemoryItem
from open_memory_suite.benchmark.trace import TraceLogger


# Remove custom event loop - let pytest-asyncio handle it


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
async def faiss_adapter(temp_dir: Path) -> AsyncGenerator[FAISStoreAdapter, None]:
    """Provide a FAISS adapter for testing."""
    adapter = FAISStoreAdapter(
        name="test_faiss",
        index_path=temp_dir / "test_index"
    )
    
    async with adapter:
        yield adapter


@pytest.fixture
async def trace_logger(temp_dir: Path) -> AsyncGenerator[TraceLogger, None]:
    """Provide a trace logger for testing."""
    trace_file = temp_dir / "test_trace.jsonl"
    
    async with TraceLogger(trace_file) as logger:
        yield logger


@pytest.fixture
def sample_memory_items() -> list[MemoryItem]:
    """Provide sample memory items for testing."""
    return [
        MemoryItem(
            content="Hello, I'm Alice and I work as a software engineer.",
            speaker="user",
            turn_id="turn_1",
            metadata={"topic": "introduction", "contains_name": True}
        ),
        MemoryItem(
            content="Nice to meet you Alice! I'm an AI assistant.",
            speaker="assistant", 
            turn_id="turn_2",
            metadata={"topic": "introduction"}
        ),
        MemoryItem(
            content="I'm planning a trip to Japan next month. Any recommendations?",
            speaker="user",
            turn_id="turn_3", 
            metadata={"topic": "travel", "destination": "Japan"}
        ),
        MemoryItem(
            content="I'd recommend visiting Kyoto for temples and Tokyo for modern culture.",
            speaker="assistant",
            turn_id="turn_4",
            metadata={"topic": "travel", "destination": "Japan"}
        ),
        MemoryItem(
            content="What's the weather like in Kyoto in spring?",
            speaker="user",
            turn_id="turn_5",
            metadata={"topic": "weather", "destination": "Japan", "city": "Kyoto"}
        )
    ]


@pytest.fixture
def conversation_turns() -> list[dict]:
    """Provide sample conversation turns in session format."""
    return [
        {
            "turn_id": "turn_1",
            "speaker": "user", 
            "content": "Hello, I'm Alice and I work as a software engineer.",
            "timestamp": "2024-01-15T10:00:00Z"
        },
        {
            "turn_id": "turn_2",
            "speaker": "assistant",
            "content": "Nice to meet you Alice! I'm an AI assistant here to help.",
            "timestamp": "2024-01-15T10:00:05Z"
        },
        {
            "turn_id": "turn_3", 
            "speaker": "user",
            "content": "I'm planning a trip to Japan next month. Any recommendations?",
            "timestamp": "2024-01-15T10:01:00Z"
        },
        {
            "turn_id": "turn_4",
            "speaker": "assistant", 
            "content": "I'd recommend visiting Kyoto for traditional temples and Tokyo for modern culture and food.",
            "timestamp": "2024-01-15T10:01:10Z"
        },
        {
            "turn_id": "turn_5",
            "speaker": "user",
            "content": "What's the weather like in Kyoto in spring?",
            "timestamp": "2024-01-15T10:02:00Z"
        }
    ]