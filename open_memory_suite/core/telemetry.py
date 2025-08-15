"""Lightweight telemetry system for cost/latency measurement."""

from __future__ import annotations
import time
import json
import pathlib
from contextlib import contextmanager
from typing import Dict, Any


_LOG = pathlib.Path("./telemetry.jsonl")


def log_event(obj: dict) -> None:
    """Log a telemetry event to JSONL file."""
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    if not _LOG.exists():
        _LOG.write_text("")
    with _LOG.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@contextmanager
def probe(op: str, adapter: str, predicted_cents: int, predicted_ms: float, meta: Dict[str, Any]):
    """
    Context manager for measuring operation performance vs predictions.
    
    Args:
        op: Operation name (e.g., "store", "retrieve", "summarize")
        adapter: Adapter name (e.g., "faiss_store", "memory_store")
        predicted_cents: Cost prediction from model
        predicted_ms: Latency prediction from model
        meta: Additional metadata (k, tokens, item_count, observed_cents, etc.)
    """
    t0 = time.perf_counter()
    err = None
    try:
        yield
    except Exception as e:
        err = repr(e)
        raise
    finally:
        ms = (time.perf_counter() - t0) * 1000.0
        log_event({
            "ts": time.time(),
            "op": op,
            "adapter": adapter,
            "predicted_cents": predicted_cents,
            "predicted_ms": predicted_ms,
            "observed_ms": ms,
            "observed_cents": meta.get("observed_cents"),
            "meta": meta,
            "error": err
        })