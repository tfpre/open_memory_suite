"""Shared tokenizer for consistent token counting across the system."""

from __future__ import annotations
from typing import Optional


class TokenCounter:
    """Consistent token counting using tiktoken when available, with robust fallback."""
    
    def __init__(self, model_hint: str = "gpt-4o-mini"):
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        """Count tokens in text using tiktoken or empirical fallback."""
        if not text:
            return 0
        if self._enc:
            return len(self._enc.encode(text))
        # Fallback: empirical 4.2 chars/token; clamp to ≥1
        return max(1, round(len(text) / 4.2))