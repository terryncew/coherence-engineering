# runtime/lungs.py
# Defines the Lung interface and 3 concrete lungs:
# - FastReflexLung
# - SlowDeliberativeLung
# - ContextRetrievalLung (3rd lung)

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass
class LungOutput:
    content: str
    confidence: float
    tokens_used: int
    processing_time: float
    mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Lung(Protocol):
    async def run(self, request: str, context: Dict[str, Any]) -> LungOutput:
        ...


class FastReflexLung:
    """Cheap, bounded, always-on. Templates, cached knowledge, low temp."""
    def __init__(self, max_tokens: int = 160, temperature: float = 0.25):
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def run(self, request: str, context: Dict[str, Any]) -> LungOutput:
        start = time.time()
        # Simulate quick bounded processing
        await asyncio.sleep(0.1)
        content = (
            f"[FAST] Concise, safe draft for: {request[:64]}..."
            " (templated, low-temp, bounded)"
        )
        return LungOutput(
            content=content,
            confidence=0.70,
            tokens_used=min(self.max_tokens, 120),
            processing_time=time.time() - start,
            mode="fast_reflex",
            metadata={"temperature": self.temperature, "max_tokens": self.max_tokens},
        )


class SlowDeliberativeLung:
    """Deeper reasoning, longer chains, more context/tools. Opportunistic."""
    def __init__(self, base_tokens: int = 800, temperature: float = 0.8):
        self.base_tokens = base_tokens
        self.temperature = temperature

    async def run(self, request: str, context: Dict[str, Any]) -> LungOutput:
        start = time.time()
        delay = 0.8 + 0.6  # simulate deeper latency; tune as needed
        await asyncio.sleep(delay)
        content = (
            f"[SLOW] Rich analysis for: {request[:64]}..."
            " (multi-angle reasoning, verification hints)"
        )
        return LungOutput(
            content=content,
            confidence=0.85,
            tokens_used=int(self.base_tokens * 0.35),
            processing_time=time.time() - start,
            mode="slow_deliberative",
            metadata={"temperature": self.temperature, "max_tokens": self.base_tokens},
        )


class ContextRetrievalLung:
    """
    “3rd lung” that adds contextual flexibility:
    - lightweight retrieval / tool-ish enrichment
    - helps when the system is stable enough to add context
    """
    def __init__(self, max_tokens: int = 320):
        self.max_tokens = max_tokens

    async def run(self, request: str, context: Dict[str, Any]) -> LungOutput:
        start = time.time()
        # Simulate quick retrieval latency (index/cache lookup)
        await asyncio.sleep(0.25)
        # In a real system, use vector search / RAG / tools here.
        snippet = context.get("retrieved_snippet") or "no-index (demo snippet)"
        content = (
            f"[CTX] Contextual add-on for: {request[:64]}..."
            f" [context: {snippet[:80]}]"
        )
        return LungOutput(
            content=content,
            confidence=0.78,
            tokens_used=min(self.max_tokens, 160),
            processing_time=time.time() - start,
            mode="context_retrieval",
            metadata={"retrieval_source": "demo_cache", "max_tokens": self.max_tokens},
        )
