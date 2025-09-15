# open_line/dual_channel.py  — stdlib-only, CI-friendly
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Deque
from enum import Enum
from collections import deque, Counter
from random import random, gauss, randint, choice
from statistics import mean, pstdev
from math import log2

class ChannelState(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"

# -------- Physics (low-bandwidth) --------

@dataclass
class PhysicsSignal:
    type: str  # "rhythm", "compression", "pointing"
    value: Any
    timestamp: float = field(default_factory=time.time)
    sequence_id: int = 0

class RhythmSignal(PhysicsSignal):
    def __init__(self, bpm: float = 60.0, sequence_id: int = 0):
        period = max(0.001, 60.0 / max(0.001, bpm))
        phase = (time.time() % period) / period  # 0..1
        super().__init__("rhythm", {"bpm": float(bpm), "phase": phase}, sequence_id=sequence_id)

class CompressionSignal(PhysicsSignal):
    def __init__(self, original: str, compressed: str, ratio: float, sequence_id: int = 0):
        super().__init__("compression", {
            "original_length": len(original),
            "compressed_length": len(compressed),
            "ratio": float(ratio),
            "entropy": self._entropy(original)
        }, sequence_id=sequence_id)

    def _entropy(self, text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        n = sum(counts.values())
        return float(-sum((c/n) * log2(c/n) for c in counts.values() if c))

class PointingSignal(PhysicsSignal):
    def __init__(self, target: str, confidence: float, context: Optional[Dict[str, Any]] = None, sequence_id: int = 0):
        super().__init__("pointing", {
            "target": target,
            "confidence": float(confidence),
            "context": context or {}
        }, sequence_id=sequence_id)

# -------- Semantics (high-bandwidth) --------

@dataclass
class SemanticSignal:
    type: str  # "narrative", "instruction", "tool_call", "query"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sequence_id: int = 0

class NarrativeSignal(SemanticSignal):
    def __init__(self, content: str, narrative_type: str = "general", sequence_id: int = 0):
        super().__init__("narrative", content, {
            "narrative_type": narrative_type,
            "word_count": len(content.split()),
            "complexity": self._complexity(content)
        }, sequence_id=sequence_id)

    def _complexity(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        avg_len = mean(len(w) for w in words)
        sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
        return float(avg_len * (len(words) / sentence_count))

# -------- Metrics --------

@dataclass
class ChannelMetrics:
    mpi: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    signal_quality: float = 1.0
    last_successful: float = field(default_factory=time.time)

# -------- Protocol --------

class DualChannelProtocol:
    """Dual-channel communication protocol with physics + semantic layers."""
    def __init__(self,
                 physics_interval: float = 1.0,
                 semantic_timeout: float = 10.0,
                 mpi_threshold: float = 0.3):
        self.physics_interval = physics_interval
        self.semantic_timeout = semantic_timeout
        self.mpi_threshold = mpi_threshold

        self.physics_state = ChannelState.ACTIVE
        self.semantic_state = ChannelState.ACTIVE

        self.physics_metrics = ChannelMetrics()
        self.semantic_metrics = ChannelMetrics()

        self.physics_history: Deque[PhysicsSignal] = deque(maxlen=100)
        self.semantic_history: Deque[SemanticSignal] = deque(maxlen=50)

        self.physics_seq = 0
        self.semantic_seq = 0

        self._running = False
        self._physics_task: Optional[asyncio.Task] = None
        self._metric_task: Optional[asyncio.Task] = None

        self._semantic_ever = False  # avoid marking FAILED if never used

        self.physics_handlers: List[Callable[[PhysicsSignal], Any]] = []
        self.semantic_handlers: List[Callable[[SemanticSignal], Any]] = []

    async def start(self):
        if self._running:
            return
        self._running = True
        self._physics_task = asyncio.create_task(self._physics_loop(), name="physics-loop")
        self._metric_task = asyncio.create_task(self._metric_loop(), name="metric-loop")

    async def stop(self):
        if not self._running:
            return
        self._running = False
        tasks = [t for t in (self._physics_task, self._metric_task) if t]
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._physics_task = self._metric_task = None

    def register_physics_handler(self, handler: Callable[[PhysicsSignal], Any]):
        self.physics_handlers.append(handler)

    def register_semantic_handler(self, handler: Callable[[SemanticSignal], Any]):
        self.semantic_handlers.append(handler)

    async def _physics_loop(self):
        try:
            while self._running:
                # 1) Rhythm heartbeat (sequence groups physics signals)
                rhythm = RhythmSignal(bpm=60 + gauss(0, 5), sequence_id=self.physics_seq)
                await self._send_physics_signal(rhythm)

                # 2) Occasionally report compression/entropy
                if random() < 0.3:
                    s = self._generate_test_string()
                    c = self._compress_string(s)
                    await self._send_physics_signal(CompressionSignal(
                        s, c, len(c) / max(1, len(s)), sequence_id=self.physics_seq
                    ))

                # 3) Point toward latest semantic activity (if any)
                if self.semantic_history:
                    recent = self.semantic_history[-1]
                    await self._send_physics_signal(PointingSignal(
                        target=recent.type, confidence=0.8,
                        context={"last_semantic": recent.content[:50]},
                        sequence_id=self.physics_seq
                    ))

                self.physics_seq += 1
                await asyncio.sleep(self.physics_interval)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"Physics loop error: {e}")

    async def _metric_loop(self):
        try:
            while self._running:
                self.physics_metrics.mpi = self._mpi(self.physics_history)
                self.semantic_metrics.mpi = self._mpi(self.semantic_history)
                self._update_channel_states()
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"Metric loop error: {e}")

    # ---- helpers ----

    def _mpi(self, history: Deque[Any]) -> float:
        if len(history) < 10:
            return 0.0
        recent = list(history)[-10:]
        intervals = []
        for i in range(1, len(recent)):
            t1 = getattr(recent[i], "timestamp", None)
            t0 = getattr(recent[i-1], "timestamp", None)
            if t1 is not None and t0 is not None:
                intervals.append(t1 - t0)
        if not intervals:
            return 0.0
        mean_interval = max(1e-3, mean(intervals))
        sigma = pstdev(intervals) if len(intervals) > 1 else 0.0
        regularity = 1.0 / (1.0 + sigma / mean_interval)
        return float(min(regularity, 1.0))

    def _update_channel_states(self):
        # Physics channel: becomes DEGRADED if irregular
        if self.physics_metrics.mpi > self.mpi_threshold:
            self.physics_state = (ChannelState.RECOVERING
                                  if self.physics_state == ChannelState.RECOVERING
                                  else ChannelState.ACTIVE)
        else:
            self.physics_state = ChannelState.DEGRADED

        # Semantic channel: only fail if we've ever used it
        if self._semantic_ever:
            if time.time() - self.semantic_metrics.last_successful > self.semantic_timeout:
                self.semantic_state = ChannelState.FAILED
            elif self.semantic_metrics.mpi > self.mpi_threshold:
                self.semantic_state = ChannelState.ACTIVE
            else:
                self.semantic_state = ChannelState.DEGRADED

    async def _send_physics_signal(self, signal: PhysicsSignal):
        start = time.time()
        try:
            self.physics_history.append(signal)
            for handler in self.physics_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(signal)
                    else:
                        handler(signal)
                except Exception as e:
                    print(f"Physics handler error: {e}")
                    self.physics_metrics.error_rate += 0.01
            self.physics_metrics.latency = time.time() - start
            self.physics_metrics.last_successful = time.time()
            first_ts = self.physics_history[0].timestamp if self.physics_history else time.time()
            span = max(1.0, time.time() - first_ts)
            self.physics_metrics.throughput = len(self.physics_history) / span
        except Exception as e:
            print(f"Physics signal send error: {e}")
            self.physics_metrics.error_rate += 0.05

    async def send_semantic_signal(self, signal: SemanticSignal):
        if self.semantic_state == ChannelState.FAILED:
            raise RuntimeError("Semantic channel failed - using physics only")

        start = time.time()
        signal.sequence_id = self.semantic_seq
        try:
            self.semantic_history.append(signal)
            for handler in self.semantic_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(signal)
                    else:
                        handler(signal)
                except Exception as e:
                    print(f"Semantic handler error: {e}")
                    self.semantic_metrics.error_rate += 0.01
            self.semantic_metrics.latency = time.time() - start
            self.semantic_metrics.last_successful = time.time()
            self.semantic_seq += 1
            first_ts = self.semantic_history[0].timestamp if self.semantic_history else time.time()
            span = max(1.0, time.time() - first_ts)
            self.semantic_metrics.throughput = len(self.semantic_history) / span
            self._semantic_ever = True
        except Exception as e:
            print(f"Semantic signal send error: {e}")
            self.semantic_metrics.error_rate += 0.05
            if self.semantic_state == ChannelState.ACTIVE:
                self.semantic_state = ChannelState.DEGRADED

    # --- test-string + compression (simple RLE) ---

    def _generate_test_string(self) -> str:
        patterns = [
            "ABCABC" * randint(3, 7),
            "Hello world! " * randint(2, 4),
            "".join(choice("ABC") for _ in range(randint(10, 30))),
            "The quick brown fox jumps over the lazy dog. " * randint(1, 2),
        ]
        return choice(patterns)

    def _compress_string(self, s: str) -> str:
        if not s:
            return s
        out: List[str] = []
        cur = s[0]; count = 1
        for ch in s[1:]:
            if ch == cur:
                count += 1
            else:
                out.append(f"{cur}{count}" if count > 3 else cur * count)
                cur, count = ch, 1
        out.append(f"{cur}{count}" if count > 3 else cur * count)
        return "".join(out)

    # --- public status ---

    def get_channel_status(self) -> Dict[str, Any]:
        return {
            "physics": {
                "state": self.physics_state.value,
                "mpi": round(self.physics_metrics.mpi, 3),
                "throughput": round(self.physics_metrics.throughput, 3),
                "error_rate": round(self.physics_metrics.error_rate, 3),
            },
            "semantic": {
                "state": self.semantic_state.value,
                "mpi": round(self.semantic_metrics.mpi, 3),
                "throughput": round(self.semantic_metrics.throughput, 3),
                "error_rate": round(self.semantic_metrics.error_rate, 3),
            },
        }
