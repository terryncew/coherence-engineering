# open_line/dual_channel.py
import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Deque
from enum import Enum
from collections import deque

class ChannelState(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"

@dataclass
class PhysicsSignal:
    """Low-bandwidth physics channel signals"""
    type: str  # "rhythm", "compression", "pointing"
    value: Any
    timestamp: float = field(default_factory=time.time)
    sequence_id: int = 0

class RhythmSignal(PhysicsSignal):
    def __init__(self, bpm: float = 60.0, sequence_id: int = 0):
        phase = (time.time() % max(0.001, (60.0 / max(0.001, bpm))))
        super().__init__("rhythm", {"bpm": bpm, "phase": phase}, sequence_id=sequence_id)

class CompressionSignal(PhysicsSignal):
    def __init__(self, original: str, compressed: str, ratio: float, sequence_id: int = 0):
        super().__init__("compression", {
            "original_length": len(original),
            "compressed_length": len(compressed),
            "ratio": ratio,
            "entropy": self._calculate_entropy(original)
        }, sequence_id=sequence_id)

    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        length = len(text)
        probs = [text.count(c) / length for c in set(text)]
        return float(-sum(p * np.log2(p) for p in probs if p > 0))

class PointingSignal(PhysicsSignal):
    def __init__(self, target: str, confidence: float, context: Optional[Dict[str, Any]] = None, sequence_id: int = 0):
        super().__init__("pointing", {
            "target": target,
            "confidence": float(confidence),
            "context": context or {}
        }, sequence_id=sequence_id)

@dataclass
class SemanticSignal:
    """High-bandwidth semantic channel signals"""
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
            "complexity": self._estimate_complexity(content)
        }, sequence_id=sequence_id)

    def _estimate_complexity(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        avg_word_len = float(np.mean([len(w) for w in words]))
        sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
        return float(avg_word_len * (len(words) / sentence_count))

@dataclass
class ChannelMetrics:
    """Metrics for each communication channel"""
    mpi: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    signal_quality: float = 1.0
    last_successful: float = field(default_factory=time.time)

class DualChannelProtocol:
    """Dual-channel communication protocol with physics + semantic layers"""
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

        self.physics_handlers: List[Callable[[PhysicsSignal], Any]] = []
        self.semantic_handlers: List[Callable[[SemanticSignal], Any]] = []

    async def start(self):
        self._running = True
        self._physics_task = asyncio.create_task(self._physics_loop())
        self._metric_task = asyncio.create_task(self._metric_loop())

    async def stop(self):
        self._running = False
        if self._physics_task:
            self._physics_task.cancel()
        if self._metric_task:
            self._metric_task.cancel()

    def register_physics_handler(self, handler: Callable[[PhysicsSignal], Any]):
        self.physics_handlers.append(handler)

    def register_semantic_handler(self, handler: Callable[[SemanticSignal], Any]):
        self.semantic_handlers.append(handler)

    async def _physics_loop(self):
        while self._running:
            try:
                rhythm = RhythmSignal(bpm=60 + float(np.random.normal(0, 5)), sequence_id=self.physics_seq)
                await self._send_physics_signal(rhythm)

                if np.random.random() < 0.3:
                    test_string = self._generate_test_string()
                    compressed = self._compress_string(test_string)
                    compression = CompressionSignal(
                        test_string, compressed,
                        len(compressed) / max(1, len(test_string)),
                        sequence_id=self.physics_seq
                    )
                    await self._send_physics_signal(compression)

                if len(self.semantic_history) > 0:
                    recent_sem = self.semantic_history[-1]
                    pointing = PointingSignal(
                        target=recent_sem.type,
                        confidence=0.8,
                        context={"last_semantic": recent_sem.content[:50]},
                        sequence_id=self.physics_seq
                    )
                    await self._send_physics_signal(pointing)

                self.physics_seq += 1
                await asyncio.sleep(self.physics_interval)

            except Exception as e:
                print(f"Physics loop error: {e}")
                self.physics_state = ChannelState.DEGRADED
                await asyncio.sleep(self.physics_interval * 2)

    async def _metric_loop(self):
        while self._running:
            try:
                self.physics_metrics.mpi = self._calculate_mpi(self.physics_history)
                self.semantic_metrics.mpi = self._calculate_mpi(self.semantic_history)
                self._update_channel_states()
                await asyncio.sleep(2.0)
            except Exception as e:
                print(f"Metric loop error: {e}")
                await asyncio.sleep(5.0)

    def _calculate_mpi(self, signal_history: Deque[Any]) -> float:
        if len(signal_history) < 10:
            return 0.0
        recent = list(signal_history)[-10:]
        intervals = []
        for i in range(1, len(recent)):
            t1 = getattr(recent[i], "timestamp", None)
            t0 = getattr(recent[i-1], "timestamp", None)
            if t1 is not None and t0 is not None:
                intervals.append(t1 - t0)
        if not intervals:
            return 0.0
        interval_std = np.std(intervals) if len(intervals) > 1 else 0.0
        mean_interval = max(1e-3, float(np.mean(intervals)))
        regularity = 1.0 / (1.0 + interval_std / mean_interval)
        return float(min(regularity, 1.0))

    def _update_channel_states(self):
        # Physics
        if self.physics_metrics.mpi > self.mpi_threshold:
            self.physics_state = ChannelState.ACTIVE if self.physics_state != ChannelState.RECOVERING else ChannelState.RECOVERING
        else:
            self.physics_state = ChannelState.DEGRADED

        # Semantics
        time_since_last = time.time() - self.semantic_metrics.last_successful
        if time_since_last > self.semantic_timeout:
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
            first_ts = self.semantic_history[0].timestamp if self.semantic_history else time.time()
            span = max(1.0, time.time() - first_ts)
            self.semantic_metrics.throughput = len(self.semantic_history) / span
            self.semantic_seq += 1
        except Exception as e:
            print(f"Semantic signal send error: {e}")
            self.semantic_metrics.error_rate += 0.05
            if self.semantic_state == ChannelState.ACTIVE:
                self.semantic_state = ChannelState.DEGRADED

    def _generate_test_string(self) -> str:
        patterns = [
            "ABCABC" * int(np.random.randint(3, 8)),
            "Hello world! " * int(np.random.randint(2, 5)),
            "".join(np.random.choice(list("ABC"), size=int(np.random.randint(10, 30)))),
            "The quick brown fox jumps over the lazy dog. " * int(np.random.randint(1, 3))
        ]
        return str(np.random.choice(patterns))

    def _compress_string(self, s: str) -> str:
        if not s:
            return s
        out = []
        cur = s[0]
        count = 1
        for ch in s[1:]:
            if ch == cur:
                count += 1
            else:
                out.append(f"{cur}{count}" if count > 3 else cur * count)
                cur = ch
                count = 1
        out.append(f"{cur}{count}" if count > 3 else cur * count)
        return "".join(out)

    def get_channel_status(self) -> Dict[str, Any]:
        return {
            "physics": {
                "state": self.physics_state.value,
                "mpi": self.physics_metrics.mpi,
                "throughput": self.physics_metrics.throughput,
                "error_rate": self.physics_metrics.error_rate
            },
            "semantic": {
                "state": self.semantic_state.value,
                "mpi": self.semantic_metrics.mpi,
                "throughput": self.semantic_metrics.throughput,
                "error_rate": self.semantic_metrics.error_rate
            }
        }
