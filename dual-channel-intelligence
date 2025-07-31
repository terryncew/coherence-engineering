“””
Dual Channel Protocol: Physics + Semantics Communication
Implements two-channel handshake following the microbe signaling metaphor

Physics Channel: rhythmic pings, compression games, pointing (always-on)
Semantic Channel: narratives, instructions, tool calls (opportunistic)

Maintains MPI (Mutual Predictive Information) across both channels
“””

import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, AsyncGenerator
from enum import Enum
import json
from collections import deque
import threading

class ChannelState(Enum):
ACTIVE = “active”
DEGRADED = “degraded”
FAILED = “failed”
RECOVERING = “recovering”

@dataclass
class PhysicsSignal:
“”“Low-bandwidth physics channel signals”””
type: str  # “rhythm”, “compression”, “pointing”
value: Any
timestamp: float = field(default_factory=time.time)
sequence_id: int = 0

class RhythmSignal(PhysicsSignal):
“”“Rhythmic heartbeat signal”””
def **init**(self, bpm: float = 60.0, sequence_id: int = 0):
super().**init**(“rhythm”, {“bpm”: bpm, “phase”: time.time() % (60/bpm)}, sequence_id=sequence_id)

class CompressionSignal(PhysicsSignal):
“”“Compression game signal”””
def **init**(self, original: str, compressed: str, ratio: float, sequence_id: int = 0):
super().**init**(“compression”, {
“original_length”: len(original),
“compressed_length”: len(compressed),
“ratio”: ratio,
“entropy”: self._calculate_entropy(original)
}, sequence_id=sequence_id)

```
def _calculate_entropy(self, text: str) -> float:
    """Simple entropy calculation"""
    chars = set(text)
    length = len(text)
    return -sum((text.count(c)/length) * np.log2(text.count(c)/length) for c in chars)
```

class PointingSignal(PhysicsSignal):
“”“Pointing/attention signal”””
def **init**(self, target: str, confidence: float, context: Dict[str, Any] = None, sequence_id: int = 0):
super().**init**(“pointing”, {
“target”: target,
“confidence”: confidence,
“context”: context or {}
}, sequence_id=sequence_id)

@dataclass
class SemanticSignal:
“”“High-bandwidth semantic channel signal”””
type: str  # “narrative”, “instruction”, “tool_call”, “query”
content: str
metadata: Dict[str, Any] = field(default_factory=dict)
timestamp: float = field(default_factory=time.time)
sequence_id: int = 0

class NarrativeSignal(SemanticSignal):
“”“Rich narrative communication”””
def **init**(self, content: str, narrative_type: str = “general”, sequence_id: int = 0):
super().**init**(“narrative”, content, {
“narrative_type”: narrative_type,
“word_count”: len(content.split()),
“complexity”: self._estimate_complexity(content)
}, sequence_id=sequence_id)

```
def _estimate_complexity(self, text: str) -> float:
    """Rough complexity estimate"""
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    return avg_word_len * (len(words) / max(sentence_count, 1))
```

@dataclass
class ChannelMetrics:
“”“Metrics for each communication channel”””
mpi: float = 0.0  # Mutual Predictive Information
latency: float = 0.0
error_rate: float = 0.0
throughput: float = 0.0
signal_quality: float = 1.0
last_successful: float = field(default_factory=time.time)

class DualChannelProtocol:
“””
Dual-channel communication protocol with physics + semantic layers
“””

```
def __init__(self, 
             physics_interval: float = 1.0,  # Physics heartbeat interval
             semantic_timeout: float = 10.0,  # Semantic channel timeout
             mpi_threshold: float = 0.3):     # Minimum MPI to stay active
    
    self.physics_interval = physics_interval
    self.semantic_timeout = semantic_timeout
    self.mpi_threshold = mpi_threshold
    
    # Channel states
    self.physics_state = ChannelState.ACTIVE
    self.semantic_state = ChannelState.ACTIVE
    
    # Metrics
    self.physics_metrics = ChannelMetrics()
    self.semantic_metrics = ChannelMetrics()
    
    # Signal histories for MPI calculation
    self.physics_history = deque(maxlen=100)
    self.semantic_history = deque(maxlen=50)
    
    # Sequence counters
    self.physics_seq = 0
    self.semantic_seq = 0
    
    # Control
    self._running = False
    self._physics_task = None
    self._metric_task = None
    
    # Callbacks
    self.physics_handlers: List[Callable] = []
    self.semantic_handlers: List[Callable] = []
    
async def start(self):
    """Start both communication channels"""
    self._running = True
    self._physics_task = asyncio.create_task(self._physics_loop())
    self._metric_task = asyncio.create_task(self._metric_loop())
    
async def stop(self):
    """Stop communication channels"""
    self._running = False
    if self._physics_task:
        self._physics_task.cancel()
    if self._metric_task:
        self._metric_task.cancel()
        
def register_physics_handler(self, handler: Callable[[PhysicsSignal], None]):
    """Register handler for physics signals"""
    self.physics_handlers.append(handler)
    
def register_semantic_handler(self, handler: Callable[[SemanticSignal], None]):
    """Register handler for semantic signals"""
    self.semantic_handlers.append(handler)
    
async def _physics_loop(self):
    """Continuous physics channel loop"""
    while self._running:
        try:
            # Generate rhythmic heartbeat
            rhythm = RhythmSignal(bpm=60 + np.random.normal(0, 5), sequence_id=self.physics_seq)
            await self._send_physics_signal(rhythm)
            
            # Occasional compression games
            if np.random.random() < 0.3:
                test_string = self._generate_test_string()
                compressed = self._compress_string(test_string)
                compression = CompressionSignal(
                    test_string, compressed, 
                    len(compressed) / len(test_string),
                    sequence_id=self.physics_seq
                )
                await self._send_physics_signal(compression)
            
            # Pointing signals based on recent activity
            if len(self.semantic_history) > 0:
                recent_semantic = self.semantic_history[-1]
                pointing = PointingSignal(
                    target=recent_semantic.type,
                    confidence=0.8,
                    context={"last_semantic": recent_semantic.content[:50]},
                    sequence_id=self.physics_seq
                )
                await self._send_physics_signal(pointing)
            
            self.physics_seq += 1
            await asyncio.sleep(self.physics_interval)
            
        except Exception as e:
            print(f"Physics loop error: {e}")
            self.physics_state = ChannelState.DEGRADED
            await asyncio.sleep(self.physics_interval * 2)  # Back off
            
async def _metric_loop(self):
    """Calculate and update channel metrics"""
    while self._running:
        try:
            # Update MPI for both channels
            self.physics_metrics.mpi = self._calculate_mpi(self.physics_history)
            self.semantic_metrics.mpi = self._calculate_mpi(self.semantic_history)
            
            # Update channel states based on MPI
            self._update_channel_states()
            
            await asyncio.sleep(2.0)  # Update metrics every 2 seconds
            
        except Exception as e:
            print(f"Metric loop error: {e}")
            await asyncio.sleep(5.0)
            
def _calculate_mpi(self, signal_history: deque) -> float:
    """Calculate Mutual Predictive Information for a signal sequence"""
    if len(signal_history) < 10:
        return 0.0
        
    # Simplified MPI calculation
    # In practice, this would be more sophisticated
    recent_signals = list(signal_history)[-10:]
    
    # Look for predictable patterns
    intervals = []
    for i in range(1, len(recent_signals)):
        if hasattr(recent_signals[i], 'timestamp') and hasattr(recent_signals[i-1], 'timestamp'):
            intervals.append(recent_signals[i].timestamp - recent_signals[i-1].timestamp)
    
    if not intervals:
        return 0.0
        
    # Higher MPI for more regular intervals
    interval_std = np.std(intervals) if len(intervals) > 1 else 0
    mean_interval = np.mean(intervals)
    regularity = 1.0 / (1.0 + interval_std / max(mean_interval, 0.001))
    
    return min(regularity, 1.0)

def _update_channel_states(self):
    """Update channel states based on current metrics"""
    # Physics channel state
    if self.physics_metrics.mpi > self.mpi_threshold:
        if self.physics_state in [ChannelState.DEGRADED, ChannelState.RECOVERING]:
            self.physics_state = ChannelState.RECOVERING
        else:
            self.physics_state = ChannelState.ACTIVE
    else:
        self.physics_state = ChannelState.DEGRADED
        
    # Semantic channel state
    time_since_last = time.time() - self.semantic_metrics.last_successful
    if time_since_last > self.semantic_timeout:
        self.semantic_state = ChannelState.FAILED
    elif self.semantic_metrics.mpi > self.mpi_threshold:
        self.semantic_state = ChannelState.ACTIVE
    else:
        self.semantic_state = ChannelState.DEGRADED
        
async def _send_physics_signal(self, signal: PhysicsSignal):
    """Send a physics signal and update metrics"""
    start_time = time.time()
    
    try:
        # Add to history
        self.physics_history.append(signal)
        
        # Call handlers
        for handler in self.physics_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(signal)
                else:
                    handler(signal)
            except Exception as e:
                print(f"Physics handler error: {e}")
                self.physics_metrics.error_rate += 0.01
                
        # Update metrics
        self.physics_metrics.latency = time.time() - start_time
        self.physics_metrics.last_successful = time.time()
        self.physics_metrics.throughput = len(self.physics_history) / max(time.time() - (self.physics_history[0].timestamp if self.physics_history else time.time()), 1)
        
    except Exception as e:
        print(f"Physics signal send error: {e}")
        self.physics_metrics.error_rate += 0.05
        
async def send_semantic_signal(self, signal: SemanticSignal):
    """Send a semantic signal"""
    if self.semantic_state == ChannelState.FAILED:
        raise Exception("Semantic channel failed - using physics only")
        
    start_time = time.time()
    signal.sequence_id = self.semantic_seq
    
    try:
        # Add to history
        self.semantic_history.append(signal)
        
        # Call handlers
        for handler in self.semantic_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(signal)
                else:
                    handler(signal)
            except Exception as e:
                print(f"Semantic handler error: {e}")
                self.semantic_metrics.error_rate += 0.01
                
        # Update metrics
        self.semantic_metrics.latency = time.time() - start_time
        self.semantic_metrics.last_successful = time.time()
        self.semantic_metrics.throughput = len(self.semantic_history) / max(time.time() - (self.semantic_history[0].timestamp if self.semantic_history else time.time()), 1)
        
        self.semantic_seq += 1
        
    except Exception as e:
        print(f"Semantic signal send error: {e}")
        self.semantic_metrics.error_rate += 0.05
        if self.semantic_state == ChannelState.ACTIVE:
            self.semantic_state = ChannelState.DEGRADED
            
def _generate_test_string(self) -> str:
    """Generate test string for compression games"""
    patterns = [
        "ABCABC" * np.random.randint(3, 8),
        "Hello world! " * np.random.randint(2, 5),
        "".join(np.random.choice(['A', 'B', 'C'], size=np.random.randint(10, 30))),
        "The quick brown fox jumps over the lazy dog. " * np.random.randint(1, 3)
    ]
    return np.random.choice(patterns)

def _compress_string(self, s: str) -> str:
    """Simple compression simulation"""
    # Run-length encoding simulation
    if not s:
        return s
        
    compressed = []
    current_char = s[0]
    count = 1
    
    for char in s[1:]:
        if char == current_char:
            count += 1
        else:
            if count > 3:
                compressed.append(f"{current_char}{count}")
            else:
                compressed.append(current_char * count)
            current_char = char
            count = 1
            
    if count > 3:
        compressed.append(f"{current_char}{count}")
    else:
        compressed.append(current_char * count)
        
    return "".join(compressed)

def get_channel_status(self) -> Dict[str, Any]:
    """Get current status of both channels"""
    return {
```
