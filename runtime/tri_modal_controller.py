# runtime/tri_modal_controller.py
# A 2–3 lung controller that:
# - runs fast & slow lungs in parallel (always)
# - optionally runs a 3rd Context/Retrieval lung
# - uses a simple Φ* / κ / ε heuristic to compute weights
# - mixes outputs based on weights
#
# Keep it simple and copy-pasteable.

from __future__ import annotations
import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from .lungs import Lung, LungOutput, FastReflexLung, SlowDeliberativeLung, ContextRetrievalLung


@dataclass
class BoundarySignals:
    # Core Bridge-Equation inputs
    kappa_hat: float = 0.0   # stress / curvature proxy (0–1)
    epsilon: float = 0.0     # entropy leak proxy (0–1)
    contradiction_rate: float = 0.0  # (0–1)
    vkd: float = 1.0         # >0 safe, <0 danger
    phi_star: float = 0.0    # computed (coherence per cost)
    timestamp: float = field(default_factory=time.time)

    # optional telemetry (not strictly used in control below)
    token_rate: float = 0.0
    queue_depth: int = 0


class TriModalController:
    """
    Controller for 2–3 lungs (reflex, deliberative, context).
    If ctx_lung is None, it degenerates cleanly to a bi-modal setup.
    """

    def __init__(
        self,
        fast_lung: Optional[Lung] = None,
        slow_lung: Optional[Lung] = None,
        ctx_lung: Optional[Lung] = None,
        alpha_kappa: float = 0.5,
        alpha_epsilon: float = 0.3,
        delta: float = 0.1,
        softmax_temp: float = 1.2,
    ):
        self.fast_lung = fast_lung or FastReflexLung()
        self.slow_lung = slow_lung or SlowDeliberativeLung()
        self.ctx_lung = ctx_lung  # pass ContextRetrievalLung() to enable 3rd lung

        self.alpha_kappa = alpha_kappa
        self.alpha_epsilon = alpha_epsilon
        self.delta = delta
        self.softmax_temp = softmax_temp

        self._last_phi_star: float = 0.0

    # ----- Φ* (Bridge Equation) -----
    def compute_phi_star(self, signals: BoundarySignals) -> float:
        # I_c proxy: less contradiction => more coherence capacity
        I_c = max(0.0, 1.0 - signals.contradiction_rate)
        denom = 1.0 + self.alpha_kappa * signals.kappa_hat + self.alpha_epsilon * signals.epsilon + self.delta
        phi_star = I_c / denom
        return float(np.clip(phi_star, 0.0, 1.0))

    # ----- simple weight policy (2 or 3 lungs) -----
    def _score_lungs(self, s: BoundarySignals) -> List[float]:
        """
        Compute unnormalized scores for [fast, slow, (ctx?)].
        Heuristic:
          - More stress (κ) => favor fast
          - More entropy (ε) => damp slow a bit
          - Higher VKD (<0 danger) => strongly favor fast, suppress ctx/slow
          - Rising Φ* can afford more slow/ctx
        """
        # baseline scores
        score_fast = 0.5
        score_slow = 0.5
        score_ctx = 0.3 if self.ctx_lung is not None else None

        # stress pushes toward fast
        score_fast += 0.8 * s.kappa_hat
        score_slow -= 0.6 * s.kappa_hat

        # entropy leak penalizes slow
        score_slow -= 0.5 * s.epsilon

        # VKD danger penalizes slow & ctx, boosts fast
        if s.vkd < 0.0:
            score_fast += 1.2
            score_slow -= 1.0
            if score_ctx is not None:
                score_ctx -= 1.0

        # better φ* allows slower/contextual lungs to contribute
        dphi = s.phi_star - self._last_phi_star
        score_slow += 0.6 * max(0.0, dphi)
        if score_ctx is not None:
            # only let context contribute when relatively safe
            safe_bonus = max(0.0, s.vkd) * 0.4 + max(0.0, dphi) * 0.3
            score_ctx += safe_bonus
            # context slightly penalized by stress
            score_ctx -= 0.4 * s.kappa_hat

        scores = [score_fast, score_slow]
        if score_ctx is not None:
            scores.append(score_ctx)
        return scores

    def _softmax(self, xs: List[float], temp: float) -> List[float]:
        xs = np.array(xs, dtype=np.float64)
        xs = xs / max(1e-6, temp)
        xs = xs - xs.max()  # numerical stability
        exps = np.exp(xs)
        probs = exps / np.clip(exps.sum(), 1e-9, None)
        return probs.tolist()

    def _mix_contents(self, parts: List[Tuple[float, LungOutput]]) -> LungOutput:
        """
        Simple textual mixing: if one weight dominates, prefer that content;
        else return a labeled blend. Confidence = weighted sum.
        """
        parts = sorted(parts, key=lambda p: p[0], reverse=True)
        weights = [w for w, _ in parts]
        outputs = [o for _, o in parts]
        top_w, top_out = parts[0]

        # If top weight is strong enough, return that output
        if top_w >= 0.66:
            return LungOutput(
                content=top_out.content,
                confidence=sum(w * o.confidence for w, o in parts),
                tokens_used=sum(o.tokens_used for o in outputs),
                processing_time=max(o.processing_time for o in outputs),
                mode=f"tri_modal_top:{top_out.mode}",
                metadata={"weights": weights, "modes": [o.mode for o in outputs]},
            )

        # Otherwise, return a readable blend
        blended = " | ".join(f"{w:.2f}:{o.content}" for w, o in parts)
        return LungOutput(
            content=f"[BLEND] {blended}",
            confidence=sum(w * o.confidence for w, o in parts),
            tokens_used=sum(o.tokens_used for o in outputs),
            processing_time=max(o.processing_time for o in outputs),
            mode="tri_modal_blend",
            metadata={"weights": weights, "modes": [o.mode for o in outputs]},
        )

    async def route_request(
        self,
        request: str,
        signals: BoundarySignals,
        context: Optional[Dict[str, Any]] = None,
        slow_timeout: float = 8.0,
        ctx_timeout: float = 3.0,
    ) -> Tuple[LungOutput, List[float]]:
        """
        Run 2–3 lungs concurrently, compute weights from signals, and mix outputs.
        Returns (mixed_output, weights).
        """
        context = context or {}
        # compute φ* and remember last for dφ
        signals.phi_star = self.compute_phi_star(signals)
        weights: List[float] = [0.0, 0.0] if self.ctx_lung is None else [0.0, 0.0, 0.0]

        # emergency damper: if in danger, only fast
        if signals.vkd < 0.0:
            fast_out = await self.fast_lung.run(request, context)
            self._last_phi_star = signals.phi_star
            return fast_out, [1.0, 0.0] if self.ctx_lung is None else [1.0, 0.0, 0.0]

        # launch lungs
        tasks = {
            "fast": asyncio.create_task(self.fast_lung.run(request, context)),
            "slow": asyncio.create_task(self.slow_lung.run(request, context)),
        }
        if self.ctx_lung is not None:
            tasks["ctx"] = asyncio.create_task(self.ctx_lung.run(request, context))

        # await fast (should be quick)
        fast_out = await tasks["fast"]

        # await slow with timeout
        try:
            slow_out = await asyncio.wait_for(tasks["slow"], timeout=slow_timeout)
        except asyncio.TimeoutError:
            slow_out = LungOutput(
                content="[SLOW][TIMEOUT]",
                confidence=0.0,
                tokens_used=0,
                processing_time=slow_timeout,
                mode="slow_timeout",
            )

        # await ctx with timeout (if present)
        ctx_out: Optional[LungOutput] = None
        if "ctx" in tasks:
            try:
                ctx_out = await asyncio.wait_for(tasks["ctx"], timeout=ctx_timeout)
            except asyncio.TimeoutError:
                ctx_out = LungOutput(
                    content="[CTX][TIMEOUT]",
                    confidence=0.0,
                    tokens_used=0,
                    processing_time=ctx_timeout,
                    mode="context_timeout",
                )

        # compute weights from signals
        scores = self._score_lungs(signals)
        weights = self._softmax(scores, temp=self.softmax_temp)

        # degrade weights if outputs timed out
        if slow_out.mode.endswith("timeout"):
            if len(weights) == 2:
                weights = [1.0, 0.0]
            else:
                weights = [min(1.0, weights[0] + 0.3), 0.0, weights[2] * 0.5]
                s = sum(weights)
                weights = [w / s for w in weights]

        if ctx_out is not None and ctx_out.mode.endswith("timeout"):
            if len(weights) == 3:
                weights[2] = 0.0
                s = max(1e-6, weights[0] + weights[1])
                weights[0] /= s
                weights[1] /= s

        # build parts and mix
        parts: List[Tuple[float, LungOutput]] = [(weights[0], fast_out), (weights[1], slow_out)]
        if ctx_out is not None:
            parts.append((weights[2], ctx_out))

        mixed = self._mix_contents(parts)
        self._last_phi_star = signals.phi_star
        return mixed, weights
