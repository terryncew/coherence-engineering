# examples/tri_modal_quickstart.py
import asyncio
import numpy as np

from runtime.tri_modal_controller import TriModalController, BoundarySignals
from runtime.lungs import ContextRetrievalLung


async def main():
    controller = TriModalController(ctx_lung=ContextRetrievalLung())

    test_requests = [
        "Summarize the main differences between supervised and reinforcement learning.",
        "Draft a safe medical-style explanation about fever in children (non-diagnostic).",
        "Plan a 3-step troubleshooting guide for a failing unit test in Python.",
    ]

    for i, req in enumerate(test_requests, 1):
        # Simulate telemetry (tweak these to see the mixer move)
        signals = BoundarySignals(
            kappa_hat=float(np.clip(np.random.beta(2, 5), 0, 1)),     # stress
            epsilon=float(np.clip(np.random.beta(1, 4), 0, 1)),       # entropy leak
            contradiction_rate=float(np.clip(np.random.beta(1, 10), 0, 1)),
            vkd=float(np.clip(np.random.beta(5, 2), 0, 1)),           # >0 safe
        )

        # Demo context a retrieval lung could use
        context = {"retrieved_snippet": "Cached: RL uses reward signals; SL uses labeled pairs."}

        mixed, weights = await controller.route_request(req, signals, context=context)
        print(f"\n--- Request {i} ---")
        print(f"κ={signals.kappa_hat:.2f}, ε={signals.epsilon:.2f}, VKD={signals.vkd:.2f}, Φ*={signals.phi_star:.3f}")
        print(f"Weights (fast, slow, ctx): {weights}")
        print(f"Mode: {mixed.mode}")
        print(f"Output: {mixed.content[:180]}...")
        print(f"Confidence: {mixed.confidence:.2f}, Time: {mixed.processing_time:.2f}s, Tokens: {mixed.tokens_used}")

if __name__ == "__main__":
    asyncio.run(main())
