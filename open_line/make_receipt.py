# open_line/make_receipt.py
import asyncio, time
from .dual_channel import DualChannelProtocol, NarrativeSignal
from .olp_adapter import snapshot_frame, build_receipt, write_receipt

async def main():
    proto = DualChannelProtocol(physics_interval=0.2)
    await proto.start()

    # Feed a couple semantic events so the physics pointer has context
    await proto.send_semantic_signal(NarrativeSignal("Booting dual-channel proto"))
    await asyncio.sleep(1.0)
    await proto.send_semantic_signal(NarrativeSignal("Warm start sequence ready"))

    # Let it run briefly
    await asyncio.sleep(3.0)

    # Snapshot -> frame -> receipt
    frame = snapshot_frame(proto, claim="Dual-channel link is steady")
    receipt = build_receipt(frame)
    path = write_receipt(receipt)
    print("[ok] wrote", path)

    await proto.stop()

if __name__ == "__main__":
    asyncio.run(main())
