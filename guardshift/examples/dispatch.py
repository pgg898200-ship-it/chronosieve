"""Batch two telemetry writes before dispatching an agent."""
import asyncio
from guardshift import Index


async def worker(inbox):
    event = await inbox.get()
    print("batch worker received", event)


async def main():
    # Start the batch worker when queue depth >= 100 and GPU usage <= 70%.
    guards = [[(0, 100, 2**63 - 1), (1, 0, 70)]]
    index = Index([95, 60], guards)
    inbox = asyncio.Queue()
    task = asyncio.create_task(worker(inbox))

    for patch in [[(0, 101), (1, 85)], [(1, 65)]]:
        for guard_id, active in index.apply(patch):
            if active:
                await inbox.put({"guard_id": guard_id, "patch": patch})
    await task


if __name__ == "__main__":
    asyncio.run(main())
