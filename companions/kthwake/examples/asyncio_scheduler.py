"""Run a live processing-time scheduler with one timer for a stream of agent results.

Install the local Python package, then run python3 examples/asyncio_scheduler.py.
Each arrivals(n) call is a separate atomic batch. The callback receives logical
times in nanoseconds; a busy event loop can deliver callbacks later.
"""
import asyncio
import math
from kthwake import Index


class WakeStream:
    def __init__(self, rules, on_change):
        self.loop = asyncio.get_running_loop()
        self.epoch = self.loop.time()
        self.index = Index(rules)
        self.on_change = on_change
        self.timer = None
        self.armed = None

    def now(self):
        return math.floor((self.loop.time() - self.epoch) * 1_000_000_000)

    def _step(self, now, count):
        for rule, active in self.index.step(now, count):
            self.on_change(now, rule, active)

    def _update(self, now, count):
        # Expiries strictly before the incoming tick belong to earlier snapshots.
        deadline = self.index.next_expiry()
        while deadline is not None and deadline < now:
            self._step(deadline, 0)
            deadline = self.index.next_expiry()
        self._step(now, count)
        deadline = self.index.next_expiry()
        if deadline == self.armed:
            return
        self.close()
        if deadline is not None:
            self.armed = deadline
            self.timer = self.loop.call_at(self.epoch + deadline / 1_000_000_000, self._idle)

    def arrivals(self, count=1):
        self._update(self.now(), count)

    def _idle(self):
        self.timer = self.armed = None
        self._update(self.now(), 0)

    def close(self):
        if self.timer is not None:
            self.timer.cancel()
        self.timer = self.armed = None


async def main():
    def on_change(t, rule, active):
        print(f"{t / 1_000_000:9.3f} ms  guard {rule}  active={active}")

    stream = WakeStream([(3, 20_000_000), (1, 5_000_000)], on_change)
    try:
        stream.arrivals(2)
        await asyncio.sleep(0.002)
        stream.arrivals()
        await asyncio.sleep(0.025)
    finally:
        stream.close()


if __name__ == "__main__":
    asyncio.run(main())
