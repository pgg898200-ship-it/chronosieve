"""Dispatch a detected shift only when a worker is available."""
import array

import chronosieve as cs
from guardshift import Index

samples = array.array("d", [0.0] * 1000 + [2.5] * 100)
alarms = bytearray(len(samples))
detector = cs.EDetector(family="catoni", mu0=0.0, sigma=1.0, threshold=500.0)
detector.ingest_batch(samples, out_alarms=alarms)

# Signals are [alarm on this tick, available workers]. Schedule on entry.
gate = Index([0, 1], [[(0, 1, 1), (1, 1, 64)]])
wake_ticks = []
for tick, alarm in enumerate(alarms):
    for guard_id, active in gate.apply([(0, alarm)]):
        if active:
            wake_ticks.append(tick)

assert wake_ticks and wake_ticks[0] == alarms.index(1)
print("First dispatch at tick", wake_ticks[0])
