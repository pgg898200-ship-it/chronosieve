"""Turn repeated ChronoSieve detector alarms into exact event-time incident updates."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from itertools import groupby
from operator import itemgetter

from chronosieve import EDetector
from kthwake import Index

# Each row supplies an incident name, an alarm count, and a window in milliseconds.
POLICIES = (("watch", 1, 3), ("escalate", 3, 9))


def incident_updates(observations: Iterable[tuple[int, float]]) -> Iterator[dict]:
    """Replay (milliseconds, residual) in order and drain the last active incidents."""
    detector = EDetector(family="catoni", side="upper", mu0=0.0, sigma=1.0,
                         threshold=20.0, auto_reset=True)
    index = Index([(count, window) for _, count, window in POLICIES])

    def apply(time_ms: int, alarms: int) -> Iterator[dict]:
        for guard_id, active in index.step(time_ms, alarms):
            yield {"t_ms": time_ms, "policy": POLICIES[guard_id][0], "active": active}

    def checked_observations() -> Iterator[tuple[int, float]]:
        previous = 0
        for row_number, (time_ms, residual) in enumerate(observations, start=1):
            if type(time_ms) is not int or not previous <= time_ms < 2**64:
                raise ValueError(f"observation {row_number}: timestamps must be nondecreasing "
                                 "unsigned 64-bit milliseconds")
            previous = time_ms
            yield time_ms, residual

    for time_ms, readings in groupby(checked_observations(), key=itemgetter(0)):
        # Idle deadlines tied with new input belong to the same snapshot.
        while (deadline := index.next_expiry()) is not None and deadline < time_ms:
            yield from apply(deadline, 0)
        alarms = sum(detector.step(residual)["alarmed"] for _, residual in readings)
        yield from apply(time_ms, alarms)

    while (deadline := index.next_expiry()) is not None:
        yield from apply(deadline, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdin", action="store_true",
                        help='read newline-delimited {"t_ms": 0, "residual": 0.0} observations')
    args = parser.parse_args()
    if args.stdin:
        rows = (json.loads(line) for line in sys.stdin if line.strip())
        series = ((row["t_ms"], row["residual"]) for row in rows)
    else:
        series = enumerate([0.0] * 5 + [3.0] * 11)
    try:
        for update in incident_updates(series):
            print(json.dumps(update))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
