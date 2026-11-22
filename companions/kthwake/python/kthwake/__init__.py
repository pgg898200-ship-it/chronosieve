"""Exact event-time count thresholds on one ordered stream."""
from . import _native

class Index:
    """Compile (k, window) rules. A rule ID is its zero-based input position."""
    __slots__ = ("_index",)

    def __init__(self, guards):
        self._index = _native.create(guards)

    def step(self, now, arrivals=0):
        """Apply one atomic batch and return changed (id, active) pairs."""
        return _native.step(self._index, now, arrivals)

    def next_expiry(self):
        """Earliest idle transition time, or None if no representable one exists."""
        return _native.next_expiry(self._index)
