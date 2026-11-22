"""Edge-triggered dispatch for integer interval guards."""
from . import _native


class Index:
    """Compile a fixed set of conjunctions of (signal, low, high) closed ranges."""

    __slots__ = ("_handle",)

    def __init__(self, initial, guards):
        self._handle = _native.build(initial, guards)

    def apply(self, patch):
        """Apply (signal, value) writes and return sorted (guard_id, active) edges."""
        return _native.apply(self._handle, patch)

    def active(self, guard_id):
        """Read the current truth of one guard."""
        return _native.active(self._handle, guard_id)
