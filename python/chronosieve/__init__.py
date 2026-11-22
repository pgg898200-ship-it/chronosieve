"""ChronoSieve: Zero-dependency C11/SIMD streaming inference & multi-agent digest engine."""

from ._core import (
    FLAG_ANOMALY,
    FLAG_PINNED,
    FLAG_ROGUE,
    FLAG_SHIFT,
    ConformalBatchPID,
    ConformalPID,
    EDetector,
    EFusion,
    SleepingBOA,
    StreamDigest,
    coma_merge,
    simd_backend,
    version,
)
from .pipeline import ChronoSieve

__version__ = version()

__all__ = [
    "ChronoSieve",
    "ConformalBatchPID",
    "ConformalPID",
    "EDetector",
    "EFusion",
    "FLAG_ANOMALY",
    "FLAG_PINNED",
    "FLAG_ROGUE",
    "FLAG_SHIFT",
    "SleepingBOA",
    "StreamDigest",
    "__version__",
    "coma_merge",
    "simd_backend",
    "version",
]
