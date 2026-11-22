"""Type stubs for chronosieve._core C11/SIMD extension."""

from collections.abc import Sequence
from typing import Any

Buffer = Any

FLAG_ANOMALY: int
FLAG_SHIFT: int
FLAG_ROGUE: int
FLAG_PINNED: int

def version() -> str: ...
def simd_backend() -> str: ...

def coma_merge(
    lowers: Buffer | Sequence[float],
    uppers: Buffer | Sequence[float],
    weights: Buffer | Sequence[float] | None = None,
    tau: float = 0.5,
    randomized: bool = False,
    seed: int = 0x243F6A8885A308D3,
) -> dict[str, Any]: ...

def coma_merge_batch(
    lowers_2d: Buffer | Sequence[float],
    uppers_2d: Buffer | Sequence[float],
    weights_2d: Buffer | Sequence[float] | None = None,
    n_agents: int = 0,
    tau: float = 0.5,
    randomized: bool = False,
    seed: int = 0x243F6A8885A308D3,
    out_hull_lo: Buffer | None = None,
    out_hull_hi: Buffer | None = None,
    out_max_agreement: Buffer | None = None,
    out_n_intervals: Buffer | None = None,
) -> dict[str, Any]: ...

class EDetector:
    stat: float
    step_count: int
    effective_family: str
    threshold: float

    def __init__(
        self,
        family: str = "auto",
        accum: str = "esr",
        side: str = "two_sided",
        threshold: float = 1000.0,
        mu0: float = ...,
        sigma: float = ...,
        max_prefix: int = 65536,
        seed: int = 0x517CC1B727220A95,
        auto_reset: bool = False,
    ) -> None: ...

    @classmethod
    def calibrate_catoni(
        cls,
        warmup: Buffer | Sequence[float],
        threshold: float = 1000.0,
        side: str = "two_sided",
        inflation_z: float = 2.5,
        auto_reset: bool = False,
    ) -> EDetector: ...

    def step(self, x: float) -> dict[str, Any]: ...
    def ingest_batch(
        self,
        xs: Buffer | Sequence[float],
        out_stats: Buffer | None = None,
        out_alarms: Buffer | None = None,
    ) -> dict[str, Any]: ...
    def reset(self) -> None: ...
    def reset_accum(self) -> None: ...

class EFusion:
    def __init__(
        self,
        n_agents: int,
        mode: str = "step",
        accum: str = "esr",
        family: str = "catoni",
        side: str = "two_sided",
        threshold: float = 1000.0,
        mu0: float = 0.0,
        sigma: float = 1.0,
        weights: Buffer | Sequence[float] | None = None,
        max_prefix: int = 65536,
        seed: int = 0x517CC1B727220A95,
        auto_reset: bool = False,
    ) -> None: ...
    def step(
        self,
        x_agents: Buffer | Sequence[float],
        weights: Buffer | Sequence[float] | None = None,
    ) -> dict[str, Any]: ...
    def ingest_batch(
        self,
        x_matrix: Buffer | Sequence[float],
        out_stats: Buffer | None = None,
        out_alarms: Buffer | None = None,
    ) -> dict[str, Any]: ...
    def reset(self) -> None: ...

class ConformalPID:
    q: float

    def __init__(
        self,
        alpha: float = 0.10,
        eta: float = 0.05,
        ki: float = 1.0,
        c_sat: float = 1.0,
        score_bound: float = 1e6,
        init_q: float = 0.0,
        edet_threshold: float = 500.0,
        warmup_cal: int = 32,
        score_window: int = 128,
        edet_max_prefix: int = 65536,
        seed: int = 0xA0761D6478BD642F,
        enable_ereset: bool = True,
    ) -> None: ...
    def step(self, y_true: float, y_pred: float) -> dict[str, Any]: ...
    def ingest_batch(
        self,
        y_true: Buffer | Sequence[float],
        y_pred: Buffer | Sequence[float],
        out_q_used: Buffer | None = None,
        out_q_next: Buffer | None = None,
        out_miscov: Buffer | None = None,
        out_resets: Buffer | None = None,
    ) -> dict[str, Any]: ...
    def reset(self) -> None: ...

class ConformalBatchPID:
    def __init__(
        self,
        n_channels: int,
        alpha: float = 0.10,
        eta: float = 0.05,
        ki: float = 1.0,
        c_sat: float = 1.0,
        score_bound: float = 1e6,
        init_q: float = 0.0,
    ) -> None: ...
    def ingest_batch(
        self,
        y_true: Buffer | Sequence[float],
        y_pred: Buffer | Sequence[float],
        out_q_next: Buffer | None = None,
        out_miscov: Buffer | None = None,
    ) -> int: ...
    def reset(self) -> None: ...

class SleepingBOA:
    def __init__(
        self,
        n_agents: int,
        share_base: float = 0.01,
        share_shift: float = 0.35,
        shift_var_decay: float = 0.25,
        rogue_threshold: float = 200.0,
        rogue_mu0: float = ...,
        rogue_sigma: float = ...,
        cal_window: int = 48,
        use_abs_loss: bool = False,
        quarantine_rogue: bool = True,
        peer_relative_rogue: bool = True,
    ) -> None: ...
    def predict(
        self,
        preds: Buffer | Sequence[float],
        active: Buffer | Sequence[int | bool] | None = None,
    ) -> float: ...
    def step(
        self,
        preds: Buffer | Sequence[float],
        y_true: float,
        active: Buffer | Sequence[int | bool] | None = None,
        shift_detected: bool = False,
    ) -> dict[str, Any]: ...
    def fuse_batch(
        self,
        preds_matrix: Buffer | Sequence[float],
        y_true: Buffer | Sequence[float],
        active_matrix: Buffer | Sequence[int | bool] | None = None,
        shifts: Buffer | Sequence[int | bool] | None = None,
        out_preds: Buffer | None = None,
        out_rogue_any: Buffer | None = None,
    ) -> dict[str, Any]: ...
    def weights(self) -> list[float]: ...
    def rogue_stats(self) -> tuple[list[float], list[bool]]: ...
    def clear_rogue(self, agent_idx: int = -1) -> None: ...
    def reset(self) -> None: ...

class StreamDigest:
    epsilon: float
    segment_count: int
    anomaly_count: int
    ring_size: int

    def __init__(
        self,
        epsilon: float = 0.05,
        ring_capacity: int = 65536,
        max_segments: int = 4096,
        max_anomalies: int = 1024,
        wake_cooldown: int = 0,
        anomaly_burst: int = 1,
    ) -> None: ...
    def push(
        self,
        t: float,
        y: float,
        y_hat: float | None = None,
        q_t: float = ...,
        edet_stat: float = 0.0,
        flags: int = 0,
    ) -> dict[str, Any]: ...
    def digest_batch(
        self,
        ts: Buffer | Sequence[float],
        ys: Buffer | Sequence[float],
        y_hats: Buffer | Sequence[float] | None = None,
        qs: Buffer | Sequence[float] | None = None,
        edet_stats: Buffer | Sequence[float] | None = None,
        flags: Buffer | Sequence[int] | None = None,
        out_wakes: Buffer | None = None,
    ) -> dict[str, Any]: ...
    def flush(self) -> None: ...
    def segments(self, max_count: int = 0) -> list[dict[str, Any]]: ...
    def anomalies(self, max_count: int = 0) -> list[dict[str, Any]]: ...
    def zoom(self, t0: float, t1: float) -> dict[str, Any]: ...
    def reconstruct(
        self,
        ts: Buffer | Sequence[float],
        out_y: Buffer | None = None,
    ) -> list[float] | None: ...
    def format_markdown(
        self,
        channel: str = "stream",
        emp_coverage: float = 1.0,
        current_q: float = 0.0,
        edet_stat: float = 0.0,
        edet_threshold: float = 500.0,
        rogue_agent: int = -1,
        max_segments: int = 6,
        max_anomalies: int = 5,
    ) -> str: ...
    def reset(self) -> None: ...
