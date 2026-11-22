"""Unified multi-agent ChronoSieve pipeline fusing all 4 C11/SIMD modules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._core import (
    FLAG_ANOMALY,
    FLAG_ROGUE,
    FLAG_SHIFT,
    ConformalPID,
    EDetector,
    SleepingBOA,
    StreamDigest,
    coma_merge,
)


class ChronoSieve:
    """End-to-end streaming inference, conformal uncertainty, fusion & LLM wake-up gate.

    Fuses:
      1. Zero-exp SIMD E-Detector (`EDetector`) on stream or residuals
      2. Pre-Control E-Reset Conformal PID (`ConformalPID`) + Online COMA (`coma_merge`)
      3. NEON Sleeping Fixed-Share BOA (`SleepingBOA`) + per-agent Catoni drift attribution
      4. Anomaly-Preserving Swing PLA + Wake-Up Gate + Ring-Buffer `zoom()` (`StreamDigest`)
    """

    def __init__(
        self,
        channel: str = "default",
        *,
        n_agents: int = 1,
        alpha: float = 0.10,
        eta: float = 0.05,
        epsilon: float = 0.05,
        edet_threshold: float = 500.0,
        rogue_threshold: float = 200.0,
        ring_capacity: int = 65536,
        warmup_cal: int = 32,
        anomaly_burst: int = 1,
        wake_cooldown: int = 0,
        monitor_level: bool = False,
        seed: int = 0x517CC1B727220A95,
    ) -> None:
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        self.channel = channel
        self.n_agents = n_agents
        self.edet_threshold = edet_threshold
        self.monitor_level = monitor_level
        self.warmup_cal = max(32, warmup_cal)

        # If monitor_level=True, split false-alarm budget 50/50 (2*gamma each) for combined ARL >= gamma
        eff_edet_thr = (2.0 * edet_threshold) if monitor_level else edet_threshold
        self.conformal = ConformalPID(
            alpha=alpha,
            eta=eta,
            edet_threshold=eff_edet_thr,
            warmup_cal=warmup_cal,
            seed=seed,
            enable_ereset=True,
        )
        self.detector = EDetector(
            family="vovk_rank",
            accum="esr",
            side="two_sided",
            threshold=eff_edet_thr,
            seed=seed ^ 0x9E3779B97F4A7C15,
            auto_reset=False,
        )
        self.fuser: SleepingBOA | None = (
            SleepingBOA(
                n_agents=n_agents,
                rogue_threshold=rogue_threshold,
                quarantine_rogue=True,
            )
            if n_agents > 1
            else None
        )
        self.digest = StreamDigest(
            epsilon=epsilon,
            ring_capacity=ring_capacity,
            wake_cooldown=wake_cooldown,
            anomaly_burst=anomaly_burst,
        )
        self._last_t: float | None = None
        self._last_y: float | None = None
        self._holt_level: float = 0.0
        self._holt_trend: float = 0.0
        self._last_cov: float = 1.0
        self._last_q: float = 0.0
        self._last_edet: float = 0.0
        self._last_rogue: int = -1
        self._known_rogues: set[int] = set()
        self._ed_warmup_left: int = self.warmup_cal
        self._step_count: int = 0

    def step(
        self,
        t: float,
        y: float,
        *,
        preds: Sequence[float] | None = None,
        active: Sequence[int | bool] | None = None,
        agent_intervals: Sequence[tuple[float, float]] | None = None,
        coma_tau: float = 0.5,
        coma_randomized: bool = False,
    ) -> dict[str, Any]:
        """Process one timestamped tick across all 4 modules."""
        # 0. Pre-validate inputs before mutating ANY module state (prevents partial state desync)
        t_f = float(t)
        y_f = float(y)
        if not math.isfinite(t_f) or not math.isfinite(y_f):
            raise ValueError("t and y must be finite numbers")
        if self._last_t is not None and t_f < self._last_t:
            raise ValueError(f"timestamps must be monotonically non-decreasing ({t_f} < {self._last_t})")

        if self.fuser is not None:
            if preds is None:
                raise ValueError("preds must be provided when n_agents > 1")
            if len(preds) != self.n_agents:
                raise ValueError(f"preds length ({len(preds)}) must equal n_agents ({self.n_agents})")
            if active is not None and len(active) != self.n_agents:
                raise ValueError(f"active length ({len(active)}) must equal n_agents ({self.n_agents})")
            for k in range(self.n_agents):
                if (active is None or active[k]) and not math.isfinite(float(preds[k])):
                    raise ValueError(f"active agent {k} prediction must be finite")
        elif preds is not None:
            if len(preds) == 0:
                raise ValueError("preds must be non-empty when provided")
            if not math.isfinite(float(preds[0])):
                raise ValueError("preds[0] must be finite")

        # 1. Prediction / Multi-agent fusion (or damped Holt linear-trend System-1 default)
        #    Capture strictly F_{t-1}-predictable BOA weights BEFORE observing y_t for COMA!
        fuser_res: dict[str, Any] | None = None
        pre_coma_weights: list[float] | None = None
        if self.fuser is not None:
            assert preds is not None
            y_hat = self.fuser.predict(preds, active)
            if agent_intervals is not None and len(agent_intervals) == self.n_agents:
                pre_coma_weights = self.fuser.weights()
        else:
            if preds is not None:
                y_hat = float(preds[0])
            elif self._last_y is None:
                y_hat = y_f
                self._holt_level = y_f
                self._holt_trend = 0.0
            else:
                y_hat = self._holt_level + 0.92 * self._holt_trend

        if not math.isfinite(y_f - y_hat):
            raise ValueError("non-finite residual y - y_hat")

        self._step_count += 1

        # 2. Pre-control E-reset Conformal PID (strictly preserves ARL >= edet_threshold).
        #    Optional level E-detector runs only when monitor_level=True (with post-reset warmup).
        cp_res = self.conformal.step(y_f, y_hat)
        ed_alarm = False
        ed_stat = 0.0
        if self.monitor_level:
            ed_res = self.detector.step(y_f)
            ed_stat = float(ed_res["stat"])
            if self._ed_warmup_left > 0:
                self._ed_warmup_left -= 1
                if self._ed_warmup_left == 0:
                    # Keep the W_cal-point B-tree baseline and zero only the accumulators!
                    self.detector.reset_accum()
                    ed_stat = 0.0
            elif ed_res["alarmed"]:
                ed_alarm = True
                self.detector.reset()
                self._ed_warmup_left = self.warmup_cal
        shift_detected = bool(cp_res["shift_reset"] or ed_alarm)

        # Update robust damped Holt state when running predictor-free single stream
        if self.fuser is None and preds is None and self._last_y is not None:
            innov = y_f - y_hat
            q_ref = float(cp_res["q_used"])
            # Huber-clip single-tick extreme spikes unless a sustained shift just fired
            if not shift_detected and self._step_count > 32 and q_ref > 0.0:
                clip = 3.0 * q_ref
                if innov > clip:
                    innov = clip
                elif innov < -clip:
                    innov = -clip
            prev_level = self._holt_level
            new_level = y_hat + 0.25 * innov
            self._holt_trend = 0.08 * (new_level - prev_level) + 0.92 * 0.92 * self._holt_trend
            self._holt_level = new_level

        # 3. Update multi-agent Sleeping BOA + per-agent Catoni drift attribution
        rogue_agent = -1
        any_rogue = False
        newly_rogue = False
        if self.fuser is not None and preds is not None:
            fuser_res = self.fuser.step(preds, y_f, active, shift_detected)
            any_rogue = bool(fuser_res["any_rogue"])
            if any_rogue:
                rogue_agent = int(fuser_res["worst_agent"])
                _, rflags = self.fuser.rogue_stats()
                cur_rogues = {i for i, fl in enumerate(rflags) if fl}
                if cur_rogues - self._known_rogues:
                    newly_rogue = True
                self._known_rogues = cur_rogues
            else:
                self._known_rogues.clear()

        # 4. Optional online COMA interval consensus (strictly F_{t-1}-predictable weights, no lookahead!)
        coma_res: dict[str, Any] | None = None
        if agent_intervals is not None and len(agent_intervals) > 0:
            lowers = [iv[0] for iv in agent_intervals]
            uppers = [iv[1] for iv in agent_intervals]
            coma_res = coma_merge(
                lowers,
                uppers,
                weights=pre_coma_weights,
                tau=coma_tau,
                randomized=coma_randomized,
                seed=(0x243F6A8885A308D3 ^ self._step_count),
            )

        # 5. Anomaly-Preserving Swing PLA + Wake-Up Gate + Ring Buffer
        flags = 0
        if cp_res["miscovered"] and self._step_count > 16:
            flags |= FLAG_ANOMALY
        if shift_detected:
            flags |= FLAG_SHIFT
        if newly_rogue:
            flags |= FLAG_ROGUE

        # When monitor_level=True, each detector uses 2 * edet_threshold for combined ARL >= edet_threshold;
        # normalize the combined statistic by 0.5 so edet_stat >= edet_threshold iff a shift actually fired.
        raw_edet_stat = max(float(cp_res["edet_stat"]), ed_stat)
        edet_stat = (0.5 * raw_edet_stat) if self.monitor_level else raw_edet_stat
        dig_res = self.digest.push(
            t_f,
            y_f,
            y_hat=y_hat,
            q_t=float(cp_res["q_used"]),
            edet_stat=edet_stat,
            flags=flags,
        )

        self._last_t = t_f
        self._last_y = y_f
        self._last_cov = float(cp_res["emp_coverage"])
        self._last_q = float(cp_res["q_next"])
        self._last_edet = edet_stat
        self._last_rogue = rogue_agent

        markdown: str | None = None
        if dig_res["should_wake"]:
            markdown = self.format_markdown()

        return {
            "t": t_f,
            "y": y_f,
            "y_hat": y_hat,
            "lower": float(cp_res["lower"]),
            "upper": float(cp_res["upper"]),
            "q_used": float(cp_res["q_used"]),
            "q_next": float(cp_res["q_next"]),
            "emp_coverage": self._last_cov,
            "miscovered": bool(cp_res["miscovered"]),
            "shift_detected": shift_detected,
            "edet_stat": edet_stat,
            "any_rogue": any_rogue,
            "rogue_agent": rogue_agent,
            "should_wake": bool(dig_res["should_wake"]),
            "digest_markdown": markdown,
            "fuser": fuser_res,
            "coma": coma_res,
        }

    def format_markdown(self, max_segments: int = 6, max_anomalies: int = 5) -> str:
        """Render a compact ~100-250 token Markdown digest packet for System-2 LLMs."""
        return self.digest.format_markdown(
            channel=self.channel,
            emp_coverage=self._last_cov,
            current_q=self._last_q,
            edet_stat=self._last_edet,
            edet_threshold=self.edet_threshold,
            rogue_agent=self._last_rogue,
            max_segments=max_segments,
            max_anomalies=max_anomalies,
        )

    def zoom(self, t0: float, t1: float) -> dict[str, Any]:
        """Lossless O(log N) drill-down over [t0, t1] from circular ring buffer."""
        return self.digest.zoom(t0, t1)

    def status(self) -> dict[str, Any]:
        """Return compact channel health summary."""
        segs = max(1, self.digest.segment_count)
        out: dict[str, Any] = {
            "channel": self.channel,
            "steps": self._step_count,
            "n_agents": self.n_agents,
            "emp_coverage": self._last_cov,
            "q_next": self._last_q,
            "edet_stat": self._last_edet,
            "epsilon": self.digest.epsilon,
            "segments": self.digest.segment_count,
            "anomalies": self.digest.anomaly_count,
            "ring_size": self.digest.ring_size,
            "compression_ratio": (self._step_count / segs) if self._step_count > 0 else 1.0,
            "rogue_agent": self._last_rogue,
        }
        if self.fuser is not None:
            rstats, rflags = self.fuser.rogue_stats()
            out["agent_weights"] = self.fuser.weights()
            out["agent_rogue_stats"] = rstats
            out["agent_rogue_flags"] = rflags
        return out

    def reset(self) -> None:
        self.conformal.reset()
        self.detector.reset()
        if self.fuser is not None:
            self.fuser.reset()
        self.digest.reset()
        self._last_t = None
        self._last_y = None
        self._holt_level = 0.0
        self._holt_trend = 0.0
        self._last_cov = 1.0
        self._last_q = 0.0
        self._last_edet = 0.0
        self._last_rogue = -1
        self._known_rogues.clear()
        self._ed_warmup_left = self.warmup_cal
        self._step_count = 0

