#!/usr/bin/env python3
"""Verify SIMD vs portable scalar numerical parity (< 1e-12) across all modules."""

from __future__ import annotations

import math
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <simd_out.txt> <scalar_out.txt>", file=sys.stderr)
        return 2

    with open(sys.argv[1], encoding="utf-8") as f1, open(sys.argv[2], encoding="utf-8") as f2:
        lines1 = [ln.strip() for ln in f1 if ln.strip()]
        lines2 = [ln.strip() for ln in f2 if ln.strip()]

    if len(lines1) != len(lines2):
        print(f"FAIL: line count mismatch ({len(lines1)} vs {len(lines2)})", file=sys.stderr)
        return 1

    b1 = lines1[0].split()[1]
    b2 = lines2[0].split()[1]
    if b2 != "SCALAR":
        print(f"FAIL: second binary must be SCALAR, got {b2}", file=sys.stderr)
        return 1

    max_abs = 0.0
    max_rel = 0.0
    worst_key = ""
    tol = 1.0e-12

    for ln1, ln2 in zip(lines1[1:], lines2[1:], strict=True):
        k1, v1_s = ln1.split()
        k2, v2_s = ln2.split()
        if k1 != k2:
            print(f"FAIL: key mismatch {k1} vs {k2}", file=sys.stderr)
            return 1
        v1, v2 = float(v1_s), float(v2_s)
        if not (math.isfinite(v1) and math.isfinite(v2)):
            print(f"FAIL: non-finite value at {k1}: {v1} vs {v2}", file=sys.stderr)
            return 1
        abs_d = abs(v1 - v2)
        rel_d = abs_d / max(1.0, abs(v1), abs(v2))
        if abs_d > max_abs:
            max_abs = abs_d
        if rel_d > max_rel:
            max_rel = rel_d
            worst_key = k1
        if rel_d > tol:
            print(
                f"FAIL: {k1}: {b1}={v1:.17g} vs SCALAR={v2:.17g} (rel={rel_d:.3e} > {tol})",
                file=sys.stderr,
            )
            return 1

    print(
        f"[PASS] SIMD ({b1}) vs SCALAR parity across {len(lines1) - 1} values: "
        f"max_rel={max_rel:.3e} < 1e-12 (at {worst_key}), max_abs={max_abs:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
