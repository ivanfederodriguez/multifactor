"""Rebalance thresholds only: the P75 short target is unchanged."""
from __future__ import annotations

POLICIES = {
    "monthly": {"mode": "monthly", "level": 0},
    "stress_original": {"mode": "stress", "level": 0, "short_delta": 0.05, "regime_cross": True},
    "stress_gap5": {"mode": "stress", "level": 1, "short_delta": 0.05, "regime_cross": False},
    "stress_gap10": {"mode": "stress", "level": 2, "short_delta": 0.10, "regime_cross": False},
    "stress_gap15": {"mode": "stress", "level": 3, "short_delta": 0.15, "regime_cross": False},
    "stress_gap20": {"mode": "stress", "level": 4, "short_delta": 0.20, "regime_cross": False},
    "drift_original": {"mode": "drift", "level": 0, "name_gap": 0.05, "global_gap": 0.10},
    "drift_7p5_15": {"mode": "drift", "level": 1, "name_gap": 0.075, "global_gap": 0.15},
    "drift_10_20": {"mode": "drift", "level": 2, "name_gap": 0.10, "global_gap": 0.20},
    "drift_15_30": {"mode": "drift", "level": 3, "name_gap": 0.15, "global_gap": 0.30},
}


def weight_gaps(current, target):
    symbols = set(current) | set(target)
    gaps = [abs(float(current.get(s, 0.0)) - float(target.get(s, 0.0))) for s in symbols]
    return max(gaps, default=0.0), 0.5 * sum(gaps)


def short_gap_trigger(target, previous, threshold):
    return previous is not None and abs(float(target) - float(previous)) >= threshold - 1e-12


def drift_trigger(name_gap, global_gap, cfg):
    return name_gap >= cfg["name_gap"] - 1e-12 or global_gap >= cfg["global_gap"] - 1e-12
