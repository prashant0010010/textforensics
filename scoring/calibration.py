"""Calibration tools and the gate that keeps unvalidated probabilities off the screen.

A probability may be displayed only if ``config/calibration.json`` exists AND contains the evaluation record
produced by ``evaluation/framework.py`` (dataset description, sample sizes, held-out metrics). No such file
ships with the application, so :func:`calibrated_probability` always returns ``None`` out of the box.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

CAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "calibration.json")
REQUIRED_KEYS = ("dataset_description", "n_train", "n_test", "heldout_metrics", "generators", "domains")


def brier_score(y_true: Sequence[int], p: Sequence[float]) -> float:
    y, q = np.asarray(y_true, float), np.asarray(p, float)
    return float(np.mean((q - y) ** 2))


def reliability_table(y_true: Sequence[int], p: Sequence[float], n_bins: int = 10) -> List[Dict[str, float]]:
    y, q = np.asarray(y_true, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (q >= lo) & ((q < hi) if hi < 1 else (q <= hi))
        if m.sum():
            rows.append({"bin_low": float(lo), "bin_high": float(hi), "n": int(m.sum()),
                         "mean_predicted": float(q[m].mean()), "observed_rate": float(y[m].mean())})
    return rows


def expected_calibration_error(y_true: Sequence[int], p: Sequence[float], n_bins: int = 10) -> float:
    rows = reliability_table(y_true, p, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] / n * abs(r["mean_predicted"] - r["observed_rate"]) for r in rows)) if n else float("nan")


def calibration_record() -> Optional[Dict]:
    if not os.path.isfile(CAL_PATH):
        return None
    try:
        with open(CAL_PATH, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    return rec if all(k in rec for k in REQUIRED_KEYS) else None


def calibrated_probability(_features: object = None) -> Optional[float]:
    """Always ``None`` unless a documented calibration record exists. Even then, applying a trained model is a
    separate, explicitly enabled step that is out of scope for the rule-based engine."""
    return None
