"""Documented evaluation framework for use with LEGITIMATELY OBTAINED, LABELLED data.

No dataset and no result ships with this project, and none must ever be invented. An evaluation on one dataset
does not establish accuracy on other documents, domains, authors, or AI systems.

Input records (list of dicts or CSV columns):
    id, label (0 = human, 1 = AI/mixed positive class), score (any monotone score, or a probability),
    author, generator (for AI texts; 'human' otherwise), domain, n_words, edited (0/1 for edited/mixed-origin text)

Command line:
    python -m evaluation.framework --csv labelled.csv --threshold 0.5 [--probabilities]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scoring.calibration import brier_score, expected_calibration_error, reliability_table


# ---------------------------------------------------------------- metrics
def confusion(y: Sequence[int], pred: Sequence[int]) -> Dict[str, int]:
    y, p = np.asarray(y), np.asarray(pred)
    return {"tp": int(((y == 1) & (p == 1)).sum()), "fp": int(((y == 0) & (p == 1)).sum()),
            "tn": int(((y == 0) & (p == 0)).sum()), "fn": int(((y == 1) & (p == 0)).sum())}


def _div(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def classification_metrics(y: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, Any]:
    pred = (np.asarray(scores, float) >= threshold).astype(int)
    c = confusion(y, pred)
    prec, rec = _div(c["tp"], c["tp"] + c["fp"]), _div(c["tp"], c["tp"] + c["fn"])
    f1 = _div(2 * prec * rec, prec + rec) if prec is not None and rec is not None else None
    return {**c, "n": len(y), "false_positive_rate": _div(c["fp"], c["fp"] + c["tn"]),
            "false_negative_rate": _div(c["fn"], c["fn"] + c["tp"]), "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": roc_auc(y, scores)}


def roc_auc(y: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """AUC via the Mann-Whitney U statistic with average ranks for ties."""
    y, s = np.asarray(y), np.asarray(scores, float)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    u = ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def bootstrap_ci(metric: Callable[[np.ndarray, np.ndarray], Optional[float]], y: Sequence[int], scores: Sequence[float],
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Optional[Tuple[float, float]]:
    """Stratified percentile bootstrap CI (resamples within each class)."""
    y, s = np.asarray(y), np.asarray(scores, float)
    rng = np.random.default_rng(seed)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    if pi.size < 2 or ni.size < 2:
        return None
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pi, pi.size), rng.choice(ni, ni.size)])
        v = metric(y[idx], s[idx])
        if v is not None and math.isfinite(v):
            vals.append(v)
    if len(vals) < 10:
        return None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------- splits
def group_split(records: List[Dict[str, Any]], group_key: str, test_fraction: float = 0.3, seed: int = 0):
    """Train/test split in which no group (author, generator or domain) appears on both sides."""
    groups = sorted({str(r[group_key]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_fraction)))
    test_g = set(groups[:n_test])
    train = [r for r in records if str(r[group_key]) not in test_g]
    test = [r for r in records if str(r[group_key]) in test_g]
    assert not ({str(r[group_key]) for r in train} & {str(r[group_key]) for r in test})
    return train, test


def leave_one_group_out(records: List[Dict[str, Any]], group_key: str):
    """Yield (held_out_group, train, test) for cross-generator / cross-domain generalisation tests."""
    for g in sorted({str(r[group_key]) for r in records}):
        yield g, [r for r in records if str(r[group_key]) != g], [r for r in records if str(r[group_key]) == g]


# ---------------------------------------------------------------- evaluation
LENGTH_BINS = [(0, 250), (250, 500), (500, 1000), (1000, 2500), (2500, 10 ** 9)]


def evaluate(records: List[Dict[str, Any]], threshold: float, probabilities: bool = False, n_boot: int = 1000,
             seed: int = 0) -> Dict[str, Any]:
    """Overall and stratified metrics with bootstrap CIs; calibration if scores are probabilities."""
    def block(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        y = [int(r["label"]) for r in rs]
        s = [float(r["score"]) for r in rs]
        m = classification_metrics(y, s, threshold)
        m["roc_auc_ci95"] = bootstrap_ci(roc_auc, y, s, n_boot, seed=seed)
        def fpr_fn(yy, ss):
            return classification_metrics(yy.tolist(), ss.tolist(), threshold)["false_positive_rate"]
        m["false_positive_rate_ci95"] = bootstrap_ci(fpr_fn, y, s, n_boot, seed=seed)
        return m

    out: Dict[str, Any] = {"threshold": threshold, "overall": block(records),
                           "caveat": "Results describe only this dataset. They do not establish accuracy on other documents, domains, "
                                     "authors or AI systems."}
    for key in ("domain", "generator", "author"):
        if records and key in records[0]:
            groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in records:
                groups[str(r[key])].append(r)
            out[f"by_{key}"] = {g: block(rs) for g, rs in groups.items() if len({int(r['label']) for r in rs}) == 2 or key == "generator"}
    if records and "n_words" in records[0]:
        out["by_length"] = {}
        for lo, hi in LENGTH_BINS:
            rs = [r for r in records if lo <= float(r["n_words"]) < hi]
            if rs:
                out["by_length"][f"{lo}-{hi if hi < 10**9 else 'inf'}"] = block(rs)
    if records and "edited" in records[0]:
        for flag, name in ((1, "edited_or_mixed"), (0, "unedited")):
            rs = [r for r in records if int(r["edited"]) == flag]
            if rs:
                out.setdefault("by_editing", {})[name] = block(rs)
    if probabilities:
        y = [int(r["label"]) for r in records]
        p = [float(r["score"]) for r in records]
        out["calibration"] = {"brier": brier_score(y, p), "ece": expected_calibration_error(y, p), "reliability": reliability_table(y, p)}
    return out


def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--probabilities", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    print(json.dumps(evaluate(load_csv(a.csv), a.threshold, a.probabilities, a.bootstrap), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
