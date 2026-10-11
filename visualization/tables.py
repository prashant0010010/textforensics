"""Flattening helpers for displaying results as tables."""
from __future__ import annotations

from typing import Any, Dict, List

from core.schemas import CATEGORIES, FeatureResult, KIND_LABELS, Signal


def fmt(value: Any, depth: int = 0) -> str:
    """Compact human-readable string for any measured value."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, dict):
        parts = [f"{k}: {fmt(v, depth + 1)}" for k, v in list(value.items())[:14]]
        more = " ..." if len(value) > 14 else ""
        s = "; ".join(parts) + more
        return f"{{{s}}}" if depth else s
    if isinstance(value, (list, tuple)):
        s = ", ".join(fmt(v, depth + 1) for v in list(value)[:12])
        return s + (" ..." if len(value) > 12 else "")
    return str(value)


def feature_rows(features: List[FeatureResult]) -> List[Dict[str, Any]]:
    return [{"Module": f.module, "Feature": f.name, "Value": fmt(f.value), "Unit": f.unit,
             "Type": KIND_LABELS.get(f.kind, f.kind), "Strength": f.strength, "Baseline": f.baseline,
             "Interpretation": f.interpretation, "What it measures": f.explanation, "Limitations": f.limitations,
             "Alternative explanations": "; ".join(f.alternatives)} for f in features]


def signal_rows(signals: List[Signal]) -> List[Dict[str, Any]]:
    return [{"ID": s.id, "Category": CATEGORIES.get(s.category, s.category), "Feature": s.feature, "Level": s.level,
             "Strength": s.strength, "Paragraph": s.para_index + 1, "Chars": f"{s.start}-{s.end}",
             "Observed": fmt(s.observed), "Baseline": s.baseline, "Why": s.why,
             "Alternatives": "; ".join(s.alternatives), "Needs": s.needs_more_evidence} for s in signals]
