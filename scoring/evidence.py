"""Evidence primitives: strength grading of heuristic indicators and helper constructors."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from core.schemas import FeatureResult, STRENGTH_RANK

UNCALIBRATED = ("Strength labels come from conventional thresholds in config/settings.py. They are "
                "uncalibrated heuristics, not validated error rates.")


def grade_high(value: Optional[float], cuts: Tuple[float, float, float]) -> str:
    """Strength when a *high* value is the AI-associated direction."""
    if value is None:
        return "n/a"
    weak, mod, strong = cuts
    if value >= strong:
        return "strong"
    if value >= mod:
        return "moderate"
    if value >= weak:
        return "weak"
    return "none"


def grade_low(value: Optional[float], cuts: Tuple[float, float, float]) -> str:
    """Strength when a *low* value is the AI-associated direction."""
    if value is None:
        return "n/a"
    weak, mod, strong = cuts
    if value <= strong:
        return "strong"
    if value <= mod:
        return "moderate"
    if value <= weak:
        return "weak"
    return "none"


def strength_from_z(z: Optional[float], cuts: Tuple[float, float, float]) -> str:
    """Map |robust z| to weak/moderate/strong (``none`` below the first cut-off)."""
    if z is None:
        return "n/a"
    a = abs(z)
    if a >= cuts[2]:
        return "strong"
    if a >= cuts[1]:
        return "moderate"
    if a >= cuts[0]:
        return "weak"
    return "none"


def max_strength(strengths: Sequence[str]) -> str:
    best = "n/a"
    for s in strengths:
        if STRENGTH_RANK.get(s, -1) > STRENGTH_RANK.get(best, -1):
            best = s
    return best


def unavailable(module: str, name: str, reason: str) -> FeatureResult:
    return FeatureResult(module=module, name=name, value=None, kind="unavailable",
                         interpretation=f"Not computed: {reason}", strength="n/a",
                         limitations=reason)
