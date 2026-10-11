"""Module B.8 (part 1) - function-word frequency analysis."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from config.lexicons import FUNCTION_WORDS, FW_CATEGORIES
from config.settings import AnalysisSettings
from core.schemas import AnalyzerOutput, Document, FeatureResult
from scoring.evidence import unavailable

MODULE = "function_words"


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    words = doc.words()
    N = len(words)
    if N < 100:
        out.features.append(unavailable(MODULE, "Function-word analysis", f"only {N} analysable words (need >= 100)"))
        return out
    c = Counter(words)
    out.tables["function_word_profile"] = {w: c[w] for w, _ in Counter({w: c[w] for w in FUNCTION_WORDS if c[w]}).most_common(60)}
    ctx["fw_counts"] = Counter({w: c[w] for w in FUNCTION_WORDS if c[w]})
    total_fw = sum(c[w] for w in FUNCTION_WORDS)
    out.features.append(FeatureResult(MODULE, "Function-word share", total_fw / N, "share", "measured", "", "",
                                      "Share of tokens that are function words (closed-class words).", None, "n/a", "",
                                      "Depends on register: conversational text has higher shares than technical text.", []))
    for cat, ws in FW_CATEGORIES.items():
        n = sum(c[w] for w in ws)
        top = [(w, c[w]) for w in sorted(ws, key=lambda x: -c[x])[:8] if c[w]]
        out.features.append(FeatureResult(
            MODULE, f"{cat.capitalize()} frequency", {"per_1000_words": 1000.0 * n / N, "count": n, "top": top},
            "per 1000 words", "measured", "", "", f"Rate of {cat} (word lists in config/lexicons.py).", None, "n/a", "",
            "Several function words are ambiguous (e.g. 'that', 'as', 'for'); counts are lexical, not syntactic.", []))
    out.tables["function_word_categories"] = {cat: sum(c[w] for w in ws) for cat, ws in FW_CATEGORIES.items()}
    return out
