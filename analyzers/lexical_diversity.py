"""Module B.3 - lexical diversity (length-aware)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict

from config.lexicons import FUNCTION_WORDS
from config.settings import AnalysisSettings
from core import mathutils as mu
from core.features import mattr, mtld
from core.schemas import AnalyzerOutput, Document, FeatureResult
from scoring.evidence import unavailable

MODULE = "lexical_diversity"


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    words = doc.words()
    N = len(words)
    if N < 25:
        out.features.append(unavailable(MODULE, "Lexical diversity", f"only {N} analysable words"))
        return out
    c = Counter(words)
    V = len(c)
    hapax = sum(1 for v in c.values() if v == 1)
    dis = sum(1 for v in c.values() if v == 2)
    # Yule's K: 10^4 * (sum m^2 V_m - N) / N^2 (length-robust concentration measure)
    spectrum = Counter(c.values())
    yule_k = 1e4 * (sum(m * m * vm for m, vm in spectrum.items()) - N) / (N * N)
    top10 = sum(v for _, v in c.most_common(10)) / N
    ctx["word_counts"] = c

    def add(name, value, expl, unit="", lim="", alts=None, interp=""):
        out.features.append(FeatureResult(MODULE, name, value, unit, "measured", "", interp, expl, None, "n/a", "",
                                          lim, alts or []))

    add("Type-token ratio (raw)", V / N, f"{V} types / {N} tokens.", "ratio",
        "Strongly decreases as text length grows; do not compare documents of different lengths with raw TTR.")
    mt = {}
    for w in cfg.mattr_windows:
        m = mattr(words, w)
        mt[w] = m
        add(f"MATTR (window={w})", m, "Mean TTR over all sliding windows of fixed size (Covington & McFall 2010); "
            "largely length-independent.", "ratio",
            "Undefined when the text is shorter than the window." if m is None else
            "Genre, topic breadth and register strongly influence MATTR; no human/AI threshold is implied.")
    md = mtld(words, cfg.mtld_threshold)
    add("MTLD", md, f"Measure of Textual Lexical Diversity (factor threshold {cfg.mtld_threshold}; mean of forward and "
        "backward passes with partial factors, McCarthy & Jarvis 2010). Needs >= 50 tokens.", "tokens/factor",
        "Less stable below ~100 tokens." if N < 100 else "")
    add("Hapax legomena ratio", {"of_types": hapax / V, "of_tokens": hapax / N},
        f"{hapax} words occur exactly once.", "ratio", "Grows/shrinks with length; compare only within similar-length texts.")
    add("Dis legomena ratio (of types)", dis / V, f"{dis} words occur exactly twice.", "ratio")
    add("Herdan's C (log V / log N)", math.log(V) / math.log(N) if N > 1 else None,
        "Log-transformed TTR, less length-sensitive than raw TTR.", "")
    add("Yule's K (vocabulary concentration)", yule_k,
        "Higher K = more repetition of words; designed to be approximately length-independent.", "")
    add("Top-10 word share", top10, "Share of tokens covered by the 10 most frequent words (mostly function words).", "share")
    content = Counter({w: v for w, v in c.items() if w not in FUNCTION_WORDS and len(w) > 2})
    n_content = sum(content.values())
    rep = [(w, v) for w, v in content.most_common(15)]
    add("Most repeated content words (top 15)", [{"word": w, "count": v, "per_1000": 1000 * v / N} for w, v in rep],
        "Frequency of non-function words.", "", "Topic words are expected to repeat in focused writing.",
        ["Topic-focused or technical subject matter"])
    add("Rare-word proportion (hapax share of content tokens)", (sum(1 for v in content.values() if v == 1) / n_content) if n_content else None,
        "Share of content-word tokens that are unique in this document. Rarity is relative to the document, not a general frequency list.", "share",
        "Unusual vocabulary is not evidence of human authorship, and common vocabulary is not evidence of AI.")
    # Neighbouring paragraph lexical overlap
    paras = [p for p in doc.included_paragraphs() if p.kind in ("prose", "bullet")]
    sets = [set(w for w in doc.paragraph_words(p) if w not in FUNCTION_WORDS and len(w) > 2) for p in paras]
    jac = [len(a & b) / len(a | b) for a, b in zip(sets, sets[1:]) if a and b and len(a | b) > 0]
    if jac:
        add("Lexical overlap between neighbouring paragraphs (mean Jaccard, content words)", mu.mean(jac),
            "Shared content vocabulary between consecutive paragraphs.", "ratio",
            "Cohesive writing is expected to share vocabulary.")
    out.tables["lexical"] = {"types": V, "tokens": N, "hapax": hapax, "yule_k": yule_k, "mattr": {str(k): v for k, v in mt.items()}, "mtld": md}
    return out
