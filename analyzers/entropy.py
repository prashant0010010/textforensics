"""Module B.4 - entropy and information distribution.

Probability model: empirical (maximum-likelihood) distributions of the
observed units in the analysed text. Plug-in estimates are biased low for small
samples; the Miller-Madow correction H + (K-1)/(2 N ln 2) is reported next to
the plug-in value for word-level distributions. No additive smoothing is used.
Entropy here is a property of the text's own symbol statistics and is NOT a
probability of AI authorship.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

from config.settings import AnalysisSettings
from core import mathutils as mu
from core.features import char_entropy_bits, compression_ratio, word_ngrams
from core.schemas import AnalyzerOutput, Document, FeatureResult
from scoring.evidence import unavailable

MODULE = "entropy"


def _cond_entropy(seq: List, order: int) -> float | None:
    """H(X_n | X_{n-1..n-order}) = H(joint order+1) - H(context order), plug-in estimate."""
    if len(seq) < order + 20:
        return None
    joint = Counter(tuple(seq[i:i + order + 1]) for i in range(len(seq) - order))
    ctx = Counter(tuple(seq[i:i + order]) for i in range(len(seq) - order))
    hj, hc = mu.shannon_entropy(joint.values()), mu.shannon_entropy(ctx.values())
    return None if hj is None or hc is None else max(0.0, hj - hc)


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    words = doc.words()
    N = len(words)
    if N < 50:
        out.features.append(unavailable(MODULE, "Entropy measures", f"only {N} analysable words (need >= 50)"))
        return out
    text = " ".join(s.text for s in doc.included_sentences())

    def add(name, value, expl, unit="bits", lim="", alts=None, interp=""):
        out.features.append(FeatureResult(MODULE, name, value, unit, "measured", "", interp, expl, None, "n/a", "",
                                          lim, alts or []))

    ch = char_entropy_bits(text)
    add("Character entropy", ch, "Shannon entropy of lower-cased characters (whitespace collapsed), bits per character.",
        "bits/char", "Depends on language and alphabet; typically ~4.0-4.4 for English prose.")
    cw = Counter(words)
    hw = mu.shannon_entropy(cw.values())
    add("Word entropy (unigram, plug-in)", hw, "Shannon entropy of the word-frequency distribution.", "bits/word",
        "Grows with vocabulary size and length; compare only like with like.")
    add("Word entropy (Miller-Madow corrected)", mu.miller_madow_entropy(cw.values()),
        "Plug-in entropy plus (K-1)/(2N ln 2) bias correction.", "bits/word")
    add("Normalised word entropy (H / log2 V)", mu.normalized_entropy(cw.values()),
        "1 = all words equally frequent; lower = more concentrated vocabulary.", "ratio")
    for n in (2, 3):
        g = Counter(word_ngrams(words, n))
        add(f"Word {n}-gram entropy (plug-in / Miller-Madow)",
            {"plugin": mu.shannon_entropy(g.values()), "miller_madow": mu.miller_madow_entropy(g.values())},
            f"Entropy of the {n}-gram distribution.", "bits",
            "Plug-in estimates are strongly biased low because most n-grams occur once; treat as descriptive.")
    ce1 = _cond_entropy(words, 1)
    add("Conditional word entropy H(W_n | W_{n-1})", ce1,
        "H(bigram) - H(unigram contexts). Lower = next word more predictable from the previous one *within this text*.",
        "bits", "Severely biased low for short texts (sparse bigrams); not a language-model predictability measure.")
    chars = list(re.sub(r"\s+", " ", text.lower()))
    for o in (1, 2):
        add(f"Conditional character entropy (order {o})", _cond_entropy(chars, o),
            f"H(C_n | previous {o} char(s)).", "bits/char", "Order-2 estimates need several thousand characters.")
    cr = compression_ratio(text)
    add("Compression ratio (zlib)", cr, "Compressed size / raw size; lower = more repetition.", "ratio",
        "Also affected by length: very short texts compress poorly.")
    add("Repeated-token fraction (tokens/types)", N / len(cw), "Average number of times each word type is used.", "ratio")

    # Local entropy over fixed-size token windows (fixed size removes the length effect)
    w = cfg.entropy_window_tokens
    local = []
    for i in range(0, N - w + 1, max(1, w // 2)):
        local.append(mu.shannon_entropy(Counter(words[i:i + w]).values()))
    local = [x for x in local if x is not None]
    if len(local) >= 3:
        add(f"Local word entropy ({w}-token windows, 50% overlap)",
            {"mean": mu.mean(local), "std": mu.std(local), "min": min(local), "max": max(local), "n_windows": len(local)},
            "Entropy within sliding windows of fixed size.", "bits",
            "Window entropy is bounded by log2(window size).")
        out.tables["local_entropy"] = local
    # Entropy variation between paragraphs
    pe = []
    for p in doc.included_paragraphs():
        pw = doc.paragraph_words(p)
        if len(pw) >= 40:
            pe.append(mu.normalized_entropy(Counter(pw).values()))
    pe = [x for x in pe if x is not None]
    if len(pe) >= 3:
        add("Entropy variation between paragraphs (normalised word entropy: mean / CV)",
            {"mean": mu.mean(pe), "cv": mu.coefficient_of_variation(pe), "n": len(pe)},
            "Normalised word entropy per paragraph of >= 40 words.", "ratio")
    return out
