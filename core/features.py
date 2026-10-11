"""Reusable low-level feature computations shared by analysers and multi-scale code."""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter, deque
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config.lexicons import DISCOURSE_CATEGORIES, FUNCTION_WORDS, HEDGES, QUALIFICATIONS
from core import mathutils as mu
from core.schemas import Document, Sentence

PUNCT_CLASSES = {
    "comma": re.compile(r","), "semicolon": re.compile(r";"), "colon": re.compile(r":"),
    "dash": re.compile(r"—|–|(?<=\s)-(?=\s)|--"), "paren_open": re.compile(r"\("),
    "quote": re.compile(r"[\"“”]"), "apostrophe": re.compile(r"['’]"),
    "question": re.compile(r"\?"), "exclaim": re.compile(r"!"), "ellipsis": re.compile(r"…|\.\.\."),
}


@lru_cache(maxsize=1)
def marker_patterns() -> Dict[str, re.Pattern]:
    pats = {}
    for cat, items in DISCOURSE_CATEGORIES.items():
        alt = "|".join(re.escape(i) for i in sorted(items, key=len, reverse=True))
        pats[cat] = re.compile(r"(?<![\w])(?:" + alt + r")(?![\w])", re.I)
    pats["hedging"] = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(i) for i in sorted(HEDGES, key=len, reverse=True)) + r")(?![\w])", re.I)
    pats["qualification"] = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(i) for i in sorted(QUALIFICATIONS, key=len, reverse=True)) + r")(?![\w])", re.I)
    return pats


@lru_cache(maxsize=1)
def initial_marker_pattern() -> re.Pattern:
    items = [m for cat, lst in DISCOURSE_CATEGORIES.items() if cat != "exemplification" for m in lst]
    items += ["it is important to note", "it is worth noting", "that being said", "with that said"]
    alt = "|".join(re.escape(i) for i in sorted(items, key=len, reverse=True))
    return re.compile(r"^[\s\"“‘'(\[]*(?:" + alt + r")(?![\w])\s*,?", re.I)


def mattr(tokens: Sequence[str], window: int) -> Optional[float]:
    """Moving-average type-token ratio (Covington & McFall 2010).

    Returns ``None`` if the text is shorter than the window (MATTR is only
    well-defined for N >= window)."""
    n = len(tokens)
    if window < 2 or n < window:
        return None
    counts: Dict[str, int] = {}
    types = 0
    total = 0.0
    n_windows = n - window + 1
    for i, t in enumerate(tokens):
        if counts.get(t, 0) == 0:
            types += 1
        counts[t] = counts.get(t, 0) + 1
        if i >= window:
            old = tokens[i - window]
            counts[old] -= 1
            if counts[old] == 0:
                types -= 1
        if i >= window - 1:
            total += types / window
    return total / n_windows


def _mtld_one_direction(tokens: Sequence[str], threshold: float) -> Optional[float]:
    factors = 0.0
    seen: set = set()
    count = 0
    for t in tokens:
        count += 1
        seen.add(t)
        if len(seen) / count <= threshold:
            factors += 1.0
            seen = set()
            count = 0
    if count > 0:
        ttr = len(seen) / count
        factors += (1.0 - ttr) / (1.0 - threshold)   # partial factor (McCarthy & Jarvis 2010)
    if factors <= 0:
        return None
    return len(tokens) / factors


def mtld(tokens: Sequence[str], threshold: float = 0.72) -> Optional[float]:
    """Measure of Textual Lexical Diversity: mean of forward and backward passes. Needs >= 50 tokens."""
    if len(tokens) < 50:
        return None
    f = _mtld_one_direction(tokens, threshold)
    b = _mtld_one_direction(list(reversed(tokens)), threshold)
    vals = [v for v in (f, b) if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def word_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    return list(zip(*(tokens[i:] for i in range(n)))) if len(tokens) >= n else []


def char_entropy_bits(text: str) -> Optional[float]:
    """Plug-in Shannon entropy (bits/char) over lower-cased text with whitespace collapsed."""
    t = re.sub(r"\s+", " ", text.lower())
    if len(t) < 2:
        return None
    return mu.shannon_entropy(Counter(t).values())


def compression_ratio(text: str) -> Optional[float]:
    """zlib compressed size / raw UTF-8 size: a repetition-sensitive measure."""
    b = text.encode("utf-8")
    if len(b) < 64:
        return None
    return len(zlib.compress(b, 9)) / len(b)


def marker_counts(text: str) -> Counter:
    c: Counter = Counter()
    for cat, pat in marker_patterns().items():
        for m in pat.finditer(text):
            c[(cat, m.group(0).lower())] += 1
    return c


def starts_with_marker(sentence_text: str) -> Optional[str]:
    m = initial_marker_pattern().match(sentence_text)
    return re.sub(r"[\s,\"“‘'(\[]+", " ", m.group(0)).strip().lower() if m else None


def repeated_ngram_share(tokens: Sequence[str], n: int) -> Tuple[float, int]:
    """Share of n-gram positions belonging to an n-gram type that occurs >= 2 times."""
    grams = word_ngrams(tokens, n)
    if not grams:
        return 0.0, 0
    c = Counter(grams)
    rep = sum(v for v in c.values() if v >= 2)
    return rep / len(grams), sum(1 for v in c.values() if v >= 2)


def fw_profile(tokens: Iterable[str]) -> Counter:
    return Counter(t for t in tokens if t in FUNCTION_WORDS)


def segment_features(doc: Document, sents: Sequence[Sentence], mattr_window: int = 25) -> Dict[str, Optional[float]]:
    """Compact feature vector for any group of sentences (sentence, paragraph, window, ...)."""
    sents = [s for s in sents if s.included]
    toks: List[str] = []
    lens: List[int] = []
    for s in sents:
        w = doc.sentence_words(s)
        toks.extend(w)
        lens.append(len(w))
    n = len(toks)
    text = " ".join(s.text for s in sents)
    f: Dict[str, Optional[float]] = {"n_words": n, "n_sentences": len(sents)}
    f["mean_sentence_len"] = mu.mean(lens)
    f["sentence_len_cv"] = mu.coefficient_of_variation(lens) if len(lens) >= 3 else None
    f["mattr25"] = mattr(toks, min(mattr_window, n)) if n >= mattr_window else None
    f["char_entropy"] = char_entropy_bits(text) if len(text) >= 200 else None
    f["comma_per_sentence"] = (text.count(",") / len(sents)) if sents else None
    f["function_word_share"] = (sum(1 for t in toks if t in FUNCTION_WORDS) / n) if n else None
    f["marker_per_100w"] = (100.0 * sum(marker_counts(text).values()) / n) if n else None
    f["repeated_trigram_share"] = repeated_ngram_share(toks, 3)[0] if n >= 30 else None
    return f


def included_tokens(doc: Document):
    """Token objects (with offsets) for included sentences, aligned with ``doc.words()``."""
    if "itoks" not in doc.cache:
        out = []
        for s in doc.included_sentences():
            out.extend(doc.tokens[s.tok_start:s.tok_end])
        doc.cache["itoks"] = out
    return doc.cache["itoks"]
