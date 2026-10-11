"""Module I - optional comparison against a user-supplied reference writing sample.

Every metric is reported separately. There is deliberately NO combined 'same author' score:
style similarity does not prove authorship and dissimilarity does not prove AI use.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from analyzers.syntactic_features import get_tagger
from config.lexicons import FUNCTION_WORDS
from config.settings import AnalysisSettings
from core import mathutils as mu
from core.features import marker_counts, word_ngrams
from core.schemas import AnalyzerOutput, Document, FeatureResult
from scoring.evidence import unavailable

MODULE = "reference_comparison"
LIM = ("Differences can reflect topic, genre, audience, time, editing tools, drafting conditions or sample size, "
       "not just authorship. This comparison cannot prove authorship or AI use.")


def _words(doc: Document) -> List[str]:
    return doc.words()


def _sent_lens(doc: Document) -> List[int]:
    return [s.n_words for s in doc.included_sentences() if s.n_words]


def _rel(c: Counter, keys) -> Dict[str, float]:
    n = sum(c.values()) or 1
    return {k: c.get(k, 0) / n for k in keys}


def compare_documents(a: Document, b: Document, cfg: AnalysisSettings) -> List[FeatureResult]:
    out: List[FeatureResult] = []
    wa, wb = _words(a), _words(b)
    ca, cb = Counter(wa), Counter(wb)

    def add(name, value, expl, unit="", lim=LIM):
        out.append(FeatureResult(MODULE, name, value, unit, "measured", "", "", expl, None, "n/a", "", lim, []))

    add("Sample sizes (words: submitted / reference)", {"submitted": len(wa), "reference": len(wb)},
        "Number of analysable words in each text.", "words",
        "Reference samples under ~1,000 words give unstable comparisons." if min(len(wa), len(wb)) < 1000 else LIM)
    # vocabulary
    va, vb = set(ca), set(cb)
    ct_a, ct_b = {w for w in va if w not in FUNCTION_WORDS}, {w for w in vb if w not in FUNCTION_WORDS}
    top = [w for w, _ in (ca + cb).most_common(500)]
    add("Vocabulary overlap", {"jaccard_all_types": len(va & vb) / len(va | vb) if va | vb else None,
                               "jaccard_content_types": len(ct_a & ct_b) / len(ct_a | ct_b) if ct_a | ct_b else None,
                               "cosine_top500_relative_freq": mu.cosine_similarity(_rel(ca, top), _rel(cb, top)),
                               "js_distance_word_distribution": mu.js_distance(ca, cb)},
        "Overlap of word types and similarity of frequency profiles (JS distance 0 = identical, 1 = disjoint).", "")
    # function words
    fa, fb = Counter({w: c for w, c in ca.items() if w in FUNCTION_WORDS}), Counter({w: c for w, c in cb.items() if w in FUNCTION_WORDS})
    add("Function-word profile", {"js_distance": mu.js_distance(fa, fb), "cosine": mu.cosine_similarity(_rel(fa, FUNCTION_WORDS), _rel(fb, FUNCTION_WORDS))},
        "Similarity of closed-class word usage (a classic stylometric signal; topic-independent but not author-proof).", "")
    # sentence-length distribution
    la, lb = _sent_lens(a), _sent_lens(b)
    ks = mu.ks_two_sample(la, lb)
    add("Sentence-length distribution", {"submitted": {"mean": mu.mean(la), "median": mu.median(la), "cv": mu.coefficient_of_variation(la)},
                                         "reference": {"mean": mu.mean(lb), "median": mu.median(lb), "cv": mu.coefficient_of_variation(lb)},
                                         "ks_D": ks[0] if ks else None, "ks_p_asymptotic": ks[1] if ks else None},
        "Two-sample Kolmogorov-Smirnov comparison of sentence lengths.", "words",
        LIM + " The KS p-value assumes independent sentences, which is only approximately true.")
    # punctuation
    ta = " ".join(s.text for s in a.included_sentences())
    tb = " ".join(s.text for s in b.included_sentences())
    marks = {"comma": ",", "semicolon": ";", "colon": ":", "dash": "—|–| - ", "paren": r"\(", "quote": "[\"“”]", "question": r"\?", "exclaim": "!"}
    pa = {k: 1000.0 * len(re.findall(p, ta)) / max(1, len(wa)) for k, p in marks.items()}
    pb = {k: 1000.0 * len(re.findall(p, tb)) / max(1, len(wb)) for k, p in marks.items()}
    add("Punctuation rates per 1000 words", {"submitted": pa, "reference": pb,
                                              "mean_abs_difference": mu.mean([abs(pa[k] - pb[k]) for k in marks]),
                                              "cosine": mu.cosine_similarity(pa, pb)}, "Per-mark punctuation rates.", "per 1000 words")
    # POS distribution
    name, approx, tagger = get_tagger(prefer_libs=not cfg.lightweight)

    def pos(doc: Document) -> Counter:
        c: Counter = Counter()
        for s in doc.included_sentences():
            surf = [doc.text[t.start:t.end] for t in doc.tokens[s.tok_start:s.tok_end]]
            if surf:
                try:
                    c.update(tagger(surf))
                except Exception:
                    pass
        return c

    pa_, pb_ = pos(a), pos(b)
    add("POS distribution", {"tagger": name, "approximate": approx, "js_distance": mu.js_distance(pa_, pb_)},
        "Similarity of coarse part-of-speech mixes.", "JS distance",
        LIM + (" POS tags are a rule-based approximation." if approx else ""))
    # n-grams
    def charprof(t: str, n: int = 3) -> Counter:
        t = re.sub(r"\s+", " ", t.lower())
        return Counter(t[i:i + n] for i in range(len(t) - n + 1))
    cga, cgb = charprof(ta), charprof(tb)
    bga, bgb = Counter(word_ngrams(wa, 2)), Counter(word_ngrams(wb, 2))
    tba, tbb = {g for g, _ in bga.most_common(500)}, {g for g, _ in bgb.most_common(500)}
    add("N-gram usage", {"char_3gram_cosine": mu.cosine_similarity(cga, cgb),
                         "word_bigram_jaccard_top500": len(tba & tbb) / len(tba | tbb) if tba | tbb else None},
        "Character 3-gram profile similarity; overlap of the 500 most frequent word bigrams.", "")
    # paragraph structure
    def para_stats(doc: Document):
        ps = [p for p in doc.included_paragraphs() if p.kind in ("prose", "bullet") and doc.paragraph_words(p)]
        wl = [len(doc.paragraph_words(p)) for p in ps]
        sc = [sum(1 for s in doc.sentences[p.sent_start:p.sent_end] if s.included and s.n_words) for p in ps]
        return {"n_paragraphs": len(ps), "mean_words": mu.mean(wl), "cv_words": mu.coefficient_of_variation(wl), "mean_sentences": mu.mean(sc)}
    add("Paragraph structure", {"submitted": para_stats(a), "reference": para_stats(b)}, "Paragraph lengths and sentences per paragraph.", "")
    # discourse markers
    ma, mb = sum(marker_counts(ta).values()), sum(marker_counts(tb).values())
    add("Discourse-marker rate per 1000 words", {"submitted": 1000.0 * ma / max(1, len(wa)), "reference": 1000.0 * mb / max(1, len(wb))},
        "Overall density of configured discourse markers.", "per 1000 words")
    return out


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    ref: Optional[Document] = ctx.get("reference_doc")
    if ref is None:
        out.features.append(unavailable(MODULE, "Reference-writing comparison", "no reference sample supplied"))
        return out
    if not cfg.reference_confirmed:
        out.features.append(unavailable(MODULE, "Reference-writing comparison",
                                        "the user has not confirmed that the sample is appropriate for comparison"))
        return out
    out.features.extend(compare_documents(doc, ref, cfg))
    return out
