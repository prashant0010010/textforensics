"""Module B.2 - paragraph structure."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np

from config.settings import AnalysisSettings
from core import mathutils as mu
from core.features import mattr, starts_with_marker
from core.schemas import AnalyzerOutput, Document, FeatureResult, Signal
from scoring.evidence import UNCALIBRATED, grade_high, grade_low, unavailable

MODULE = "paragraph_statistics"


def _vec(doc: Document, p) -> List[float]:
    sents = [s for s in doc.sentences[p.sent_start:p.sent_end] if s.included and s.n_words]
    lens = [s.n_words for s in sents]
    return [len(sents), mu.mean(lens) or 0.0, sum(lens)]


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    paras = [p for p in doc.included_paragraphs() if p.kind in ("prose", "bullet", "quote", "reference")
             and doc.paragraph_words(p)]
    n = len(paras)
    if n < 2:
        out.features.append(unavailable(MODULE, "Paragraph structure", f"only {n} analysable paragraph(s)"))
        return out
    wl = [len(doc.paragraph_words(p)) for p in paras]
    sc = [sum(1 for s in doc.sentences[p.sent_start:p.sent_end] if s.included and s.n_words) for p in paras]
    cv_w = mu.coefficient_of_variation(wl)
    cv_s = mu.coefficient_of_variation(sc)
    out.tables["paragraph_lengths"] = wl
    out.tables["paragraph_sentence_counts"] = sc

    def add(name, value, expl, unit="", strength="n/a", baseline="", interp="", lim="", alts=None, fam="", kind="measured"):
        out.features.append(FeatureResult(MODULE, name, value, unit, kind, baseline, interp, expl, None,
                                          strength, fam, lim, alts or []))

    cuts = cfg.thresholds.paragraph_len_cv_low
    st = grade_low(cv_w, cuts) if n >= 6 else "n/a"
    add("Paragraph length (words): mean / median / SD",
        {"mean": mu.mean(wl), "median": mu.median(wl), "std": mu.std(wl), "variance": mu.variance(wl)},
        "Distribution of paragraph lengths.", "words")
    add("Coefficient of variation of paragraph length", cv_w,
        "Paragraph-length spread relative to the mean.", "ratio", st,
        f"Heuristic cut-offs: <= {cuts[0]}/{cuts[1]}/{cuts[2]} (needs >= 6 paragraphs)",
        ("Not graded (fewer than 6 paragraphs). " if n < 6 else "") + UNCALIBRATED,
        "Uniform paragraphs are normal in templated, journalistic or instructional text and in short answers.",
        ["Assignment word-count constraints", "Template or style guide", "Editing for consistent length"],
        "paragraph_structure", "heuristic" if st != "n/a" else "measured")
    add("Mean sentences per paragraph", mu.mean(sc), "Average sentence count per paragraph.", "sentences")
    add("Sentence-count consistency (CV)", cv_s, "Spread of sentences-per-paragraph relative to mean.", "ratio")

    # Opening patterns
    openings = []
    for p in paras:
        w = doc.paragraph_words(p)
        openings.append(" ".join(w[:2]))
    oc = Counter(openings)
    top, top_n = oc.most_common(1)[0]
    add("Paragraph opening-pattern repetition (max share of identical 2-word openings)", top_n / n,
        f"Most frequent opening: '{top}' ({top_n}/{n} paragraphs).", "share", lim="Meaningful only with >= 6 paragraphs.")
    # Transition patterns
    marks = [starts_with_marker(doc.sentences[p.sent_start].text) if p.sent_end > p.sent_start else None for p in paras]
    mshare = sum(1 for m in marks if m) / n
    add("Paragraphs opening with a discourse marker", mshare,
        "Share of paragraphs whose first sentence starts with a transition (e.g. 'However,', 'In addition,').", "share",
        interp="Descriptive. Compare against your own genre expectations.",
        alts=["Academic and essay genres teach explicit signposting"])
    # Structural similarity between paragraphs
    V = np.array([_vec(doc, p) for p in paras], dtype=float)
    scale = V.max(axis=0) - V.min(axis=0)
    scale[scale == 0] = 1.0
    Z = (V - V.min(axis=0)) / scale
    if n >= 3:
        dists = np.sqrt(((Z[1:] - Z[:-1]) ** 2).sum(axis=1)) / np.sqrt(Z.shape[1])
        neigh_sim = float(1 - dists.mean())
        full = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)) / np.sqrt(Z.shape[1])
        mean_sim = float(1 - full[np.triu_indices(n, 1)].mean())
        add("Paragraph structural similarity (neighbouring / all pairs)", {"neighbouring": neigh_sim, "all_pairs": mean_sim},
            "1 - normalised Euclidean distance between (sentence count, mean sentence length, word count) vectors, min-max scaled within the document.",
            lim="Relative to this document's own range; a uniform document scores high even when normal.")
    # Paragraph-level lexical variation
    mt = [mattr(doc.paragraph_words(p), 25) for p in paras]
    mt = [m for m in mt if m is not None]
    if len(mt) >= 3:
        add("Paragraph-level lexical variation (MATTR-25 mean / CV)", {"mean": mu.mean(mt), "cv": mu.coefficient_of_variation(mt)},
            "Lexical diversity computed inside each paragraph of >= 25 words.")
    # Structural repetition: runs of paragraphs with nearly identical shape
    run_start = 0
    for i in range(1, n + 1):
        same = i < n and sc[i] == sc[i - 1] and abs(wl[i] - wl[i - 1]) <= 0.15 * max(wl[i], wl[i - 1])
        if not same:
            run_len = i - run_start
            if run_len >= 4 and sc[run_start] >= 2:
                a, b = paras[run_start], paras[i - 1]
                strength = "moderate" if run_len >= 6 else "weak"
                out.signals.append(Signal(
                    "structural", "Paragraph structural repetition", "paragraph", a.start, b.end,
                    doc.sentences[a.sent_start].index, doc.sentences[b.sent_end - 1].index, a.index,
                    float(run_len), "run of consecutive paragraphs with identical sentence count and similar length",
                    strength,
                    f"{run_len} consecutive paragraphs have the same sentence count ({sc[run_start]}) and word counts within 15% of each other.",
                    ["Assignment or template format", "Short-answer exam style", "Deliberately parallel structure"],
                    "Read the paragraphs for substance and compare with the author's other work.",
                    "paragraph_structure", {"paragraph_indices": [p.index for p in paras[run_start:i]]}))
            run_start = i
    return out
