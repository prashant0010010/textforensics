"""Module B.9 - transitions and discourse analysis."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List

from config.lexicons import CEE_EVIDENCE, CEE_EXPLANATION
from config.settings import AnalysisSettings
from core import mathutils as mu
from core.features import marker_counts, marker_patterns, starts_with_marker
from core.schemas import AnalyzerOutput, Document, FeatureResult, Signal
from scoring.evidence import UNCALIBRATED, grade_high, unavailable

MODULE = "discourse"
ALTS = ["Academic essays are explicitly taught to signpost transitions", "Genre convention (reports, instructions)",
        "Writer style, or editing for flow", "Non-native writers often use explicit connectors"]


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    sents = [s for s in doc.included_sentences() if s.n_words]
    N = sum(s.n_words for s in sents)
    if N < 100 or len(sents) < 5:
        out.features.append(unavailable(MODULE, "Discourse analysis", f"only {N} words / {len(sents)} sentences"))
        return out
    text = " ".join(s.text for s in sents)
    mc = marker_counts(text)
    by_cat: Counter = Counter()
    for (cat, _), v in mc.items():
        by_cat[cat] += v
    total = sum(mc.values())
    rates = {cat: 1000.0 * by_cat.get(cat, 0) / N for cat in marker_patterns()}
    out.tables["discourse_rates"] = rates
    out.tables["discourse_markers"] = [{"category": c, "marker": m, "count": v} for (c, m), v in mc.most_common(40)]

    def add(name, value, expl, unit="", base="", interp="", lim="", alts=None, strength="n/a", fam="", kind="measured"):
        out.features.append(FeatureResult(MODULE, name, value, unit, kind, base, interp, expl, None, strength, fam, lim,
                                          alts or ALTS))

    for cat in ("additive", "contrastive", "causal", "conclusive", "sequential", "exemplification", "qualification"):
        add(f"{cat.capitalize()} markers", {"per_1000_words": rates[cat], "count": by_cat.get(cat, 0)},
            f"Frequency of {cat} discourse markers (configurable lists in config/lexicons.py).", "per 1000 words",
            lim="Several markers are ambiguous ('then', 'yet', 'first', 'also'); counts are lexical.")
    cuts_h = cfg.thresholds.hedge_per_1000_high
    st_h = grade_high(rates["hedging"], cuts_h) if N >= 500 else "n/a"
    add("Hedging expressions", {"per_1000_words": rates["hedging"], "count": by_cat.get("hedging", 0)},
        "Frequency of hedges ('may', 'might', 'generally', ...).", "per 1000 words",
        f"Heuristic cut-offs: >= {cuts_h[0]}/{cuts_h[1]}/{cuts_h[2]} (needs >= 500 words)",
        ("Not graded (fewer than 500 words). " if N < 500 else "") + UNCALIBRATED,
        "Modal words like 'may' and 'can be' are ubiquitous in scientific writing.", ["Scientific/cautious register",
        "Academic convention of epistemic modesty"], st_h, "discourse_rhetoric", "heuristic" if st_h != "n/a" else "measured")
    distinct = len(mc)
    add("Marker diversity", {"distinct_markers": distinct, "total": total,
                             "type_token_ratio": (distinct / total) if total else None,
                             "entropy_bits": mu.shannon_entropy(mc.values())},
        "How varied the markers are (distinct / total, entropy).", "", lim="Small counts give unstable diversity values.")
    # sentence-initial transitions
    init_marks = [starts_with_marker(s.text) for s in sents]
    rate_init = sum(1 for m in init_marks if m) / len(sents)
    cuts = cfg.thresholds.transition_initial_rate_high
    st = grade_high(rate_init, cuts) if len(sents) >= 20 else "n/a"
    add("Sentence-initial transition rate", rate_init, "Share of sentences that begin with a discourse marker.", "share",
        f"Heuristic cut-offs: >= {cuts[0]}/{cuts[1]}/{cuts[2]} (needs >= 20 sentences)",
        ("Not graded (fewer than 20 sentences). " if len(sents) < 20 else "") + UNCALIBRATED, "", None, st,
        "discourse_rhetoric", "heuristic" if st != "n/a" else "measured")
    # distribution across paragraphs
    per_p = []
    for p in doc.included_paragraphs():
        ps = [s for s in doc.sentences[p.sent_start:p.sent_end] if s.included and s.n_words]
        if len(ps) >= 3:
            per_p.append(sum(1 for s in ps if starts_with_marker(s.text)) / len(ps))
    if len(per_p) >= 3:
        add("Distribution of initial transitions across paragraphs (mean / CV)",
            {"mean": mu.mean(per_p), "cv": mu.coefficient_of_variation(per_p)}, "Evenness of transition use across paragraphs.", "share")
    # repeated topic-sentence structure and claim-evidence-explanation
    paras = [p for p in doc.included_paragraphs() if p.kind == "prose"]
    firsts = []
    cee = 0
    cee_elig = 0
    cee_paras = []
    for p in paras:
        ps = [s for s in doc.sentences[p.sent_start:p.sent_end] if s.included and s.n_words]
        if not ps:
            continue
        firsts.append(" ".join(doc.sentence_words(ps[0])[:3]))
        if len(ps) >= 3:
            cee_elig += 1
            ev = [i for i, s in enumerate(ps) if any(re.search(x, s.text, re.I) for x in CEE_EVIDENCE)]
            ex = [i for i, s in enumerate(ps) if any(re.search(x, s.text, re.I) for x in CEE_EXPLANATION)]
            if ev and ex and ev[0] >= 1 and ex[-1] > ev[0]:
                cee += 1
                cee_paras.append(p)
    if firsts:
        fc = Counter(firsts)
        add("Repeated topic-sentence openings (share of paragraphs whose first 3 words repeat)",
            sum(v for v in fc.values() if v >= 2) / len(firsts), f"Most common: {fc.most_common(2)}", "share")
    if cee_elig >= 4:
        frac = cee / cee_elig
        cuts_c = cfg.thresholds.cee_fraction_high
        st_c = grade_high(frac, cuts_c)
        add("Claim-evidence-explanation paragraph pattern (approximate)", frac,
            f"{cee}/{cee_elig} paragraphs of >= 3 sentences contain, in order, an opening claim, a sentence with an evidence cue "
            "('for example', 'according to', figures, ...) and a later explanation cue ('this suggests', 'therefore', ...).",
            "share", f"Heuristic cut-offs: >= {cuts_c[0]}/{cuts_c[1]}/{cuts_c[2]}", UNCALIBRATED,
            "Cue-word matching only; cannot verify the paragraph actually reasons this way.",
            ["Standard academic paragraph structure (PEEL / PEE) is explicitly taught"], st_c, "discourse_rhetoric", "heuristic")
    # clustered initial transitions -> signals
    k = 0
    for i in range(len(sents)):
        win = list(range(i, min(i + 5, len(sents))))
        if len(win) == 5 and sum(1 for j in win if init_marks[j]) >= 4 and (i == 0 or sum(1 for j in range(i - 1, i + 4) if init_marks[j]) < 4):
            hits = [j for j in win if init_marks[j]]
            a, b = sents[win[0]], sents[win[-1]]
            out.signals.append(Signal(
                "structural", "Clustered sentence-initial transitions", "window", a.start, b.end, a.index, b.index, a.para_index,
                float(len(hits)), f"document sentence-initial transition rate {rate_init:.2f}", "weak",
                f"{len(hits)} of 5 consecutive sentences begin with a discourse marker ({', '.join(init_marks[j] for j in hits)}).",
                ALTS, "Check whether the signposting is purposeful or mechanical.", "discourse_rhetoric", {"markers": [init_marks[j] for j in hits]}))
    return out
