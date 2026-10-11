"""Module B.6 - punctuation and formatting analysis."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict

from config.lexicons import RHETORICAL_TEMPLATES
from config.settings import AnalysisSettings
from core import mathutils as mu
from core.features import PUNCT_CLASSES
from core.schemas import AnalyzerOutput, Document, FeatureResult
from scoring.evidence import UNCALIBRATED, grade_high, unavailable

MODULE = "punctuation"
_BULLET = re.compile(r"^\s*([-*•◦▪‣–]|\d{1,3}[.)]|[a-zA-Z][.)])\s+")


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    sents = [s for s in doc.included_sentences() if s.n_words]
    N = len(doc.words())
    if N < 50 or not sents:
        out.features.append(unavailable(MODULE, "Punctuation analysis", f"only {N} analysable words"))
        return out
    text = " ".join(s.text for s in sents)
    counts = {k: len(p.findall(text)) for k, p in PUNCT_CLASSES.items()}
    counts["quote"] = counts["quote"]
    per1000 = {k: 1000.0 * v / N for k, v in counts.items()}
    per_sent = {k: v / len(sents) for k, v in counts.items()}
    out.tables["punctuation_rates"] = {"counts": counts, "per_1000_words": per1000, "per_sentence": per_sent}

    def add(name, value, expl, unit="", lim="", alts=None, interp="", strength="n/a", base="", fam="", kind="measured"):
        out.features.append(FeatureResult(MODULE, name, value, unit, kind, base, interp, expl, None, strength, fam,
                                          lim, alts or []))

    for k, label in (("comma", "Comma"), ("semicolon", "Semicolon"), ("colon", "Colon"), ("dash", "Dash"),
                     ("paren_open", "Parenthetical"), ("quote", "Quotation mark")):
        add(f"{label} frequency", {"count": counts[k], "per_1000_words": per1000[k], "per_sentence": per_sent[k]},
            f"Occurrences of {label.lower()} characters.", "",
            "Punctuation habits vary by author, editor, style guide and keyboard/software autoformatting; "
            "no single mark is a reliable AI fingerprint.")
    pe = mu.shannon_entropy([v for v in counts.values() if v > 0])
    add("Punctuation entropy", pe, "Shannon entropy of the distribution over punctuation classes (comma, semicolon, ...).", "bits",
        "Depends on how many classes occur at all; descriptive only.")
    # sentence-boundary patterns
    ends = Counter()
    for s in sents:
        m = re.search(r"([.!?…]+)[\"'”’)\]]*$", s.text)
        ends[m.group(1) if m else "(none)"] += 1
    add("Sentence-final punctuation distribution", dict(ends), "How sentences end.", "count",
        "Headings, bullets and lists often lack final punctuation.")
    # formatting consistency
    kinds = Counter(p.kind for p in doc.paragraphs)
    add("Block-type distribution", dict(kinds), "Counts of detected block types (prose, heading, bullet, quote, ...).", "count")
    markers = Counter()
    for p in doc.paragraphs:
        if p.kind == "bullet":
            for line in p.text.split("\n"):
                m = _BULLET.match(line)
                if m:
                    markers["numbered" if m.group(1)[0].isdigit() else "lettered" if m.group(1)[0].isalpha() else m.group(1)] += 1
    if markers:
        add("List-marker consistency", dict(markers), "Distinct bullet/number styles used.", "count",
            "One consistent marker style is normal.")
        items = [s.n_words for s in doc.sentences if s.kind == "bullet"]
        add("Repeated list structure (bullet item length CV)", mu.coefficient_of_variation(items),
            "Spread of list-item lengths; low CV = parallel items.", "ratio",
            "Parallel list items are good writing practice.")
    # rhetorical templates
    tcounts = {}
    for name, pat in RHETORICAL_TEMPLATES.items():
        tcounts[name] = len(re.findall(pat, text, re.I))
    total_tpl = sum(v for k, v in tcounts.items() if not k.startswith("tricolon"))
    rate = 1000.0 * total_tpl / N
    cuts = cfg.thresholds.template_per_1000_high
    st = grade_high(rate, cuts) if N >= 500 else "n/a"
    add("Rhetorical-template frequency (excluding tricolons, per 1000 words)", {"total_per_1000": rate, "by_template": tcounts,
                                                                              "tricolon_per_1000": 1000.0 * tcounts.get("tricolon list (X, Y, and Z)", 0) / N},
        "Counts of configurable rhetorical frames defined in config/lexicons.py (contrast-correction, importance framing, ...).",
        "per 1000 words", "These frames are widespread in persuasive, marketing and academic human writing. The list is "
        "configurable and is not a validated set of AI fingerprints. " + UNCALIBRATED,
        ["Persuasive or promotional genre", "Style taught in composition courses"],
        ("Not graded (fewer than 500 words). " if N < 500 else "") + "Descriptive frequency of editable patterns.", st,
        f"Heuristic cut-offs: >= {cuts[0]}/{cuts[1]}/{cuts[2]} per 1000 words", "discourse_rhetoric",
        "heuristic" if st != "n/a" else "measured")
    out.tables["templates"] = tcounts
    return out
