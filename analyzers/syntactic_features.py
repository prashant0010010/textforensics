"""Module B.8 (part 2) - POS and syntactic-pattern analysis.

Taggers (best available, never required):
  1. spaCy ``en_core_web_sm`` if installed        (not loaded in lightweight mode)
  2. NLTK averaged perceptron tagger if its data is present
  3. Built-in rule-based fallback (APPROXIMATE: closed-class lexicons + suffix rules)
The tagger used is always reported.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple

from config.lexicons import ARTICLES, AUXILIARIES, CONJUNCTIONS, PREPOSITIONS, PRONOUNS
from config.settings import AnalysisSettings
from core import mathutils as mu
from core.schemas import AnalyzerOutput, Document, FeatureResult, Signal
from scoring.evidence import unavailable

MODULE = "syntactic_features"
_PASSIVE = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?\w+(?:ed|en)\b", re.I)
_PAST = re.compile(r"\b\w+ed\b|\b(?:was|were|had|did|went|made|took|saw|said|found|came|gave)\b", re.I)
_PRESENT = re.compile(r"\b(?:is|are|has|have|does|do)\b|\b\w+(?:s)\b(?=\s+(?:the|a|an|to)\b)", re.I)
_FUTURE = re.compile(r"\b(?:will|shall|going to)\b", re.I)
_NUM = re.compile(r"^\d[\d,.]*$")


def _rule_tag(word: str, is_sentence_start: bool) -> str:
    w = word.lower()
    if _NUM.match(w):
        return "NUM"
    if w in ARTICLES:
        return "DET"
    if w in PRONOUNS and w not in ("that", "which", "one", "ones", "each", "other", "another"):
        return "PRON"
    if w in ("this", "that", "these", "those", "each", "another", "some", "any", "no", "all", "many", "much", "few", "more", "most"):
        return "DET"
    if w in AUXILIARIES:
        return "AUX"
    if w in PREPOSITIONS:
        return "ADP"
    if w in CONJUNCTIONS:
        return "CONJ"
    if w in ("not", "very", "too", "also", "just", "only", "then", "there", "here", "when", "where", "why", "how", "however", "thus", "therefore"):
        return "ADV"
    if w.endswith("ly") and len(w) > 4:
        return "ADV"
    if w.endswith(("ing", "ed")) and len(w) > 4:
        return "VERB"
    if w.endswith(("ous", "ful", "ive", "al", "able", "ible", "ic", "less", "ish", "ary")) and len(w) > 4:
        return "ADJ"
    if word[:1].isupper() and not is_sentence_start:
        return "PROPN"
    if w.endswith(("ize", "ise", "ify", "ate")) and len(w) > 5:
        return "VERB"
    return "NOUN"


def get_tagger(prefer_libs: bool) -> Tuple[str, bool, Callable[[List[str]], List[str]]]:
    """Return (name, is_approximate, tagger(list_of_words)->coarse tags)."""
    if prefer_libs:
        try:  # pragma: no cover - optional dependency
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "parser"])

            def spacy_tag(ws: List[str]) -> List[str]:
                from spacy.tokens import Doc
                d = Doc(nlp.vocab, words=ws)
                for _, proc in nlp.pipeline:
                    d = proc(d)
                return [t.pos_ if t.pos_ not in ("CCONJ", "SCONJ") else "CONJ" for t in d]
            return "spaCy en_core_web_sm", False, spacy_tag
        except Exception:
            pass
        try:  # pragma: no cover - optional dependency
            import nltk
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
            m = {"NN": "NOUN", "VB": "VERB", "JJ": "ADJ", "RB": "ADV", "PR": "PRON", "DT": "DET", "IN": "ADP", "CC": "CONJ", "MD": "AUX", "CD": "NUM"}

            def nltk_tag(ws: List[str]) -> List[str]:
                return [m.get(t[:2], "X") if t != "NNP" else "PROPN" for _, t in nltk.pos_tag(ws)]
            return "NLTK averaged perceptron", False, nltk_tag
        except Exception:
            pass

    def rule_tag(ws: List[str]) -> List[str]:
        return [_rule_tag(w, i == 0) for i, w in enumerate(ws)]
    return "built-in rule-based fallback (approximate)", True, rule_tag


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    sents = [s for s in doc.included_sentences() if s.n_words]
    N = sum(s.n_words for s in sents)
    if N < 100:
        out.features.append(unavailable(MODULE, "POS / syntactic analysis", f"only {N} analysable words (need >= 100)"))
        return out
    name, approx, tagger = get_tagger(prefer_libs=not cfg.lightweight)
    # Use original-case surface forms for the tagger.
    sent_tags: List[List[str]] = []
    for s in sents:
        surf = [doc.text[t.start:t.end] for t in doc.tokens[s.tok_start:s.tok_end]]
        try:
            tags = tagger(surf)
        except Exception:
            tags = [_rule_tag(w, i == 0) for i, w in enumerate(surf)]
            name, approx = "built-in rule-based fallback (approximate)", True
        sent_tags.append(tags)
    ctx["pos_tagger"] = name
    flat = [t for ts in sent_tags for t in ts]
    dist = Counter(flat)
    ctx["pos_counts"] = dist
    lim = ("POS tags come from a rule-based approximation (closed-class lexicons and suffix rules); noun/verb/adjective "
           "assignments are error-prone." if approx else "Tagger accuracy varies by domain.")

    def add(nm, value, expl, unit="", interp="", alts=None, strength="n/a", fam=""):
        out.features.append(FeatureResult(MODULE, nm, value, unit, "measured", "", interp, expl, None, strength, fam, lim,
                                          alts or []))

    add("POS tagger used", {"tagger": name, "approximate": approx}, "Which tagger produced the tags below.")
    add("POS distribution", {k: v / len(flat) for k, v in dist.most_common()}, "Share of each coarse POS tag.", "share")
    for n in (2, 3):
        grams = Counter(tuple(ts[i:i + n]) for ts in sent_tags for i in range(len(ts) - n + 1))
        add(f"POS {n}-gram entropy", mu.shannon_entropy(grams.values()),
            f"Entropy of POS {n}-gram distribution (plug-in).", "bits")
        add(f"Most frequent POS {n}-grams", [(" ".join(g), c) for g, c in grams.most_common(5)], "Dominant local grammatical patterns.")
    # POS-distribution variation across paragraphs
    para_js = []
    for p in doc.included_paragraphs():
        idxs = [i for i, s in enumerate(sents) if s.para_index == p.index]
        tags = [t for i in idxs for t in sent_tags[i]]
        if len(tags) >= 40:
            para_js.append(mu.js_distance(Counter(tags), dist))
    para_js = [x for x in para_js if x is not None]
    if len(para_js) >= 3:
        add("POS-distribution variation across paragraphs (mean / SD of JS distance to document)",
            {"mean": mu.mean(para_js), "std": mu.std(para_js)}, "How far each paragraph's POS mix is from the whole document's.", "JS distance")
    # Repeated syntactic templates: POS pattern of the first 5 tokens
    templ = Counter(" ".join(ts[:5]) for ts in sent_tags if len(ts) >= 5)
    nt = sum(templ.values())
    if nt:
        top, topn = templ.most_common(1)[0]
        add("Repeated syntactic templates (share of sentences with the most common 5-tag opening)", topn / nt,
            f"Most common opening POS pattern: {top} ({topn}/{nt}).", "share",
            "Descriptive; short sentences and noun-phrase openings make some templates common.",
            ["Genre conventions", "Tagger approximation"])
    # Tense and construction distributions (regex approximations)
    text = " ".join(s.text for s in sents)
    tense = {"past_like": len(_PAST.findall(text)), "present_like": len(_PRESENT.findall(text)), "future": len(_FUTURE.findall(text))}
    add("Tense-marker distribution (approximate regex counts)", tense, "Counts of past/present/future surface markers.", "count",
        "Rough surface heuristics; not a grammatical analysis.")
    add("Passive-construction rate (approximate, per 100 sentences)", 100.0 * len(_PASSIVE.findall(text)) / len(sents),
        "Be-verb + participle-like word pattern.", "per 100 sentences", "Regex approximation; irregular participles are missed.")
    # Sentence-initial word class distribution
    init = Counter(ts[0] for ts in sent_tags if ts)
    add("Sentence-initial POS distribution", {k: v / sum(init.values()) for k, v in init.most_common()}, "Word class opening each sentence.", "share")
    out.tables["pos_distribution"] = {k: v / len(flat) for k, v in dist.most_common()}
    return out
