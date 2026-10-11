"""Transparent evidence aggregation (Module F).

No weights, no probabilities. The evidence label follows explicit counting rules over *feature families*:

  family level  = strongest heuristic indicator in the family, raised by passage-level signals
                  (strong: >=2 strong signals; moderate: >=1 strong or >=2 moderate; weak: >=1 moderate or >=3 weak)
  Category 5  Insufficient data to assess      : too few analysable words/sentences
  Category 4  Strong convergence               : >=4 families moderate+, >=2 strong, >=5 families evaluated
  Category 3  Several features associated      : >=2 families moderate+, or >=3 families weak+
  Category 2  Mixed or inconclusive            : >=2 families weak+, or >=1 family moderate+
  Category 1  Little evidence                  : otherwise

Families are NOT statistically independent (e.g. repetition and lexical diversity are correlated); the label
describes how many different kinds of measurement agree, nothing more. The label never states authorship.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from config.settings import AnalysisSettings
from core.schemas import FeatureResult, ModuleStatus, STRENGTH_RANK, Signal, TAXONOMY

AI_ASSOCIATED_FAMILIES = {
    "sentence_variability": "Uniformity of sentence length",
    "paragraph_structure": "Uniform paragraph structure",
    "repetition": "Repetition of phrases / openings",
    "discourse_rhetoric": "Transition, hedging and rhetorical-template density",
    "lexical_anomaly": "Locally reduced lexical diversity",
    "internal_predictability": "Locally low character-level entropy",
    "model_predictability": "Locally high predictability to a reference model",
}
DESCRIPTIVE_FAMILIES = {"internal_inconsistency": "Stylometric inconsistency within the document (possible mixed origin or editing; "
                                                  "not specific to AI)"}
_ANOMALY_FAMILIES = {"sentence_variability", "lexical_anomaly", "internal_predictability", "repetition", "discourse_rhetoric"}
DISCLAIMER = ("These labels describe the strength of statistical evidence from the features tested. They do not establish "
              "authorship, intent or misconduct, and no calibrated probability of AI authorship is produced.")


def _level_from_signals(sigs: List[Signal]) -> str:
    strong = sum(1 for s in sigs if s.strength == "strong")
    mod = sum(1 for s in sigs if s.strength == "moderate")
    weak = sum(1 for s in sigs if s.strength == "weak")
    if strong >= 2:
        return "strong"
    if strong >= 1 or mod >= 2:
        return "moderate"
    if mod >= 1 or weak >= 3:
        return "weak"
    return "none"


def _max(a: str, b: str) -> str:
    return a if STRENGTH_RANK[a] >= STRENGTH_RANK[b] else b


def aggregate(doc_stats: Dict[str, Any], features: List[FeatureResult], signals: List[Signal],
              statuses: List[ModuleStatus], cfg: AnalysisSettings, watermark_ran: bool = False) -> Dict[str, Any]:
    by_fam_ind: Dict[str, List[FeatureResult]] = defaultdict(list)
    for f in features:
        if f.family and f.strength in ("none", "weak", "moderate", "strong"):
            by_fam_ind[f.family].append(f)
    by_fam_sig: Dict[str, List[Signal]] = defaultdict(list)
    for s in signals:
        by_fam_sig[s.family].append(s)
    ran = {st.name for st in statuses if st.status == "ran"}
    evaluated = set(by_fam_ind)
    if "internal_anomalies" in ran:
        evaluated |= _ANOMALY_FAMILIES
    if "perplexity" in ran:
        evaluated.add("model_predictability")
    all_fams = list(AI_ASSOCIATED_FAMILIES) + list(DESCRIPTIVE_FAMILIES)
    table = []
    levels: Dict[str, str] = {}
    for fam in all_fams:
        lvl = "n/a" if fam not in evaluated and not by_fam_sig.get(fam) else "none"
        for f in by_fam_ind.get(fam, []):
            lvl = _max(lvl, f.strength)
        if by_fam_sig.get(fam):
            lvl = _max(lvl, _level_from_signals(by_fam_sig[fam]))
        levels[fam] = lvl
        table.append({
            "family": fam, "description": {**AI_ASSOCIATED_FAMILIES, **DESCRIPTIVE_FAMILIES}[fam], "level": lvl,
            "counts_toward_label": fam in AI_ASSOCIATED_FAMILIES,
            "indicators": [{"name": f.name, "strength": f.strength, "value": f.value if not isinstance(f.value, (dict, list)) else None}
                           for f in by_fam_ind.get(fam, [])],
            "signals": {k: sum(1 for s in by_fam_sig.get(fam, []) if s.strength == k) for k in ("weak", "moderate", "strong")},
        })
    counted = {f: l for f, l in levels.items() if f in AI_ASSOCIATED_FAMILIES}
    n_eval = sum(1 for l in counted.values() if l != "n/a")
    n_any = sum(1 for l in counted.values() if STRENGTH_RANK[l] >= 1)
    n_mod = sum(1 for l in counted.values() if STRENGTH_RANK[l] >= 2)
    n_strong = sum(1 for l in counted.values() if l == "strong")
    words, sents = doc_stats.get("words_analysed", 0), doc_stats.get("sentences_analysed", 0)
    rationale: List[str] = []
    if words < cfg.min_words_for_assessment or sents < cfg.min_sentences_for_assessment:
        cat = 5
        rationale.append(f"Only {words} analysable words and {sents} sentences; at least {cfg.min_words_for_assessment} words and "
                         f"{cfg.min_sentences_for_assessment} sentences are required for an assessment.")
    elif n_eval == 0:
        cat = 5
        rationale.append("No feature family could be evaluated.")
    elif n_mod >= 4 and n_strong >= 2 and n_eval >= 5:
        cat = 4
    elif n_mod >= 2 or n_any >= 3:
        cat = 3
    elif n_any >= 2 or n_mod >= 1:
        cat = 2
    else:
        cat = 1
    if cat != 5:
        rationale.append(f"{n_eval} feature families evaluated; {n_any} show at least weak evidence, {n_mod} at least moderate, {n_strong} strong.")
        if cat == 1:
            rationale.append("Absence of flagged features is not evidence of human authorship: the tested features cannot detect "
                             "edited, paraphrased or carefully prompted machine text.")
        if cat in (2, 3, 4):
            rationale.append("Each contributing family and its limitations are listed below; consider alternative explanations such as genre, "
                             "editing, assignment constraints and non-native writing.")
    triggered = [f for f in features if f.kind == "heuristic" and f.strength in ("none", "weak", "moderate", "strong")]
    n_trig = sum(1 for f in triggered if STRENGTH_RANK[f.strength] >= 1)
    experimental = {
        "label": "Experimental indicator count (NOT a probability)",
        "indicators_evaluated": len(triggered), "indicators_triggered_at_least_weak": n_trig,
        "how_calculated": "Count of document-level heuristic indicators (conventional thresholds in config/settings.py) that reach at least "
                          "'weak', out of those that could be evaluated for this document. Unweighted; not validated or calibrated.",
    }
    not_assessed = [{"module": st.name, "status": st.status, "detail": st.detail} for st in statuses if st.status in ("failed", "unavailable", "skipped")]
    return {
        "category": cat, "label": TAXONOMY[cat], "rationale": rationale, "family_table": table,
        "n_families_evaluated": n_eval, "n_weak_or_more": n_any, "n_moderate_or_more": n_mod, "n_strong": n_strong,
        "experimental_index": experimental, "not_assessed": not_assessed, "disclaimer": DISCLAIMER,
        "evidence_types": {"directly_measured": sum(1 for f in features if f.kind == "measured"),
                           "heuristic": sum(1 for f in features if f.kind == "heuristic"),
                           "model_based": sum(1 for f in features if f.kind == "model"),
                           "watermark_tests_run": int(watermark_ran),
                           "unavailable": sum(1 for f in features if f.kind == "unavailable")},
        "calibrated_probability": None,
        "calibration_note": "No calibrated probability is displayed: no classifier has been trained and evaluated on documented labelled data.",
    }
