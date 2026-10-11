"""Module H - human-review recommendations generated from the actual findings."""
from __future__ import annotations

from typing import Any, Dict, List

from core.schemas import AnalysisResult, FeatureResult, STRENGTH_RANK, Signal


def _loc(s: Signal) -> str:
    return f"paragraph {s.para_index + 1}, characters {s.start}-{s.end}"


def _signal_rec(s: Signal) -> str:
    f = s.feature
    key = s.details.get("feature")
    if f == "Recurring phrase":
        return (f"Passage ({_loc(s)}): the phrase '{s.details.get('phrase')}' recurs {s.details.get('count')} times. "
                "Check whether it is required terminology, quoted from the prompt, or a stylistic habit before drawing any conclusion.")
    if f == "Repeated sentence opening":
        return (f"Passage ({_loc(s)}): several sentences open with '{s.details.get('opening')}'. Read the surrounding sentences to judge "
                "whether the parallel structure is purposeful, and compare with how the same author structures sentences elsewhere.")
    if f == "Paragraph structural repetition":
        return (f"Paragraphs {', '.join(str(i + 1) for i in s.details.get('paragraph_indices', [])[:8])}: identical sentence counts and lengths. "
                "Check whether the assignment or a template prescribes this structure.")
    if f == "Clustered sentence-initial transitions":
        return f"Passage ({_loc(s)}): clustered transition openers ({', '.join(s.details.get('markers', []))}). Judge whether signposting is purposeful."
    if key == "sentence_len_cv":
        return (f"Passage ({_loc(s)}): sentence lengths are unusually uniform relative to surrounding prose ({s.baseline}). "
                "Examine the sentence structures and whether the passage is a list, definition or constrained summary.")
    if key in ("mattr25", "char_entropy", "repeated_trigram_share"):
        return (f"Passage ({_loc(s)}): compare its vocabulary distribution with the rest of the document ({s.baseline}); look for reused "
                "phrases and check whether technical terminology explains the pattern.")
    if key == "marker_per_100w":
        return f"Passage ({_loc(s)}): dense transitions ({s.baseline}). Check whether this is an introduction/conclusion or a mechanical connector pattern."
    if key == "mean_surprisal":
        return (f"Passage ({_loc(s)}): unusually predictable to the reference model ({s.baseline}). Repeat with a different reference model and "
                "compare with known human text of the same genre; predictability is model-dependent.")
    if s.category == "stylometric":
        return (f"Passage ({_loc(s)}): stylistic profile differs from the rest of the document ({s.baseline}). Check for quoted or pasted source "
                "text, a change of section type, collaboration or editing.")
    return f"Passage ({_loc(s)}): {s.why} {s.needs_more_evidence}"


def build_recommendations(res: AnalysisResult, max_passages: int = 12) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    sigs = sorted(res.signals, key=lambda s: (-STRENGTH_RANK.get(s.strength, 0), s.start))
    for s in sigs[:max_passages]:
        recs.append({"type": "passage", "signal_id": s.id, "strength": s.strength, "text": _signal_rec(s)})
    trig = [f for f in res.features if f.kind == "heuristic" and f.strength in ("weak", "moderate", "strong")]
    for f in trig:
        recs.append({"type": "document", "strength": f.strength,
                     "text": f"Document-level indicator '{f.name}' reached '{f.strength}' (value {f.value:.3f}). "
                             f"Consider: {'; '.join(f.alternatives[:2]) or 'genre and editing effects'}."
                     if isinstance(f.value, (int, float)) else f"Indicator '{f.name}' reached '{f.strength}'."})
    excl = res.doc_stats.get("excluded_blocks", {})
    if sigs:
        recs.append({"type": "verification", "text": "For each flagged passage, verify factual claims and citations and check that each "
                     "claim corresponds to the cited source; statistical patterns say nothing about accuracy."})
        notes = []
        if excl:
            notes.append("these block types were excluded from analysis: " + ", ".join(f"{k} ({v})" for k, v in excl.items()))
        recs.append({"type": "context", "text": "Consider whether flagged passages are quoted, technical, formulaic or heavily constrained by the assignment"
                     + (" (" + "; ".join(notes) + ")" if notes else "") + ", and gather additional context before reaching conclusions."})
        if not any(f.module == "reference_comparison" and f.kind != "unavailable" for f in res.features):
            recs.append({"type": "reference", "text": "If a legitimately supplied sample of the author's earlier writing exists, use the optional reference "
                         "comparison to see whether the flagged passage differs from it (still not proof either way)."})
        recs.append({"type": "context", "text": "Where appropriate, review drafting or version history and discuss the work with the author; "
                     "such process evidence is more informative than text statistics."})
    else:
        recs.append({"type": "context", "text": "No passage was flagged. This does not show the text is human-written; it only means none of the tested "
                     "patterns stood out against the document's own baseline."})
    recs.append({"type": "caution", "text": "Do not treat this statistical evidence alone as proof of misconduct. False positives occur, particularly for "
                 "non-native English writers, formulaic genres and heavily edited text."})
    return recs
