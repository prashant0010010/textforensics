"""Plain-text report."""
from __future__ import annotations

import textwrap
from typing import List

from core.schemas import AnalysisResult, CATEGORIES, KIND_LABELS
from reporting.common import LIMITATIONS, METHODOLOGY, NOT_EQUIVALENT, modules_summary, selected_signals
from visualization.tables import fmt


def _wrap(s: str, indent: str = "") -> str:
    return textwrap.fill(s, 100, initial_indent=indent, subsequent_indent=indent + "  ")


def build_text(res: AnalysisResult) -> str:
    L: List[str] = []
    h = lambda t: L.extend(["", t, "=" * len(t)])
    ev = res.evidence
    L.append("TEXTFORENSICS LOCAL - ANALYSIS REPORT")
    L.append(_wrap(NOT_EQUIVALENT))
    h("1. Executive summary")
    L.append(f"Evidence category {ev['category']}: {ev['label']}")
    for r in ev["rationale"]:
        L.append(_wrap("- " + r))
    L.append(_wrap(ev["disclaimer"]))
    L.append(_wrap(ev["calibration_note"]))
    ei = ev["experimental_index"]
    L.append(_wrap(f"{ei['label']}: {ei['indicators_triggered_at_least_weak']} of {ei['indicators_evaluated']} evaluable heuristic indicators reached at least 'weak'. {ei['how_calculated']}"))
    h("2. Document statistics")
    for k, v in res.doc_stats.items():
        L.append(f"  {k}: {fmt(v)}")
    L.append(f"  source: {res.ingest.get('source')} | encoding: {res.ingest.get('encoding')}")
    for w in res.ingest.get("warnings", []):
        L.append(_wrap("  warning: " + w))
    h("3. Analysis modules")
    for m in modules_summary(res):
        L.append(f"  [{m['status']:>11}] {m['module']} ({m['seconds']}s) {m['detail']}")
    h("4. Evidence by feature family")
    for f in ev["family_table"]:
        L.append(f"  {f['family']:<24} {f['level']:<9} {'(counts toward label)' if f['counts_toward_label'] else '(descriptive only)'}  {f['description']}")
    h("5. Feature measurements")
    for f in res.features:
        L.append(f"- {f.name} [{KIND_LABELS.get(f.kind, f.kind)}; strength: {f.strength}]")
        L.append(_wrap(f"value: {fmt(f.value)} {f.unit}", "    "))
        if f.baseline:
            L.append(_wrap("baseline: " + f.baseline, "    "))
        if f.interpretation:
            L.append(_wrap("interpretation: " + f.interpretation, "    "))
        if f.limitations:
            L.append(_wrap("limitations: " + f.limitations, "    "))
        if f.alternatives:
            L.append(_wrap("alternative explanations: " + "; ".join(f.alternatives), "    "))
    chosen, skipped = selected_signals(res)
    h("6. Passage-level findings")
    L.append(f"{len(res.signals)} findings; {len(chosen)} highlighted; {len(skipped)} not highlighted to avoid over-marking the text.")
    text = res.document.text
    for s in res.signals:
        L.append("")
        L.append(f"[#{s.id}] {CATEGORIES.get(s.category, s.category)} - {s.feature} ({s.strength}, {s.level}-level)"
                 f"{'' if s in chosen else ' [not highlighted]'}")
        L.append(f"    location: paragraph {s.para_index + 1}, characters {s.start}-{s.end}")
        L.append(_wrap("passage: " + text[s.start:s.end].replace("\n", " ")[:600], "    "))
        L.append(_wrap(f"observed: {fmt(s.observed)}; comparison: {s.baseline}", "    "))
        L.append(_wrap("why: " + s.why, "    "))
        L.append(_wrap("alternative explanations: " + "; ".join(s.alternatives), "    "))
        L.append(_wrap("additional evidence needed: " + s.needs_more_evidence, "    "))
    h("7. Watermark results")
    for r in res.tables.get("watermark", []):
        L.append(_wrap(f"- {r['scheme']}: {'RAN' if r['ran'] else 'not run'} - {r['message']}"))
    insp = res.inspection
    L.append(_wrap(f"Text inspection (not proof of AI authorship): {insp['total_invisible']} invisible/unusual whitespace characters found."))
    h("8. Alternative explanations and limitations")
    for x in LIMITATIONS:
        L.append(_wrap("- " + x))
    h("9. Human-review recommendations")
    for r in res.recommendations:
        L.append(_wrap(f"- [{r['type']}] {r['text']}"))
    h("10. Technical methodology")
    for a, b in METHODOLOGY:
        L.append(_wrap(f"- {a}: {b}"))
    h("11. Reproducibility")
    for k, v in res.reproducibility.items():
        L.append(f"  {k}: {fmt(v)}")
    L.append("  settings: " + fmt(res.settings))
    return "\n".join(L) + "\n"
