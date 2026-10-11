"""Self-contained HTML report (no external resources)."""
from __future__ import annotations

import html
from typing import List

from core.schemas import AnalysisResult, CATEGORIES, KIND_LABELS
from reporting.common import LIMITATIONS, METHODOLOGY, NOT_EQUIVALENT, modules_summary, selected_signals
from visualization import charts
from visualization.highlights import interactive_panel_html
from visualization.tables import fmt

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#1f2430;line-height:1.5}
h1{font-size:26px}h2{border-bottom:2px solid #e3e6ec;padding-bottom:4px;margin-top:34px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border:1px solid #d9dce3;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f1f3f7}.notice{background:#fff7e0;border:1px solid #e8d48a;padding:10px 14px;border-radius:6px}
.cat{font-size:20px;font-weight:600}.muted{color:#5a6270;font-size:12px}.ok{color:#2b7a3d}.bad{color:#a33}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
"""


def _table(headers: List[str], rows: List[List[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{tr}</table>"


def build_html(res: AnalysisResult) -> str:
    ev = res.evidence
    chosen, skipped = selected_signals(res)
    text = res.document.text
    P: List[str] = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>TextForensics Local report</title><style>{CSS}</style></head><body>"]
    P.append("<h1>TextForensics Local &mdash; analysis report</h1>")
    P.append(f"<div class='notice'>{html.escape(NOT_EQUIVALENT)}</div>")
    P.append("<h2>1. Executive summary</h2>")
    P.append(f"<div class='cat'>Evidence category {ev['category']}: {html.escape(ev['label'])}</div>")
    P.append("<ul>" + "".join(f"<li>{html.escape(r)}</li>" for r in ev["rationale"]) + "</ul>")
    P.append(f"<p>{html.escape(ev['disclaimer'])}</p><p class='muted'>{html.escape(ev['calibration_note'])}</p>")
    ei = ev["experimental_index"]
    P.append(f"<p><b>{html.escape(ei['label'])}:</b> {ei['indicators_triggered_at_least_weak']} of {ei['indicators_evaluated']} evaluable heuristic indicators reached at least 'weak'. "
             f"<span class='muted'>{html.escape(ei['how_calculated'])}</span></p>")
    P.append(_table(["Feature family", "Level", "Counts toward label", "Description"],
                    [[f["family"], f["level"], "yes" if f["counts_toward_label"] else "no (descriptive)", f["description"]] for f in ev["family_table"]]))
    P.append("<h2>2. Document statistics</h2>")
    P.append(_table(["Item", "Value"], [[k, fmt(v)] for k, v in res.doc_stats.items()] +
                    [["source", str(res.ingest.get("source"))], ["encoding", str(res.ingest.get("encoding"))]] +
                    [["warning", w] for w in res.ingest.get("warnings", [])]))
    P.append("<h2>3. Analysis modules executed</h2>")
    P.append(_table(["Module", "Status", "Seconds", "Detail"], [[m["module"], m["status"], m["seconds"], m["detail"]] for m in modules_summary(res)]))
    P.append("<h2>4. Charts</h2><div class='charts'>")
    P.append(charts.histogram(res.tables.get("sentence_lengths", []), "Sentence lengths (words)", xlabel="words per sentence"))
    P.append(charts.bar_chart(res.tables.get("paragraph_lengths", []), "Paragraph lengths (words) in order", xlabel="paragraph #"))
    P.append(charts.zipf_plot(res.tables.get("zipf")))
    P.append(charts.anomaly_map(res.tables.get("window_z", []), len(text), res.settings["thresholds"]["anomaly_z"][0]))
    P.append("</div>")
    P.append("<h2>5. Statistical feature tables</h2>")
    for kind in ("measured", "heuristic", "model", "watermark", "unavailable"):
        fs = [f for f in res.features if f.kind == kind]
        if fs:
            P.append(f"<h3>{html.escape(KIND_LABELS[kind])}</h3>")
            P.append(_table(["Module", "Feature", "Value", "Strength", "Baseline", "Interpretation", "Limitations", "Alternative explanations"],
                            [[f.module, f.name, f"{fmt(f.value)} {f.unit}".strip(), f.strength, f.baseline, f.interpretation, f.limitations, "; ".join(f.alternatives)] for f in fs]))
    P.append("<h2>6. Highlighted text and segment-level findings</h2>")
    P.append(f"<p>{len(res.signals)} findings; {len(chosen)} highlighted; {len(skipped)} not highlighted so that no more than "
             f"{int(res.settings.get('max_highlight_coverage', .35) * 100)}% of the text is marked. Click a finding to jump to its passage.</p>")
    P.append(interactive_panel_html(text, chosen, height_px=None))
    if skipped:
        P.append("<h3>Findings not highlighted</h3>" + _table(["ID", "Category", "Feature", "Strength", "Location", "Why"],
                 [[s.id, CATEGORIES.get(s.category, s.category), s.feature, s.strength, f"paragraph {s.para_index + 1}, chars {s.start}-{s.end}", s.why] for s in skipped]))
    P.append("<h2>7. Internal consistency results</h2>")
    ic = [f for f in res.features if f.module == "internal_anomalies"]
    P.append(_table(["Feature", "Value", "Baseline", "Limitations"], [[f.name, fmt(f.value), f.baseline, f.limitations] for f in ic]) if ic else "<p>Not run.</p>")
    P.append("<h2>8. Optional reference-model results</h2>")
    pm = [f for f in res.features if f.module == "perplexity"]
    P.append(_table(["Feature", "Value", "Limitations"], [[f.name, fmt(f.value), f.limitations] for f in pm]) if pm else "<p>Not run.</p>")
    P.append("<h2>9. Optional reference-writing comparison</h2>")
    rc = [f for f in res.features if f.module == "reference_comparison"]
    P.append(_table(["Metric", "Value", "Limitations"], [[f.name, fmt(f.value), f.limitations] for f in rc]) if rc else "<p>Not run.</p>")
    P.append("<h2>10. Watermark results</h2>")
    P.append(_table(["Scheme", "Ran", "Status", "Eligible tokens", "Statistic", "p-value (adj.)", "Message"],
                    [[r["scheme"], r["ran"], r["config_status"], fmt(r["eligible_tokens"]), fmt(r["statistic"]),
                      f"{fmt(r['p_value'])} ({fmt(r['adjusted_p_value'])})", r["message"]] for r in res.tables.get("watermark", [])]))
    insp = res.inspection
    P.append(f"<p class='muted'>Text inspection (characters only; not evidence of AI authorship): {insp['total_invisible']} invisible/unusual whitespace "
             f"characters; counts: {html.escape(fmt(insp['invisible_counts']))}.</p>")
    P.append("<h2>11. Alternative explanations, confidence and limitations</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in LIMITATIONS) + "</ul>")
    P.append("<h2>12. Human-review recommendations</h2><ul>" + "".join(f"<li>{html.escape(r['text'])}</li>" for r in res.recommendations) + "</ul>")
    P.append("<h2>13. Technical methodology</h2><ul>" + "".join(f"<li><b>{html.escape(a)}:</b> {html.escape(b)}</li>" for a, b in METHODOLOGY) + "</ul>")
    P.append("<h2>14. Reproducibility</h2>" + _table(["Item", "Value"], [[k, fmt(v)] for k, v in res.reproducibility.items()] + [["settings", fmt(res.settings)]]))
    P.append("</body></html>")
    return "".join(P)
