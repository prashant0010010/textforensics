"""PDF report via ReportLab (optional local dependency; no network)."""
from __future__ import annotations

import io
import re
from typing import List

from core.schemas import AnalysisResult, CATEGORIES
from reporting.common import LIMITATIONS, NOT_EQUIVALENT, selected_signals
from visualization.highlights import COLORS, build_pieces
from visualization.tables import fmt


class PdfUnavailable(Exception):
    """ReportLab is not installed."""


def _safe(s: str) -> str:
    """Escape for ReportLab mini-markup; replace glyphs the built-in Helvetica cannot encode."""
    s = s.encode("cp1252", errors="replace").decode("cp1252")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def is_available() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


def build_pdf(res: AnalysisResult) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors
    except ImportError as e:
        raise PdfUnavailable("PDF export needs the optional 'reportlab' package (pip install reportlab).") from e
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title="TextForensics Local report")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5)
    story: List = [Paragraph("TextForensics Local - analysis report", ss["Title"]), Paragraph(_safe(NOT_EQUIVALENT), body), Spacer(1, 6)]
    ev = res.evidence
    story += [Paragraph("1. Executive summary", ss["Heading2"]),
              Paragraph(f"<b>Evidence category {ev['category']}: {_safe(ev['label'])}</b>", body)]
    for r in ev["rationale"]:
        story.append(Paragraph("&bull; " + _safe(r), body))
    story += [Paragraph(_safe(ev["disclaimer"]), body), Paragraph(_safe(ev["calibration_note"]), small)]
    fam = [["Feature family", "Level", "Counts toward label"]] + [[Paragraph(_safe(f["description"]), small), f["level"], "yes" if f["counts_toward_label"] else "no"] for f in ev["family_table"]]
    t = Table(fam, colWidths=[110 * mm, 25 * mm, 40 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story += [t, Paragraph("2. Document statistics", ss["Heading2"])]
    for k, v in res.doc_stats.items():
        story.append(Paragraph(f"{_safe(k)}: {_safe(fmt(v))}", small))
    story.append(Paragraph("3. Measured features", ss["Heading2"]))
    rows = [["Feature", "Value", "Strength"]] + [[Paragraph(_safe(f.name), small), Paragraph(_safe((fmt(f.value) + " " + f.unit)[:260]), small), f.strength] for f in res.features if f.kind != "unavailable"]
    t2 = Table(rows, colWidths=[85 * mm, 75 * mm, 20 * mm], repeatRows=1)
    t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke), ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [t2, Paragraph("4. Highlighted text", ss["Heading2"])]
    chosen, skipped = selected_signals(res)
    story.append(Paragraph("Colours: " + ", ".join(f'<font backColor="{COLORS[c]}">{_safe(l)}</font>' for c, l in CATEGORIES.items()) +
                           ". Highlights mark statistical patterns, not confirmed AI-written text.", small))
    text = res.document.text
    by_id = {s.id: s for s in chosen}
    paras: List[List[str]] = [[]]
    for a, b, ids in build_pieces(text, chosen):
        col = COLORS.get(by_id[ids[0]].category, "#dddddd") if ids else None
        for k, part in enumerate(re.split(r"\n\s*\n", text[a:b])):
            if k > 0:
                paras.append([])          # blank line = new paragraph; tags never span paragraphs
            if part:
                chunk = _safe(part).replace("\n", " ")
                paras[-1].append(f'<font backColor="{col}">{chunk}</font>' if col else chunk)
    for para in paras:
        markup = "".join(para)
        if markup.strip():
            story.append(Paragraph(markup, body))
    story.append(Paragraph("5. Passage findings", ss["Heading2"]))
    for s in res.signals:
        story.append(Paragraph(f"<b>#{s.id} {_safe(CATEGORIES.get(s.category, s.category))} - {_safe(s.feature)} ({s.strength})</b> "
                               f"paragraph {s.para_index + 1}, chars {s.start}-{s.end}. {_safe(s.why)} Alternatives: {_safe('; '.join(s.alternatives))}. "
                               f"Additional evidence: {_safe(s.needs_more_evidence)}", small))
    story.append(Paragraph("6. Limitations", ss["Heading2"]))
    for x in LIMITATIONS:
        story.append(Paragraph("&bull; " + _safe(x), body))
    story.append(Paragraph("7. Recommendations", ss["Heading2"]))
    for r in res.recommendations:
        story.append(Paragraph("&bull; " + _safe(r["text"]), body))
    story.append(Paragraph("Note: the built-in PDF font cannot display all Unicode characters; unsupported characters are shown as '?'. "
                           "Use the HTML or JSON report for exact text.", small))
    doc.build(story)
    return buf.getvalue()
