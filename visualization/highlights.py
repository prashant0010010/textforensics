"""Module G - passage highlighting mapped exactly onto the original text."""
from __future__ import annotations

import html
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from core.schemas import CATEGORIES, STRENGTH_RANK, Signal
from visualization.tables import fmt

COLORS = {
    "predictability": "#bfe0ff", "repetition": "#ffd9a8", "structural": "#cdeec3", "lexical": "#f4cdef",
    "stylometric": "#fff0a0", "model": "#d4cffa", "watermark": "#ffc9c9",
}
_LEVEL_SPECIFICITY = {"sentence": 0, "paragraph": 1, "window": 2, "section": 3}


def select_signals(signals: Sequence[Signal], text_len: int, max_coverage: float = 0.35,
                   max_signals: int = 60) -> Tuple[List[Signal], List[Signal]]:
    """Choose which signals to highlight without painting the whole document.

    Strongest and most specific (sentence-level) signals are kept first; a signal is skipped when it would push
    the highlighted share of the text above ``max_coverage``. Skipped signals remain listed in tables/reports."""
    covered = np.zeros(max(text_len, 1), dtype=bool)
    chosen: List[Signal] = []
    skipped: List[Signal] = []
    order = sorted(signals, key=lambda s: (-STRENGTH_RANK.get(s.strength, 0), _LEVEL_SPECIFICITY.get(s.level, 9), s.end - s.start, s.start))
    budget = max_coverage * text_len
    for s in order:
        if len(chosen) >= max_signals:
            skipped.append(s)
            continue
        trial = covered.copy()
        trial[s.start:s.end] = True
        if trial.sum() <= budget:
            covered = trial
            chosen.append(s)
        else:
            skipped.append(s)
    chosen.sort(key=lambda s: (s.start, s.end))
    return chosen, skipped


def build_pieces(text: str, signals: Sequence[Signal]) -> List[Tuple[int, int, List[int]]]:
    """Partition ``text`` into contiguous pieces; each piece lists the ids of the signals covering it."""
    cuts = {0, len(text)}
    for s in signals:
        cuts.add(max(0, min(len(text), s.start)))
        cuts.add(max(0, min(len(text), s.end)))
    bounds = sorted(cuts)
    pieces = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a == b:
            continue
        ids = [s.id for s in signals if s.start <= a and s.end >= b]
        pieces.append((a, b, ids))
    return pieces


def verify_pieces(text: str, pieces: Sequence[Tuple[int, int, List[int]]]) -> bool:
    """True when the pieces tile the original text exactly (used by tests and self-checks)."""
    return "".join(text[a:b] for a, b, _ in pieces) == text and all(pieces[i][1] == pieces[i + 1][0] for i in range(len(pieces) - 1))


def render_text_html(text: str, signals: Sequence[Signal]) -> str:
    by_id = {s.id: s for s in signals}
    starts: Dict[int, List[int]] = {}
    for s in signals:
        starts.setdefault(s.start, []).append(s.id)
    out: List[str] = []
    for a, b, ids in build_pieces(text, signals):
        anchors = "".join(f'<span id="sig-{i}"></span>' for i in starts.get(a, []))
        chunk = html.escape(text[a:b])
        if not ids:
            out.append(anchors + chunk)
            continue
        ss = sorted((by_id[i] for i in ids), key=lambda s: (-STRENGTH_RANK.get(s.strength, 0), s.start))
        prim = ss[0]
        style = f"background:{COLORS.get(prim.category, '#ddd')};"
        if len(ss) > 1 and ss[1].category != prim.category:
            style += f"box-shadow:inset 0 -4px 0 {COLORS.get(ss[1].category, '#999')};"
        tip = html.escape(" | ".join(f"[{s.strength}] {s.feature}" for s in ss), quote=True)
        out.append(f'{anchors}<span class="hl s-{prim.strength}" data-ids="{",".join(str(i) for i in ids)}" style="{style}" title="{tip}">{chunk}</span>')
    return "".join(out)


def legend_html() -> str:
    items = "".join(f'<span class="chip" style="background:{COLORS[c]}">{html.escape(l)}</span>' for c, l in CATEGORIES.items())
    return (f'<div class="legend">{items}</div><div class="legend-note">Underline weight shows strength: dotted = weak, solid = moderate, thick = strong. '
            'Watermark results are document-level and are not highlighted on passages. Highlights mark statistical or linguistic patterns, '
            'not confirmed AI-written text.</div>')


def finding_card_html(s: Signal) -> str:
    alts = "".join(f"<li>{html.escape(a)}</li>" for a in s.alternatives)
    return (
        f'<div class="finding" data-sig="{s.id}" tabindex="0">'
        f'<div class="fhead"><span class="chip" style="background:{COLORS.get(s.category, "#ddd")}">{html.escape(CATEGORIES.get(s.category, s.category))}</span> '
        f'<b>{html.escape(s.feature)}</b> <span class="str str-{s.strength}">{s.strength}</span></div>'
        f'<div class="floc">paragraph {s.para_index + 1} &middot; characters {s.start}&ndash;{s.end} &middot; {s.level}-level</div>'
        f'<div><b>Observed:</b> {html.escape(fmt(s.observed))}</div>'
        f'<div><b>Document comparison:</b> {html.escape(s.baseline)}</div>'
        f'<div><b>Why it may be associated with machine-generated writing:</b> {html.escape(s.why)}</div>'
        f'<details><summary>Alternative explanations and additional evidence needed</summary><ul>{alts}</ul>'
        f'<div><b>Additional evidence needed:</b> {html.escape(s.needs_more_evidence)}</div>'
        f'<div class="muted">Strength labels are heuristic and uncalibrated.</div></details></div>')


CSS = """
body{font-family:Segoe UI,Arial,sans-serif;font-size:14px;color:#1f2430;margin:0}
.wrap{display:flex;gap:14px;align-items:flex-start}.pane{flex:3;min-width:0}.side{flex:2;min-width:0}
.textpane{white-space:pre-wrap;line-height:1.65;border:1px solid #d9dce3;border-radius:6px;padding:14px;background:#fff;overflow:auto}
.hl{border-radius:2px;padding:0 1px;cursor:pointer;border-bottom:1px dotted #555}.hl.s-moderate{border-bottom:2px solid #555}.hl.s-strong{border-bottom:3px solid #222}
.flash{outline:3px solid #e4572e;transition:outline .2s}
.finding{border:1px solid #d9dce3;border-radius:6px;padding:8px 10px;margin-bottom:8px;background:#fafbfc;cursor:pointer}.finding:hover,.finding:focus{border-color:#4a6fa5}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin:0 6px 4px 0;color:#1f2430}
.legend{margin-bottom:4px}.legend-note,.muted,.floc{font-size:12px;color:#5a6270}.floc{margin:2px 0 4px}
.str{font-size:12px;border-radius:4px;padding:0 6px;border:1px solid #999}.str-strong{background:#f6c6c6}.str-moderate{background:#fbe3b6}.str-weak{background:#eef2c4}
.sidepane{overflow:auto}details{margin-top:4px}ul{margin:4px 0 4px 18px;padding:0}
"""

JS = """
(function(){
 function flash(el){el.classList.add('flash');setTimeout(function(){el.classList.remove('flash')},1600)}
 document.querySelectorAll('.finding').forEach(function(f){f.addEventListener('click',function(){
   var t=document.getElementById('sig-'+f.dataset.sig); if(!t) return;
   var target=t.nextElementSibling||t.parentElement; (target.scrollIntoView?target:t).scrollIntoView({behavior:'smooth',block:'center'});
   var ids=(target.dataset&&target.dataset.ids)||''; flash(target);})});
 document.querySelectorAll('.hl').forEach(function(h){h.addEventListener('click',function(){
   var id=h.dataset.ids.split(',')[0]; var c=document.querySelector('.finding[data-sig="'+id+'"]');
   if(c){c.scrollIntoView({behavior:'smooth',block:'center'});flash(c)}})});
})();
"""


def interactive_panel_html(text: str, signals: Sequence[Signal], height_px: int | None = 720, note: str = "") -> str:
    """Self-contained HTML: highlighted text + clickable findings (also embedded in the HTML report)."""
    h = f"max-height:{height_px}px;" if height_px else ""
    cards = "".join(finding_card_html(s) for s in signals) or '<div class="muted">No passage-level findings to display.</div>'
    return (f"<style>{CSS}</style>{legend_html()}{('<div class=muted>' + html.escape(note) + '</div>') if note else ''}"
            f'<div class="wrap"><div class="pane"><div class="textpane" style="{h}">{render_text_html(text, signals)}</div></div>'
            f'<div class="side"><div class="sidepane" style="{h}">{cards}</div></div></div><script>{JS}</script>')
