"""Paragraph / sentence / section / token segmentation with exact offsets.

All offsets index into the canonical text (``Document.text``). Segmentation
never modifies the text.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from config.lexicons import ABBREVIATIONS, REFERENCE_HEADINGS
from config.settings import AnalysisSettings
from core.schemas import Document, Paragraph, Section, Sentence, Token

WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_END = re.compile(r"([.!?…]+)([\"'”’)\]]*)(\s+)")
_BULLET = re.compile(r"^\s*(?:[-*•◦▪‣–]|\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+\S")
_TERMINAL = re.compile(r"[.!?…][\"'”’)\]]*$")
_REF_LINE = re.compile(
    r"^\s*(?:\[\d+\]\s*|\d+\.\s+)?(?:"
    r"[A-Z][\w'’\-]+,\s+(?:[A-Z]\.\s?){1,3}.*\b(?:19|20)\d{2}\b"      # Smith, J., ... 2019
    r"|[A-Z][\w'’\-]+(?: et al\.)?,?\s+\((?:19|20)\d{2}[a-z]?\)\."       # Smith (2019). Title ...  (period after the year = reference style)
    r"|doi:\s*\S+)")
_NUMBERED_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXL]+\.)\s+[A-Z]")
_OPEN_QUOTES = "\"“‘'"
_CLOSE_QUOTES = "\"”’'"


def split_paragraph_spans(text: str) -> List[Tuple[int, int]]:
    """Split on blank lines; return trimmed (start, end) spans."""
    raw: List[Tuple[int, int]] = []
    pos = 0
    for m in re.finditer(r"\n[ \t]*\n+", text):
        raw.append((pos, m.start()))
        pos = m.end()
    raw.append((pos, len(text)))
    out = []
    for s, e in raw:
        seg = text[s:e]
        ls = len(seg) - len(seg.lstrip())
        rs = len(seg) - len(seg.rstrip())
        if e - rs > s + ls:
            out.append((s + ls, e - rs))
    return out


def _lines_with_offsets(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    spans = []
    pos = start
    for m in re.finditer(r"\n", text[start:end]):
        spans.append((pos, start + m.start()))
        pos = start + m.end()
    spans.append((pos, end))
    out = []
    for s, e in spans:
        seg = text[s:e]
        ls = len(seg) - len(seg.lstrip())
        rs = len(seg) - len(seg.rstrip())
        if e - rs > s + ls:
            out.append((s + ls, e - rs))
    return out


def _looks_like_reference(line: str) -> bool:
    return bool(_REF_LINE.search(line))


def _is_heading(text: str, start: int, end: int, refs_active: bool) -> bool:
    seg = text[start:end]
    if "\n" in seg.strip():
        return False
    s = seg.strip()
    words = len(WORD_RE.findall(s))
    if words == 0 or words > 12:
        return False
    if s.startswith("#"):
        return True
    if _TERMINAL.search(s) or s[-1] in ",;:":
        return False
    if refs_active and (re.search(r"\b(?:19|20)\d{2}\b", s) or words > 6):
        return False
    if s.lower().strip(" :") in REFERENCE_HEADINGS:
        return True
    if _NUMBERED_HEADING.match(s) or (s.isupper() and words >= 1):
        return True
    # Title-like short unterminated line.
    cap_words = sum(1 for w in WORD_RE.findall(s) if w[0].isupper() or w.lower() in
                    {"of", "the", "and", "a", "an", "in", "on", "for", "to", "with", "vs"})
    return words <= 10 and cap_words >= max(1, words - 1)


def _classify(text: str, start: int, end: int, refs_active: bool, hint: Optional[str]) -> str:
    seg = text[start:end]
    lines = [ln for ln in seg.split("\n") if ln.strip()]
    if hint:
        return hint
    if seg.lstrip().startswith("```") or (
            len(lines) >= 2 and all(re.match(r"^( {4}|\t)", ln) for ln in lines)
            and sum(1 for ln in lines if re.search(r"[{}();=<>]", ln)) >= 0.3 * len(lines)):
        return "code"
    if len(lines) >= 2 and all(ln.count("|") >= 1 or ln.count("\t") >= 1 for ln in lines):
        return "table"
    if _is_heading(text, start, end, refs_active):
        return "heading"
    if lines and all(ln.lstrip().startswith(">") for ln in lines):
        return "quote"
    if lines and sum(1 for ln in lines if _BULLET.match(ln)) >= max(1, 0.5 * len(lines)) and \
            not (len(lines) == 1 and len(WORD_RE.findall(lines[0])) > 40):
        return "bullet"
    s = seg.strip()
    if s and s[0] in _OPEN_QUOTES and s.rstrip(".!?…")[-1:] in _CLOSE_QUOTES and \
            len(WORD_RE.findall(s)) >= 12 and s.count('"') + s.count("“") + s.count("”") <= 2:
        return "quote"
    if refs_active:
        return "reference"
    if lines and sum(1 for ln in lines if _looks_like_reference(ln)) >= 0.6 * len(lines):
        return "reference"
    return "prose"


def split_sentences(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    """Rule-based sentence boundary detection with abbreviation handling."""
    seg = text[start:end]
    spans: List[Tuple[int, int]] = []
    cur = 0
    for m in _END.finditer(seg):
        term = m.group(1)
        nxt = seg[m.end():m.end() + 1]
        before = seg[cur:m.start(1)]
        prev_m = re.search(r"([\w.’'\-]+)$", before)
        prev = prev_m.group(1).lower().strip(".") if prev_m else ""
        if nxt and nxt.islower() and term == "." and len(prev) <= 3:
            continue          # likely an abbreviation ("etc. and", "fig. five"); longer words end sentences
        if term == "." and (prev in ABBREVIATIONS or (len(prev) == 1 and prev.isalpha() and prev_m and
                                                      before[prev_m.start():prev_m.start() + 1].isupper())):
            continue
        spans.append((cur, m.end(2)))
        cur = m.end()
    if cur < len(seg):
        spans.append((cur, len(seg)))
    out = []
    for s, e in spans:
        piece = seg[s:e]
        ls = len(piece) - len(piece.lstrip())
        rs = len(piece) - len(piece.rstrip())
        if e - rs > s + ls:
            out.append((start + s + ls, start + e - rs))
    return out


def segment_document(text: str, cfg: AnalysisSettings,
                     hints: Sequence[Tuple[int, int, str]] = ()) -> Document:
    """Segment ``text`` into paragraphs, sentences, sections and tokens."""
    inc_map = {"prose": True, "quote": cfg.include_quotes, "reference": cfg.include_references,
               "bullet": cfg.include_bullets, "code": cfg.include_code, "table": cfg.include_tables,
               "heading": cfg.include_headings}
    raw_spans = split_paragraph_spans(text)
    # Optionally treat single newlines as paragraph breaks.
    spans: List[Tuple[int, int]] = []
    for s, e in raw_spans:
        lines = _lines_with_offsets(text, s, e)
        mode = cfg.single_newline_paragraphs
        if len(lines) > 1 and mode != "no":
            seg_lines = [text[a:b] for a, b in lines]
            term_frac = sum(1 for ln in seg_lines if _TERMINAL.search(ln)) / len(lines)
            bullets = sum(1 for ln in seg_lines if _BULLET.match(ln)) / len(lines)
            if mode == "yes" or (term_frac >= 0.6 and bullets < 0.5):
                spans.extend(lines)
                continue
        spans.append((s, e))

    hint_list = sorted(hints)

    def hint_for(s: int, e: int) -> Optional[str]:
        for hs, he, kind in hint_list:
            if hs <= s and e <= he:
                return kind
        return None

    paragraphs: List[Paragraph] = []
    sentences: List[Sentence] = []
    tokens: List[Token] = []
    sections: List[Section] = []
    refs_active = False
    sec_idx = 0
    sec_title = "Document"
    sec_start = 0
    sec_para_start = 0

    for pi, (s, e) in enumerate(spans):
        kind = _classify(text, s, e, refs_active, hint_for(s, e))
        if kind == "heading":
            if pi > sec_para_start or sections:
                prev_end = paragraphs[-1].end if paragraphs else 0
                if pi > sec_para_start:
                    sections.append(Section(sec_idx, sec_title, sec_start, prev_end, sec_para_start, pi))
                    sec_idx += 1
            sec_title = text[s:e].strip().lstrip("#").strip()[:120]
            sec_start = s
            sec_para_start = pi
            refs_active = text[s:e].strip().lstrip("#").strip().lower().strip(" :") in REFERENCE_HEADINGS
        para = Paragraph(pi, s, e, text[s:e], kind, inc_map.get(kind, True), sec_idx, len(sentences), len(sentences))
        if kind in ("code", "table", "heading"):
            sent_spans = _lines_with_offsets(text, s, e) if kind != "heading" else [(s, e)]
        elif kind == "bullet":
            sent_spans = []
            for a, b in _lines_with_offsets(text, s, e):
                sent_spans.extend(split_sentences(text, a, b) or [(a, b)])
        else:
            sent_spans = split_sentences(text, s, e)
        for a, b in sent_spans:
            si = len(sentences)
            tstart = len(tokens)
            for m in WORD_RE.finditer(text, a, b):
                tokens.append(Token(m.group(0).lower(), m.start(), m.end(), si))
            sentences.append(Sentence(si, pi, a, b, text[a:b], tstart, len(tokens), kind, para.included))
        para.sent_end = len(sentences)
        paragraphs.append(para)

    if paragraphs:
        sections.append(Section(sec_idx, sec_title, sec_start, paragraphs[-1].end, sec_para_start, len(paragraphs)))
    return Document(text=text, paragraphs=paragraphs, sentences=sentences, sections=sections, tokens=tokens)


def document_counts(text: str, doc: Document) -> dict:
    """Counts shown in the UI before analysis (never truncates)."""
    inc = doc.included_sentences()
    words_all = len(doc.tokens)
    words_inc = sum(s.n_words for s in inc)
    n_win = max(0, (len(inc) - 1) // 4)
    return {
        "characters": len(text),
        "words_total": words_all,
        "words_analysed": words_inc,
        "paragraphs": len(doc.paragraphs),
        "paragraphs_analysed": len(doc.included_paragraphs()),
        "sentences": len(doc.sentences),
        "sentences_analysed": len(inc),
        "sections": len(doc.sections),
        "excluded_blocks": {k: sum(1 for p in doc.paragraphs if p.kind == k and not p.included)
                            for k in ("heading", "quote", "reference", "code", "table", "bullet", "prose")
                            if any(p.kind == k and not p.included for p in doc.paragraphs)},
        "estimated_windows": n_win,
        "workload": ("light" if words_inc < 3000 else "moderate" if words_inc < 12000 else "heavy"),
        "approx_lm_tokens": int(words_inc * 1.35),
    }
