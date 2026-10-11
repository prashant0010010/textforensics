"""Module D - multi-scale segment records (document, section, paragraph, sentence, window)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from config.settings import AnalysisSettings
from core.features import segment_features
from core.schemas import Document, SegmentRecord, Sentence, Signal


def _quality(n_words: int) -> tuple:
    if n_words >= 100:
        return "adequate", []
    if n_words >= 40:
        return "limited", ["Fewer than 100 words: statistics are unstable."]
    return "insufficient", ["Fewer than 40 words: statistics are descriptive only and should not be interpreted."]


def _record(doc: Document, level: str, idx: int, sents: Sequence[Sentence], para_index: int,
            start: int, end: int, cfg: AnalysisSettings) -> SegmentRecord:
    inc = [s for s in sents if s.included]
    feats = segment_features(doc, inc) if inc else {"n_words": 0, "n_sentences": 0}
    q, lim = _quality(int(feats.get("n_words") or 0))
    if sents:
        w0, w1 = sents[0].tok_start, sents[-1].tok_end
        s0, s1 = sents[0].index, sents[-1].index
    else:
        w0 = w1 = s0 = s1 = 0
    return SegmentRecord(level, idx, start, end, w0, w1, s0, s1, para_index, doc.text[start:end], feats, [], q, lim)


def build_segments(doc: Document, windows: List[Any], signals: List[Signal], cfg: AnalysisSettings) -> List[SegmentRecord]:
    segs: List[SegmentRecord] = []
    segs.append(_record(doc, "document", 0, doc.sentences, 0, 0, len(doc.text), cfg))
    for sec in doc.sections:
        ss = [s for p in doc.paragraphs[sec.para_start:sec.para_end] for s in doc.sentences[p.sent_start:p.sent_end]]
        segs.append(_record(doc, "section", sec.index, ss, doc.paragraphs[sec.para_start].index if doc.paragraphs else 0,
                            sec.start, sec.end, cfg))
    for p in doc.paragraphs:
        if p.included:
            segs.append(_record(doc, "paragraph", p.index, doc.sentences[p.sent_start:p.sent_end], p.index, p.start, p.end, cfg))
    for s in doc.included_sentences():
        segs.append(_record(doc, "sentence", s.index, [s], s.para_index, s.start, s.end, cfg))
    for i, (a, b, group) in enumerate(windows or []):
        segs.append(_record(doc, "window", i, group, group[0].para_index, a, b, cfg))
    # attach anomalies (signal ids overlapping each segment)
    order = sorted(signals, key=lambda s: s.start)
    for seg in segs:
        if seg.level == "document":
            seg.anomalies = [s.id for s in order]
        else:
            seg.anomalies = [s.id for s in order if s.start < seg.end and s.end > seg.start]
    return segs
