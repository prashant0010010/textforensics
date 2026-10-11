import unicodedata
import unittest
from html.parser import HTMLParser

from config.settings import AnalysisSettings
from core.pipeline import run_analysis
from core.schemas import Signal
from core.segmentation import segment_document
from tests import samples
from tests.synthetic import make_text, repetitive_text
from visualization.highlights import build_pieces, render_text_html, select_signals, verify_pieces


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, d):
        self.parts.append(d)


def visible_text(html_str: str) -> str:
    p = _Text()
    p.feed(html_str)
    return "".join(p.parts)


TEXTS = {
    "academic": samples.ACADEMIC, "technical": samples.TECHNICAL, "quotes": samples.QUOTES, "unicode": samples.UNICODE * 5,
    "punct": samples.PUNCT * 5, "html_chars": "Tom & Jerry <b>bold</b> said \"hi\" & left. " * 20 + "\n\n" + repetitive_text(8),
    "synthetic": make_text(11, 2500),
}


class TestOffsets(unittest.TestCase):
    def test_all_segments_map_to_original_text(self):
        for name, t in TEXTS.items():
            d = segment_document(t, AnalysisSettings())
            for p in d.paragraphs:
                self.assertEqual(t[p.start:p.end], p.text, name)
            for s in d.sentences:
                self.assertEqual(t[s.start:s.end], s.text, name)
                self.assertTrue(p_contains(d.paragraphs[s.para_index], s), name)
            for tok in d.tokens:
                self.assertEqual(t[tok.start:tok.end].lower(), tok.text, name)
                s = d.sentences[tok.sent_index]
                self.assertTrue(s.start <= tok.start and tok.end <= s.end, name)

    def test_original_text_is_never_modified(self):
        for name, t in TEXTS.items():
            before = unicodedata.normalize("NFC", t)             # ingestion applies NFC once; nothing else may change it
            res = run_analysis(t, AnalysisSettings(lightweight=True))
            self.assertEqual(res.document.text, before, name)
            self.assertEqual(len(res.document.text), len(before))

    def test_signal_and_segment_offsets_valid(self):
        for name in ("academic", "synthetic", "html_chars"):
            t = TEXTS[name]
            res = run_analysis(t, AnalysisSettings())
            for s in res.signals:
                self.assertTrue(0 <= s.start < s.end <= len(t), (name, s.feature))
                self.assertEqual(res.document.text[s.start:s.end], t[s.start:s.end])
                self.assertLessEqual(s.sent_start, s.sent_end)
            for g in res.segments:
                self.assertTrue(0 <= g.start <= g.end <= len(t))
                if g.level not in ("document",):
                    self.assertEqual(g.text, t[g.start:g.end])
                self.assertLessEqual(g.word_start, g.word_end)

    def test_pieces_tile_text(self):
        t = TEXTS["synthetic"]
        res = run_analysis(t)
        sel, _ = select_signals(res.signals, len(t))
        self.assertTrue(verify_pieces(t, build_pieces(t, sel)))

    def test_highlight_html_roundtrips_to_original(self):
        for name in ("html_chars", "unicode", "synthetic", "academic"):
            t = TEXTS[name]
            res = run_analysis(t)
            sel, _ = select_signals(res.signals, len(t))
            self.assertEqual(visible_text(render_text_html(t, sel)), t, name)

    def test_highlight_ids_and_anchors_present(self):
        t = repetitive_text(20)
        res = run_analysis(t)
        sel, _ = select_signals(res.signals, len(t))
        html_str = render_text_html(t, sel)
        for s in sel:
            self.assertIn(f'id="sig-{s.id}"', html_str)

    def test_manual_overlapping_signals(self):
        t = "abcdefghij" * 10
        mk = lambda i, a, b, st="weak", cat="repetition": Signal(cat, "f", "sentence", a, b, 0, 0, 0, 1.0, "", st, "", [], "", "x", id=i)
        sigs = [mk(0, 5, 40), mk(1, 20, 60, "strong", "lexical"), mk(2, 90, 100)]
        pieces = build_pieces(t, sigs)
        self.assertTrue(verify_pieces(t, pieces))
        self.assertEqual([ids for a, b, ids in pieces if a <= 25 < b][0], [0, 1])
        self.assertEqual(visible_text(render_text_html(t, sigs)), t)

    def test_whole_document_never_highlighted(self):
        t = repetitive_text(60)                      # extremely repetitive: many signals
        res = run_analysis(t)
        sel, skipped = select_signals(res.signals, len(t), 0.35, 60)
        covered = set()
        for s in sel:
            covered.update(range(s.start, s.end))
        self.assertLessEqual(len(covered) / len(t), 0.35 + 1e-9)
        self.assertLessEqual(len(sel), 60)
        self.assertEqual(len(sel) + len(skipped), len(res.signals))


def p_contains(p, s):
    return p.start <= s.start and s.end <= p.end


if __name__ == "__main__":
    unittest.main()
