import unittest

from config.settings import AnalysisSettings
from core.segmentation import segment_document, split_paragraph_spans
from tests import samples


def seg(text, **kw):
    cfg = AnalysisSettings()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return segment_document(text, cfg)


class TestSegmentation(unittest.TestCase):
    def test_paragraphs_preserved(self):
        d = seg("One para here.\n\nSecond para here.\n\n\nThird para here.")
        self.assertEqual(len(d.paragraphs), 3)
        self.assertEqual([p.text for p in d.paragraphs][1], "Second para here.")

    def test_single_paragraph(self):
        d = seg(samples.SINGLE_PARA)
        self.assertEqual(len(d.paragraphs), 1)
        self.assertEqual(len(d.sentences), 6)

    def test_abbreviations_decimals_initials(self):
        d = seg("Dr. Smith measured 3.5 mm, e.g. in J. Jones's lab. Then it ended. Next, Mr. Lee arrived.")
        self.assertEqual(len(d.sentences), 3)

    def test_question_exclamation(self):
        d = seg("Is it? Yes! Then go.")
        self.assertEqual([s.text for s in d.sentences], ["Is it?", "Yes!", "Then go."])

    def test_block_classification(self):
        d = seg(samples.ACADEMIC)
        kinds = [p.kind for p in d.paragraphs]
        self.assertEqual(kinds.count("heading"), 4)
        self.assertEqual(kinds.count("reference"), 2)
        self.assertEqual(len(d.sections), 4)
        self.assertEqual(d.sections[0].title, "Introduction")
        d2 = seg(samples.TECHNICAL)
        self.assertIn("code", [p.kind for p in d2.paragraphs])
        self.assertIn("table", [p.kind for p in d2.paragraphs])
        d3 = seg(samples.QUOTES)
        self.assertEqual([p.kind for p in d3.paragraphs].count("quote"), 2)

    def test_excluded_blocks_not_analysed_by_default(self):
        d = seg(samples.QUOTES)
        quoted = [s for s in d.sentences if s.kind == "quote"]
        self.assertTrue(quoted and all(not s.included for s in quoted))
        d2 = seg(samples.QUOTES, include_quotes=True)
        self.assertTrue(all(s.included for s in d2.sentences if s.kind == "quote"))

    def test_references_toggle(self):
        d = seg(samples.ACADEMIC, include_references=True)
        self.assertTrue(all(s.included for s in d.sentences if s.kind == "reference"))

    def test_bullets_one_sentence_per_line(self):
        d = seg("- alpha item\n- beta item\n- gamma item")
        self.assertEqual(len(d.sentences), 3)
        self.assertEqual(d.paragraphs[0].kind, "bullet")

    def test_single_newline_auto_mode(self):
        d = seg("First line ends here.\nSecond line ends here.\nThird line ends here.")
        self.assertEqual(len(d.paragraphs), 3)
        d2 = seg("a hard wrapped line that continues\nonto the next line without stopping\nand ends here.")
        self.assertEqual(len(d2.paragraphs), 1)

    def test_boundaries_cover_no_overlap(self):
        d = seg(samples.ACADEMIC)
        for a, b in zip(d.sentences, d.sentences[1:]):
            self.assertLessEqual(a.end, b.start)
        for a, b in zip(d.paragraphs, d.paragraphs[1:]):
            self.assertLessEqual(a.end, b.start)

    def test_unicode_and_punctuation_do_not_crash(self):
        for t in (samples.UNICODE, samples.PUNCT, samples.SHORT):
            d = seg(t)
            self.assertGreaterEqual(len(d.sentences), 1)


if __name__ == "__main__":
    unittest.main()
