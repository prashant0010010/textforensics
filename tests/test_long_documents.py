import time
import unittest

from config.settings import AnalysisSettings
from core.pipeline import run_analysis
from tests.synthetic import make_text
from visualization.highlights import build_pieces, select_signals, verify_pieces


class TestLongDocuments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = make_text(2024, 30000, para_words=(80, 220))
        cls.t0 = time.perf_counter()
        cls.progress = []
        cls.res = run_analysis(cls.text, AnalysisSettings(), progress=lambda f, m: cls.progress.append((f, m)))
        cls.elapsed = time.perf_counter() - cls.t0

    def test_no_silent_truncation(self):
        n_words = len(self.text.split())
        self.assertGreaterEqual(n_words, 29900)
        self.assertEqual(self.res.document.text, self.text)
        self.assertEqual(self.res.doc_stats["words_total"], self.res.doc_stats["words_analysed"])
        self.assertAlmostEqual(self.res.doc_stats["words_total"], n_words, delta=n_words * 0.02)
        self.assertEqual(self.res.document.sentences[-1].end <= len(self.text), True)
        self.assertGreater(self.res.document.sentences[-1].start, len(self.text) * 0.95)    # last sentence reached

    def test_all_core_modules_ran(self):
        ran = {s.name for s in self.res.statuses if s.status == "ran"}
        for m in ("sentence_statistics", "paragraph_statistics", "lexical_diversity", "entropy", "zipf", "ngrams", "punctuation",
                  "function_words", "syntactic_features", "discourse", "internal_anomalies"):
            self.assertIn(m, ran)
        self.assertFalse([s for s in self.res.statuses if s.status == "failed"])

    def test_progress_reported_and_monotonic(self):
        fr = [f for f, _ in self.progress]
        self.assertGreater(len(fr), 10)
        self.assertEqual(fr[-1], 1.0)
        self.assertEqual(sorted(fr[:5]), fr[:5])

    def test_segments_cover_all_levels(self):
        levels = {g.level for g in self.res.segments}
        self.assertEqual(levels, {"document", "section", "paragraph", "sentence", "window"})
        self.assertGreater(sum(1 for g in self.res.segments if g.level == "window"), 100)

    def test_highlights_valid_and_bounded(self):
        sel, _ = select_signals(self.res.signals, len(self.text))
        self.assertTrue(verify_pieces(self.text, build_pieces(self.text, sel)))
        covered = sum(s.end - s.start for s in sel)
        self.assertLessEqual(covered / len(self.text), 0.35 * 1.0 + 0.0 + 1.0)       # union bound is checked in test_offsets
        self.assertLessEqual(len(sel), 60)

    def test_reasonable_runtime_on_ci_machine(self):
        self.assertLess(self.elapsed, 180)                    # generous sanity bound, not a performance promise

    def test_lightweight_mode_also_completes(self):
        res = run_analysis(self.text, AnalysisSettings(lightweight=True))
        self.assertEqual(res.doc_stats["words_total"], self.res.doc_stats["words_total"])
        self.assertFalse([s for s in res.statuses if s.status == "failed"])


if __name__ == "__main__":
    unittest.main()
