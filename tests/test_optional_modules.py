import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from analyzers import perplexity
from analyzers.perplexity import HFCausalLMScorer, ModelUnavailable, TokenScores, summarise_scores, window_surprisal
from config.settings import AnalysisSettings
from core.pipeline import run_analysis
from tests.synthetic import make_text
from watermark import registry
from watermark.base import UNAVAILABLE_MESSAGE
from watermark.green_list import GreenListZTest, green_list_z_test


class FakeScorer:
    """SYNTHETIC scorer used only to test the arithmetic/plumbing of the model module (not a language model)."""

    def __init__(self, nll_value=math.log(4.0)):
        self.v = nll_value

    def score(self, text, progress=None):
        import re
        ms = list(re.finditer(r"\w+", text))
        n = len(ms)
        nll = np.full(n, self.v)
        nll[: n // 2] += 1.0                       # first half less predictable
        return TokenScores(nll, np.array([m.start() for m in ms]), np.array([m.end() for m in ms]),
                           np.ones(n, dtype=int) * 3, np.full(n, 2.0), {"model_name": "synthetic-fake", "tokenizer": "regex",
                                                                           "device": "cpu", "n_text_tokens": n})


class TestModelModule(unittest.TestCase):
    def test_summary_math(self):
        sc = TokenScores(np.full(200, math.log(4.0)), np.arange(200), np.arange(200) + 1, np.ones(200, int), np.full(200, 1.0))
        s = summarise_scores(sc)
        self.assertAlmostEqual(s["perplexity"], 4.0)
        self.assertAlmostEqual(s["cross_entropy_bits_per_token"], 2.0)
        self.assertAlmostEqual(s["surprisal_variance"], 0.0)
        self.assertEqual(s["rank_stats"]["top1_share"], 1.0)
        self.assertEqual(len(s["segment_perplexity"]), 5)

    def test_summary_too_few_tokens(self):
        sc = TokenScores(np.ones(5), np.arange(5), np.arange(5) + 1)
        self.assertEqual(summarise_scores(sc), {"n_tokens": 5})

    def test_window_surprisal_mapping(self):
        sc = TokenScores(np.concatenate([np.ones(20), np.full(20, 3.0)]), np.arange(40) * 2, np.arange(40) * 2 + 1)
        self.assertAlmostEqual(window_surprisal(sc, 0, 40), 1.0)
        self.assertAlmostEqual(window_surprisal(sc, 40, 80), 3.0)
        self.assertIsNone(window_surprisal(sc, 0, 10))              # < 15 tokens

    def test_missing_model_is_unavailable_and_nothing_else_breaks(self):
        t = make_text(3, 1500)
        for path in ("", "/definitely/not/a/model"):
            res = run_analysis(t, AnalysisSettings(enable_model=True, model_path=path))
            st = {s.name: s for s in res.statuses}
            self.assertEqual(st["perplexity"].status, "unavailable")
            self.assertEqual(st["sentence_statistics"].status, "ran")
            self.assertTrue(res.features)
            self.assertFalse([f for f in res.features if f.module == "perplexity" and f.kind == "model"])

    def test_missing_torch_is_reported(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules, {"torch": None, "transformers": None}):
            with self.assertRaises(ModelUnavailable) as cm:
                HFCausalLMScorer(d).load()
            self.assertIn("not installed", str(cm.exception))

    def test_loader_never_downloads(self):
        import re
        with open(perplexity.__file__, encoding="utf-8") as fh:
            src = fh.read()
        calls = re.findall(r"from_pretrained\([^)]*\)", src)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("local_files_only=True" in c for c in calls))
        self.assertNotIn("local_files_only=False", src)

    def test_pipeline_with_injected_scorer_adds_model_features_and_signals(self):
        t = make_text(5, 2500)
        res = run_analysis(t, AnalysisSettings(enable_model=True), scorer_override=FakeScorer())
        names = {f.name for f in res.features if f.module == "perplexity"}
        self.assertIn("Perplexity", names)
        ppl = next(f for f in res.features if f.name == "Perplexity")
        self.assertEqual(ppl.kind, "model")
        self.assertIn("reference model", (ppl.limitations + ppl.baseline).lower())
        self.assertEqual(ppl.strength, "n/a")                     # no calibrated threshold: no grade
        self.assertIn("window_baselines", res.tables)
        self.assertIn("mean_surprisal", res.tables["window_baselines"])


class TestWatermark(unittest.TestCase):
    def test_z_test_math(self):
        always = lambda p, t: True
        never = lambda p, t: False
        ids = list(range(200))
        r = green_list_z_test(ids, always, 0.25)
        self.assertEqual(r["T"], 199)
        self.assertAlmostEqual(r["z"], (199 - 0.25 * 199) / math.sqrt(199 * 0.25 * 0.75))
        self.assertLess(r["p"], 1e-30)
        self.assertGreater(green_list_z_test(ids, never, 0.25)["p"], 0.99)
        half = green_list_z_test(ids, lambda p, t: t % 2 == 0, 0.5)
        self.assertLess(abs(half["z"]), 1.0)
        self.assertGreater(half["p"], 0.2)

    def test_dedupe_pairs(self):
        ids = [1, 2] * 100
        self.assertEqual(green_list_z_test(ids, lambda p, t: True, 0.5, dedupe_pairs=True)["T"], 2)
        self.assertEqual(green_list_z_test(ids, lambda p, t: True, 0.5, dedupe_pairs=False)["T"], 199)

    def test_gamma_validation(self):
        with self.assertRaises(ValueError):
            green_list_z_test([1, 2, 3], lambda p, t: True, 1.5)

    def test_no_config_reports_unavailable(self):
        res = registry.run_all("some text " * 50)
        self.assertTrue(res and all(not r.ran for r in res))
        self.assertTrue(all(UNAVAILABLE_MESSAGE in r.message for r in res))
        self.assertIn("SynthID-Text", [r.scheme for r in res])

    def test_invalid_configs(self):
        d = GreenListZTest()
        for cfg in ({}, {"gamma": 2}, {"gamma": 0.25}, {"gamma": 0.25, "scheme_module": "/nope.py", "tokenizer_path": "/nope"}):
            r = d.detect("text", cfg)
            self.assertFalse(r.ran)
            self.assertIn(UNAVAILABLE_MESSAGE, r.message)
            self.assertIsNone(r.p_value)

    def test_runs_with_injected_tokenizer_and_reports_full_record(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [hash(w) % 1000 for w in text.split()]
        cfg = {"gamma": 0.25, "is_green": lambda p, t: True, "tokenizer": Tok()}
        text = " ".join("w%d" % i for i in range(300))
        r = GreenListZTest().detect(text, cfg, n_tests=2)
        self.assertTrue(r.ran)
        self.assertGreaterEqual(r.eligible_tokens, 25)
        self.assertIn("H0", r.null_hypothesis)
        self.assertLessEqual(r.p_value, r.adjusted_p_value)
        self.assertIn("Bonferroni", r.multiple_testing)
        self.assertTrue(r.limitations and r.version and r.implementation)

    def test_too_few_tokens_not_run(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]
        r = GreenListZTest().detect("a b c", {"gamma": 0.25, "is_green": lambda p, t: True, "tokenizer": Tok()})
        self.assertFalse(r.ran)

    def test_pipeline_reports_unavailable_watermark_not_success(self):
        res = run_analysis(make_text(2, 1200))
        wm = [f for f in res.features if f.module == "watermark"]
        self.assertTrue(wm and all(f.kind == "unavailable" for f in wm))
        self.assertEqual({s.name: s.status for s in res.statuses}["watermark"], "unavailable")

    def test_no_fake_detection_helpers(self):
        # the watermark package must not inspect punctuation / token frequencies / hidden characters
        for fn in ("base.py", "green_list.py", "registry.py"):
            with open(os.path.join(os.path.dirname(registry.__file__), fn), encoding="utf-8") as fh:
                src = fh.read().lower()
            self.assertNotIn("zero-width", src)
            self.assertNotIn("u+200b", src)

    def test_text_inspection_is_separate_and_disclaimed(self):
        res = run_analysis("Hello\u200b world. " * 60)
        self.assertGreaterEqual(res.inspection["total_invisible"], 60)
        self.assertIn("not proof of AI authorship", res.inspection["note"])


if __name__ == "__main__":
    unittest.main()
