import random
import unittest

from analyzers import reference_comparison
from config.settings import AnalysisSettings
from core.pipeline import run_analysis
from core.segmentation import segment_document
from evaluation import framework as ev
from tests import samples
from tests.synthetic import make_text


class TestReferenceComparison(unittest.TestCase):
    def setUp(self):
        self.a = make_text(1, 2500)
        self.b = make_text(1, 2500)
        self.c = samples.ACADEMIC * 12

    def test_requires_confirmation(self):
        res = run_analysis(self.a, AnalysisSettings(enable_reference=True, reference_confirmed=False), reference=self.b)
        f = [x for x in res.features if x.module == "reference_comparison"]
        self.assertTrue(f and all(x.kind == "unavailable" for x in f))
        self.assertIn("confirmed", f[0].interpretation)

    def test_no_reference_unavailable(self):
        res = run_analysis(self.a, AnalysisSettings(enable_reference=True, reference_confirmed=True))
        self.assertTrue(all(x.kind == "unavailable" for x in res.features if x.module == "reference_comparison"))

    def test_identical_vs_different_samples(self):
        cfg = AnalysisSettings(lightweight=True)
        da, db, dc = (segment_document(t, cfg) for t in (self.a, self.b, self.c))
        same = {f.name: f.value for f in reference_comparison.compare_documents(da, db, cfg)}
        diff = {f.name: f.value for f in reference_comparison.compare_documents(da, dc, cfg)}
        self.assertAlmostEqual(same["Function-word profile"]["js_distance"], 0.0, places=9)
        self.assertGreater(diff["Function-word profile"]["js_distance"], 0.05)
        self.assertEqual(same["Sentence-length distribution"]["ks_D"], 0.0)
        self.assertGreater(same["Vocabulary overlap"]["jaccard_all_types"], diff["Vocabulary overlap"]["jaccard_all_types"])

    def test_metrics_reported_separately_without_verdict(self):
        res = run_analysis(self.a, AnalysisSettings(enable_reference=True, reference_confirmed=True, lightweight=True), reference=self.c)
        rc = [f for f in res.features if f.module == "reference_comparison"]
        names = " ".join(f.name.lower() for f in rc)
        for part in ("vocabulary", "function-word", "sentence-length", "punctuation", "pos", "n-gram", "paragraph"):
            self.assertIn(part, names)
        self.assertNotIn("same author", names)
        self.assertNotIn("match", names)
        self.assertTrue(all("cannot prove authorship" in f.limitations or "unstable" in f.limitations or "Differences" in f.limitations for f in rc))

    def test_small_reference_flagged(self):
        cfg = AnalysisSettings(lightweight=True)
        out = reference_comparison.compare_documents(segment_document(self.a, cfg), segment_document(samples.SINGLE_PARA, cfg), cfg)
        self.assertIn("unstable", out[0].limitations)


class TestEvaluationFramework(unittest.TestCase):
    def test_auc_known(self):
        self.assertEqual(ev.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(ev.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(ev.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertIsNone(ev.roc_auc([1, 1], [0.1, 0.2]))

    def test_metrics(self):
        m = ev.classification_metrics([0, 0, 0, 1, 1, 1], [0.1, 0.6, 0.2, 0.7, 0.4, 0.9], 0.5)
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (2, 1, 2, 1))
        self.assertAlmostEqual(m["false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(m["false_negative_rate"], 1 / 3)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)

    def _records(self):
        rng = random.Random(0)
        return [{"id": i, "label": i % 2, "score": rng.random() * 0.5 + 0.4 * (i % 2), "author": f"a{i % 9}", "generator": f"g{i % 4}" if i % 2 else "human",
                 "domain": f"d{i % 3}", "n_words": 200 + 40 * i, "edited": (i // 2) % 2} for i in range(300)]

    def test_group_split_is_disjoint(self):
        for key in ("author", "generator", "domain"):
            tr, te = ev.group_split(self._records(), key, 0.3, seed=1)
            self.assertFalse({r[key] for r in tr} & {r[key] for r in te})
            self.assertTrue(tr and te)

    def test_leave_one_group_out(self):
        for g, tr, te in ev.leave_one_group_out(self._records(), "domain"):
            self.assertTrue(all(r["domain"] == g for r in te))
            self.assertTrue(all(r["domain"] != g for r in tr))

    def test_evaluate_strata_and_ci(self):
        out = ev.evaluate(self._records(), 0.5, probabilities=True, n_boot=100)
        for k in ("overall", "by_domain", "by_generator", "by_length", "by_editing", "calibration", "caveat"):
            self.assertIn(k, out)
        lo, hi = out["overall"]["roc_auc_ci95"]
        self.assertLessEqual(lo, out["overall"]["roc_auc"])
        self.assertGreaterEqual(hi, out["overall"]["roc_auc"])
        self.assertIn("do not establish accuracy", out["caveat"])

    def test_no_dataset_or_results_shipped(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertEqual([f for f in os.listdir(os.path.join(root, "evaluation")) if f.endswith((".csv", ".json"))], [])


if __name__ == "__main__":
    unittest.main()
