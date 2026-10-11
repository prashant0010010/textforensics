import math
import unittest

import numpy as np
from scipy import stats as sps

from analyzers import (discourse, entropy, function_words, lexical_diversity, ngrams, paragraph_statistics, punctuation,
                       sentence_statistics, syntactic_features, zipf)
from config.settings import AnalysisSettings
from core import mathutils as mu
from core.features import compression_ratio, mattr, mtld
from core.segmentation import segment_document
from tests import samples
from tests.synthetic import make_text, repetitive_text


def run(mod, text, **kw):
    cfg = AnalysisSettings()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return mod.analyze(segment_document(text, cfg), cfg, {})


def feat(out, name_part):
    return next(f for f in out.features if name_part in f.name)


class TestMath(unittest.TestCase):
    def test_skew_kurtosis_match_scipy(self):
        x = [3, 5, 8, 8, 9, 14, 21, 30, 4, 7]
        self.assertAlmostEqual(mu.skewness(x), float(sps.skew(x, bias=False)), places=9)
        self.assertAlmostEqual(mu.excess_kurtosis(x), float(sps.kurtosis(x, bias=False)), places=9)

    def test_variance_cv_iqr(self):
        x = [2, 4, 4, 4, 5, 5, 7, 9]
        self.assertAlmostEqual(mu.variance(x), float(np.var(x, ddof=1)))
        self.assertAlmostEqual(mu.coefficient_of_variation(x), float(np.std(x, ddof=1) / np.mean(x)))
        self.assertAlmostEqual(mu.iqr(x), float(np.subtract(*np.percentile(x, [75, 25]))))

    def test_undefined_returns_none_not_nan(self):
        self.assertIsNone(mu.mean([]))
        self.assertIsNone(mu.variance([1]))
        self.assertIsNone(mu.skewness([1, 2]))
        self.assertIsNone(mu.skewness([5, 5, 5, 5]))           # zero variance
        self.assertIsNone(mu.excess_kurtosis([1, 2, 3]))
        self.assertIsNone(mu.coefficient_of_variation([0, 0, 0]))   # zero mean
        self.assertIsNone(mu.shannon_entropy([]))
        self.assertIsNone(mu.autocorrelation([1, 2, 3]))
        self.assertIsNone(mu.robust_z(1.0, 0.0, 0.0))

    def test_nan_inf_are_dropped(self):
        self.assertEqual(mu.mean([1, 2, float("nan"), float("inf")]), 1.5)
        self.assertIsNone(mu.finite_or_none(float("inf")))
        self.assertEqual(mu.safe_div(1, 0, None), None)
        self.assertEqual(mu.safe_div(1, 0), 0.0)

    def test_autocorrelation_alternating(self):
        self.assertLess(mu.autocorrelation([1, 9] * 10, 1), -0.8)
        self.assertGreater(mu.autocorrelation(list(range(30)), 1), 0.8)

    def test_entropy_values(self):
        self.assertAlmostEqual(mu.shannon_entropy([1, 1, 1, 1]), 2.0)
        self.assertAlmostEqual(mu.shannon_entropy([10]), 0.0)
        self.assertAlmostEqual(mu.normalized_entropy([5, 5]), 1.0)
        self.assertGreater(mu.miller_madow_entropy([3, 2, 1, 1]), mu.shannon_entropy([3, 2, 1, 1]))

    def test_entropy_numerically_stable_for_huge_counts(self):
        h = mu.shannon_entropy([10 ** 15, 10 ** 15 + 1, 3])
        self.assertTrue(math.isfinite(h))

    def test_js(self):
        self.assertAlmostEqual(mu.js_divergence({"a": 1, "b": 1}, {"a": 2, "b": 2}), 0.0, places=12)
        self.assertAlmostEqual(mu.js_divergence({"a": 1}, {"b": 1}), 1.0, places=12)
        self.assertIsNone(mu.js_divergence({}, {}))

    def test_linear_fit_exact(self):
        f = mu.linear_fit([1, 2, 3, 4], [3, 5, 7, 9])
        self.assertAlmostEqual(f["slope"], 2.0)
        self.assertAlmostEqual(f["intercept"], 1.0)
        self.assertAlmostEqual(f["r2"], 1.0)
        self.assertIsNone(mu.linear_fit([1, 1, 1], [1, 2, 3]))

    def test_ks(self):
        d, p = mu.ks_two_sample([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        self.assertEqual(d, 0.0)
        self.assertEqual(p, 1.0)
        d, p = mu.ks_two_sample(list(range(50)), list(range(100, 150)))
        self.assertEqual(d, 1.0)
        self.assertLess(p, 1e-6)

    def test_robust_scale_fallback(self):
        x = [1] * 10 + [2] * 3
        self.assertGreaterEqual(mu.robust_scale(x), 0.0)
        self.assertEqual(mu.robust_scale([5, 5, 5, 5]), 0.0)


class TestLexicalMeasures(unittest.TestCase):
    def test_mattr_known(self):
        self.assertAlmostEqual(mattr(list("abcdefghij"), 5), 1.0)
        self.assertAlmostEqual(mattr(["a"] * 10, 5), 0.2)
        self.assertIsNone(mattr(["a", "b"], 5))

    def test_mattr_matches_bruteforce(self):
        toks = list(np.random.default_rng(1).choice(list("abcdefgh"), 80))
        brute = np.mean([len(set(toks[i:i + 10])) / 10 for i in range(len(toks) - 9)])
        self.assertAlmostEqual(mattr(toks, 10), float(brute))

    def test_mtld(self):
        self.assertIsNone(mtld(["w%d" % i for i in range(100)]))          # TTR never drops: undefined
        self.assertIsNone(mtld(["a"] * 10))                               # < 50 tokens
        v = mtld(list("abcdefghij") * 10)
        self.assertAlmostEqual(v, 14.0, delta=1.0)                        # ~14 tokens per factor (TTR 10/14 < 0.72)
        self.assertLess(mtld(["a", "b"] * 50), mtld(list("abcdefghij") * 10))

    def test_length_dependence_of_ttr_but_not_mattr(self):
        long_ = make_text(3, 6000).lower().split()
        short = long_[:600]
        ttr = lambda t: len(set(t)) / len(t)
        self.assertGreater(ttr(short) - ttr(long_), 0.1)
        self.assertLess(abs(mattr(short, 50) - mattr(long_, 50)), 0.03)


class TestAnalyzers(unittest.TestCase):
    def test_sentence_statistics_known_lengths(self):
        sents = []
        for i in range(30):
            n = 5 if i % 2 == 0 else 15
            sents.append(" ".join(["word"] * n) + ".")
        out = run(sentence_statistics, "\n\n".join(sents))
        st = out.tables["sentence_length_stats"]
        self.assertAlmostEqual(st["mean"], 10.0)
        self.assertAlmostEqual(st["median"], 10.0)
        self.assertAlmostEqual(st["mean_abs_consecutive_diff"], 10.0)
        self.assertLess(st["lag1_autocorrelation"], -0.8)
        self.assertEqual(st["share_short"], 0.5)

    def test_sentence_cv_graded_for_uniform_text(self):
        uniform = " ".join("one two three four five six seven eight nine ten eleven twelve." for _ in range(40))
        f = feat(run(sentence_statistics, uniform), "Coefficient of variation")
        self.assertEqual(f.strength, "strong")
        self.assertEqual(f.kind, "heuristic")
        self.assertIn("uncalibrated", f.interpretation.lower())

    def test_variable_text_not_flagged(self):
        f = feat(run(sentence_statistics, make_text(5, 3000, sent_words=(4, 40))), "Coefficient of variation")
        self.assertIn(f.strength, ("none", "weak"))

    def test_too_short_is_unavailable_not_fabricated(self):
        out = run(sentence_statistics, samples.SHORT)
        self.assertEqual(out.features[0].kind, "unavailable")
        for mod in (lexical_diversity, entropy, zipf, ngrams, punctuation, function_words, syntactic_features, discourse):
            o = run(mod, samples.SHORT)
            self.assertTrue(all(f.kind == "unavailable" for f in o.features), mod.__name__)

    def test_paragraph_statistics(self):
        out = run(paragraph_statistics, samples.ACADEMIC, include_headings=False)
        f = feat(out, "Mean sentences per paragraph")
        self.assertGreater(f.value, 1)
        out2 = run(paragraph_statistics, samples.SINGLE_PARA)
        self.assertEqual(out2.features[0].kind, "unavailable")

    def test_paragraph_structural_repetition_signal(self):
        paras = ["\n\n".join(" ".join("alpha beta gamma delta epsilon zeta eta theta." for _ in range(3)) for _ in range(7))][0]
        out = run(paragraph_statistics, paras)
        self.assertTrue(any(s.feature == "Paragraph structural repetition" for s in out.signals))

    def test_lexical_zero_division_safe(self):
        out = run(lexical_diversity, "same " * 60)
        for f in out.features:
            self._assert_finite(f.value)

    def _assert_finite(self, v):
        if isinstance(v, float):
            self.assertTrue(math.isfinite(v))
        elif isinstance(v, dict):
            for x in v.values():
                self._assert_finite(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                self._assert_finite(x)

    def test_all_outputs_finite_on_varied_text(self):
        for t in (make_text(2, 2500), repetitive_text(), samples.ACADEMIC * 4, samples.PUNCT * 40, "same " * 400):
            for mod in (sentence_statistics, paragraph_statistics, lexical_diversity, entropy, zipf, ngrams, punctuation,
                        function_words, syntactic_features, discourse):
                for f in run(mod, t).features:
                    self._assert_finite(f.value)

    def test_entropy_repetition_sensitive(self):
        varied = run(entropy, make_text(4, 2000))
        rep = run(entropy, ("the cat sat on the mat. " * 200))
        self.assertLess(feat(rep, "Compression ratio").value, feat(varied, "Compression ratio").value)
        self.assertLess(feat(rep, "Character entropy").value, feat(varied, "Character entropy").value + 0.5)
        self.assertLess(feat(rep, "Word entropy (unigram").value, feat(varied, "Word entropy (unigram").value)
        self.assertIsNotNone(feat(varied, "Conditional word entropy").value)

    def test_zipf_fit_on_zipfian_sample(self):
        rng = np.random.default_rng(0)
        V = 800
        p = 1 / np.arange(1, V + 1)
        p /= p.sum()
        words = ["w%d" % i for i in rng.choice(V, 20000, p=p)]
        text = " ".join(w + "." if i % 12 == 11 else w for i, w in enumerate(words))
        out = run(zipf, text)
        fit = out.tables["zipf"]["fits"]["ranks_with_freq>=2"]
        self.assertTrue(-1.3 < fit["slope"] < -0.7, fit)
        self.assertGreater(fit["r2"], 0.85)
        f = feat(out, "Zipf slope (ranks_with_freq>=2)")
        self.assertIn("Clauset", f.limitations)

    def test_ngram_repetition_exceeds_shuffle_baseline(self):
        out = run(ngrams, repetitive_text(30))
        row = next(r for r in out.tables["word_ngram_summary"] if r["n"] == 4)
        self.assertGreater(row["repeated_share"], 0.9)
        self.assertGreater(row["repeated_share"], row["shuffled_mean"])
        self.assertTrue(out.signals)
        varied = run(ngrams, make_text(8, 3000, sent_words=(6, 30)))
        row2 = next(r for r in varied.tables["word_ngram_summary"] if r["n"] == 4)
        self.assertLess(row2["repeated_share"], row["repeated_share"])

    def test_ngram_deterministic_with_seed(self):
        a = run(ngrams, repetitive_text(10), seed=5).tables["word_ngram_summary"]
        b = run(ngrams, repetitive_text(10), seed=5).tables["word_ngram_summary"]
        self.assertEqual(a, b)

    def test_punctuation_counts(self):
        text = "Yes, indeed; however: (see note) \u201cquoted\u201d \u2014 done. " * 10
        out = run(punctuation, text)
        c = out.tables["punctuation_rates"]["counts"]
        self.assertEqual(c["comma"], 10)
        self.assertEqual(c["semicolon"], 10)
        self.assertEqual(c["colon"], 10)
        self.assertEqual(c["dash"], 10)
        self.assertEqual(c["paren_open"], 10)
        self.assertIn("Do not treat" if False else "reliable AI fingerprint", feat(out, "Comma").limitations)

    def test_discourse_markers_and_alternatives(self):
        out = run(discourse, repetitive_text(10))
        rates = out.tables["discourse_rates"]
        self.assertGreater(rates["additive"], 0)
        self.assertGreater(rates["contrastive"], 0)
        self.assertGreater(rates["causal"], 0)
        f = feat(out, "Sentence-initial transition rate")
        self.assertGreater(f.value, 0.5)
        self.assertTrue(f.alternatives)

    def test_function_words_and_pos_fallback_labelled_approximate(self):
        out = run(function_words, samples.ACADEMIC * 3)
        self.assertGreater(feat(out, "Function-word share").value, 0.2)
        syn = run(syntactic_features, make_text(2, 1500), lightweight=True)
        info = feat(syn, "POS tagger used").value
        self.assertTrue(info["approximate"])
        self.assertIn("approximation", feat(syn, "POS distribution").limitations)

    def test_punctuation_heavy_and_unicode_inputs(self):
        for t in (samples.PUNCT * 30, samples.UNICODE * 30):
            for mod in (punctuation, sentence_statistics, lexical_diversity, entropy):
                run(mod, t)


if __name__ == "__main__":
    unittest.main()


class TestAnomalyRobustness(unittest.TestCase):
    def test_homogeneous_document_does_not_produce_strong_anomalies(self):
        from analyzers import internal_anomalies
        text = make_text(43, 1500, uniform=True)
        cfg = AnalysisSettings()
        out = internal_anomalies.analyze(segment_document(text, cfg), cfg, {})
        self.assertFalse([s for s in out.signals if s.strength == "strong"], [(s.feature, s.details.get("z")) for s in out.signals])
        base = out.tables["window_baselines"]["marker_per_100w"]
        self.assertIn("floor_applied", base)

    def test_zipf_handles_flat_frequency_distribution(self):
        words = ["w%d" % i for i in range(60)] * 5          # every word exactly 5 times: zero variance in log-frequency
        text = " ".join(w + ("." if i % 10 == 9 else "") for i, w in enumerate(words))
        out = run(zipf, text * 1)
        self.assertTrue(out.features)

    def test_anomaly_window_flagged_in_otherwise_uniform_text(self):
        from analyzers import internal_anomalies
        base = make_text(50, 3000, sent_words=(10, 18))
        odd = " ".join("Go." for _ in range(60))                    # very short sentences inserted mid-document
        paras = base.split("\n\n")
        text = "\n\n".join(paras[:len(paras) // 2] + [odd] + paras[len(paras) // 2:])
        cfg = AnalysisSettings()
        out = internal_anomalies.analyze(segment_document(text, cfg), cfg, {})
        pos = text.index(odd)
        hits = [s for s in out.signals if s.start <= pos + len(odd) and s.end >= pos]
        self.assertTrue(hits, "the inserted passage should deviate from the document baseline")
