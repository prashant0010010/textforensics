import os
import unittest

from config.settings import AnalysisSettings
from core.pipeline import run_analysis
from core.schemas import FeatureResult, ModuleStatus, Signal, TAXONOMY
from scoring import calibration
from scoring.aggregation import aggregate
from scoring.evidence import grade_high, grade_low, max_strength, strength_from_z
from tests import samples
from tests.synthetic import make_text, repetitive_text

CFG = AnalysisSettings()
STATS_OK = {"words_analysed": 2000, "sentences_analysed": 120}
RAN = [ModuleStatus("internal_anomalies", "ran")]


def ind(family, strength, value=0.1):
    return FeatureResult("m", f"ind-{family}", value, kind="heuristic", strength=strength, family=family)


def sig(family, strength):
    return Signal("repetition", "x", "sentence", 0, 10, 0, 0, 0, 1.0, "b", strength, "w", ["a"], "n", family)


class TestGrading(unittest.TestCase):
    def test_grade_high_low(self):
        self.assertEqual(grade_high(0.5, (0.2, 0.3, 0.4)), "strong")
        self.assertEqual(grade_high(0.25, (0.2, 0.3, 0.4)), "weak")
        self.assertEqual(grade_high(0.1, (0.2, 0.3, 0.4)), "none")
        self.assertEqual(grade_low(0.1, (0.45, 0.38, 0.30)), "strong")
        self.assertEqual(grade_low(0.5, (0.45, 0.38, 0.30)), "none")
        self.assertEqual(grade_low(None, (0.45, 0.38, 0.30)), "n/a")

    def test_z_strength(self):
        cuts = (3, 4, 5)
        self.assertEqual(strength_from_z(-5.5, cuts), "strong")
        self.assertEqual(strength_from_z(3.2, cuts), "weak")
        self.assertEqual(strength_from_z(1.0, cuts), "none")
        self.assertEqual(strength_from_z(None, cuts), "n/a")
        self.assertEqual(max_strength(["weak", "strong", "none"]), "strong")


class TestTaxonomy(unittest.TestCase):
    def test_five_labels_exact(self):
        self.assertEqual(len(TAXONOMY), 5)
        self.assertEqual(TAXONOMY[5], "Insufficient data to assess")
        self.assertEqual(TAXONOMY[1], "Little evidence from the tested features")

    def test_insufficient_data(self):
        ev = aggregate({"words_analysed": 80, "sentences_analysed": 5}, [], [], RAN, CFG)
        self.assertEqual(ev["category"], 5)

    def test_little_evidence(self):
        feats = [ind("sentence_variability", "none"), ind("repetition", "none")]
        self.assertEqual(aggregate(STATS_OK, feats, [], RAN, CFG)["category"], 1)

    def test_single_weak_still_little(self):
        feats = [ind("sentence_variability", "weak"), ind("repetition", "none")]
        self.assertEqual(aggregate(STATS_OK, feats, [], RAN, CFG)["category"], 1)

    def test_mixed(self):
        feats = [ind("sentence_variability", "moderate"), ind("repetition", "none")]
        self.assertEqual(aggregate(STATS_OK, feats, [], RAN, CFG)["category"], 2)

    def test_several(self):
        feats = [ind("sentence_variability", "moderate"), ind("repetition", "moderate")]
        self.assertEqual(aggregate(STATS_OK, feats, [], RAN, CFG)["category"], 3)
        feats = [ind("sentence_variability", "weak"), ind("repetition", "weak"), ind("discourse_rhetoric", "weak")]
        self.assertEqual(aggregate(STATS_OK, feats, [], RAN, CFG)["category"], 3)

    def test_strong_convergence_needs_many_families(self):
        fams = ["sentence_variability", "repetition", "discourse_rhetoric", "paragraph_structure", "lexical_anomaly"]
        feats = [ind(f, "strong") for f in fams[:2]] + [ind(f, "moderate") for f in fams[2:]]
        self.assertEqual(aggregate(STATS_OK, feats, [], RAN, CFG)["category"], 4)
        # three strong families alone are not "strong convergence"
        feats = [ind(f, "strong") for f in fams[:3]]
        self.assertEqual(aggregate(STATS_OK, feats, [], RAN, CFG)["category"], 3)

    def test_inconsistency_family_does_not_count_toward_label(self):
        feats = [ind("internal_inconsistency", "strong")] + [sig_f for sig_f in []]
        sigs = [sig("internal_inconsistency", "strong") for _ in range(3)]
        ev = aggregate(STATS_OK, feats, sigs, RAN, CFG)
        self.assertEqual(ev["category"], 1)
        row = next(f for f in ev["family_table"] if f["family"] == "internal_inconsistency")
        self.assertFalse(row["counts_toward_label"])

    def test_signals_raise_family_level(self):
        sigs = [sig("repetition", "strong"), sig("repetition", "strong")]
        ev = aggregate(STATS_OK, [], sigs, RAN, CFG)
        self.assertEqual(next(f for f in ev["family_table"] if f["family"] == "repetition")["level"], "strong")

    def test_never_returns_probability(self):
        ev = aggregate(STATS_OK, [ind("repetition", "strong")], [], RAN, CFG)
        self.assertIsNone(ev["calibrated_probability"])
        self.assertIn("No calibrated probability", ev["calibration_note"])
        self.assertIn("experimental", ev["experimental_index"]["label"].lower())
        self.assertIn("NOT a probability", ev["experimental_index"]["label"])

    def test_evidence_types_are_distinguished(self):
        feats = [FeatureResult("m", "a", 1, kind="measured"), ind("repetition", "weak"),
                 FeatureResult("p", "b", 1, kind="model"), FeatureResult("w", "c", None, kind="unavailable")]
        ev = aggregate(STATS_OK, feats, [], RAN, CFG)["evidence_types"]
        self.assertEqual((ev["directly_measured"], ev["heuristic"], ev["model_based"], ev["unavailable"]), (1, 1, 1, 1))


class TestEndToEndEvidence(unittest.TestCase):
    def test_empty_like_short_input_is_insufficient(self):
        r = run_analysis(samples.SHORT)
        self.assertEqual(r.evidence["category"], 5)

    def test_every_finding_has_explanation_fields(self):
        r = run_analysis(repetitive_text(30))
        self.assertTrue(r.signals)
        for s in r.signals:
            self.assertTrue(s.why and s.baseline and s.alternatives and s.needs_more_evidence)
            self.assertIn(s.strength, ("weak", "moderate", "strong"))
        for f in r.features:
            self.assertTrue(f.name and f.kind in ("measured", "heuristic", "model", "watermark", "unavailable"))
            if f.kind == "heuristic":
                self.assertIn("uncalibrated", (f.interpretation + f.limitations).lower())

    def test_recommendations_reference_actual_findings(self):
        r = run_analysis(repetitive_text(30))
        texts = " ".join(x["text"] for x in r.recommendations)
        self.assertIn("recurs", texts)
        self.assertIn("proof of misconduct", texts)
        phrase = next(s.details["phrase"] for s in r.signals if s.feature == "Recurring phrase")
        self.assertIn(phrase, texts)
        calm = run_analysis(make_text(3, 2500, sent_words=(4, 40), para_words=(40, 200)))
        if not calm.signals:
            self.assertIn("No passage was flagged", " ".join(x["text"] for x in calm.recommendations))

    def test_calibration_gate_closed_by_default(self):
        self.assertFalse(os.path.exists(calibration.CAL_PATH))
        self.assertIsNone(calibration.calibration_record())
        self.assertIsNone(calibration.calibrated_probability({}))

    def test_calibration_metrics(self):
        y = [0, 0, 1, 1]
        p = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(calibration.brier_score(y, p), (0.01 + 0.04 + 0.04 + 0.01) / 4)
        self.assertLess(calibration.expected_calibration_error(y, p), 0.2)


if __name__ == "__main__":
    unittest.main()
