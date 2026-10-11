"""Module B.1 - sentence-length statistics."""
from __future__ import annotations

from typing import Any, Dict, List

from config.settings import AnalysisSettings
from core import mathutils as mu
from core.schemas import AnalyzerOutput, Document, FeatureResult
from scoring.evidence import UNCALIBRATED, grade_low, unavailable

MODULE = "sentence_statistics"


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    sents = [s for s in doc.included_sentences() if s.n_words > 0]
    lens = [s.n_words for s in sents]
    n = len(lens)
    if n < 3:
        out.features.append(unavailable(MODULE, "Sentence-length statistics", f"only {n} analysable sentence(s)"))
        return out
    short = sum(1 for x in lens if x <= cfg.short_sentence_max) / n
    long_ = sum(1 for x in lens if x >= cfg.long_sentence_min) / n
    diffs = [abs(lens[i + 1] - lens[i]) for i in range(n - 1)]
    cv = mu.coefficient_of_variation(lens)
    ac = mu.autocorrelation(lens, 1)
    stats = {
        "n_sentences": n, "mean": mu.mean(lens), "median": mu.median(lens), "std": mu.std(lens),
        "variance": mu.variance(lens), "iqr": mu.iqr(lens), "cv": cv, "skewness": mu.skewness(lens),
        "excess_kurtosis": mu.excess_kurtosis(lens) if n >= 8 else None,
        "share_short": short, "share_medium": 1 - short - long_, "share_long": long_,
        "mean_abs_consecutive_diff": mu.mean(diffs), "lag1_autocorrelation": ac,
        "min": min(lens), "max": max(lens),
    }
    ctx["sentence_stats"] = stats
    out.tables["sentence_length_stats"] = stats
    out.tables["sentence_lengths"] = lens

    def add(name, value, explanation, unit="", strength="n/a", baseline="", interp="", lim="", alts=None, kind="measured"):
        out.features.append(FeatureResult(MODULE, name, value, unit, kind, baseline, interp, explanation,
                                          None, strength, "sentence_variability" if strength != "n/a" else "",
                                          lim, alts or []))

    add("Mean sentence length", stats["mean"], "Average words per sentence.", "words")
    add("Median sentence length", stats["median"], "Middle value; robust to a few very long sentences.", "words")
    add("Sentence-length standard deviation", stats["std"], "Spread of sentence lengths (sample SD).", "words")
    add("Sentence-length variance", stats["variance"], "Squared spread (sample variance).", "words^2")
    add("Sentence-length IQR", stats["iqr"], "Interquartile range: spread of the middle 50% of sentences.", "words")
    low = cfg.thresholds.sentence_len_cv_low
    strength = grade_low(cv, low) if n >= 20 else "n/a"
    interp = ("Not graded (fewer than 20 sentences)." if n < 20 else
              {"none": "Sentence lengths vary widely; this feature gives no indication.",
               "weak": "Sentence lengths are somewhat uniform.",
               "moderate": "Sentence lengths are noticeably uniform.",
               "strong": "Sentence lengths are very uniform."}[strength])
    add("Coefficient of variation of sentence length", cv,
        "SD divided by mean: spread relative to typical length (sometimes called burstiness).", "ratio",
        strength, f"Heuristic cut-offs (weak/moderate/strong): <= {low[0]}/{low[1]}/{low[2]}",
        interp + " " + UNCALIBRATED,
        "Uniform sentence length also occurs in legal, technical, instructional, simplified-language, and "
        "heavily edited human writing; high variance does not indicate human authorship.",
        ["Genre or style guide imposing short, regular sentences", "Non-native or simplified-register writing",
         "Heavy editing or templated text"], kind="heuristic" if strength != "n/a" else "measured")
    add("Skewness of sentence length", stats["skewness"], "Asymmetry of the length distribution (G1).", "",
        lim="Sensitive to a few extreme sentences.")
    add("Excess kurtosis of sentence length", stats["excess_kurtosis"],
        "Tail heaviness relative to a normal distribution (G2); reported only for n >= 8.", "",
        lim="Needs a reasonably large sample to be stable.")
    add("Share of short / medium / long sentences",
        {"short": short, "medium": stats["share_medium"], "long": long_},
        f"Short <= {cfg.short_sentence_max} words, long >= {cfg.long_sentence_min} words.")
    add("Mean absolute consecutive length difference", stats["mean_abs_consecutive_diff"],
        "Average |length(i+1) - length(i)|: how much neighbouring sentences differ.", "words")
    add("Lag-1 autocorrelation of sentence length", ac,
        "Correlation between a sentence's length and the next one's (needs >= 10 sentences).", "",
        lim="Unstable for short documents; no validated interpretation threshold is applied.")
    return out
