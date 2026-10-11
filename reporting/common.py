"""Shared helpers for report generation (all local, no network)."""
from __future__ import annotations

from typing import Any, Dict, List

from config.settings import APP_NAME, APP_VERSION
from core.schemas import AnalysisResult
from visualization.highlights import select_signals

NOT_EQUIVALENT = (f"{APP_NAME} is an independent, transparent statistical tool. It is not equivalent to, validated against, or "
                  "affiliated with Turnitin, GPTZero, ZeroGPT or any other proprietary detector, and its output must not be "
                  "treated as proof of AI use or misconduct.")

LIMITATIONS = [
    "No labelled-data validation has been performed by this software; no accuracy, false-positive or false-negative rate is claimed.",
    "Statistical regularities (uniform sentences, repetition, predictability) also occur in human writing, especially formulaic genres, "
    "technical prose, non-native English and heavily edited text.",
    "Machine text that has been edited, paraphrased, or produced with careful prompting can show none of the measured patterns.",
    "Heuristic strength labels use conventional thresholds (config/settings.py) that are uncalibrated.",
    "Sentence/paragraph segmentation and POS tagging (when the built-in fallback is used) are approximations.",
    "Reference-model results depend on the chosen model; watermark tests are valid only for the exact scheme and configuration tested.",
]

METHODOLOGY = [
    ("Segmentation", "Paragraphs split on blank lines (optionally single newlines); rule-based sentence splitter with abbreviation handling; "
                     "regex word tokeniser; offsets always refer to the canonical (NFC, LF) text."),
    ("Sentence/paragraph statistics", "Sample mean, SD, variance, IQR, CV, adjusted Fisher-Pearson skewness, unbiased excess kurtosis, lag-1 autocorrelation."),
    ("Lexical diversity", "TTR, MATTR (Covington & McFall), MTLD (McCarthy & Jarvis, bidirectional with partial factors), hapax ratios, Herdan's C, Yule's K."),
    ("Entropy", "Plug-in Shannon entropy over empirical distributions with Miller-Madow bias correction; conditional entropy as H(joint)-H(context); "
                "fixed-size token windows for local entropy; zlib compression ratio."),
    ("N-grams", "Repeated-type share for word n-grams (2-5) compared with a seeded word-shuffle baseline; opening/ending repetition; "
                "local vs global repetition; character n-gram profiles."),
    ("Zipf", "OLS on log-frequency vs log-rank, with R^2/RMSE and a hapax-trimmed fit; reported as descriptive."),
    ("Function words / POS / discourse", "Closed-class lexicon rates; POS via spaCy/NLTK if present else a rule-based fallback; configurable discourse-marker lists."),
    ("Internal anomalies", "Sliding sentence windows; robust z-scores (median, 1.4826*MAD) against the document's own distribution; adjacent flagged windows merged."),
    ("Evidence aggregation", "Counting rules over feature families (see scoring/aggregation.py); no weights or probabilities."),
    ("Optional modules", "Reference LM: log-softmax of local causal-LM logits (never downloaded automatically). Watermark: real statistical test only when scheme and configuration are supplied."),
]


def report_title(res: AnalysisResult) -> str:
    return f"{APP_NAME} report"


def selected_signals(res: AnalysisResult):
    cfg = res.settings
    return select_signals(res.signals, len(res.document.text), cfg.get("max_highlight_coverage", 0.35), cfg.get("max_highlighted_signals", 60))


def modules_summary(res: AnalysisResult) -> List[Dict[str, Any]]:
    return [{"module": s.name, "status": s.status, "detail": s.detail, "seconds": round(s.seconds, 3)} for s in res.statuses]
