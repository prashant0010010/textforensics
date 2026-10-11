"""Centralised, editable configuration.

Heuristic thresholds below are *conventions chosen for transparency*, not
values validated against labelled data. They control only the strength labels
of heuristic indicators and can be edited by the user.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

APP_NAME = "TextForensics Local"
APP_VERSION = "1.0.0"
HARD_LIMIT_WORDS = 200_000       # refused with an explicit error (never silently truncated)
SOFT_LIMIT_WORDS = 30_000        # above this a warning is shown; analysis still runs


@dataclass
class Thresholds:
    """(weak, moderate, strong) cut-offs. ``low`` indicators fire when value <= cut-off."""

    sentence_len_cv_low: Tuple[float, float, float] = (0.45, 0.38, 0.30)
    paragraph_len_cv_low: Tuple[float, float, float] = (0.30, 0.22, 0.15)
    opening_repeat_share_high: Tuple[float, float, float] = (0.15, 0.25, 0.40)
    transition_initial_rate_high: Tuple[float, float, float] = (0.20, 0.30, 0.40)
    hedge_per_1000_high: Tuple[float, float, float] = (12.0, 20.0, 30.0)
    repeated_ngram_share_high: Tuple[float, float, float] = (0.05, 0.10, 0.20)
    template_per_1000_high: Tuple[float, float, float] = (3.0, 6.0, 10.0)
    cee_fraction_high: Tuple[float, float, float] = (0.60, 0.75, 0.90)
    # robust z-scores for document-internal anomalies
    anomaly_z: Tuple[float, float, float] = (3.0, 4.0, 5.0)


@dataclass
class AnalysisSettings:
    # module switches
    enable_statistical: bool = True
    enable_lexical: bool = True
    enable_structural: bool = True
    enable_linguistic: bool = True
    enable_internal: bool = True
    enable_model: bool = False
    enable_reference: bool = False
    enable_watermark: bool = True
    lightweight: bool = False

    # treatment of special blocks (excluded blocks are displayed but not analysed)
    include_quotes: bool = False
    include_references: bool = False
    include_bullets: bool = True
    include_code: bool = False
    include_tables: bool = False
    include_headings: bool = False
    single_newline_paragraphs: str = "auto"       # auto | yes | no

    # sentence statistics
    short_sentence_max: int = 10
    long_sentence_min: int = 26

    # lexical
    mattr_windows: Tuple[int, ...] = (25, 50, 100)
    mtld_threshold: float = 0.72
    # n-grams
    word_ngram_min: int = 2
    word_ngram_max: int = 5
    char_ngram_sizes: Tuple[int, ...] = (3, 4, 5)
    shuffle_iterations: int = 20
    local_repeat_gap_tokens: int = 200
    # entropy
    entropy_window_tokens: int = 100
    # windows (multi-scale)
    window_sentences: int = 8
    window_step: int = 4
    min_window_words: int = 80
    min_windows_for_baseline: int = 8
    # evidence
    min_words_for_assessment: int = 300
    min_sentences_for_assessment: int = 15
    max_highlight_coverage: float = 0.35
    max_highlighted_signals: int = 60
    seed: int = 1234
    thresholds: Thresholds = field(default_factory=Thresholds)

    # optional reference language model (local path only; never downloaded automatically)
    model_path: str = ""
    model_device: str = "auto"
    model_max_length: int = 1024
    model_stride: int = 512
    # optional reference writing
    reference_confirmed: bool = False
    # optional watermark detector configuration (JSON dict)
    watermark_configs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def effective(self) -> "AnalysisSettings":
        """Return settings adjusted for lightweight mode (copy)."""
        if not self.lightweight:
            return self
        import copy

        s = copy.deepcopy(self)
        s.shuffle_iterations = min(s.shuffle_iterations, 8)
        s.char_ngram_sizes = (4,)
        s.window_step = max(s.window_step, s.window_sentences)
        s.word_ngram_max = min(s.word_ngram_max, 4)
        return s
