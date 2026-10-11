"""Structured data models shared by every module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

STRENGTH_RANK = {"n/a": -1, "none": 0, "weak": 1, "moderate": 2, "strong": 3}

# Evidence taxonomy (describes evidence, never authorship or misconduct).
TAXONOMY = {
    1: "Little evidence from the tested features",
    2: "Mixed or inconclusive evidence",
    3: "Several features associated with AI-generated text",
    4: "Strong convergence of multiple informative features",
    5: "Insufficient data to assess",
}

CATEGORIES = {
    "predictability": "Statistical predictability",
    "repetition": "Repetition",
    "structural": "Structural regularity",
    "lexical": "Lexical anomaly",
    "stylometric": "Stylometric inconsistency",
    "model": "Model-based evidence",
    "watermark": "Known-watermark test result",
}

KIND_LABELS = {
    "measured": "Directly measured statistic",
    "heuristic": "Heuristic interpretation",
    "model": "Model-based result",
    "watermark": "Watermark test result",
    "unavailable": "Unavailable / not run",
}


@dataclass(slots=True)
class Token:
    text: str          # lower-cased surface form
    start: int
    end: int
    sent_index: int


@dataclass(slots=True)
class Sentence:
    index: int
    para_index: int
    start: int
    end: int
    text: str
    tok_start: int
    tok_end: int
    kind: str
    included: bool

    @property
    def n_words(self) -> int:
        return self.tok_end - self.tok_start


@dataclass(slots=True)
class Paragraph:
    index: int
    start: int
    end: int
    text: str
    kind: str
    included: bool
    section_index: int
    sent_start: int = 0
    sent_end: int = 0   # exclusive


@dataclass(slots=True)
class Section:
    index: int
    title: str
    start: int
    end: int
    para_start: int
    para_end: int       # exclusive


@dataclass
class Document:
    """Segmented document. ``text`` is never modified after construction."""

    text: str
    paragraphs: List[Paragraph]
    sentences: List[Sentence]
    sections: List[Section]
    tokens: List[Token]
    cache: Dict[str, Any] = field(default_factory=dict)

    def included_sentences(self) -> List[Sentence]:
        if "inc_sent" not in self.cache:
            self.cache["inc_sent"] = [s for s in self.sentences if s.included]
        return self.cache["inc_sent"]

    def sentence_words(self, s: Sentence) -> List[str]:
        return [t.text for t in self.tokens[s.tok_start:s.tok_end]]

    def words(self) -> List[str]:
        """Lower-cased words in included sentences, in document order."""
        if "words" not in self.cache:
            out: List[str] = []
            for s in self.included_sentences():
                out.extend(t.text for t in self.tokens[s.tok_start:s.tok_end])
            self.cache["words"] = out
        return self.cache["words"]

    def included_paragraphs(self) -> List[Paragraph]:
        return [p for p in self.paragraphs if p.included]

    def paragraph_words(self, p: Paragraph) -> List[str]:
        out: List[str] = []
        for s in self.sentences[p.sent_start:p.sent_end]:
            if s.included:
                out.extend(self.sentence_words(s))
        return out


@dataclass
class FeatureResult:
    module: str
    name: str
    value: Any
    unit: str = ""
    kind: str = "measured"          # measured | heuristic | model | watermark | unavailable
    baseline: str = ""
    interpretation: str = ""
    explanation: str = ""           # what the statistic is
    location: Optional[Dict[str, Any]] = None
    strength: str = "n/a"           # n/a | none | weak | moderate | strong
    family: str = ""                # feature family used for convergence counting
    limitations: str = ""
    alternatives: List[str] = field(default_factory=list)


@dataclass
class Signal:
    """A passage-level finding that can be highlighted."""

    category: str
    feature: str
    level: str                      # sentence | paragraph | window
    start: int
    end: int
    sent_start: int
    sent_end: int                   # inclusive sentence indices
    para_index: int
    observed: Optional[float]
    baseline: str
    strength: str
    why: str
    alternatives: List[str]
    needs_more_evidence: str
    family: str
    details: Dict[str, Any] = field(default_factory=dict)
    id: int = -1


@dataclass
class AnalyzerOutput:
    features: List[FeatureResult] = field(default_factory=list)
    signals: List[Signal] = field(default_factory=list)
    tables: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class ModuleStatus:
    name: str
    status: str                     # ran | skipped | failed | unavailable
    detail: str = ""
    seconds: float = 0.0


@dataclass
class SegmentRecord:
    level: str                      # document | section | paragraph | sentence | window
    index: int
    start: int
    end: int
    word_start: int
    word_end: int
    sent_start: int
    sent_end: int                   # inclusive
    para_index: int
    text: str
    features: Dict[str, Any]
    anomalies: List[int] = field(default_factory=list)
    evidence_quality: str = "adequate"
    limitations: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    document: Document
    ingest: Dict[str, Any]
    doc_stats: Dict[str, Any]
    features: List[FeatureResult]
    signals: List[Signal]
    segments: List[SegmentRecord]
    tables: Dict[str, Any]
    statuses: List[ModuleStatus]
    evidence: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    inspection: Dict[str, Any]
    notes: List[str]
    settings: Dict[str, Any]
    reproducibility: Dict[str, Any]
