"""Analysis pipeline: orchestrates all modules with per-module error isolation and progress reporting."""
from __future__ import annotations

import hashlib
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from analyzers import (discourse, entropy, function_words, internal_anomalies, lexical_diversity, ngrams,
                       paragraph_statistics, perplexity, punctuation, reference_comparison, sentence_statistics,
                       syntactic_features, zipf)
from config.settings import APP_NAME, APP_VERSION, AnalysisSettings
from core.ingestion import IngestResult, from_pasted_text
from core.multiscale import build_segments
from core.normalization import inspect_text_characters
from core.schemas import AnalysisResult, AnalyzerOutput, FeatureResult, ModuleStatus
from core.segmentation import document_counts, segment_document
from scoring.aggregation import aggregate
from scoring.recommendations import build_recommendations
from watermark import registry as wm_registry

log = logging.getLogger("textforensics")   # never logs submitted text

ProgressFn = Callable[[float, str], None]

PIPELINE = [
    # (name, switch attribute or None, function)
    ("sentence_statistics", "enable_statistical", sentence_statistics.analyze),
    ("paragraph_statistics", "enable_structural", paragraph_statistics.analyze),
    ("lexical_diversity", "enable_lexical", lexical_diversity.analyze),
    ("entropy", "enable_statistical", entropy.analyze),
    ("zipf", "enable_statistical", zipf.analyze),
    ("ngrams", "enable_lexical", ngrams.analyze),
    ("punctuation", "enable_statistical", punctuation.analyze),
    ("function_words", "enable_linguistic", function_words.analyze),
    ("syntactic_features", "enable_linguistic", syntactic_features.analyze),
    ("discourse", "enable_linguistic", discourse.analyze),
    ("perplexity", "enable_model", perplexity.analyze),
    ("internal_anomalies", "enable_internal", internal_anomalies.analyze),
    ("reference_comparison", "enable_reference", reference_comparison.analyze),
]


def _pkg_versions() -> Dict[str, str]:
    out = {}
    for name in ("numpy", "streamlit", "docx", "pypdf", "reportlab", "torch", "transformers", "spacy", "nltk"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            out[name] = "not installed"
    return out


def run_analysis(source: Union[str, IngestResult], cfg: Optional[AnalysisSettings] = None,
                 reference: Optional[Union[str, IngestResult]] = None, progress: Optional[ProgressFn] = None,
                 scorer_override: Any = None) -> AnalysisResult:
    cfg = (cfg or AnalysisSettings()).effective()
    prog: ProgressFn = progress or (lambda f, m: None)
    ing = from_pasted_text(source) if isinstance(source, str) else source
    prog(0.02, "Segmenting document")
    doc = segment_document(ing.text, cfg, ing.hints)
    doc_stats = document_counts(ing.text, doc)
    inspection = inspect_text_characters(ing.text)
    ctx: Dict[str, Any] = {"scorer_override": scorer_override}
    if reference is not None:
        ref_ing = from_pasted_text(reference) if isinstance(reference, str) else reference
        ctx["reference_doc"] = segment_document(ref_ing.text, cfg, ref_ing.hints)
    statuses: List[ModuleStatus] = []
    features: List[FeatureResult] = []
    signals = []
    tables: Dict[str, Any] = {}
    notes: List[str] = list(ing.warnings)
    n = len(PIPELINE) + 2
    for i, (name, switch, fn) in enumerate(PIPELINE):
        frac = 0.05 + 0.80 * i / n
        if switch and not getattr(cfg, switch):
            statuses.append(ModuleStatus(name, "skipped", f"disabled in settings ({switch})"))
            continue
        prog(frac, f"Running {name.replace('_', ' ')}")
        ctx["progress_inner"] = lambda x, _f=frac: prog(_f + 0.8 / n * x, f"Running {name.replace('_', ' ')} ({int(x * 100)}%)")
        t0 = time.perf_counter()
        try:
            out: AnalyzerOutput = fn(doc, cfg, ctx)
        except Exception as e:  # isolate failures; never include text in messages
            log.error("module %s failed (%s)", name, type(e).__name__)   # no message/traceback: they could echo document text
            statuses.append(ModuleStatus(name, "failed", f"{type(e).__name__}: {str(e)[:200]}", time.perf_counter() - t0))
            continue
        dt = time.perf_counter() - t0
        if out.features and all(f.kind == "unavailable" for f in out.features) and not out.signals:
            statuses.append(ModuleStatus(name, "unavailable", out.features[0].limitations, dt))
        else:
            statuses.append(ModuleStatus(name, "ran", "", dt))
        features.extend(out.features)
        signals.extend(out.signals)
        for k, v in out.tables.items():
            tables[k] = v
        notes.extend(out.notes)
    # Watermark tests
    wm_results = []
    if cfg.enable_watermark:
        prog(0.88, "Running available watermark tests")
        t0 = time.perf_counter()
        try:
            wm_results = wm_registry.run_all(doc.text, cfg.watermark_configs)
            statuses.append(ModuleStatus("watermark", "ran" if any(r.ran for r in wm_results) else "unavailable",
                                         "" if any(r.ran for r in wm_results) else "No supported watermark detector is configured.",
                                         time.perf_counter() - t0))
        except Exception as e:
            statuses.append(ModuleStatus("watermark", "failed", f"{type(e).__name__}: {str(e)[:200]}"))
    else:
        statuses.append(ModuleStatus("watermark", "skipped", "disabled in settings"))
    for r in wm_results:
        features.append(FeatureResult(
            "watermark", f"Watermark test: {r.scheme}", r.to_dict(), "", "watermark" if r.ran else "unavailable", "",
            (f"z={r.statistic:.2f}, p={r.p_value:.3g} (adjusted {r.adjusted_p_value:.3g}). " if r.ran and r.p_value is not None else "") + r.message,
            "Statistical watermark detection test (only run when its real algorithm and configuration are available).",
            None, "n/a", "", r.limitations, []))
    tables["watermark"] = [r.to_dict() for r in wm_results]
    # Ids, segments, evidence
    prog(0.92, "Building multi-scale segments")
    signals.sort(key=lambda s: (s.start, s.end))
    for i, s in enumerate(signals):
        s.id = i
    segments = build_segments(doc, ctx.get("windows", []), signals, cfg)
    prog(0.96, "Aggregating evidence")
    evidence = aggregate(doc_stats, features, signals, statuses, cfg, watermark_ran=any(r.ran for r in wm_results))
    repro = {
        "application": APP_NAME, "version": APP_VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(), "packages": _pkg_versions(),
        "random_seed": cfg.seed, "text_sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
        "pos_tagger": ctx.get("pos_tagger", "not run"), "network_access_used": False,
    }
    res = AnalysisResult(
        document=doc, ingest={"source": ing.source, "encoding": ing.encoding, "normalization": ing.normalization,
                              "warnings": ing.warnings, "meta": ing.meta},
        doc_stats=doc_stats, features=features, signals=signals, segments=segments, tables=tables, statuses=statuses,
        evidence=evidence, recommendations=[], inspection=inspection, notes=notes, settings=cfg.to_dict(), reproducibility=repro)
    res.recommendations = build_recommendations(res)
    prog(1.0, "Done")
    return res
