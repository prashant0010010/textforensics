"""Module E - internal consistency / anomaly detection (document-specific baseline).

Every sliding window of sentences is compared with the *document's own* distribution of window-level
features using robust statistics (median and 1.4826*MAD, falling back to IQR/1.349). A window is only
flagged when it deviates by >= the configured robust z-score in a direction that is meaningful for the
feature. This is an internal-consistency analysis: an unusual passage is not necessarily AI-written, and a
uniform document is not necessarily human-written.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from config.lexicons import FUNCTION_WORDS
from config.settings import AnalysisSettings
from core import mathutils as mu
from core.features import segment_features
from core.schemas import AnalyzerOutput, Document, FeatureResult, Sentence, Signal
from scoring.evidence import strength_from_z, unavailable

MODULE = "internal_anomalies"

# key: (label, category, family, direction, interpretation of the deviation, alternative explanations)
FEATURE_SPECS: Dict[str, Tuple[str, str, str, str, str, List[str]]] = {
    "sentence_len_cv": ("Sentence-length variability", "structural", "sentence_variability", "low",
                        "sentence lengths are unusually uniform compared with the rest of the document",
                        ["Short list-like or definitional passage", "Different genre section", "Editing or templated text"]),
    "mattr25": ("Lexical diversity (MATTR-25)", "lexical", "lexical_anomaly", "low",
                "vocabulary is unusually repetitive compared with the rest of the document",
                ["Topic requiring fixed terminology", "Technical or formulaic passage", "Definitions or lists"]),
    "char_entropy": ("Character entropy", "predictability", "internal_predictability", "low",
                     "character-level information content is unusually low (more predictable text) relative to the document",
                     ["Repetitive or code-like content", "Lists, numbers or quotations", "Different language or formatting"]),
    "repeated_trigram_share": ("Repeated word-trigram share", "repetition", "repetition", "high",
                               "an unusually high share of three-word sequences repeats inside the passage",
                               ["Formulaic phrases", "Deliberate repetition", "Topic terminology"]),
    "marker_per_100w": ("Discourse-marker density", "structural", "discourse_rhetoric", "high",
                        "discourse markers/transitions are unusually dense relative to the document",
                        ["Signposting in an introduction or conclusion", "Genre convention", "Writer's style shift"]),
    "mean_sentence_len": ("Mean sentence length", "stylometric", "internal_inconsistency", "both",
                          "average sentence length differs markedly from the rest of the document",
                          ["Quotation, list or technical section", "Different drafting session", "Mixed authorship or editing"]),
    "comma_per_sentence": ("Commas per sentence", "stylometric", "internal_inconsistency", "both",
                           "comma usage differs markedly from the rest of the document",
                           ["Editor or software changes", "Different section type", "Mixed authorship"]),
    "function_word_share": ("Function-word share", "stylometric", "internal_inconsistency", "both",
                            "the proportion of function words differs markedly from the rest of the document",
                            ["Register shift (technical vs narrative)", "Quoted material", "Mixed authorship"]),
    "fw_js": ("Function-word profile distance", "stylometric", "internal_inconsistency", "high",
              "the function-word profile differs from the document's overall profile (Jensen-Shannon distance)",
              ["Topic or register change", "Quoted source text", "Mixed authorship or heavy editing"]),
    "mean_surprisal": ("Mean token surprisal (reference model)", "model", "model_predictability", "low",
                       "the passage is more predictable to the reference model than most of the document",
                       ["Formulaic or well-known text", "Text memorised by the model", "Domain match with the model"]),
}


# Minimum spread (per feature) used when the document's own robust spread is smaller. Without this floor, a very
# homogeneous document (tiny MAD) would turn trivial differences into huge z-scores (a false-positive hazard).
MIN_SCALE: Dict[str, float] = {
    "sentence_len_cv": 0.05, "mattr25": 0.03, "char_entropy": 0.05, "repeated_trigram_share": 0.03,
    "marker_per_100w": 0.75, "mean_sentence_len": 1.5, "comma_per_sentence": 0.4, "function_word_share": 0.03,
    "fw_js": 0.03, "mean_surprisal": 0.15,
}


def make_windows(doc: Document, cfg: AnalysisSettings) -> List[Tuple[int, int, List[Sentence]]]:
    """Overlapping windows of >= ``window_sentences`` sentences that also contain >= ``min_window_words`` words.

    Windows grow past the nominal sentence count when sentences are short, so documents with short sentences are not
    silently excluded; windows that cannot reach the word minimum (end of document) are skipped."""
    inc = [s for s in doc.included_sentences() if s.n_words]
    W, step = max(3, cfg.window_sentences), max(1, cfg.window_step)
    prefix = [0]
    for s in inc:
        prefix.append(prefix[-1] + s.n_words)
    n = len(inc)

    def end_for(a: int) -> int:
        e = min(n, a + W)
        while e < n and prefix[e] - prefix[a] < cfg.min_window_words:
            e += 1
        return e

    starts = list(range(0, max(1, n - W + 1), step))
    wins: List[Tuple[int, int, List[Sentence]]] = []
    seen = set()
    for a in starts:
        e = end_for(a)
        if prefix[e] - prefix[a] < cfg.min_window_words or (a, e) in seen:
            continue
        seen.add((a, e))
        group = inc[a:e]
        wins.append((group[0].start, group[-1].end, group))
    # make sure the tail of the document is covered by one final window
    if n >= W:
        a = n - W
        while a > 0 and prefix[n] - prefix[a] < cfg.min_window_words:
            a -= 1
        if prefix[n] - prefix[a] >= cfg.min_window_words and (a, n) not in seen:
            group = inc[a:n]
            wins.append((group[0].start, group[-1].end, group))
    return wins


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    wins = make_windows(doc, cfg)
    ctx["windows"] = wins
    if len(wins) < cfg.min_windows_for_baseline:
        out.features.append(unavailable(
            MODULE, "Internal consistency / anomaly analysis",
            f"only {len(wins)} usable window(s); at least {cfg.min_windows_for_baseline} windows of >= {cfg.min_window_words} words "
            "are needed to form a document-specific baseline"))
        return out
    doc_fw = ctx.get("fw_counts") or Counter(w for w in doc.words() if w in FUNCTION_WORDS)
    sc = ctx.get("model_scores")
    rows: List[Dict[str, Optional[float]]] = []
    prog = ctx.get("progress_inner")
    for i, (a, b, group) in enumerate(wins):
        f = segment_features(doc, group)
        toks = [t for s in group for t in doc.sentence_words(s)]
        f["fw_js"] = mu.js_distance(Counter(t for t in toks if t in FUNCTION_WORDS), doc_fw)
        if sc is not None:
            from analyzers.perplexity import window_surprisal
            f["mean_surprisal"] = window_surprisal(sc, a, b)
        rows.append(f)
        if prog and i % 25 == 0:
            prog(i / len(wins))
    baselines: Dict[str, Dict[str, float]] = {}
    z_by_feature: Dict[str, List[Optional[float]]] = {}
    for key in FEATURE_SPECS:
        vals = [r.get(key) for r in rows]
        valid = [v for v in vals if v is not None]
        if len(valid) < cfg.min_windows_for_baseline:
            continue
        med, raw = float(mu.median(valid)), mu.robust_scale(valid)
        scale = max(raw, MIN_SCALE.get(key, 0.0))
        baselines[key] = {"median": med, "robust_sd": scale, "raw_robust_sd": raw, "floor_applied": bool(raw < scale),
                          "n_windows": len(valid)}
        z_by_feature[key] = [mu.robust_z(v, med, scale) if v is not None else None for v in vals]
    out.tables["window_features"] = [{"start": w[0], "end": w[1], **{k: v for k, v in r.items()}} for w, r in zip(wins, rows)]
    out.tables["window_z"] = [{"start": w[0], "end": w[1], "z": {k: z_by_feature[k][i] for k in z_by_feature}} for i, w in enumerate(wins)]
    out.tables["window_baselines"] = baselines
    out.features.append(FeatureResult(
        MODULE, "Document-specific baseline (window level)", baselines, "", "measured",
        f"{len(wins)} windows of {cfg.window_sentences} sentences (step {cfg.window_step}), >= {cfg.min_window_words} words each",
        "Median and robust SD of each window-level feature across this document.",
        "Median/MAD-based baseline built from the submitted document only; a per-feature minimum spread prevents near-constant baselines from inflating z-scores.", None, "n/a", "",
        "Windows overlap, so neighbouring windows are not independent. A homogeneous document produces few anomalies "
        "whether human or machine written.", ["Document genre shifts across sections", "Quoted or pasted material"]))

    cuts = cfg.thresholds.anomaly_z
    flagged_counts: Dict[str, int] = {}
    for key, zs in z_by_feature.items():
        label, cat, fam, direction, meaning, alts = FEATURE_SPECS[key]
        base = baselines[key]
        runs: List[List[int]] = []
        cur: List[int] = []
        sign = 0
        for i, z in enumerate(zs):
            hit = z is not None and abs(z) >= cuts[0] and (direction == "both" or (direction == "low" and z < 0) or (direction == "high" and z > 0))
            if hit and (not cur or (i - cur[-1] <= 1 and (z > 0) == (sign > 0))):
                cur.append(i)
                sign = 1 if z > 0 else -1
            else:
                if cur:
                    runs.append(cur)
                cur = [i] if hit else []
                sign = (1 if (hit and z > 0) else -1) if hit else 0
        if cur:
            runs.append(cur)
        flagged_counts[key] = sum(len(r) for r in runs)
        for r in runs:
            best = max(r, key=lambda j: abs(zs[j]))  # type: ignore[arg-type]
            z, v = zs[best], rows[best][key]
            first, last = wins[r[0]][2][0], wins[r[-1]][2][-1]
            side = "below" if z < 0 else "above"
            out.signals.append(Signal(
                cat, label, "window", first.start, last.end, first.index, last.index, first.para_index, v,
                f"document median {base['median']:.3f} (robust SD {base['robust_sd']:.3f}{' - minimum-spread floor applied' if base['floor_applied'] else ''}, "
                f"{int(base['n_windows'])} windows); "
                f"passage value {v:.3f}; robust z = {z:+.1f}",
                strength_from_z(z, cuts),
                f"{label} is {abs(z):.1f} robust standard deviations {side} the document median: {meaning}.",
                alts + ["Internal deviation alone does not indicate origin"],
                "Compare with other writing by the same author and check the passage's content, sources and context.",
                fam, {"feature": key, "z": z, "n_windows_merged": len(r)}))
    out.features.append(FeatureResult(
        MODULE, "Windows flagged per feature", flagged_counts, "windows", "measured",
        f"Flag threshold: |robust z| >= {cuts[0]} (weak), {cuts[1]} (moderate), {cuts[2]} (strong) in the stated direction",
        "Number of windows deviating from the document's own profile.", "Internal-consistency result.", None, "n/a", "",
        "With ~hundreds of windows a few threshold crossings are expected by chance in any document.", []))
    return out
