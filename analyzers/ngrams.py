"""Module B.5 - n-gram analysis (word and character level)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from config.lexicons import FUNCTION_WORDS
from config.settings import AnalysisSettings
from core import mathutils as mu
from core.features import included_tokens, word_ngrams
from core.schemas import AnalyzerOutput, Document, FeatureResult, Signal
from scoring.evidence import UNCALIBRATED, grade_high, unavailable

MODULE = "ngrams"
ALTS_REPEAT = ["Technical, legal or formulaic genres repeat fixed phrases", "Required terminology or quoted prompt wording",
               "Deliberate rhetorical repetition", "Short documents repeat topic phrases naturally"]


_P = np.uint64(0x9E3779B97F4A7C15)


def _repeat_share_ids(ids: np.ndarray, n: int) -> float:
    """Repeated-n-gram share on integer-coded tokens using a 64-bit polynomial hash (exact up to negligible collisions)."""
    L = len(ids) - n + 1
    if L <= 0:
        return 0.0
    key = np.zeros(L, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            key = key * _P + ids[j:j + L].astype(np.uint64) + np.uint64(1)
    _, counts = np.unique(key, return_counts=True)
    return float(counts[counts >= 2].sum() / L)


def _repeat_share(grams: List[Tuple[str, ...]]) -> float:
    if not grams:
        return 0.0
    c = Counter(grams)
    return sum(v for v in c.values() if v >= 2) / len(grams)


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    words = doc.words()
    N = len(words)
    if N < 100:
        out.features.append(unavailable(MODULE, "N-gram analysis", f"only {N} analysable words (need >= 100)"))
        return out
    toks = included_tokens(doc)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    ids = np.fromiter((vocab[w] for w in words), dtype=np.int64, count=N)
    perms = [ids[rng.permutation(N)] for _ in range(cfg.shuffle_iterations)]   # one shuffle per iteration, reused for every n
    for n in range(cfg.word_ngram_min, cfg.word_ngram_max + 1):
        grams = word_ngrams(words, n)
        c = Counter(grams)
        obs = _repeat_share(grams)
        base = [_repeat_share_ids(pm, n) for pm in perms]
        bm, bs = float(np.mean(base)), float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
        z = mu.robust_z(obs, bm, bs) if bs > 0 else None
        top10 = sum(v for _, v in c.most_common(10)) / len(grams)
        hhi = sum((v / len(grams)) ** 2 for v in c.values())
        rows.append({"n": n, "types": len(c), "repeated_types": sum(1 for v in c.values() if v >= 2),
                     "repeated_share": obs, "shuffled_mean": bm, "shuffled_sd": bs, "z_vs_shuffled": z,
                     "top10_share": top10, "hhi": hhi})
    out.tables["word_ngram_summary"] = rows
    for r in rows:
        out.features.append(FeatureResult(
            MODULE, f"Repeated word {r['n']}-grams", r, "share", "measured",
            f"Word-shuffled baseline over {cfg.shuffle_iterations} permutations (seed {cfg.seed}): "
            f"mean share {r['shuffled_mean']:.4f}",
            "Observed share exceeding the shuffled baseline indicates repetition beyond what vocabulary frequency alone "
            "produces by chance." if r["repeated_share"] > r["shuffled_mean"] else "At or below the chance baseline.",
            f"Share of {r['n']}-gram positions belonging to a type seen >= 2 times; concentration via top-10 share and "
            "Herfindahl index. The shuffle baseline destroys grammar, so it is a rough chance reference only.",
            None, "n/a", "", "The baseline preserves word frequencies but not syntax.", ALTS_REPEAT))

    ref_n = 4 if cfg.word_ngram_min <= 4 <= cfg.word_ngram_max else rows[-1]["n"]
    ref_row = next(r for r in rows if r["n"] == ref_n)
    cuts = cfg.thresholds.repeated_ngram_share_high
    st = grade_high(ref_row["repeated_share"], cuts) if N >= 500 else "n/a"
    out.features.append(FeatureResult(
        MODULE, f"Concentration of repeated {ref_n}-grams (heuristic indicator)", ref_row["repeated_share"], "share",
        "heuristic" if st != "n/a" else "measured", f"Heuristic cut-offs: >= {cuts[0]}/{cuts[1]}/{cuts[2]} (needs >= 500 words)",
        ("Not graded (fewer than 500 words). " if N < 500 else "") + UNCALIBRATED,
        f"Share of {ref_n}-gram positions that repeat elsewhere in the document.", None, st, "repetition",
        "Repetition is genre-dependent.", ALTS_REPEAT))

    # Sentence openings / endings
    inc = [s for s in doc.included_sentences() if s.n_words >= 4]
    openings = [(" ".join(doc.sentence_words(s)[:3])) for s in inc]
    endings = [(" ".join(doc.sentence_words(s)[-2:])) for s in inc]
    oc, ec = Counter(openings), Counter(endings)
    rep_open = sum(v for v in oc.values() if v >= 3) / len(inc) if inc else 0.0
    rep_end = sum(v for v in ec.values() if v >= 3) / len(inc) if inc else 0.0
    cuts_o = cfg.thresholds.opening_repeat_share_high
    st_o = grade_high(rep_open, cuts_o) if len(inc) >= 20 else "n/a"
    out.features.append(FeatureResult(
        MODULE, "Repeated sentence openings (share of sentences whose first 3 words occur >= 3 times)", rep_open, "share",
        "heuristic" if st_o != "n/a" else "measured", f"Heuristic cut-offs: >= {cuts_o[0]}/{cuts_o[1]}/{cuts_o[2]} (needs >= 20 sentences)",
        ("Not graded (fewer than 20 sentences). " if len(inc) < 20 else "") + UNCALIBRATED,
        f"Most common openings: {', '.join(f'{k!r} x{v}' for k, v in oc.most_common(3))}.", None, st_o, "repetition",
        "Parallel openings are common in lists, instructions and rhetorical emphasis.", ALTS_REPEAT))
    out.features.append(FeatureResult(
        MODULE, "Repeated sentence endings (share of sentences whose last 2 words occur >= 3 times)", rep_end, "share",
        "measured", "", "", f"Most common endings: {', '.join(f'{k!r} x{v}' for k, v in ec.most_common(3))}.", None, "n/a", "",
        "", ALTS_REPEAT))
    out.tables["openings"] = oc.most_common(15)

    # Local vs global repetition for 4-grams
    g4 = word_ngrams(words, 4)
    pos: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for i, g in enumerate(g4):
        pos[g].append(i)
    local = glob = 0
    for g, p in pos.items():
        if len(p) < 2:
            continue
        for a, b in zip(p, p[1:]):
            if b - a <= cfg.local_repeat_gap_tokens:
                local += 1
            else:
                glob += 1
    tot = local + glob
    out.features.append(FeatureResult(
        MODULE, "Local vs global repetition of 4-grams", {"local_repeats": local, "global_repeats": glob,
                                                         "local_share": (local / tot) if tot else None},
        "share", "measured", "", "", f"Repeats are 'local' when consecutive occurrences are <= {cfg.local_repeat_gap_tokens} tokens apart.",
        None, "n/a", "", "Local repetition clusters may reflect topic focus or copy-editing patterns.", ALTS_REPEAT))

    # Recurring phrases (4-grams with >= 1 content word, count >= 3)
    phrases = []
    for g, p in pos.items():
        if len(p) >= 3 and any(w not in FUNCTION_WORDS for w in g):
            phrases.append((g, p))
    phrases.sort(key=lambda x: -len(x[1]))
    out.tables["recurring_phrases"] = [{"phrase": " ".join(g), "count": len(p), "first_offset": toks[p[0]].start}
                                       for g, p in phrases[:25]]
    out.features.append(FeatureResult(
        MODULE, "Recurring phrases (4-grams, >= 3 occurrences, with content word)",
        out.tables["recurring_phrases"][:10], "", "measured", "", "", "Top recurring phrases and counts.", None, "n/a", "",
        "Phrases from the assignment prompt or key terms recur legitimately.", ALTS_REPEAT))

    # Character n-grams
    if not cfg.lightweight or cfg.char_ngram_sizes:
        text = re.sub(r"\s+", " ", " ".join(s.text for s in doc.included_sentences()).lower())
        crows = []
        for n in cfg.char_ngram_sizes:
            grams = [text[i:i + n] for i in range(len(text) - n + 1)]
            c = Counter(grams)
            crows.append({"n": n, "types": len(c), "repeated_share": _repeat_share(grams),
                          "top10_share": sum(v for _, v in c.most_common(10)) / max(1, len(grams)),
                          "entropy_bits": mu.shannon_entropy(c.values())})
        out.tables["char_ngram_summary"] = crows
        out.features.append(FeatureResult(MODULE, "Character n-gram summary", crows, "", "measured", "", "",
                                          "Repeated share, concentration and entropy of character n-grams.", None, "n/a", "",
                                          "Dominated by common English morphology; descriptive only.", []))

    # --- passage-level signals ----------------------------------------------------------------------------
    sent_of_tok = [t.sent_index for t in toks]
    flagged: Dict[int, Tuple[int, str]] = {}
    for g, p in phrases:
        cnt = len(p)
        for i in p:
            si = sent_of_tok[i]
            if si not in flagged or flagged[si][0] < cnt:
                flagged[si] = (cnt, " ".join(g))
    for si, (cnt, phrase) in flagged.items():
        s = doc.sentences[si]
        strength = "strong" if cnt >= 8 else "moderate" if cnt >= 5 else "weak"
        out.signals.append(Signal(
            "repetition", "Recurring phrase", "sentence", s.start, s.end, si, si, s.para_index, float(cnt),
            f"phrase appears {cnt} times in the document; document 4-gram repeat share {ref_row['repeated_share']:.3f}",
            strength, f"The phrase '{phrase}' occurs {cnt} times across the document, including in this sentence.",
            ALTS_REPEAT, "Check whether the repeated phrase is required terminology or from the prompt.", "repetition",
            {"phrase": phrase, "count": cnt}))
    # clustered openings
    by_open: Dict[str, List[int]] = defaultdict(list)
    for k, (s, o) in enumerate(zip(inc, openings)):
        by_open[o].append(k)
    for o, ks in by_open.items():
        if len(ks) < 3:
            continue
        marked = set()
        for a in range(len(ks) - 2):
            if ks[a + 2] - ks[a] <= 9:
                marked.update(ks[a:a + 3])
        if not marked:
            continue
        for k in sorted(marked):
            window_hits = sum(1 for j in ks if abs(j - k) <= 9)
            s = inc[k]
            strength = "strong" if window_hits >= 6 else "moderate" if window_hits >= 4 else "weak"
            out.signals.append(Signal(
                "repetition", "Repeated sentence opening", "sentence", s.start, s.end, s.index, s.index, s.para_index,
                float(window_hits), f"opening '{o}' occurs {len(ks)} times in the document",
                strength, f"The opening '{o}' is used {window_hits} times within about 10 sentences of this one.",
                ["Parallel structure for emphasis", "List-like or instructional content", "Template or style requirement"],
                "Read the surrounding sentences to see whether the parallelism is purposeful.", "repetition",
                {"opening": o}))
    return out
