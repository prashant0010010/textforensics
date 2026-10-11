"""Module B.7 - Zipf / frequency-distribution analysis."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict

import numpy as np

from config.settings import AnalysisSettings
from core import mathutils as mu
from core.schemas import AnalyzerOutput, Document, FeatureResult
from scoring.evidence import unavailable

MODULE = "zipf"


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    words = doc.words()
    N = len(words)
    if N < 200:
        out.features.append(unavailable(MODULE, "Zipf analysis", f"only {N} analysable words (need >= 200 for a meaningful fit)"))
        return out
    c = ctx.get("word_counts") or Counter(words)
    freqs = sorted(c.values(), reverse=True)
    V = len(freqs)
    ranks = np.arange(1, V + 1)
    logr, logf = np.log(ranks), np.log(freqs)
    # Fit on ranks 1..R where R excludes the long hapax tail (flat plateau biases the slope).
    R_full = V
    R_trim = max(10, sum(1 for f in freqs if f >= 2))
    fits = {}
    for label, R in (("all_ranks", R_full), ("ranks_with_freq>=2", R_trim)):
        fit = mu.linear_fit(logr[:R], logf[:R])
        if fit:
            fit["fit_ranks"] = int(R)
            fits[label] = fit
    out.tables["zipf"] = {"log_rank": logr[:2000].tolist(), "log_freq": logf[:2000].tolist(), "fits": fits}

    def add(name, value, expl, unit="", lim="", alts=None, interp=""):
        out.features.append(FeatureResult(MODULE, name, value, unit, "measured", "", interp, expl, None, "n/a", "",
                                          lim, alts or []))

    for label, fit in fits.items():
        r2 = fit["r2"]
        fmt3 = lambda v: "n/a" if v is None else f"{v:.3f}"
        quality = "n/a" if r2 is None else ("good (R^2>=0.95)" if r2 >= 0.95 else "moderate (0.85<=R^2<0.95)" if r2 >= 0.85 else "poor (R^2<0.85)")
        add(f"Zipf slope ({label})", fit["slope"],
            f"OLS slope of log(frequency) vs log(rank); ideal Zipf slope is about -1. Fit ranks: {fit['fit_ranks']}; "
            f"R^2={fmt3(r2)}, RMSE(log)={fmt3(fit['rmse'])}; fit quality: {quality}.", "",
            "OLS on log-log data is a biased estimator of a power-law exponent (Clauset et al. 2009); the tail of "
            "tied low frequencies flattens the fit; short documents deviate strongly from Zipf. Descriptive only.",
            ["Document length", "Genre and topic", "Tokenisation choices"],
            "Nothing in this result implies that AI or human writing follows Zipf's law more closely.")
    top = freqs[:10]
    add("Frequency concentration (top 10 / top 50 types share)", {"top10": sum(top) / N, "top50": sum(freqs[:50]) / N},
        "Token share held by the most frequent word types.", "share")
    add("Rare-word proportions", {"hapax_share_of_types": sum(1 for f in freqs if f == 1) / V,
                                  "tokens_in_types_with_freq<=2": sum(f for f in freqs if f <= 2) / N},
        "Share of types that are hapax; share of tokens in types with frequency <= 2.", "share")
    return out
