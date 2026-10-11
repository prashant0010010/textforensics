"""Module C - optional local reference-language-model analysis.

* Works only if the user points to a *locally available* causal LM
  (``local_files_only=True`` is always used: nothing is ever downloaded).
* Requires the optional packages ``torch`` and ``transformers``.
* Every number reported is computed from the model's actual next-token
  log-probabilities; there are no heuristic stand-ins.
* Perplexity depends on the chosen reference model, its tokenizer and training
  data. It is not a universal measure of AI authorship.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

from config.settings import AnalysisSettings
from core import mathutils as mu
from core.schemas import AnalyzerOutput, Document, FeatureResult
from scoring.evidence import unavailable

MODULE = "perplexity"


class ModelUnavailable(Exception):
    """Raised when the optional model module cannot run."""


@dataclass
class TokenScores:
    """Per-token results aligned to character offsets of the analysed text."""

    nll: np.ndarray                       # negative log-likelihood in nats
    starts: np.ndarray                    # char start offsets
    ends: np.ndarray
    ranks: Optional[np.ndarray] = None    # 1 = most probable token
    pred_entropy: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class TokenScorer(Protocol):
    def score(self, text: str, progress: Optional[Callable[[float], None]] = None) -> TokenScores: ...


class HFCausalLMScorer:
    """Scores text with a local Hugging Face causal LM using overlapping windows.

    Each token is scored exactly once, with up to ``max_length - 1`` tokens of preceding context
    (windows advance by ``stride``; only the not-yet-scored tail of each window contributes)."""

    def __init__(self, model_path: str, device: str = "auto", max_length: int = 1024, stride: int = 512):
        self.model_path, self.device_pref = model_path, device
        self.max_length, self.stride = max_length, min(stride, max_length)
        self._model = self._tok = self._torch = None
        self.device = "cpu"

    def load(self) -> None:
        if not self.model_path or not os.path.isdir(self.model_path):
            raise ModelUnavailable("Model path does not exist or is not a local directory. "
                                   "TextForensics Local never downloads models automatically.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ModelUnavailable("Optional packages 'torch' and 'transformers' are not installed "
                                   "(see README: optional reference model).") from e
        try:
            self._tok = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
        except Exception as e:
            raise ModelUnavailable(f"The model could not be loaded from the local path: {e}") from e
        if not getattr(self._tok, "is_fast", False):
            raise ModelUnavailable("A fast tokenizer is required to map tokens back to character offsets.")
        pref = self.device_pref
        self.device = "cuda" if (pref == "auto" and torch.cuda.is_available()) or pref == "cuda" else "cpu"
        try:
            self._model.to(self.device).eval()
        except Exception as e:
            raise ModelUnavailable(f"Could not move the model to device '{self.device}': {e}") from e
        self._torch = torch

    def score(self, text: str, progress: Optional[Callable[[float], None]] = None) -> TokenScores:
        torch = self._torch
        if torch is None:
            self.load()
            torch = self._torch
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids: List[int] = enc["input_ids"]
        offs = enc["offset_mapping"]
        if len(ids) < 2:
            raise ModelUnavailable("Text is too short to score.")
        bos = getattr(self._tok, "bos_token_id", None)
        prefix = 1 if bos is not None else 0
        full = ([bos] if prefix else []) + ids       # bos lets the first real token be scored
        L = len(full)
        nll: List[float] = []
        ranks: List[int] = []
        ent: List[float] = []
        starts: List[int] = []
        ends: List[int] = []
        scored_upto = max(prefix, 1)                # absolute index of first token still to score
        begin = 0
        while True:
            end = min(begin + self.max_length, L)
            chunk = torch.tensor([full[begin:end]], device=self.device)
            with torch.no_grad():
                logits = self._model(chunk).logits[0, :-1].float()
            logp = torch.log_softmax(logits, dim=-1)
            tgt = chunk[0, 1:]
            lp_t = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            rk = (logp > lp_t.unsqueeze(1)).sum(dim=1) + 1
            pe = -(logp.exp() * logp).sum(dim=1)
            for j in range(1, end - begin):
                absj = begin + j
                if absj < scored_upto:
                    continue
                ti = absj - prefix
                nll.append(-float(lp_t[j - 1]))
                ranks.append(int(rk[j - 1]))
                ent.append(float(pe[j - 1]))
                starts.append(offs[ti][0])
                ends.append(offs[ti][1])
            scored_upto = end
            if progress:
                progress(min(1.0, end / L))
            if end >= L:
                break
            begin += self.stride
        meta = {"model_name": os.path.basename(os.path.normpath(self.model_path)),
                "tokenizer": type(self._tok).__name__, "n_text_tokens": len(ids), "n_scored_tokens": len(nll),
                "device": self.device, "max_length": self.max_length, "stride": self.stride,
                "method": "log-softmax of causal-LM logits; sliding windows with overlap context; each token scored once; "
                          + ("BOS token prepended so first token is scored" if prefix else "first token unscored (no BOS)")}
        return TokenScores(np.array(nll), np.array(starts), np.array(ends), np.array(ranks), np.array(ent), meta)


def summarise_scores(sc: TokenScores, mask: Optional[np.ndarray] = None, local_window: int = 50) -> Dict[str, Any]:
    """Pure NumPy summary of per-token surprisal (testable without any model)."""
    nll = sc.nll if mask is None else sc.nll[mask]
    n = int(nll.size)
    if n < 20:
        return {"n_tokens": n}
    mean_nll = float(nll.mean())
    res: Dict[str, Any] = {
        "n_tokens": n, "mean_nll_nats": mean_nll, "cross_entropy_bits_per_token": mean_nll / math.log(2),
        "perplexity": float(math.exp(min(mean_nll, 700))), "median_surprisal_nats": float(np.median(nll)),
        "surprisal_std": float(nll.std(ddof=1)), "surprisal_variance": float(nll.var(ddof=1)),
        "surprisal_quantiles": mu.quantiles(nll, (5, 25, 50, 75, 95)),
    }
    if n >= 2 * local_window:
        wm = [float(nll[i:i + local_window].mean()) for i in range(0, n - local_window + 1, local_window)]
        res["local_window_tokens"] = local_window
        res["local_mean_surprisal_std"] = float(np.std(wm, ddof=1))
        res["local_mean_surprisal_range"] = [min(wm), max(wm)]
    if sc.ranks is not None:
        rk = sc.ranks if mask is None else sc.ranks[mask]
        res["rank_stats"] = {"median_rank": float(np.median(rk)), "mean_log_rank": float(np.log(rk).mean()),
                             "top1_share": float((rk == 1).mean()), "top10_share": float((rk <= 10).mean())}
    if sc.pred_entropy is not None:
        pe = sc.pred_entropy if mask is None else sc.pred_entropy[mask]
        res["mean_predictive_entropy_nats"] = float(pe.mean())
    k = 5
    if n >= k * 40:
        edges = np.linspace(0, n, k + 1).astype(int)
        res["segment_perplexity"] = [float(math.exp(min(float(nll[a:b].mean()), 700))) for a, b in zip(edges[:-1], edges[1:])]
    return res


def window_surprisal(sc: TokenScores, start: int, end: int) -> Optional[float]:
    """Mean per-token NLL for tokens whose first character lies in [start, end)."""
    m = (sc.starts >= start) & (sc.starts < end)
    return float(sc.nll[m].mean()) if int(m.sum()) >= 15 else None


def analyze(doc: Document, cfg: AnalysisSettings, ctx: Dict[str, Any]) -> AnalyzerOutput:
    out = AnalyzerOutput()
    scorer: Optional[TokenScorer] = ctx.get("scorer_override")
    if scorer is None:
        if not cfg.model_path:
            out.features.append(unavailable(MODULE, "Reference language-model analysis",
                                            "no local model path configured (the module is optional)"))
            return out
        scorer = HFCausalLMScorer(cfg.model_path, cfg.model_device, cfg.model_max_length, cfg.model_stride)
        try:
            scorer.load()  # type: ignore[attr-defined]
        except ModelUnavailable as e:
            out.features.append(unavailable(MODULE, "Reference language-model analysis", str(e)))
            return out
    try:
        sc = scorer.score(doc.text, ctx.get("progress_inner"))
    except ModelUnavailable as e:
        out.features.append(unavailable(MODULE, "Reference language-model analysis", str(e)))
        return out
    # mask: only tokens inside included sentences
    mask = np.zeros(sc.starts.shape[0], dtype=bool)
    for s in doc.included_sentences():
        mask |= (sc.starts >= s.start) & (sc.starts < s.end)
    summ = summarise_scores(sc, mask)
    ctx["model_scores"] = sc
    out.tables["model_summary"] = summ
    out.tables["model_meta"] = sc.meta
    lim = ("Perplexity and surprisal depend on the reference model, its tokenizer and training data; low perplexity is "
           "also produced by famous, formulaic, or heavily edited human text and by text the model memorised. "
           "No calibrated threshold exists for this model, so no strength grade is assigned at document level.")
    alts = ["Formulaic, technical or common-knowledge prose", "Text present in the model's training data",
            "Short documents (high variance of estimates)", "Domain mismatch with the reference model"]
    out.features.append(FeatureResult(MODULE, "Reference model", sc.meta, "", "model", "", "",
                                      "Model name, tokenizer, token counts, device and calculation method.", None, "n/a", "", lim, []))
    if summ.get("n_tokens", 0) < 20:
        out.features.append(unavailable(MODULE, "Surprisal statistics", "fewer than 20 scored tokens"))
        return out
    items = [("Mean negative log-likelihood", summ["mean_nll_nats"], "nats/token"),
             ("Cross-entropy", summ["cross_entropy_bits_per_token"], "bits/token"),
             ("Perplexity", summ["perplexity"], "")]
    for name, val, unit in items:
        out.features.append(FeatureResult(MODULE, name, val, unit, "model", "reference-model specific",
                                          "", f"{name} of the scored tokens under the reference model.", None, "n/a", "", lim, alts))
    for key, label in (("surprisal_std", "Surprisal standard deviation"), ("surprisal_variance", "Surprisal variance"),
                       ("surprisal_quantiles", "Surprisal quantiles (nats)"), ("rank_stats", "Token-rank statistics"),
                       ("local_mean_surprisal_std", "Local variation of mean surprisal"),
                       ("segment_perplexity", "Perplexity across five document segments")):
        if key in summ:
            out.features.append(FeatureResult(MODULE, label, summ[key], "", "model", "", "", f"{label}.", None, "n/a", "", lim, alts))
    return out
