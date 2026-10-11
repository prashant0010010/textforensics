"""Green-list watermark z-test (Kirchenbauer et al., 2023, "A Watermark for Large Language Models").

What this detector DOES: given (a) the tokenizer used at generation time, (b) the
fraction ``gamma`` of the vocabulary that is "green" at each step, and (c) the
exact green-list rule of the generating system (a user-supplied Python file that
defines ``is_green(prev_token_id, token_id) -> bool``), it counts green tokens and
performs the one-sided z-test

    z = (|s|_G - gamma * T) / sqrt(T * gamma * (1 - gamma))

H0: the text was NOT generated with this watermark, so each scored token is green
independently with probability gamma. The p-value is the upper tail of the
standard normal.

What it does NOT do: it cannot find an unknown key or scheme. Without the exact
rule and tokenizer it reports "Watermark testing unavailable for this scheme."
"""
from __future__ import annotations

import importlib.util
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from watermark.base import WatermarkDetector, WatermarkResult

MIN_TOKENS = 25


def green_list_z_test(token_ids: Sequence[int], is_green: Callable[[int, int], bool], gamma: float,
                      dedupe_pairs: bool = True) -> Dict[str, Any]:
    """Core statistic; independent of any tokenizer (unit-testable)."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    pairs = list(zip(token_ids[:-1], token_ids[1:]))
    if dedupe_pairs:
        pairs = list(dict.fromkeys(pairs))     # score each (prev, token) pair once
    T = len(pairs)
    green = sum(1 for p, t in pairs if is_green(p, t))
    if T == 0:
        return {"T": 0, "green": 0, "z": None, "p": None}
    z = (green - gamma * T) / math.sqrt(T * gamma * (1 - gamma))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return {"T": T, "green": green, "z": z, "p": p}


class GreenListZTest(WatermarkDetector):
    scheme = "Green-list (KGW-style) z-test"
    implementation = "textforensics_local.watermark.green_list"
    version = "1.0"
    requires = ("transformers (tokenizer)", "scheme_module: python file defining is_green(prev_id, tok_id)")
    description = "Tests for a green-list watermark when the exact generation rule and tokenizer are supplied."

    def _load_rule(self, config: Dict[str, Any]) -> Callable[[int, int], bool]:
        if callable(config.get("is_green")):
            return config["is_green"]
        path = config.get("scheme_module", "")
        spec = importlib.util.spec_from_file_location("tf_user_watermark_scheme", path)
        if spec is None or spec.loader is None:
            raise ValueError("scheme_module could not be loaded")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)        # executes the user's own file locally
        fn = getattr(mod, "is_green", None)
        if not callable(fn):
            raise ValueError("scheme_module must define is_green(prev_token_id, token_id)")
        return fn

    def check_config(self, config: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        if not config:
            return False, "no configuration supplied"
        g = config.get("gamma")
        if not isinstance(g, (int, float)) or not 0 < g < 1:
            return False, "'gamma' must be a number in (0, 1)"
        if not (callable(config.get("is_green")) or os.path.isfile(str(config.get("scheme_module", "")))):
            return False, "'scheme_module' must be the path of an existing Python file defining is_green()"
        if config.get("tokenizer") is None:
            tp = str(config.get("tokenizer_path", ""))
            if not os.path.isdir(tp):
                return False, "'tokenizer_path' must be an existing local tokenizer directory"
            try:
                import transformers  # noqa: F401
            except ImportError:
                return False, "the 'transformers' package is not installed"
        return True, "ok"

    def detect(self, text: str, config: Dict[str, Any], n_tests: int = 1) -> WatermarkResult:
        ok, status = self.check_config(config)
        if not ok:
            return self.not_run("missing" if not config else "invalid", status)
        try:
            tok = config.get("tokenizer")
            if tok is None:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
            rule = self._load_rule(config)
            ids: List[int] = list(tok.encode(text, add_special_tokens=False))
        except Exception as e:
            return self.not_run("invalid", f"configuration could not be used: {e}")
        gamma = float(config["gamma"])
        res = green_list_z_test(ids, rule, gamma, bool(config.get("dedupe_pairs", True)))
        if res["T"] < MIN_TOKENS:
            r = self.not_run("ok", f"only {res['T']} eligible tokens (minimum {MIN_TOKENS}).")
            r.eligible_tokens = res["T"]
            return r
        adj = min(1.0, res["p"] * max(1, n_tests))
        return WatermarkResult(
            self.scheme, self.implementation, self.version, "ok", True, "Test executed.", res["T"],
            "z-score of green-token count", res["z"],
            f"H0: tokens are green independently with probability gamma={gamma} (text not generated with this watermark).",
            res["p"], adj,
            f"Bonferroni adjustment for {n_tests} watermark test(s) run in this analysis; windowed re-testing would require further correction.",
            "Valid only for text produced with exactly this scheme/tokenizer. Paraphrasing, translation and editing reduce "
            "detectability; a non-significant result does not show the text is unwatermarked, and a significant one "
            "does not identify who used the model.",
            {"green_tokens": res["green"], "gamma": gamma, "dedupe_pairs": bool(config.get("dedupe_pairs", True))})
