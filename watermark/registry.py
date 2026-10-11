"""Registry of locally executable watermark detectors."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from watermark.base import UNAVAILABLE_MESSAGE, WatermarkDetector, WatermarkResult
from watermark.green_list import GreenListZTest

_REGISTRY: Dict[str, WatermarkDetector] = {}

# Schemes people ask about that this application deliberately does NOT emulate.
UNSUPPORTED_SCHEMES = {
    "SynthID-Text": "Detection requires the secret watermarking keys/configuration and the official detector; "
                    "none is bundled, so no result is produced.",
    "Proprietary / vendor watermarks": "Algorithms and keys are not public.",
}


def register(detector: WatermarkDetector) -> None:
    """Add a detector. Only register detectors whose real algorithm is implemented."""
    _REGISTRY[detector.scheme] = detector


def list_detectors() -> List[WatermarkDetector]:
    return list(_REGISTRY.values())


def run_all(text: str, configs: Optional[Dict[str, Dict[str, Any]]] = None) -> List[WatermarkResult]:
    configs = configs or {}
    results: List[WatermarkResult] = []
    runnable = [d for d in _REGISTRY.values() if d.check_config(configs.get(d.scheme))[0]]
    n_tests = max(1, len(runnable))
    for d in _REGISTRY.values():
        cfg = configs.get(d.scheme)
        ok, status = d.check_config(cfg)
        results.append(d.detect(text, cfg or {}, n_tests) if ok else d.not_run("missing" if not cfg else "invalid", status))
    for name, reason in UNSUPPORTED_SCHEMES.items():
        results.append(WatermarkResult(name, "none", "-", "n/a", False, f"{UNAVAILABLE_MESSAGE} {reason}", limitations=reason))
    return results


register(GreenListZTest())
