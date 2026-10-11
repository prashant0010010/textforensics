"""JSON report (machine-readable)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any, Dict

import numpy as np

from core.schemas import AnalysisResult
from reporting.common import LIMITATIONS, NOT_EQUIVALENT, METHODOLOGY, modules_summary, selected_signals


def _clean(o: Any) -> Any:
    """Make values JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (bool, int, str)) or o is None:
        return o
    return str(o)


def build_json(res: AnalysisResult, include_text: bool = True) -> Dict[str, Any]:
    """``include_text=False`` omits passages and the full text (offsets and hash only)."""
    chosen, skipped = selected_signals(res)
    text = res.document.text
    sigs = []
    for s in res.signals:
        d = asdict(s)
        d["passage"] = text[s.start:s.end] if include_text else None
        d["highlighted"] = s in chosen
        sigs.append(d)
    segs = []
    for g in res.segments:
        d = asdict(g)
        # Offsets identify sentence/window/document segments; their text is omitted to keep the file compact.
        if not include_text or g.level in ("sentence", "window", "document"):
            d["text"] = None
        segs.append(d)
    return _clean({
        "notice": NOT_EQUIVALENT,
        "evidence": res.evidence,
        "document": {"stats": res.doc_stats, "ingest": res.ingest, "inspection": res.inspection,
                     "text": text if include_text else None},
        "modules": modules_summary(res),
        "features": [asdict(f) for f in res.features],
        "signals": sigs,
        "segments": segs,
        "tables": {k: v for k, v in res.tables.items() if k not in ("window_features",)},
        "recommendations": res.recommendations,
        "limitations": LIMITATIONS,
        "methodology": [{"topic": a, "description": b} for a, b in METHODOLOGY],
        "settings": res.settings,
        "reproducibility": res.reproducibility,
    })


def to_json_string(res: AnalysisResult, include_text: bool = True) -> str:
    return json.dumps(build_json(res, include_text), indent=2, ensure_ascii=False, allow_nan=False)
