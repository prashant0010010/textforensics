"""Dependency-free SVG charts for reports and the interface."""
from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence

import numpy as np

W, H, PAD = 560, 220, 36


def _wrap(title: str, body: str, w: int = W, h: int = H) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(title)}" '
            f'font-family="Segoe UI,Arial,sans-serif" font-size="11"><text x="{w/2}" y="14" text-anchor="middle" font-size="12" font-weight="600">{html.escape(title)}</text>{body}</svg>')


def histogram(values: Sequence[float], title: str, bins: int = 20, xlabel: str = "") -> str:
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size < 2:
        return _wrap(title, f'<text x="{W/2}" y="{H/2}" text-anchor="middle">not enough data</text>')
    counts, edges = np.histogram(v, bins=min(bins, max(3, int(np.sqrt(v.size)) + 2)))
    mx = counts.max() or 1
    bw = (W - 2 * PAD) / len(counts)
    bars = "".join(f'<rect x="{PAD + i * bw:.1f}" y="{H - PAD - (c / mx) * (H - 2 * PAD):.1f}" width="{bw - 1:.1f}" '
                   f'height="{(c / mx) * (H - 2 * PAD):.1f}" fill="#4a6fa5"><title>{edges[i]:.1f}-{edges[i+1]:.1f}: {c}</title></rect>' for i, c in enumerate(counts))
    axis = (f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#555"/>'
            f'<text x="{PAD}" y="{H-PAD+14}">{edges[0]:.0f}</text><text x="{W-PAD}" y="{H-PAD+14}" text-anchor="end">{edges[-1]:.0f}</text>'
            f'<text x="{W/2}" y="{H-6}" text-anchor="middle">{html.escape(xlabel)}</text>')
    return _wrap(title, bars + axis)


def bar_chart(values: Sequence[float], title: str, xlabel: str = "") -> str:
    v = [float(x) for x in values]
    if not v:
        return _wrap(title, f'<text x="{W/2}" y="{H/2}" text-anchor="middle">no data</text>')
    mx = max(v) or 1
    bw = (W - 2 * PAD) / len(v)
    bars = "".join(f'<rect x="{PAD + i * bw:.1f}" y="{H - PAD - (x / mx) * (H - 2 * PAD):.1f}" width="{max(1, bw - 1):.1f}" height="{(x / mx) * (H - 2 * PAD):.1f}" fill="#5b8c5a"><title>#{i+1}: {x:.0f}</title></rect>' for i, x in enumerate(v))
    return _wrap(title, bars + f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#555"/><text x="{W/2}" y="{H-6}" text-anchor="middle">{html.escape(xlabel)}</text>')


def zipf_plot(table: Optional[Dict]) -> str:
    title = "Zipf plot: log frequency vs log rank"
    if not table or not table.get("log_rank"):
        return _wrap(title, f'<text x="{W/2}" y="{H/2}" text-anchor="middle">not computed</text>')
    x, y = np.asarray(table["log_rank"]), np.asarray(table["log_freq"])
    sx = lambda v: PAD + (v - x.min()) / (np.ptp(x) or 1) * (W - 2 * PAD)
    sy = lambda v: H - PAD - (v - y.min()) / (np.ptp(y) or 1) * (H - 2 * PAD)
    step = max(1, len(x) // 400)
    pts = "".join(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="1.8" fill="#4a6fa5"/>' for a, b in zip(x[::step], y[::step]))
    line = ""
    fit = (table.get("fits") or {}).get("ranks_with_freq>=2") or (table.get("fits") or {}).get("all_ranks")
    if fit and fit.get("slope") is not None:
        xs = np.array([x.min(), x.max()])
        ys = fit["intercept"] + fit["slope"] * xs
        line = f'<line x1="{sx(xs[0]):.1f}" y1="{sy(ys[0]):.1f}" x2="{sx(xs[1]):.1f}" y2="{sy(ys[1]):.1f}" stroke="#c0392b" stroke-width="1.5" stroke-dasharray="4 3"/>'
    return _wrap(title, pts + line + f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#555"/><line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#555"/>')


def anomaly_map(window_z: List[Dict], text_len: int, cut: float = 3.0) -> str:
    """Strip along the document; colour = largest |robust z| among AI-direction and descriptive features."""
    title = "Internal anomaly map (max |robust z| per window)"
    if not window_z or text_len <= 0:
        return _wrap(title, f'<text x="{W/2}" y="{H/2}" text-anchor="middle">not computed (too few windows)</text>', W, 90)
    rects = []
    for w in window_z:
        zs = [abs(v) for v in w["z"].values() if v is not None]
        m = max(zs) if zs else 0
        a = min(1.0, m / 6.0)
        color = f"rgba(214,64,45,{0.12 + 0.88 * a:.2f})" if m >= cut else f"rgba(74,111,165,{0.10 + 0.35 * a:.2f})"
        x = PAD + w["start"] / text_len * (W - 2 * PAD)
        wd = max(1.5, (w["end"] - w["start"]) / text_len * (W - 2 * PAD))
        rects.append(f'<rect x="{x:.1f}" y="30" width="{wd:.1f}" height="30" fill="{color}"><title>max |z| = {m:.1f}</title></rect>')
    legend = (f'<text x="{PAD}" y="78">start of document</text><text x="{W-PAD}" y="78" text-anchor="end">end</text>'
              f'<text x="{W/2}" y="78" text-anchor="middle">red = |z| &gt;= {cut:g} (overlapping windows)</text>')
    return _wrap(title, "".join(rects) + legend, W, 90)
