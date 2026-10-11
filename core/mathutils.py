"""Numerically safe statistical helpers (NumPy only).

Every function returns ``None`` (never NaN/inf) when the statistic is not
mathematically defined for the input, so callers can report "not available"
instead of silently propagating invalid numbers.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def finite_or_none(x) -> Optional[float]:
    """Return ``float(x)`` if it is finite, else ``None``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def safe_div(a: float, b: float, default: Optional[float] = 0.0) -> Optional[float]:
    """Divide ``a`` by ``b``; return ``default`` for zero / non-finite results."""
    try:
        if b == 0 or not math.isfinite(b) or not math.isfinite(a):
            return default
        r = a / b
        return r if math.isfinite(r) else default
    except (TypeError, ZeroDivisionError):
        return default


def to_array(x: Iterable[float]) -> np.ndarray:
    """Convert to a float array, dropping non-finite values."""
    a = np.asarray(list(x), dtype=float)
    return a[np.isfinite(a)] if a.size else a


def mean(x: Iterable[float]) -> Optional[float]:
    a = to_array(x)
    return float(a.mean()) if a.size else None


def median(x: Iterable[float]) -> Optional[float]:
    a = to_array(x)
    return float(np.median(a)) if a.size else None


def variance(x: Iterable[float]) -> Optional[float]:
    """Sample variance (ddof=1); needs n >= 2."""
    a = to_array(x)
    return float(a.var(ddof=1)) if a.size >= 2 else None


def std(x: Iterable[float]) -> Optional[float]:
    v = variance(x)
    return math.sqrt(v) if v is not None else None


def iqr(x: Iterable[float]) -> Optional[float]:
    a = to_array(x)
    if a.size < 2:
        return None
    q75, q25 = np.percentile(a, [75, 25])
    return float(q75 - q25)


def quantiles(x: Iterable[float], qs: Sequence[float] = (5, 25, 50, 75, 95)) -> dict:
    a = to_array(x)
    if a.size == 0:
        return {}
    return {f"q{int(q)}": float(np.percentile(a, q)) for q in qs}


def coefficient_of_variation(x: Iterable[float]) -> Optional[float]:
    """Sample SD divided by mean; undefined when the mean is zero."""
    a = to_array(x)
    if a.size < 2:
        return None
    m = float(a.mean())
    if m == 0:
        return None
    return finite_or_none(float(a.std(ddof=1)) / abs(m))


def skewness(x: Iterable[float]) -> Optional[float]:
    """Adjusted Fisher-Pearson skewness G1 (n >= 3, non-zero variance)."""
    a = to_array(x)
    n = a.size
    if n < 3:
        return None
    m2 = float(((a - a.mean()) ** 2).mean())
    if m2 <= 0:
        return None
    m3 = float(((a - a.mean()) ** 3).mean())
    g1 = m3 / m2 ** 1.5
    return finite_or_none(math.sqrt(n * (n - 1)) / (n - 2) * g1)


def excess_kurtosis(x: Iterable[float]) -> Optional[float]:
    """Unbiased excess kurtosis G2 (n >= 4, non-zero variance)."""
    a = to_array(x)
    n = a.size
    if n < 4:
        return None
    m2 = float(((a - a.mean()) ** 2).mean())
    if m2 <= 0:
        return None
    m4 = float(((a - a.mean()) ** 4).mean())
    g2 = m4 / m2 ** 2 - 3.0
    return finite_or_none((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0))


def autocorrelation(x: Iterable[float], lag: int = 1, min_n: int = 10) -> Optional[float]:
    """Lag-k sample autocorrelation; needs at least ``min_n`` points."""
    a = to_array(x)
    if a.size < max(min_n, lag + 3):
        return None
    d = a - a.mean()
    den = float((d ** 2).sum())
    if den <= 0:
        return None
    return finite_or_none(float((d[:-lag] * d[lag:]).sum()) / den)


def mad(x: Iterable[float]) -> Optional[float]:
    a = to_array(x)
    if a.size == 0:
        return None
    return float(np.median(np.abs(a - np.median(a))))


def robust_scale(x: Iterable[float]) -> float:
    """Robust SD estimate: 1.4826*MAD, falling back to IQR/1.349, else 0."""
    a = to_array(x)
    if a.size < 3:
        return 0.0
    m = float(np.median(np.abs(a - np.median(a)))) * 1.4826
    if m > 0:
        return m
    q75, q25 = np.percentile(a, [75, 25])
    return float((q75 - q25) / 1.349)


def robust_z(value: float, med: float, scale: float) -> Optional[float]:
    if scale <= 0 or not math.isfinite(value):
        return None
    return finite_or_none((value - med) / scale)


def shannon_entropy(counts: Iterable[float], base: float = 2.0) -> Optional[float]:
    """Plug-in (maximum-likelihood) Shannon entropy of a count vector."""
    c = np.asarray([v for v in counts if v > 0], dtype=float)
    n = c.sum()
    if c.size == 0 or n <= 0:
        return None
    p = c / n
    return finite_or_none(float(-(p * (np.log(p) / math.log(base))).sum()))


def miller_madow_entropy(counts: Iterable[float]) -> Optional[float]:
    """Miller-Madow bias-corrected entropy in bits: H_ML + (K-1)/(2 N ln 2)."""
    c = [v for v in counts if v > 0]
    h = shannon_entropy(c)
    if h is None:
        return None
    n = float(sum(c))
    return finite_or_none(h + (len(c) - 1) / (2.0 * n * math.log(2)))


def normalized_entropy(counts: Iterable[float]) -> Optional[float]:
    """Entropy divided by its maximum log2(K); 0 when K <= 1."""
    c = [v for v in counts if v > 0]
    if len(c) <= 1:
        return 0.0 if c else None
    h = shannon_entropy(c)
    return None if h is None else finite_or_none(h / math.log2(len(c)))


def js_divergence(p: Mapping, q: Mapping) -> Optional[float]:
    """Jensen-Shannon divergence (bits, 0..1) between two count/prob mappings."""
    keys = set(p) | set(q)
    if not keys:
        return None
    sp = float(sum(p.values()))
    sq = float(sum(q.values()))
    if sp <= 0 or sq <= 0:
        return None
    pa = np.array([p.get(k, 0.0) / sp for k in keys])
    qa = np.array([q.get(k, 0.0) / sq for k in keys])
    m = 0.5 * (pa + qa)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return finite_or_none(0.5 * kl(pa, m) + 0.5 * kl(qa, m))


def js_distance(p: Mapping, q: Mapping) -> Optional[float]:
    d = js_divergence(p, q)
    return None if d is None else math.sqrt(max(d, 0.0))


def cosine_similarity(p: Mapping, q: Mapping) -> Optional[float]:
    keys = set(p) | set(q)
    if not keys:
        return None
    a = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    b = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return finite_or_none(float(a @ b / (na * nb)))


def linear_fit(x: Sequence[float], y: Sequence[float]) -> Optional[dict]:
    """Ordinary least squares y = a + b x with R^2 and RMSE."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size < 3 or float(xa.var()) == 0:
        return None
    slope, intercept = np.polyfit(xa, ya, 1)
    pred = intercept + slope * xa
    ss_res = float(((ya - pred) ** 2).sum())
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "slope": finite_or_none(slope),
        "intercept": finite_or_none(intercept),
        "r2": finite_or_none(r2),
        "rmse": finite_or_none(math.sqrt(ss_res / xa.size)),
        "n_points": int(xa.size),
    }


def ks_two_sample(a: Iterable[float], b: Iterable[float]) -> Optional[Tuple[float, float]]:
    """Two-sample Kolmogorov-Smirnov statistic D and asymptotic p-value."""
    x, y = np.sort(to_array(a)), np.sort(to_array(b))
    n, m = x.size, y.size
    if n < 2 or m < 2:
        return None
    allv = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, allv, side="right") / n
    cdf_y = np.searchsorted(y, allv, side="right") / m
    d = float(np.abs(cdf_x - cdf_y).max())
    ne = n * m / (n + m)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    if lam < 1e-9:
        return d, 1.0
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


def counter_entropy(c: Counter) -> Optional[float]:
    return shannon_entropy(c.values())
