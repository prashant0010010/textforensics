"""Benchmark processing time and peak memory for several document lengths.

    python scripts/benchmark.py                 # 1k, 5k, 15k, 30k words
    python scripts/benchmark.py --sizes 2000 10000 --lightweight

Uses seeded synthetic text ONLY to measure speed/memory. It says nothing about detection accuracy.
Numbers depend on your computer; no fixed processing time is promised.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from config.settings import AnalysisSettings  # noqa: E402
from core.pipeline import run_analysis  # noqa: E402
from synthetic import make_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 15000, 30000])
    ap.add_argument("--lightweight", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-memory", action="store_true", help="skip the (slower) tracemalloc peak-memory run")
    a = ap.parse_args()
    print(f"{'words':>8} {'seconds':>9} {'peak MB':>9}  slowest modules")
    for n in a.sizes:
        text = make_text(a.seed, n)
        t0 = time.perf_counter()
        res = run_analysis(text, AnalysisSettings(lightweight=a.lightweight))
        dt = time.perf_counter() - t0
        peak = float("nan")
        if not a.skip_memory:                      # separate run: tracemalloc slows execution several-fold
            tracemalloc.start()
            run_analysis(text, AnalysisSettings(lightweight=a.lightweight))
            peak = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
        slow = sorted(res.statuses, key=lambda s: -s.seconds)[:3]
        print(f"{res.doc_stats['words_total']:>8} {dt:>9.2f} {peak:>9.1f}  " + ", ".join(f"{s.name} {s.seconds:.1f}s" for s in slow))


if __name__ == "__main__":
    main()
