"""Unicode / line-ending normalisation and invisible-character inspection.

The *canonical text* produced here is the single string every offset in the
application refers to. After this step the text is never altered again.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

INVISIBLE = {
    "\u200b": "ZERO WIDTH SPACE", "\u200c": "ZERO WIDTH NON-JOINER", "\u200d": "ZERO WIDTH JOINER",
    "\u2060": "WORD JOINER", "\ufeff": "ZERO WIDTH NO-BREAK SPACE / BOM", "\u00ad": "SOFT HYPHEN",
    "\u200e": "LEFT-TO-RIGHT MARK", "\u200f": "RIGHT-TO-LEFT MARK", "\u202a": "LRE", "\u202b": "RLE",
    "\u202c": "PDF", "\u202d": "LRO", "\u202e": "RLO", "\u2066": "LRI", "\u2067": "RLI", "\u2068": "FSI",
    "\u2069": "PDI", "\u00a0": "NO-BREAK SPACE", "\u202f": "NARROW NO-BREAK SPACE", "\u2009": "THIN SPACE",
    "\u2003": "EM SPACE", "\u2002": "EN SPACE", "\u3000": "IDEOGRAPHIC SPACE", "\u180e": "MONGOLIAN VOWEL SEPARATOR",
}


@dataclass
class NormalizationReport:
    text: str
    changes: Dict[str, int] = field(default_factory=dict)


def normalize_text(raw: str) -> NormalizationReport:
    """NFC-normalise, unify line endings, drop a leading BOM. Nothing else is changed."""
    changes: Dict[str, int] = {}
    t = raw
    if t.startswith("\ufeff"):
        t = t[1:]
        changes["leading_bom_removed"] = 1
    n_crlf = t.count("\r\n")
    n_cr = t.count("\r") - n_crlf
    if n_crlf or n_cr:
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        changes["line_endings_unified"] = n_crlf + n_cr
    nfc = unicodedata.normalize("NFC", t)
    if nfc != t:
        changes["nfc_composed_chars"] = abs(len(t) - len(nfc)) or 1
        t = nfc
    t = t.replace("\x00", "")
    return NormalizationReport(text=t, changes=changes)


def inspect_text_characters(text: str, max_examples: int = 50) -> Dict[str, Any]:
    """Report invisible / unusual characters. This is *text inspection only*:
    it is not evidence of AI authorship (copy-paste from web pages, Word and PDF
    routinely introduces such characters)."""
    counts: Counter = Counter()
    examples: List[Dict[str, Any]] = []
    scripts: Counter = Counter()
    ctrl = 0
    for i, ch in enumerate(text):
        if ch in INVISIBLE:
            counts[ch] += 1
            if len(examples) < max_examples:
                examples.append({"offset": i, "codepoint": f"U+{ord(ch):04X}", "name": INVISIBLE[ch]})
        elif ch not in "\n\t" and unicodedata.category(ch) in ("Cc", "Cf"):
            ctrl += 1
        if ch.isalpha():
            try:
                scripts[unicodedata.name(ch).split(" ")[0]] += 1
            except ValueError:
                scripts["UNKNOWN"] += 1
    return {
        "invisible_counts": {f"U+{ord(c):04X} {INVISIBLE[c]}": n for c, n in counts.items()},
        "total_invisible": int(sum(counts.values())),
        "other_control_or_format_chars": ctrl,
        "examples": examples,
        "alphabetic_script_counts": dict(scripts.most_common(8)),
        "mixed_scripts": sum(1 for s, n in scripts.items() if n > 0 and s not in ("UNKNOWN",)) > 1,
        "note": ("Inspection of characters only. Their presence is not proof of AI authorship and their "
                 "absence is not proof of human authorship."),
    }
