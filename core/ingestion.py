"""Text ingestion: pasted text, TXT, DOCX and PDF (extractable text only)."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.settings import HARD_LIMIT_WORDS, SOFT_LIMIT_WORDS
from core.normalization import normalize_text

MAX_FILE_BYTES = 50 * 1024 * 1024


class IngestionError(Exception):
    """Raised when input cannot be read."""


class ScannedPDFError(IngestionError):
    """Raised when a PDF has no extractable text layer (e.g. scanned pages)."""


class DocumentTooLargeError(IngestionError):
    """Raised above the hard word limit (the text is never silently truncated)."""


@dataclass
class IngestResult:
    text: str
    source: str
    encoding: str = "utf-8"
    warnings: List[str] = field(default_factory=list)
    hints: List[Tuple[int, int, str]] = field(default_factory=list)   # (start, end, kind)
    normalization: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


_WORD = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def _finalise(text: str, source: str, encoding: str, warnings: List[str],
              hints: Optional[List[Tuple[int, int, str]]] = None, meta: Optional[dict] = None) -> IngestResult:
    n = normalize_text(text)
    if n.text != text and hints:
        warnings.append("Text was normalised after extraction; structural hints were discarded.")
        hints = []
    words = count_words(n.text)
    if not n.text.strip():
        raise IngestionError("The document contains no text.")
    if words > HARD_LIMIT_WORDS:
        raise DocumentTooLargeError(
            f"The document has {words:,} words, above the {HARD_LIMIT_WORDS:,}-word safety limit. "
            "Split it into parts; the application never silently truncates text.")
    if words > SOFT_LIMIT_WORDS:
        warnings.append(f"The document has {words:,} words (> ~{SOFT_LIMIT_WORDS:,}). It will be analysed in full, "
                        "but processing may be slow and memory-hungry; consider lightweight mode.")
    return IngestResult(text=n.text, source=source, encoding=encoding, warnings=warnings,
                        hints=hints or [], normalization=n.changes, meta=meta or {})


def from_pasted_text(text: str) -> IngestResult:
    return _finalise(text, "pasted", "utf-8 (as provided)", [])


def decode_bytes(data: bytes) -> Tuple[str, str, List[str]]:
    """Decode with BOM detection, then UTF-8, then CP1252, then Latin-1."""
    warnings: List[str] = []
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8"), "utf-8-sig", warnings
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16", warnings
    try:
        return data.decode("utf-8"), "utf-8", warnings
    except UnicodeDecodeError:
        pass
    try:
        txt = data.decode("cp1252")
        warnings.append("File is not valid UTF-8; decoded as Windows-1252. Check special characters.")
        return txt, "cp1252", warnings
    except UnicodeDecodeError:
        txt = data.decode("latin-1")
        warnings.append("File is not valid UTF-8 or Windows-1252; decoded as Latin-1. Check special characters.")
        return txt, "latin-1", warnings


def from_txt_bytes(data: bytes, name: str = "file.txt") -> IngestResult:
    if len(data) > MAX_FILE_BYTES:
        raise IngestionError("File is larger than the 50 MB safety limit.")
    if not data.strip():
        raise IngestionError("The file is empty.")
    if b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise IngestionError("The file looks binary, not text.")
    text, enc, warns = decode_bytes(data)
    return _finalise(text, name, enc, warns)


def from_docx_bytes(data: bytes, name: str = "file.docx") -> IngestResult:
    try:
        import docx  # python-docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:  # pragma: no cover
        raise IngestionError("python-docx is not installed (pip install python-docx).") from e
    if len(data) > MAX_FILE_BYTES:
        raise IngestionError("File is larger than the 50 MB safety limit.")
    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as e:
        raise IngestionError(f"Could not open the DOCX file: {e}") from e
    parts: List[str] = []
    hints: List[Tuple[int, int, str]] = []
    pos = 0
    warnings: List[str] = []

    def add(block: str, kind: Optional[str]):
        nonlocal pos
        if not block.strip():
            return
        start = pos
        parts.append(block)
        pos += len(block)
        if kind:
            hints.append((start, pos, kind))
        parts.append("\n\n")
        pos += 2

    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            style = (p.style.name if p.style is not None else "") or ""
            low = style.lower()
            kind = "heading" if low.startswith("heading") or low == "title" else (
                "bullet" if "list" in low else None)
            add(p.text, kind)
        elif tag == "tbl":
            t = Table(child, d)
            rows = [" | ".join(c.text.strip().replace("\n", " ") for c in r.cells) for r in t.rows]
            add("\n".join(rows), "table")
    text = "".join(parts).rstrip("\n")
    if hints and hints[-1][1] > len(text):
        hints[-1] = (hints[-1][0], len(text), hints[-1][2])
    warnings.append("DOCX footnotes, headers/footers and text boxes are not extracted.")
    return _finalise(text, name, "docx", warnings, hints)


def _reflow_page(page_text: str) -> str:
    """Join hard-wrapped PDF lines into paragraphs (heuristic)."""
    lines = [ln.rstrip() for ln in page_text.split("\n")]
    nonempty = [len(ln) for ln in lines if ln.strip()]
    max_len = max(nonempty) if nonempty else 0
    out: List[str] = []
    cur = ""
    for i, ln in enumerate(lines):
        if not ln.strip():
            if cur:
                out.append(cur)
                cur = ""
            continue
        if not cur:
            cur = ln.strip()
            continue
        prev_end = cur[-1]
        short = len(lines[i - 1]) < 0.75 * max_len if max_len else False
        if prev_end in ".!?:\"”)" and ln.strip()[0].isupper() and short:
            out.append(cur)
            cur = ln.strip()
        elif cur.endswith("-") and ln.strip()[:1].islower():
            cur = cur[:-1] + ln.strip()
        else:
            cur += " " + ln.strip()
    if cur:
        out.append(cur)
    return "\n\n".join(out)


def from_pdf_bytes(data: bytes, name: str = "file.pdf") -> IngestResult:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise IngestionError("pypdf is not installed (pip install pypdf).") from e
    if len(data) > MAX_FILE_BYTES:
        raise IngestionError("File is larger than the 50 MB safety limit.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise IngestionError("The PDF is password-protected; remove the password and retry.")
        pages = []
        for pg in reader.pages:
            try:
                pages.append(pg.extract_text() or "")
            except Exception:
                pages.append("")
    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Could not read the PDF: {e}") from e
    n_pages = len(pages)
    total_chars = sum(len(p.strip()) for p in pages)
    if n_pages == 0 or total_chars < 25 * max(n_pages, 1):
        raise ScannedPDFError(
            "This PDF has little or no extractable text, so it appears to consist of scanned images. "
            "TextForensics Local does not perform OCR and has NOT analysed this document. Use an OCR tool "
            "to create a text layer (or paste the text) and try again.")
    empty = sum(1 for p in pages if len(p.strip()) < 25)
    warnings = ["PDF text was reflowed heuristically; headers, footers, page numbers and hyphenation "
                "may remain in the extracted text."]
    if empty:
        warnings.append(f"{empty} of {n_pages} pages had no extractable text and were not analysed.")
    text = "\n\n".join(_reflow_page(p) for p in pages if p.strip())
    return _finalise(text, name, "pdf", warnings, meta={"pages": n_pages, "pages_without_text": empty})


def from_upload(name: str, data: bytes) -> IngestResult:
    """Dispatch on file extension."""
    low = name.lower()
    if low.endswith(".docx"):
        return from_docx_bytes(data, name)
    if low.endswith(".pdf"):
        return from_pdf_bytes(data, name)
    if low.endswith((".txt", ".md", ".text")):
        return from_txt_bytes(data, name)
    raise IngestionError("Unsupported file type. Supported: .txt, .docx, .pdf (with extractable text).")
