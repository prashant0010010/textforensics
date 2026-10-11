"""TextForensics Local - Streamlit interface. Everything runs on this computer; no text is transmitted."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
import streamlit as st
import streamlit.components.v1 as components

from config.settings import APP_NAME, APP_VERSION, AnalysisSettings, SOFT_LIMIT_WORDS
from core.ingestion import IngestionError, IngestResult, ScannedPDFError, from_pasted_text, from_upload
from core.pipeline import run_analysis
from core.schemas import CATEGORIES, KIND_LABELS, TAXONOMY
from core.segmentation import document_counts, segment_document
from reporting.html_report import build_html
from reporting.json_report import to_json_string
from reporting.pdf_report import PdfUnavailable, build_pdf, is_available as pdf_available
from reporting.text_report import build_text
from visualization import charts
from visualization.highlights import interactive_panel_html
from visualization.tables import feature_rows, fmt, signal_rows
from watermark import registry as wm_registry
from watermark.base import UNAVAILABLE_MESSAGE
from reporting.common import selected_signals

st.set_page_config(page_title=APP_NAME, page_icon="🔍", layout="wide")


def render_html(html_str: str, height: int, scrolling: bool = False) -> None:
    """Render a self-contained HTML fragment (iframe so in-page anchors/JS work); fall back to markdown."""
    try:
        components.html(html_str, height=height, scrolling=scrolling)
    except Exception:  # pragma: no cover - API differences between Streamlit versions
        st.markdown(html_str, unsafe_allow_html=True)


def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@st.cache_data(show_spinner=False)
def preview_counts(text: str, settings_json: str, hints: tuple) -> dict:
    cfg = AnalysisSettings(**{k: v for k, v in json.loads(settings_json).items()})
    return document_counts(text, segment_document(text, cfg, list(hints)))


def sidebar_settings() -> AnalysisSettings:
    sb = st.sidebar
    sb.header("Analysis settings")
    s = AnalysisSettings()
    sb.subheader("Modules")
    s.enable_statistical = sb.checkbox("Statistical analysis (sentence stats, entropy, Zipf, punctuation)", True)
    s.enable_lexical = sb.checkbox("Lexical analysis (diversity, n-grams)", True)
    s.enable_structural = sb.checkbox("Structural analysis (paragraphs)", True)
    s.enable_linguistic = sb.checkbox("Linguistic analysis (function words, POS, discourse)", True)
    s.enable_internal = sb.checkbox("Internal consistency / anomaly analysis", True)
    s.enable_watermark = sb.checkbox("Available watermark tests", True)
    s.lightweight = sb.checkbox("Lightweight mode (less RAM/CPU, fewer shuffles, no spaCy/NLTK)", False)
    sb.subheader("Special text")
    s.include_quotes = sb.checkbox("Include block quotations in analysis", False)
    s.include_references = sb.checkbox("Include reference lists", False)
    s.include_bullets = sb.checkbox("Include bullet lists", True)
    s.include_code = sb.checkbox("Include code blocks", False)
    s.include_tables = sb.checkbox("Include tables", False)
    s.include_headings = sb.checkbox("Include headings", False)
    s.single_newline_paragraphs = sb.selectbox("Single line breaks are paragraph breaks", ["auto", "yes", "no"], 0)
    with sb.expander("Window and baseline sizes"):
        s.window_sentences = int(st.number_input("Window size (sentences)", 4, 30, 8))
        s.window_step = int(st.number_input("Window step (sentences)", 1, 30, 4))
        s.min_window_words = int(st.number_input("Minimum words per window", 30, 400, 80))
        s.entropy_window_tokens = int(st.number_input("Entropy window (tokens)", 40, 500, 100))
        s.shuffle_iterations = int(st.number_input("N-gram shuffle iterations", 5, 100, 20))
        s.seed = int(st.number_input("Random seed", 0, 10 ** 6, 1234))
    sb.subheader("Optional dependencies")
    for label, mod in (("torch", "torch"), ("transformers", "transformers"), ("spaCy", "spacy"), ("NLTK", "nltk"), ("ReportLab (PDF export)", "reportlab")):
        sb.caption(f"{'✅' if have(mod) else '⚪'} {label}: {'installed' if have(mod) else 'not installed (optional)'}")
    return s


def model_settings(s: AnalysisSettings) -> None:
    with st.expander("Optional: local reference language model (advanced, heavy)"):
        st.markdown("Point to a **locally stored** Hugging Face causal language model folder. Nothing is downloaded automatically. "
                    "Needs the optional packages `torch` and `transformers`. Perplexity depends on the model and is **not** a universal "
                    "measure of AI authorship.")
        s.enable_model = st.checkbox("Enable reference-model analysis", False)
        s.model_path = st.text_input("Local model folder", "", placeholder=r"C:\models\gpt2")
        c1, c2, c3 = st.columns(3)
        s.model_device = c1.selectbox("Device", ["auto", "cpu", "cuda"], 0)
        s.model_max_length = int(c2.number_input("Context length (tokens)", 128, 4096, 1024))
        s.model_stride = int(c3.number_input("Stride (tokens)", 64, 4096, 512))
        if s.enable_model and not (have("torch") and have("transformers")):
            st.warning("torch/transformers are not installed: the module will be reported as unavailable; other analyses continue.")


def reference_inputs(s: AnalysisSettings):
    ref = None
    with st.expander("Optional: reference writing sample comparison"):
        st.markdown("Upload a *legitimately supplied* sample of the author's writing. Results are reported metric by metric; style similarity "
                    "does not prove authorship and a mismatch does not prove AI use.")
        up = st.file_uploader("Reference sample (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"], key="ref_up")
        s.reference_confirmed = st.checkbox("I confirm this sample is appropriate to use for comparison", False)
        if up is not None:
            try:
                ref = from_upload(up.name, up.getvalue())
                st.caption(f"Loaded {up.name}: {len(ref.text.split()):,} words")
                s.enable_reference = True
                if not s.reference_confirmed:
                    st.info("Tick the confirmation box to run the comparison.")
            except IngestionError as e:
                st.error(str(e))
    return ref


def watermark_inputs(s: AnalysisSettings) -> None:
    with st.expander("Optional: watermark detector configuration"):
        for d in wm_registry.list_detectors():
            st.markdown(f"**{d.scheme}** - {d.description}  \nRequires: {', '.join(d.requires)}")
        st.markdown("Unsupported schemes (e.g. SynthID-Text) report: *" + UNAVAILABLE_MESSAGE + "*")
        up = st.file_uploader("Detector configuration (JSON: {scheme name: {...}})", type=["json"], key="wm_up")
        if up is not None:
            try:
                s.watermark_configs = json.loads(up.getvalue().decode("utf-8"))
            except Exception as e:
                st.error(f"Invalid configuration file: {e}")


def get_input():
    tab1, tab2 = st.tabs(["Paste text", "Upload document"])
    ing = None
    with tab1:
        text = st.text_area("Text to analyse", height=300, key="pasted", placeholder="Paste up to ~30,000 words here...")
        if text.strip():
            try:
                ing = from_pasted_text(text)
            except IngestionError as e:
                st.error(str(e))
    with tab2:
        up = st.file_uploader("TXT, DOCX or PDF (extractable text)", type=["txt", "docx", "pdf"], key="doc_up")
        if up is not None:
            try:
                ing = from_upload(up.name, up.getvalue())
            except ScannedPDFError as e:
                st.error(str(e))
            except IngestionError as e:
                st.error(str(e))
    return ing


def validation_panel(ing: IngestResult, s: AnalysisSettings) -> None:
    sj = json.dumps({k: v for k, v in s.to_dict().items() if k not in ("thresholds", "watermark_configs")})
    try:
        c = preview_counts(ing.text, sj, tuple(ing.hints))
    except Exception:
        c = document_counts(ing.text, segment_document(ing.text, s, ing.hints))
    cols = st.columns(6)
    cols[0].metric("Words", f"{c['words_total']:,}")
    cols[1].metric("Characters", f"{c['characters']:,}")
    cols[2].metric("Paragraphs", c["paragraphs"])
    cols[3].metric("Sentences", c["sentences"])
    cols[4].metric("Words analysed", f"{c['words_analysed']:,}")
    cols[5].metric("Workload", c["workload"])
    st.caption(f"Source: {ing.source} · encoding: {ing.encoding} · excluded blocks: {fmt(c['excluded_blocks']) if c['excluded_blocks'] else 'none'} · "
               f"~{c['approx_lm_tokens']:,} LM tokens if a reference model is used. Workload is relative; no processing time is promised.")
    for w in ing.warnings:
        st.warning(w)
    if c["words_total"] > SOFT_LIMIT_WORDS:
        st.warning("Very long document: analysis runs in full (never truncated) but may take a while; consider lightweight mode.")
    if c["words_analysed"] < s.min_words_for_assessment:
        st.info(f"Fewer than {s.min_words_for_assessment} analysable words: the evidence label will be 'Insufficient data to assess'.")


def run(ing: IngestResult, s: AnalysisSettings, ref) -> None:
    bar = st.progress(0.0, text="Starting")
    status = st.empty()

    def cb(frac: float, msg: str) -> None:
        bar.progress(min(1.0, max(0.0, frac)), text=msg)
        status.caption(msg)

    try:
        res = run_analysis(ing, s, reference=ref, progress=cb)
    except Exception as e:  # keep the app alive; never echo the text
        st.error(f"Analysis failed: {type(e).__name__}: {str(e)[:300]}")
        return
    bar.empty()
    status.empty()
    st.session_state["result"] = res
    st.session_state.pop("downloads", None)


def show_results(res) -> None:
    ev = res.evidence
    st.success(f"Analysis complete. Evidence category {ev['category']}: **{ev['label']}**")
    t = st.tabs(["Overview", "Evidence", "Highlights", "Features", "Anomaly map", "Model", "Watermark", "Limitations & recommendations", "Export"])
    with t[0]:
        st.subheader("Document overview")
        st.json(res.doc_stats, expanded=False)
        st.markdown("**Modules**")
        st.dataframe([{"Module": x.name, "Status": x.status, "Seconds": round(x.seconds, 3), "Detail": x.detail} for x in res.statuses])
        with st.expander("Text inspection (characters only; not evidence of AI authorship)"):
            st.write(res.inspection["note"])
            st.json({k: v for k, v in res.inspection.items() if k != "note"}, expanded=False)
        for n in res.notes:
            st.caption("• " + n)
    with t[1]:
        st.subheader(f"Category {ev['category']}: {ev['label']}")
        for r in ev["rationale"]:
            st.write("• " + r)
        st.caption(ev["disclaimer"])
        st.caption(ev["calibration_note"])
        st.markdown("**Taxonomy**: " + " · ".join(f"{k}. {v}" for k, v in TAXONOMY.items()))
        st.dataframe([{"Feature family": f["family"], "Level": f["level"], "Counts toward label": f["counts_toward_label"],
                       "Description": f["description"], "Signals w/m/s": "/".join(str(f["signals"][k]) for k in ("weak", "moderate", "strong"))}
                      for f in ev["family_table"]])
        ei = ev["experimental_index"]
        with st.expander("Experimental indicator count (not a probability)"):
            st.write(f"{ei['indicators_triggered_at_least_weak']} of {ei['indicators_evaluated']} evaluable heuristic indicators reached at least 'weak'.")
            st.caption(ei["how_calculated"])
        st.markdown("**Evidence types in this analysis**")
        st.json(ev["evidence_types"], expanded=False)
        if ev["not_assessed"]:
            st.markdown("**Not assessed**")
            st.dataframe(ev["not_assessed"])
    with t[2]:
        chosen, skipped = selected_signals(res)
        st.caption(f"{len(res.signals)} findings · {len(chosen)} highlighted · {len(skipped)} listed but not highlighted to avoid over-marking. "
                   "Click a finding card to jump to the passage; click a highlight to see its finding.")
        render_html(interactive_panel_html(res.document.text, chosen, 760), 820, True)
        if res.signals:
            with st.expander("All findings (table)"):
                st.dataframe(signal_rows(res.signals))
    with t[3]:
        kinds = ["All"] + [KIND_LABELS[k] for k in ("measured", "heuristic", "model", "watermark", "unavailable")]
        mods = ["All"] + sorted({f.module for f in res.features})
        c1, c2 = st.columns(2)
        k = c1.selectbox("Evidence type", kinds)
        m = c2.selectbox("Module", mods)
        fs = [f for f in res.features if (k == "All" or KIND_LABELS[f.kind] == k) and (m == "All" or f.module == m)]
        st.dataframe(feature_rows(fs))
        c3, c4 = st.columns(2)
        with c3:
            render_html(charts.histogram(res.tables.get("sentence_lengths", []), "Sentence lengths (words)", xlabel="words"), 260)
        with c4:
            render_html(charts.zipf_plot(res.tables.get("zipf")), 260)
    with t[4]:
        st.caption("Internal-consistency map: how each stretch of the document compares with the document's own profile. An unusual region is not "
                   "necessarily AI-written, and a uniform document is not necessarily human-written.")
        render_html(charts.anomaly_map(res.tables.get("window_z", []), len(res.document.text), res.settings["thresholds"]["anomaly_z"][0]), 120)
        if res.tables.get("window_baselines"):
            st.dataframe([{"Feature": k, **v} for k, v in res.tables["window_baselines"].items()])
    with t[5]:
        pm = [f for f in res.features if f.module == "perplexity"]
        if pm and any(f.kind == "model" for f in pm):
            st.dataframe(feature_rows(pm))
        else:
            st.info(pm[0].interpretation if pm else "Reference-model analysis was not enabled.")
        rc = [f for f in res.features if f.module == "reference_comparison"]
        if rc:
            st.subheader("Reference-writing comparison")
            st.dataframe(feature_rows(rc))
    with t[6]:
        for r in res.tables.get("watermark", []):
            (st.success if r["ran"] else st.info)(f"**{r['scheme']}** - {r['message']}")
            if r["ran"]:
                st.json({k: v for k, v in r.items() if k not in ("message",)}, expanded=False)
    with t[7]:
        st.subheader("Recommendations for human review")
        for r in res.recommendations:
            st.write(f"• {r['text']}")
        st.subheader("Limitations")
        from reporting.common import LIMITATIONS
        for x in LIMITATIONS:
            st.write("• " + x)
    with t[8]:
        st.caption("Reports are generated locally in memory and offered as downloads; nothing is written to disk unless you save the download.")
        incl = st.checkbox("Include the full text and passages in the JSON report", True)
        if st.button("Prepare downloads"):
            d = {"html": build_html(res).encode("utf-8"), "json": to_json_string(res, incl).encode("utf-8"), "txt": build_text(res).encode("utf-8")}
            if pdf_available():
                try:
                    d["pdf"] = build_pdf(res)
                except PdfUnavailable:
                    pass
            st.session_state["downloads"] = d
        d = st.session_state.get("downloads")
        if d:
            stamp = res.reproducibility["text_sha256"][:8]
            cols = st.columns(4)
            cols[0].download_button("HTML report", d["html"], f"textforensics_{stamp}.html", "text/html")
            cols[1].download_button("JSON report", d["json"], f"textforensics_{stamp}.json", "application/json")
            cols[2].download_button("TXT report", d["txt"], f"textforensics_{stamp}.txt", "text/plain")
            if "pdf" in d:
                cols[3].download_button("PDF report", d["pdf"], f"textforensics_{stamp}.pdf", "application/pdf")
            else:
                cols[3].caption("PDF export needs `pip install reportlab`.")


def main() -> None:
    st.title(f"🔍 {APP_NAME}")
    st.caption(f"Version {APP_VERSION} · fully local · no API keys · no telemetry · your text never leaves this computer")
    with st.expander("What this tool does - and does not do", expanded=False):
        st.markdown(
            "TextForensics Local measures statistical and linguistic properties of a text (sentence/paragraph structure, lexical diversity, "
            "entropy, repetition, punctuation, function words, discourse markers, internal consistency, and optionally reference-model "
            "surprisal), highlights passages with unusual patterns and explains each one.\n\n"
            "**It cannot determine whether a text was written by AI.** It does not use commercial detectors, does not produce a calibrated "
            "probability, does not detect hidden watermarks unless a real detector and its configuration are supplied, and is not equivalent "
            "to Turnitin or similar products. False positives and false negatives are expected. Findings are evidence to examine, "
            "never proof of misconduct.")
    s = sidebar_settings()
    ing = get_input()
    model_settings(s)
    ref = reference_inputs(s)
    watermark_inputs(s)
    if ing is not None:
        validation_panel(ing, s)
    if st.button("Start analysis", type="primary", disabled=ing is None):
        run(ing, s, ref)
    if st.session_state.get("result") is not None:
        show_results(st.session_state["result"])


main()
