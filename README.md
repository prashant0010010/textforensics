# TextForensics Local

A **fully local** Streamlit application that measures statistical and linguistic properties of a text, highlights passages whose patterns
are *associated* with machine-generated writing, and explains every finding with its measured values, comparison baseline, limitations
and alternative explanations.

* $0 to run - no paid APIs, no API keys, no cloud inference, no telemetry, no commercial detectors.
* Works with **no language model installed**. A local reference model is an optional, separately configured extra.
* Never claims more than it measured: there is **no AI probability**, no fake watermark detection, no simulated model output.

> **Important.** This tool cannot determine whether a text was written by AI. False positives and false negatives are expected
> (especially for non-native English writers, formulaic genres and edited text). It is not equivalent to Turnitin, GPTZero, ZeroGPT or any
> other product. Its output is evidence to examine, never proof of misconduct.

## Contents
1. [Capabilities](#capabilities) · 2. [Limitations](#limitations) · 3. [Architecture](#architecture) · 4. [Installation (Windows)](#installation-windows) ·
5. [macOS / Linux](#macos--linux) · 6. [Using the app](#using-the-app) · 7. [Optional reference model](#optional-local-reference-model) ·
8. [Watermark detectors](#watermark-detectors) · 9. [Reading the results](#reading-the-results) · 10. [Performance](#performance) · 11. [Privacy](#privacy) ·
12. [Testing](#testing) · 13. [Scientific validation](#scientific-validation) · 14. [Troubleshooting](#troubleshooting) · 15. [Standalone executable](#standalone-executable) ·
16. [Verification status](#verification-status)

## Capabilities
| Area | What is computed (all locally) |
|---|---|
| Ingestion | pasted text, TXT (UTF-8/UTF-16/CP1252/Latin-1), DOCX (headings, lists, tables), PDF with a text layer; Unicode NFC, LF line endings; ~30,000 words (soft limit; hard stop at 200,000 with an explicit error - **never silent truncation**); scanned PDFs are refused with an explanation, not "analysed" |
| Segmentation | paragraphs, sentences (abbreviation-aware), sections (headings), words; block types: prose, heading, quote, reference, bullet, code, table; quotes/references/code/tables/headings are shown but excluded from statistics unless you include them |
| Sentence statistics | mean, median, SD, variance, IQR, CV, skewness, excess kurtosis, short/medium/long shares, consecutive differences, lag-1 autocorrelation |
| Paragraph structure | length distribution/variance, sentences per paragraph, opening repetition, transitions, structural similarity, paragraph lexical variation, runs of identically shaped paragraphs |
| Lexical diversity | TTR, MATTR (configurable windows), MTLD, hapax ratios, Herdan's C, Yule's K, repeated content words, neighbouring-paragraph overlap |
| Entropy | character / word / n-gram / conditional entropy (plug-in and Miller-Madow), local entropy in fixed windows, per-paragraph variation, zlib compression ratio |
| N-grams | word 2-5-grams vs a seeded shuffle baseline, concentration, repeated openings/endings, local vs global repetition, recurring phrases, character n-grams |
| Punctuation / formatting | rates per 1000 words, punctuation entropy, sentence-final patterns, list-marker consistency, configurable rhetorical templates |
| Zipf | log-log OLS slope, R², RMSE, hapax-trimmed fit, concentration, rare-word share |
| Function words / POS / discourse | function-word categories, POS via spaCy/NLTK if installed else a rule-based fallback (labelled *approximate*), POS n-gram entropy, configurable discourse-marker categories, hedging, claim-evidence-explanation cue pattern |
| Internal anomalies | sliding sentence windows compared with the document's **own** robust baseline (median/MAD); adjacent flagged windows merged |
| Reference model (optional) | token log-probabilities, NLL, cross-entropy, perplexity, surprisal quantiles/variance, rank stats, local variation, per-segment perplexity |
| Reference writing (optional) | vocabulary, function words, sentence lengths (KS), punctuation, POS, n-grams, paragraph structure, reported metric by metric |
| Watermarks | pluggable interface + a real green-list z-test that runs only when you supply the exact scheme rule and tokenizer; everything else reports *"Watermark testing unavailable for this scheme."* |
| Reports | HTML (interactive highlights), JSON, TXT, PDF - generated locally in memory |

## Limitations
* No labelled-data validation ships with the software; **no accuracy, false-positive or false-negative rate is claimed.**
* Heuristic strength labels (weak/moderate/strong) use conventional thresholds in `config/settings.py`. They are **uncalibrated** and editable.
* Uniform sentences, repetition, transition use or low entropy also occur in legal, technical, instructional, non-native and heavily edited human writing.
* Edited, paraphrased or carefully prompted machine text may trigger nothing. "Little evidence" is **not** evidence of human authorship.
* Documents under ~300 words / 15 sentences get the label *Insufficient data to assess*.
* The rule-based POS tagger, sentence splitter (English-centric; `。` is not a boundary) and cue-word detectors are approximations.
* Reference-model results depend on the chosen model. Watermark results apply only to the exact scheme tested.

## Architecture
```
app.py                    Streamlit interface (input, settings, results dashboard, exports)
config/                   settings.py (all thresholds/switches), lexicons.py (editable word/phrase categories)
core/                     ingestion, normalization (+ invisible-character inspection), segmentation, schemas,
                          features (shared maths), multiscale (segment records), pipeline (orchestration, error isolation)
analyzers/                one module per analysis family (sentence, paragraph, lexical, entropy, ngrams, punctuation, zipf,
                          function_words, syntactic_features, discourse, perplexity, internal_anomalies, reference_comparison)
watermark/                base interface, registry, green_list (real z-test, config-gated)
scoring/                  evidence (strength grading), aggregation (taxonomy rules), calibration (gate + metrics), recommendations
visualization/            highlights (offset-exact HTML), charts (SVG), tables
reporting/                html_report, json_report, text_report, pdf_report, common
evaluation/               framework.py - splits, metrics, bootstrap CIs, calibration for YOUR labelled data (no data included)
scripts/benchmark.py      time + peak-memory benchmark on synthetic text
tests/                    139 tests (unittest-style; pytest-compatible)
examples/                 synthetic inputs and watermark-config template
```
Every detector returns structured results (`FeatureResult` / `Signal`): feature name, measured value, baseline, interpretation, location,
strength, limitations and alternative explanations. Evidence is typed: **directly measured**, **heuristic interpretation**,
**model-based**, **watermark test**, or **unavailable**. Details of each calculation: [docs/CALCULATIONS.md](docs/CALCULATIONS.md).

## Installation (Windows)
Requires **Python 3.10, 3.11 or 3.12** (3.13 should work but is untested) from <https://www.python.org/downloads/> (tick *Add python.exe to PATH*).

**Quick start:** double-click `run_windows.bat`. On first run it creates `.venv`, installs the core requirements (the only step needing
internet) and opens <http://localhost:8501>.

**Manual (PowerShell):**
```powershell
cd path\to\textforensics_local
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt       # core only: streamlit, numpy, python-docx, pypdf, reportlab
streamlit run app.py
```
Network activity: `pip install` (package downloads) and, only if *you* choose, downloading a model yourself. The application itself makes no network
requests; a test fails the build if a network library is imported.

## macOS / Linux
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Using the app
* **Paste text:** *Paste text* tab -> check the counts/validation panel -> **Start analysis**.
* **Upload documents:** *Upload document* tab (TXT, DOCX, PDF with extractable text). Scanned PDFs are rejected with an explanation (no OCR).
* **Settings (sidebar):** enable/disable statistical, lexical, structural, linguistic, internal-consistency and watermark modules; choose whether quotes,
  references, bullets, code, tables and headings are analysed; window sizes; lightweight mode.
* **Results tabs:** Overview · Evidence · Highlights (click a finding to jump to the passage, click a highlight to see its finding) · Features ·
  Anomaly map · Model · Watermark · Limitations & recommendations · Export.
* **Export:** *Export* tab -> **Prepare downloads** -> HTML / JSON / TXT / PDF. Nothing is written to disk unless you save the download.
  The JSON report can omit the full text.
* **Reference writing sample:** expander on the main page; upload a sample and tick the confirmation box. Results are shown metric by metric, with no combined verdict.

## Optional local reference model
1. `pip install -r requirements-optional.txt` (large; torch + transformers).
2. Download a causal language model yourself (for example with `huggingface-cli download gpt2 --local-dir C:\models\gpt2`) - *this is your explicit, separate step*.
3. In the app open **Optional: local reference language model**, tick *Enable*, enter the folder path, choose device/context length/stride.

The loader always uses `local_files_only=True` and never downloads anything. If torch/transformers or the model are missing the module is reported as
**unavailable** and everything else still runs. Reported: model name, tokenizer, token count, device, method, mean NLL, cross-entropy, perplexity,
surprisal variance/quantiles, rank statistics, local variation and per-segment perplexity. A fast (Rust) tokenizer is required so tokens map back to character offsets.
Perplexity is specific to the chosen model and is **not** a universal measure of AI authorship; document-level model values receive no strength grade because no calibrated threshold exists.

## Watermark detectors
Detectors are registered in `watermark/registry.py` and must subclass `WatermarkDetector` (`watermark/base.py`). Register one **only** if its real,
documented algorithm and required configuration are implemented. Never infer a watermark from punctuation, token frequencies or hidden characters.

The bundled `GreenListZTest` (Kirchenbauer et al., 2023) needs: the generator's tokenizer (local folder), `gamma`, and a Python file defining
`is_green(prev_token_id, token_id) -> bool` that reproduces the generator's exact rule. See `examples/watermark_config.example.json` and `examples/example_scheme.py`
(a toy rule, not a real scheme), then upload the JSON in the app. Reported: scheme, implementation/version, configuration status, whether it ran,
eligible tokens, z statistic, null hypothesis, p-value, Bonferroni-adjusted p-value and limitations. SynthID-Text and vendor schemes are listed as unsupported:
*"Watermark testing unavailable for this scheme."* Invisible-character inspection is shown separately and is described as text inspection only.

To add a detector: subclass `WatermarkDetector`, implement `check_config` and `detect`, call `register(MyDetector())`, and add tests.
Warning: `scheme_module` is executed as Python on your computer - load only files you trust.

## Reading the results
* **Evidence category** (1-5) - rules in `scoring/aggregation.py`: *Little evidence* · *Mixed or inconclusive* · *Several features associated with AI-generated text* ·
  *Strong convergence of multiple informative features* · *Insufficient data to assess*. These describe the **evidence**, not authorship or misconduct.
  Feature families are correlated, so "convergence" means different kinds of measurement agree - nothing more.
* **Experimental indicator count** - an unweighted count of triggered heuristic indicators. It is explicitly **not** a probability.
* **Highlights** are categories (predictability, repetition, structure, lexical, stylometric, model, watermark) with strength shown by underline. At most 35% of the
  text and 60 findings are highlighted; others stay listed in tables and reports.
* **Internal anomaly analysis** compares each window with the document's own median/MAD (with a minimum-spread floor). An unusual passage is not necessarily AI-written; a uniform document is not necessarily human-written.

## Performance
Measured with `python scripts/benchmark.py` on synthetic text in the build sandbox (single run; **your results will differ and no time is promised**):
| words | seconds | peak Python heap |
|---|---|---|
| 1,000 | 0.2 | 1.6 MB |
| 5,000 | 0.3 | 7.1 MB |
| 15,000 | 0.95 | 19.9 MB |
| 30,000 | 2.2 | 40.0 MB |

Real text has larger vocabularies and the reference model, if enabled, dominates run time (CPU inference over 30,000 words can take many minutes). Use **lightweight mode**
for limited RAM; expensive optional analyses (reference model, spaCy/NLTK tagging) are marked in the interface. Deterministic results are cached for the preview;
each analysis shows staged progress.

## Privacy
* All text stays in memory on your machine. No analytics, no telemetry (`gatherUsageStats = false`), no remote logging, no uploads.
* Logs never contain submitted text (failure logs record only module name and exception type).
* Nothing is stored unless you download a report. No temporary files are created by the analysis (tests assert this).
* Network use: only `pip install` and any model download you perform yourself.

## Testing
```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t .      # or: pytest
python scripts/benchmark.py --sizes 1000 30000
```
Tests cover empty/short/single/multi-paragraph input, punctuation-heavy, technical, academic, quotations, reference lists, Unicode, DOCX and PDF extraction (incl. scanned PDF),
30,000-word input, segmentation boundaries, character offsets, highlight correctness, missing optional dependencies and models, invalid watermark configuration,
numerical stability (zero division, NaN/inf), report generation, and privacy/offline operation. Synthetic texts test the mathematics only; they make no claim about detector accuracy.

## Scientific validation
See [evaluation/README.md](evaluation/README.md). `evaluation/framework.py` provides author-/generator-/domain-separated splits, leave-one-group-out generalisation,
FPR/FNR/precision/recall/F1/ROC-AUC with bootstrap confidence intervals, calibration (Brier, ECE, reliability) and breakdowns by length, domain, generator and edited text.
**No dataset or benchmark result is included, and none may be invented.** A result on one dataset does not establish accuracy on other documents, domains, authors or AI systems.
Probabilities are only ever shown if a documented `config/calibration.json` exists (none ships).

## Troubleshooting
| Problem | Fix |
|---|---|
| `python` not found / wrong version | install Python 3.10-3.12 and tick *Add to PATH*; use `py -3.12` |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process RemoteSigned` |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| "scanned images" error for a PDF | the PDF has no text layer; OCR it first or paste text |
| Garbled characters in a TXT | save as UTF-8; the app warns when it falls back to Windows-1252/Latin-1 |
| Model module "unavailable" | install `requirements-optional.txt`; give a local folder with a fast tokenizer; the app never downloads |
| PDF export button missing | `pip install reportlab` |
| Out of memory / slow | enable *Lightweight mode*; disable the reference model; split the document |
| Highlight panel blank | your Streamlit version may have changed the HTML component API; update the app (`render_html` in `app.py`) or use the HTML report |

## Standalone executable
Practical with PyInstaller, but **not tested in the build environment**:
```powershell
pip install pyinstaller
pyinstaller --noconfirm --onedir --name TextForensicsLocal --collect-all streamlit --collect-all docx --collect-all reportlab ^
  --add-data "app.py;." --add-data "config;config" --add-data ".streamlit;.streamlit" scripts\exe_launcher.py
```
Package source folders (`core`, `analyzers`, ...) may need `--hidden-import` / `--add-data` entries; the optional model stack makes the bundle very large.

## Verification status
Honest account of what was and was not run while this project was built (Linux sandbox, Python 3.12, no internet):
* **Executed and passing:** 139 automated tests; the benchmark; HTML/JSON/TXT/PDF generation; DOCX and PDF extraction using files created in the tests.
* **Streamlit UI:** Streamlit could not be installed offline, so `app.py` was byte-compiled and executed end-to-end only against a *fake* Streamlit stand-in
  (`tests/test_app_smoke.py`). Please run `streamlit run app.py` once and report any rendering differences.
* **Not executed here:** the Hugging Face model scorer (torch/transformers not installed; its arithmetic is tested through an injected synthetic scorer),
  the spaCy/NLTK tagger adapters, tokenizer loading for the watermark detector (tested with an injected tokenizer), `run_windows.bat`, and PyInstaller.
