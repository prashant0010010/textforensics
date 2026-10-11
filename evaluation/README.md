# Evaluation framework

This folder contains the *method* for evaluating any scorer (the rule-based indicators, a reference-model feature, or a classifier
you train yourself). **It contains no data and no results. Do not invent either.**

## What you need
Legitimately obtained, labelled human-written and AI-generated texts, with metadata columns:
`id,label,score,author,generator,domain,n_words,edited`  (`label`: 0 human / 1 AI-or-mixed positive class).

## Design rules (implemented in `framework.py`)
* **Train/test separation** - `group_split` never lets the same author / generator / domain appear on both sides.
* **Author-level separation** for human texts, **generator-level separation** for AI texts (`group_split(..., "generator")`).
* **Cross-model and cross-domain generalisation** - `leave_one_group_out`.
* **Metrics** - FPR, FNR, precision, recall, F1, ROC-AUC (Mann-Whitney), with **stratified bootstrap 95% confidence intervals**.
* **Calibration** (only for probabilistic scores) - Brier score, ECE, reliability table (`scoring/calibration.py`).
* **Strata** - by document length, writing domain, generator, and edited/mixed-origin text.
* A false-positive rate measured on one dataset says nothing certain about other populations (non-native writers, genres, edits).

## Gate for displaying probabilities
`config/calibration.json` must contain `dataset_description, n_train, n_test, heldout_metrics, generators, domains`. Without it the
application never displays a probability (see `scoring/calibration.py`).
