# Implemented calculations, experimental and unavailable features

## Notation
N = analysable words (included sentences only), V = vocabulary size, "document baseline" = statistics of the submitted document itself.

## Statistical features
| Feature | Definition |
|---|---|
| Sentence length | words per sentence; mean, median, sample SD/variance (ddof=1), IQR, CV = SD/mean, skewness = adjusted Fisher-Pearson G1, excess kurtosis = unbiased G2 (n >= 8), lag-1 autocorrelation (n >= 10) |
| Paragraph structure | word and sentence counts per paragraph, CV, first-two-word opening repetition, normalised Euclidean similarity of (sentences, mean sentence length, words) vectors, runs of >= 4 equally shaped paragraphs |
| TTR / MATTR | V/N; mean of V_w/w over all sliding windows of size w (Covington & McFall 2010) - undefined if N < w |
| MTLD | forward and backward passes; a factor completes when running TTR <= 0.72; partial factor (1-TTR)/(1-0.72); MTLD = N / factors; mean of both directions; needs >= 50 tokens |
| Hapax / Yule's K / Herdan's C | hapax/V and hapax/N; K = 10^4 (sum m^2 V_m - N)/N^2; C = ln V / ln N |
| Entropy | plug-in H = -sum p log2 p over the empirical distribution (no smoothing); Miller-Madow H + (K-1)/(2N ln 2); normalised H/log2 K; conditional H(X_n given previous k) = H(joint) - H(context); local entropy in fixed 100-token windows (50% overlap); compression ratio = zlib size / raw size |
| N-grams | share of n-gram positions in types seen >= 2 times; baseline = same statistic after random word permutations (seeded, 64-bit hashed n-grams); top-10 share, Herfindahl index; repeated first-3-word openings and last-2-word endings; local repeat = consecutive occurrences <= 200 tokens apart |
| Punctuation | counts per 1000 words and per sentence; entropy over punctuation classes; rhetorical templates = editable regexes in `config/lexicons.py` |
| Zipf | OLS of ln(freq) on ln(rank); slope, intercept, R², RMSE; fit on all ranks and on ranks with frequency >= 2 |
| Function words / POS | closed-class lexicons; POS from spaCy -> NLTK -> rule-based fallback (labelled approximate); POS n-gram entropy; JS distance of paragraph POS mix to document POS mix |
| Discourse | regex match of configurable marker categories (additive, contrastive, causal, conclusive, sequential, exemplification, hedging, qualification); sentence-initial rate; claim-evidence-explanation = ordered cue words in paragraphs of >= 3 sentences |
| Internal anomalies | windows >= 8 sentences and >= 80 words; robust z = (x - median) / max(1.4826 MAD or IQR/1.349, feature floor); flag at |z| >= 3 / 4 / 5 (weak / moderate / strong) in the feature's meaningful direction; consecutive windows merged |
| Reference comparison | Jaccard, cosine, Jensen-Shannon distance, two-sample KS statistic with asymptotic p-value; reported separately |
| Reference model | NLL_i = -log p(token_i given previous tokens); mean NLL, cross-entropy (bits) = NLL/ln 2, perplexity = exp(mean NLL), quantiles, variance, rank of the true token, predictive entropy, windowed local means, five-segment perplexities |
| Watermark z-test | z = (green - gamma T)/sqrt(T gamma (1-gamma)); one-sided normal p-value; optional unique-pair counting; Bonferroni adjustment over the number of watermark tests run |

## Feature families and evidence rules
Families: sentence variability, paragraph structure, repetition, discourse/rhetoric, lexical anomaly, internal predictability, model predictability (count toward the label);
internal inconsistency (descriptive, does not count). Rules: see `scoring/aggregation.py` docstring.

## Experimental features
* Heuristic strength thresholds (all uncalibrated; `config/settings.py`).
* Claim-evidence-explanation, tense, passive and rhetorical-template detectors (regex approximations).
* Rule-based POS tagger fallback.
* "Experimental indicator count" (unweighted count; not a probability).

## Unavailable features (by design)
* Calibrated probability of AI authorship (needs your own documented, held-out evaluation; see `evaluation/`).
* SynthID-Text and vendor watermark detection (algorithms/keys not available) - reported as unavailable.
* OCR for scanned PDFs.
* Bundled language models (you provide one).
* Non-English sentence splitting and POS (English-centric rules).
