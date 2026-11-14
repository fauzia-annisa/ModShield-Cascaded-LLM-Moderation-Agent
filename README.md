# ModShield: Cascaded LLM Moderation Agent

A content-moderation pipeline for toxic comments: cheap model first, LLM agent only on uncertain cases, with evaluation of cost, quality, bias and drift.

**Data:** Jigsaw Civil Comments (Hugging Face `google/civil_comments`, the data behind the Kaggle Jigsaw toxicity competitions). Label: toxic if crowd toxicity score >= 0.5. About 8% of comments are toxic, so I report PR-AUC and F1, not accuracy. Samples: 60k train (30k used for BERT), 5k validation, 10k test. Thresholds are tuned on validation and applied to test.

## Architecture
Comment -> Stage 1: TF-IDF + Logistic Regression -> Stage 2: DistilBERT (PyTorch) -> if score is uncertain -> Stage 3: Qwen2.5-1.5B-Instruct agent (policy prompt + 4 retrieved similar labeled cases) -> score blend -> allow / human review / remove.

## Results (test set, 10k comments)
| Model | PR-AUC | F1 | ms/comment |
|---|---|---|---|
| TF-IDF + LR | 0.584 | 0.543 | 0.16 |
| DistilBERT | 0.700 | 0.652 | 0.96 |
| Cascade, 0.4-0.6 band, w=0.5 (1.4% to LLM) | 0.699 | 0.652 | 2.22 |
| Cascade, 0.1-0.9 band, w=0.5 (8% to LLM) | 0.698 | 0.646 | 8.09 |
| LLM score only in 0.1-0.9 band | 0.428 | 0.604 | 8.09 |

**Finding:** the LLM stage did not improve on DistilBERT in any of 12 band/weight configurations, and it raised latency up to 8x. Likely reasons: a 1.5B model is weak at this task, its probabilities are uncalibrated, and crowd labels near the 0.5 cutoff are noisy. Recommendation: ship DistilBERT alone; revisit the LLM stage with a larger model and a calibrated blend. See cascade_tradeoff.png.

## Bias audit
False-positive rate on non-toxic comments that mention a group (keyword proxy, full 97k test split; overall BERT FPR is 3.0%):

| Group | n | FPR LR | FPR BERT |
|---|---|---|---|
| muslim | 987 | 0.169 | 0.070 |
| jewish | 338 | 0.118 | 0.077 |
| gay/lgbt | 396 | 0.144 | 0.081 |
| black | 175 | 0.229 | 0.269 |
| white | 164 | 0.323 | 0.232 |
| christian | 769 | 0.078 | 0.040 |
| women | 2198 | 0.084 | 0.050 |
| mental illness | 143 | 0.140 | 0.056 |

BERT lowers FPR for most groups compared with LR. The exception is comments mentioning race ("black people", "white people", "African American"), flagged at 23-27% vs 3.0% overall. I read the 15 highest-scoring false positives for the "black" group: most are non-toxic comments that discuss racism, white supremacy or the KKK, so the model appears to react to sensitive topics rather than to the identity term alone. This is a known failure mode of toxicity classifiers: condemning hate speech looks like hate speech. Mitigations I would try: counterfactual augmentation (swap identity terms), training on non-toxic comments that mention groups, and an LLM stage with a policy that explicitly allows discussion of offensive content. Caveats: keyword-based grouping, small samples (n=164-175), and I only reviewed the top 15 errors. Groups are detected by keyword, so this is a proxy; small groups have wide error bars.

## Failure analysis
- Mild insults near the label cutoff are flagged but labeled non-toxic ("Cherry is a troll.", "I'm a silly dude.").
- Contemptuous or veiled language is missed ("Hope he gets his derriere handed to him.", "I have read enough of Pio Nono's drivel.").
- Harsh political criticism is often a false positive.

## Drift monitoring
PSI of BERT scores vs validation: 0.069 at 20% toxic traffic (no alert), 0.345 at 40%, 0.786 at 60% (alert at 0.2). PSI reacts slowly to moderate shifts, so production should also track flag rate, LLM-escalation rate and human-review feedback.

## Limitations
English only; noisy crowd labels; single 1.5B LLM and a simple prompt; keyword-based bias proxy; validation set of 5k; serving manifests are provided but not deployed to a live cluster.

## Files
app.py (FastAPI), Dockerfile, k8s.yaml, notebook, cascade_tradeoff.png. Model: huggingface.co/YOUR_HF_USERNAME/modshield-distilbert
