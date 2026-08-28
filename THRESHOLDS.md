# Pre-Registered Protocols and Operating Points — v2 Detection Evaluation

**Written: 2026-08-28 14:30 (local), BEFORE any v2 detection run.**
This file fixes all evaluation design choices for `scripts/v2/detection_v2.py` in advance.
Any deviation discovered during execution must be logged in the Deviations section at the
bottom with a reason. Results reported in the manuscript must follow this file.

## 1. WESAD (stress vs baseline+amusement)

- **Protocol: leave-one-subject-out (LOSO) over all 15 subjects** (S2–S17 minus S1/S12 as in dataset).
- Per fold: train on the other 14 subjects' **baseline** windows only; score the held-out
  subject's baseline+stress+amusement windows. Positive = stress (label 2); negative =
  baseline (1) + amusement (3).
- Scaler, IF, LOF, OC-SVM, AE refit per fold on that fold's training partition only.
- **Threshold (pre-registered):** each model's training-derived decision threshold at the
  85th percentile of its own training scores (contamination 0.15), applied unchanged to the
  held-out subject. No held-out data influences any threshold.
- Report: per-subject AUC (mean ± SD), pooled AUC (all held-out predictions concatenated,
  threshold-free), pooled P/R/F1 (per-fold binary flags combined).

## 2. MIT-BIH Arrhythmia (inter-patient)

- **Excluded entirely (AAMI paced records): 102, 104, 107, 217.**
- **DS1 (22, train):** 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
  201, 203, 205, 207, 208, 209, 215, 220, 223, 230.
- **DS2 (22, test):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213,
  214, 219, 221, 222, 228, 231, 232, 233, 234.
  (Runtime assertion: 22+22 unique records, none of the excluded four, no overlap.)
- Beat labels: AAMI class N = {N, L, R, e, j} → normal; **all other beat symbols →
  anomalous** (SVEB, VEB, F, and any residual Q). Non-beat annotation symbols excluded.
- Features: 12 summary statistics per 0.8 s beat segment (lead MLII), as v1.
- Train: DS1 normal beats only. **Test: DS2 beats only.** No DS1 beat is scored.
- **Threshold (pre-registered):** 85th percentile of training (DS1-normal) scores,
  contamination 0.15, applied frozen to DS2.
- The v1 intra-patient number is retained ONLY as an appendix table explicitly labeled
  "protocol-contaminated (train ⊂ eval, intra-patient)".

## 3. PTB-XL (pathology, fold 10)

- Labels: normal iff diagnostic superclass set == {NORM}; records with no superclass
  dropped (21,799 → 21,388). Train: folds 1–8 normals. **Validation: fold 9 (all).**
  Test: fold 10 (all).
- **Threshold (pre-registered):** percentile p* selected on fold 9 by maximizing F1 over
  the integer grid p = 1..99 (scores from the model trained on folds 1–8 normals); the
  SAME p* is applied frozen to fold 10. No fold-10 label information touches any
  operating point.
- Also report the full threshold-sensitivity curve (P/R/F1 vs percentile) on fold 10 as a
  figure. AUC is threshold-free.

## 4. PPG-DaLiA alert stream (unchanged from v1, for comparability)

- 95th-percentile flag per detector (5% each), union = alert stream. The v1 flagged set
  (398 windows) is reused verbatim; v2 changes queries/explanations, not detection.

## 5. Models and hyperparameters (fixed, no tuning on any eval data)

- Isolation Forest: 100 trees, contamination as above. **10 seeds (0–9)**; report
  mean ± SD; no seed selection.
- KNN-LOF: k = 20, novelty mode, contamination as above (deterministic).
- One-Class SVM (new baseline): RBF, gamma = 'scale', nu = contamination.
- Autoencoder (new baseline): d → d/2 → d/4 → d/2 → d dense ReLU (linear output),
  MSE loss, Adam lr = 1e-3, batch 256, max 100 epochs, early stop patience 10 on a
  random 10% of training normals; score = per-feature-mean reconstruction error.
- StandardScaler fit on the training partition only, per model, per fold.

## 6. Uncertainty quantification (all datasets, all models)

- Bootstrap 95% CI on AUC: 1,000 resamples, percentile method.
- DeLong test for IF vs LOF AUC difference on each dataset (per-seed mean IF AUC used).

## 7. PTB-XL feature-group ablation (supports or refutes the "feature budget" claim)

Feature groups (drop one at a time, LOF only):
- location: mean, median, p25, p75
- spread: std, min, max, peak-to-peak
- shape: skewness, kurtosis
- dynamics: up-crossing ratio, RMS roughness

## Deviations

1. **DeLong → paired bootstrap** (2026-08-28): the DeLong covariance implementation
   produced inconsistent array shapes across unequal class sizes; the pre-registered
   IF-vs-LOF comparison is instead reported as a paired bootstrap test over the same
   1,000 resamples used for the AUC CIs (two-sided p from the resampled AUC-difference
   distribution). Same hypothesis, same alpha, documented here.
