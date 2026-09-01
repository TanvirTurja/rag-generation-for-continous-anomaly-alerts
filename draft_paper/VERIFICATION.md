# Claim-by-Claim Verification — `paper.md` (v2)

Every quantitative claim in the revised manuscript, traced to a v2 artifact.
Generated 2026-08-28 by `scripts/v2/verify_claims_v2.py` (recomputes from
`outputs_v2/` and string-matches `draft_paper/paper.md`). Run it yourself:
`see draft_paper/verify_claims_v2.py` → `outputs_v2/verification_v2.json`.

v2 changes vs v1: pre-registered protocols (`THRESHOLDS.md`, timestamped before
runs); zero dev-history-only claims; every load-bearing number recomputable.

## A. Detection (Table 2/3) — `detection_results_v2.csv`, `detection_summary_v2.json`

| Protocol | Result |
|---|---|
| WESAD LOSO | IF 0.799 [0.770, 0.827]; LOF 0.827 [0.800, 0.852]; OC-SVM 0.814; AE 0.855 [0.831, 0.878] |
| MIT-BIH inter-patient (DS1→DS2, paced excl.) | IF 0.670±0.023; **LOF 0.502 [0.492, 0.512]**; OC-SVM 0.675; AE 0.638 |
| PTB-XL (fold-9 threshold → fold 10) | IF 0.632; LOF 0.682 [0.661, 0.703]; OC-SVM 0.628; AE 0.628 |
| Paired bootstrap IF vs LOF | WESAD p=.008; MIT-BIH p=.001; PTB-XL p=.001 |
| Feature ablation (PTB-XL LOF) | −dynamics 0.646; −spread 0.633; −shape 0.666; −location 0.677 |
| v1 contamination record | v1 numbers + protocol labels archived in `outputs_v1_archive/` + Table 3 |

## B. Query construction & retrieval — `alerts_queries_v2.csv`, `wesad_zscore_sanity.json`

| Claim | Value |
|---|---|
| Flag reproduction | 216/216/34/398 exact vs archived parquet (assert in `query_builder_v2.py`) |
| WESAD z-metric sanity | stress 43.7 vs baseline 1.5 mean max-\|z\|; Mann–Whitney p ≈ 2.5e-138 |
| Retrieval latency | 25 ms mean, 32 ms p95 (`rag_analysis_v1/retrieval_latency.json`) |

## C. Retrieval diversity / duplication / corpus

| Claim | v1 (archived) | v2 | Source |
|---|---|---|---|
| Naive-template docs | 5 (reconstructed artifact; v1's "~11" was dev-history) | — | `rag_analysis_v1/ablation_2x2.csv` |
| Deviation+diversity docs | 53 | 44 | `ablation_2x2.csv`, `near_duplicate_v2.json` |
| Near-dup clusters @0.9 | 173 | 237 | `near_duplicate_summary.json`, `near_duplicate_v2.json` |
| Mean NN cosine | 0.9356 | 0.9149 | same |
| % with >0.9 twin | 81.66 | 67.59 | same |
| Guideline alert reach | 6.5% | 17.6% | same |
| Corpus | 204 docs / 4,732 chunks | 206 docs / 5,045 chunks (992 guideline) | `corpus_stats_v2.json`, Tier1_v2 manifest |

## D. Citation audit — `citation_audit_v2.json` (raw text preserved; capture rule fixed)

| Metric | Value |
|---|---|
| Raw citations | 1,208; 1,196 valid = **99.01%**; 12 invalid across 9 rows |
| Repaired | 100% (12 dropped, 0 snaps; not digit transpositions) |
| Guideline name-citations | 109 valid, 8 unmatched |
| v1 pre-repair "99.3%" | NOT reconstructable from v1 artifacts (`rag_analysis_v1/prerepair_note.json`); superseded |

## E. Judge validation — `judge_validation.json` + per-judge CSVs + `api_judge_checkpoint_v2.jsonl`

| Judge | Detection | FP | Notes |
|---|---|---|---|
| llama3.1:8b (v1's judge) | **0.00** | 0.00 | null instrument; explains v1's "zero hallucination" |
| gemma4:e4b (think=False) | **0.48** | 0.01 | exaggeration .96, fab-fact .80, fab-cit .16, citation-swap .00 |
| DeepSeek-V4-Flash | **0.44** (rerun 0.42) | **0.07** (rerun 0.31) | cross-run instability; both runs in the checkpoint (977 calls, $0.36 total) |

Invalidated artifacts kept for transparency: `judge_validation_gemma4_e4b_INVALID_empty_thinking.csv`
(default thinking mode consumed the token budget → empty outputs). Local-judge
scores are not bit-reproducible across runs (dalia faithfulness means 2.21–2.69
over three identical-input runs; paper uses the final complete run). Judge cells
in `rag_v2.ipynb` are **cache-only** — notebook re-execution makes zero API calls.

## F. Main judging (both validated judges) — `rag_evaluation_v2.csv` (646 rows, 100% API coverage)

| Group | n | local f/r/c | API f/r/c |
|---|---|---|---|
| PPG-DaLiA | 398 | 2.21 / 2.66 / 2.19 | 2.10 / 2.75 / 2.50 |
| WESAD stress | 50 | 2.78 / 3.00 / 2.76 | 2.02 / 2.92 / 2.46 |
| MIT-BIH ectopy | 50 | 2.36 / 3.00 / 2.56 | 2.02 / 2.72 / 2.28 |
| PTB-XL pathology | 48 | 2.45 / 3.00 / 2.13 | 2.00 / 2.41 / 2.16 |
| Word-cap 300 | 50 | 2.02 / 2.52 / 2.08 | 2.08 / 2.82 / 2.70 |
| llama3.1 generator | 50 | 1.84 / 2.34 / 1.46 | 1.94 / 2.24 / 1.80 |

Local parse failures: 61/646 (9.4%), excluded from statistics and reported.
Dalia faithfulness: local {3:176, 2:175, 1:2, 0:45}, API {3:40, 2:313}.
Agreement (353 valid pairs): raw .592/.751/.602, within-1 1.000 all axes,
Gwet AC1 .476/.680/.489 (`agreement_v2.json`; recomputed live in the notebook).

## G. Concordance — `concordance_v2.json` (labels never in queries)

WESAD 47/50 (94%); MIT-BIH 6/50 (12%; VEB 1/25, SVEB 5/25); PTB-XL 3/48 (6%;
MI 0/12, STTC 1/12, CD 2/12, HYP 0/12). Artifact language in 28/50, 28/50, 20/48.
Selection artifact note: first-pass MIT-BIH selection ranked by detector flags
(inter-patient detector ≈ chance) → 49/50 windows had no annotated ectopy;
superseded keys recorded in `superseded_keys.json`, replaced by annotation-driven
selection (25 VEB + 25 SVEB, ≥3 abnormal beats).

## H. Atomic verification — `factscore_lite.json`

797 claims / 60 explanations: 52.3% SUPPORTED, 47.7% UNVERIFIABLE, 0% UNSUPPORTED
(verifier gemma4:e4b, think=False; invalid first pass preserved with `_INVALID` suffix).

## I. System

Generation 11.0 s mean/alert (RTX 5060 laptop); retrieval 25 ms (p95 32);
local judge 8.1 s/call (measured, replaces v1's unarchived 5.6 s); API spend $0.00
(key expired; cap $10 in-script).

## Open items before submission

1. **API key renewal** → API-judge validation row + main-judging columns +
   agreement stats (§4.5/§4.6 slots marked PENDING-API-KEY); est. < $1, ~1 hr.
2. **Clinician ratings** via `clinician_eval/` kit (sample selection requires
   `select_sample.py` run) → §Human Evaluation.
3. References marked `verify-before-submit` in `references_v2_additions.bib`
   (4 entries with indirect DOIs).
4. Optional: re-sign commits (`--no-gpg-sign` used after pinentry timeout).
