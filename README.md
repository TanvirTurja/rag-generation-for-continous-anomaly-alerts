# Explaining Wearable Biosignal Anomaly Alerts with RAG

Code and artifacts for the paper *"Explaining Wearable Biosignal Anomaly Alerts
with RAG: A Contamination-Hardened Evaluation"* (Md Tanvir Hasan Turja, 2026).

## What this is

A fully local pipeline in which unsupervised anomaly detectors watching
wearable biosignals (PPG, ECG, EDA, temperature, respiration) automatically
trigger retrieval-augmented explanation: each flagged window becomes a
structured, plain-language alert (DETECTED / EVIDENCE / RECOMMENDATION /
DISCLAIMER) written by a local LLM from retrieved peer-reviewed guidelines and
literature only, with checkable citations and no cloud dependency.

The evaluation is pre-registered (`THRESHOLDS.md`, timestamped before any run)
and designed against three shortcuts that silently invalidate conclusions:
train/test contamination, test-derived thresholds, and unvalidated LLM judges.
A protocol-sensitivity analysis quantifies how much these choices matter.

## Headline findings

| Finding | Number |
|---|---|
| MIT-BIH LOF AUC: intra-patient vs inter-patient protocol | 0.899 vs **0.502 (chance)**; IF holds 0.670 |
| Best WESAD detector (LOSO) | autoencoder **0.855** > LOF 0.827 > IF 0.799 |
| Labeled-event explanation accuracy | stress **94%**, ectopy **12%**, pathology **6%**; 42–56% of true pathology attributed to artifact |
| Judge validation (corruption benchmark) | near-constant judge **0/100** detected; validated local judge 48% @ 1% FP; 284B API judge unstable (FP 7–31% across identical runs) |
| Raw citation accuracy (pre-repair) | **99.01%** (12 fabrications in 9/546 explanations) |
| Atomic claims verifiable from retrieved context | **52.3%** |

Every number is recomputable from the artifacts in this repository:
`python draft_paper/verify_claims_v2.py` re-checks all 59 headline claims
against raw outputs and exits nonzero on any mismatch.

## Repository layout

| Path | Contents |
|---|---|
| `detection_v2.ipynb` | Detection half: loaders, featurization, pre-registered protocols (LOSO / inter-patient / frozen thresholds), IF+LOF+OC-SVM+autoencoder, CIs, tests, figures (executed, 0 errors) |
| `rag_v2.ipynb` | Explanation half: corpus build, deviation-aware queries, grounded generation with raw-text preservation, corruption-validated judges (cache-only API cells), labeled-event concordance, FActScore-style verification (executed, 0 errors) |
| `draft_paper/` | `paper.md` (manuscript), `references.bib`, `figures_v2/`, `VERIFICATION.md` (claim-by-claim trace), `verify_claims_v2.py` |
| `THRESHOLDS.md` | Pre-registered evaluation protocols (written before any v2 run) |
| `SEARCH_PROTOCOL.md` | Documented literature search behind related work |
| `outputs_v2/` | All artifacts: feature caches, 696 explanations (raw + canonicalized), judge benchmarks, frozen API-judge checkpoint, agreement and concordance stats |
| `outputs_v1_archive/` | Original-protocol outputs, preserved for the protocol-sensitivity comparisons |
| `clinician_eval/` | Ready-to-run human-evaluation kit (60 stratified items, rubrics, forms, Fleiss' κ analysis) — future work; no clinical-adequacy claim is made |
| `scripts/fetch_tier2.py` | Tier-2 corpus builder (Europe PMC); needed only for a from-scratch rebuild |
| `archive_v1/` | Original notebooks and draft, superseded |
| `Dataset/` | Raw data (not committed; DOIs in the paper) |

## Reproducing

1. Download the four public datasets (PPG-DaLiA, WESAD, MIT-BIH, PTB-XL; DOIs
   in the paper) into `Dataset/`.
2. Install Python 3.13 with `wfdb scikit-learn torch chromadb
   sentence-transformers ollama pandas pyarrow`, and Ollama with the
   `qwen3.5:9b`, `gemma4:e4b`, and `llama3.1:8b` models.
3. Run `detection_v2.ipynb`, then `rag_v2.ipynb`, top to bottom. With the
   cached artifacts present this takes ~10–15 minutes per notebook; a cold
   rebuild is ~15 minutes (detection) and ~3 h GPU (RAG). The API-judge cells
   are cache-only: re-execution makes zero API calls.
4. `python draft_paper/verify_claims_v2.py` → expect `claims: 59 OK, 0 FAIL`.

## Data, corpus, and licensing

Datasets are public with DOIs listed in the paper. The retrieval corpus
comprises 4 society guideline PDFs, 2 wearable-relevant guidance documents
(fetched with recorded provenance, `Dataset/Tier1_v2/manifest.csv`), and 200
open-access articles (149 CC BY, 51 CC BY-NC/NC-ND) fetched by
`scripts/fetch_tier2.py`. Rebuild the vector store with the notebook; the
binary `chroma_db_v2/` is deliberately not committed.

## Citation

If you use this code, please cite the paper (bibtex in
`draft_paper/references.bib`, entry to be finalized on publication).
