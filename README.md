# Auditing LLM Explanations of Biosignal Alerts

Code and artifacts for the paper *"Auditing LLM Explanations of Biosignal Alerts"* (Md Tanvir Hasan Turja, 2026).

## What this is

An end-to-end pipeline — unsupervised anomaly detection on wearable biosignals
(PPG, ECG, EDA, temperature, respiration), alert-triggered retrieval over a
206-document peer-reviewed corpus, and grounded local LLM generation with
checkable citations — audited by instruments the literature omits:
pre-registered contamination-hard protocols, cluster-aware inference at the
true sampling unit, corruption-validated LLM judges, raw-text citation
auditing, condition-blind query controls, flag-status accounting, and a
retrieval-poisoning probe. The audit is the contribution: every headline the
naive evaluation produced changes hands under it.

## Headline findings

| Finding | Number |
|---|---|
| MIT-BIH LOF: intra-patient vs inter-patient protocol | 0.899 vs **0.502 (chance)**; record-cluster CI **[0.32, 0.75]** |
| Inter-patient MIT-BIH, all four detectors at record level | CIs span ~[0.41, 0.86]; beat-level CIs were ~20× too tight |
| WESAD winner stability | AE point estimate replicates (10 seeds: **0.846 ± 0.008**) but vs LOF **p = 0.16** (parity); IF at seed-mean **p = 0.97**; feature-set perturbation reshuffles the ranking (AE 0.855 → 0.689) |
| Inter-patient IF vs LOF, cluster-level tests | WESAD **p = 0.97**, MIT-BIH **p = 0.139**, PTB-XL **p = 0.001** — the interim table's seed-0/window-level p = 0.008/0.001/0.001 dissolved under correction |
| Equivalence semantics (no "tie" claimed) | TOST at δ = 0.05 narrowly fails for both WESAD pairs; minimum detectable AUC difference **0.088** (WESAD), **0.052** (AE vs LOF), **0.285** (MIT-BIH, 22 records) |
| Feature redundancy | 43 within-channel pairs \|r\| > 0.90; location stats on 4 Hz channels r = 1.00 |
| Query-condition leakage | stress concordance 94% → **68–72%** over three blind runs, **58–60%** under the strictest topics; decomposition banded (**80–82%** contexts fixed: echo 12–14 pts, retrieval 8–14 pts); wrong-condition echo only **2/93**; artifact attributions 70–92% |
| Retrieval-off attribution | empty context still names stress at **76%** (original topics) / **46%** (strict) — most stress naming is **parametric prior**; retrieval adds **12–18 points**; irrelevant context collapses naming to **6–10%**; pathology floor holds RAG-off (0–6%) |
| Pathology naming (labeled events) | ectopy 12%→8%, pathology 6.2%→12.5% — floor in both conditions; 12 summary statistics carry no morphology |
| Flag status of the 148 labeled events | **69/148 (47%)** under pre-registered operating points, **68/148** under the deployment-style union rule; MIT-BIH count corrected 19→18 (beat-row mapping fix, disclosed in the paper) |
| Judge validation | null judge **0/100** corruptions; validated local judge 48% @ 1% FP; 284B API judge unstable across identical runs (FP 0.07→0.31) |
| Raw citation accuracy (pre-repair) | **99.01%** (12 fabrications in 9/398) |
| Retrieval poisoning | one planted chunk in context: **96–98% adoption** across three runs, cited 49–50/50, membership audit passes, judge certifies **31–33 of 38–39** parsed as "fully grounded"; naive store seeding: **0/50** audited, **4.1%** of an alternative alert stream (stream Jaccard 0.47; optimized published attacks expected to succeed) |
| Guideline reach | 6.5% → 12.6% (corpus expansion) → 17.6% (query fix); query variants overlap at Jaccard **0.12** |

Every number is recomputable from the artifacts in this repository:
`python draft_paper/verify_claims_v2.py` re-checks all headline claims
against raw outputs and exits nonzero on any mismatch.

## Repository layout

| Path | Contents |
|---|---|
| `detection_v2.ipynb` | The complete detection pipeline: loaders, featurization, pre-registered protocols (LOSO / inter-patient / frozen thresholds), four detectors, CIs and tests, collinearity, hygiene, cluster-aware inference, seed replication, TOST/MDE, patient-level intervals — every cell compute-or-cached, executed top to bottom with 0 errors |
| `rag_v2.ipynb` | The complete explanation pipeline: corpus build, deviation-aware queries, grounded generation with raw-text preservation, corruption-validated judges, condition-blind / same-context / strict / shuffled / retrieval-off concordance arms, dual-rule flag status, alert-stream sensitivity, retrieval-poisoning probe — every cell compute-or-cached, executed top to bottom with 0 errors |
| `draft_paper/` | `paper.md` (manuscript), `references.bib`, `figures_v2/`, `verify_claims_v2.py` |
| `THRESHOLDS.md` | Pre-registered detection protocols (timestamped; scope stated in the paper) |
| `SEARCH_PROTOCOL.md` | Documented literature search behind related work |
| `scripts/robustness/` | `independent_concordance.py`: standalone reimplementation of every arm's concordance counts through a different code path (correctness instrument; the notebooks carry the computation itself) |
| `outputs_v2/robustness/` | Cached artifacts for the notebook robustness/security cells (cluster CIs, seed replication, TOST/MDE, Jaccard, flag status, all concordance arms, poisoning runs, alert-stream variant); delete any cache and the notebook cell rebuilds it from raw data |
| `outputs_v2/` | All artifacts: feature caches, explanations (raw + canonicalized), judge benchmarks, frozen API-judge checkpoint, agreement and concordance stats |
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
   rebuild is ~15 minutes (detection) and ~3 h GPU (RAG). The API-judge and
   LLM cells are resume-safe/cache-only: re-execution makes zero API calls.
4. `python draft_paper/verify_claims_v2.py` → expect all claims OK.

## Data, corpus, and licensing

Datasets are public with DOIs listed in the paper. The retrieval corpus
comprises 6 society guideline documents (4 v1 guidelines plus the 2022 EHRA
digital-devices guide and the 2023 ACC/AHA AF guideline, fetched with recorded
provenance, `Dataset/Tier1_v2/manifest.csv`) and 200 open-access articles
(149 CC BY, 51 CC BY-NC/NC-ND) fetched by `scripts/fetch_tier2.py`. Rebuild the
vector store with the notebook; the binary `chroma_db_v2/` is deliberately not
committed. The poisoning experiments run against a *copy* of the vector store
(a guarded notebook cell against a copy of the vector store); the released corpus manifest identifies
the legitimate corpus.

## Citation

If you use this code, please cite the paper (bibtex in
`draft_paper/references.bib`, entry to be finalized on publication).
