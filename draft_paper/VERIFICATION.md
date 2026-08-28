# Claim-by-Claim Verification — `paper.md`

Every quantitative claim in the draft, traced to its source artifact. Generated 2026-08-24 by
`verify_claims.py` (recomputes from `outputs/`) plus frozen notebook outputs and web-verified DOIs.

Legend: **CSV/PARQUET/JSONL** = recomputed from artifact · **NB** = frozen notebook output (detection.ipynb / rag.ipynb, also in detection.html / rag.html) · **WEB** = DOI/URL resolved at time of writing · **DESC** = project_description.md only (dev history, not re-derivable) — flag if load-bearing.

## A. Detection claims (Table 4, Abstract §1)

| # | Claim | Source | Status |
|---|---|---|---|
| A1 | WESAD IF AUC 0.874 / P 0.801 / R 0.397 / F1 0.531 | `outputs/detection_results.csv`; NB cells 72, 74 | ✅ exact |
| A2 | WESAD LOF AUC 0.910 / P 0.880 / R 0.436 / F1 0.583 | same | ✅ exact |
| A3 | MIT-BIH IF AUC 0.668 / P 0.502 / R 0.400 / F1 0.445 | same; NB cell 85 | ✅ exact |
| A4 | MIT-BIH LOF AUC 0.899 / P 0.734 / R 0.585 / F1 0.651 | same | ✅ exact |
| A5 | PTB-XL IF AUC 0.636 / P **0.658** / R 0.649 / F1 0.653 | same; NB cell 92 | ✅ exact (⚠ project_description.md §11 says 0.758 — typo there; paper uses correct 0.658) |
| A6 | PTB-XL LOF AUC 0.682 / P 0.689 / R 0.680 / F1 0.684 | same | ✅ exact |
| A7 | WESAD train = 586 baseline windows; eval = 1,106 (771 normal / 335 stress) | NB cell 69 | ✅ exact |
| A8 | MIT-BIH: 111,305 beats; 90,359 normal / 20,946 anomaly; train 90,097 (DS1 normal) | NB cells 81, 83 | ✅ exact |
| A9 | PTB-XL: 21,799 → 21,388 usable; 9,069 N / 12,319 pathology; train 7,243 (folds 1–8 normal); test fold 10 = 2,158 (912/1,246) | NB cells 89, 90 | ✅ exact |
| A10 | Thresholds: 95th pct (DaLiA), 85th (WESAD, MIT-BIH), 43rd pct (PTB-XL ≈ 57% prevalence) | NB cells 47, 72, 85, 92 | ✅ exact |
| A11 | IF: 100 trees; LOF: k=20, novelty=True; contamination 0.05 (DaLiA) / 0.15 (labeled) | NB cells 45–46, 71, 84, 91 | ✅ exact |
| A12 | No hyperparameter tuning; single run | NB (no tuning cells exist) | ✅ true by construction |

## B. PPG-DaLiA pipeline claims (§3.2–3.3, 4.2–4.3)

| # | Claim | Source | Status |
|---|---|---|---|
| B1 | 15 subjects; chest 700 Hz; BVP 64 Hz; wrist EDA/TEMP 4 Hz | NB cells 10, 20 | ✅ exact |
| B2 | Chest EDA + chest TEMP constant (dead) for all 15 subjects | NB audit cells 31, 36 (Y/– matrix) | ✅ exact |
| B3 | 5 channels kept: ECG, RESP, BVP, wrist EDA, wrist TEMP | NB cell 37 | ✅ exact |
| B4 | 4,308 windows; 60 features (5 ch × 12); 30 s non-overlapping | NB cells 41–42 | ✅ exact |
| B5 | 12 feature names (mean…roughness) | NB cell 38 | ✅ exact |
| B6 | IF flagged 216; LOF 216; both 34; union 398; Jaccard 0.085 | NB cells 47–48; `flagged_windows.parquet` recomputed: 216/216/34, 398 rows | ✅ exact |
| B7 | `flagged_windows.parquet` = 398 × 69; all 15 subjects present | PARQUET recomputed | ✅ exact |
| B8 | 50 confirmed WESAD stress windows exported | `confirmed_anomalies_for_rag.csv` = 50 rows, all WESAD/stress | ✅ exact |
| B9 | MIT-BIH beat segment ±144 samples (0.8 s); AAMI normal {N,L,R,e} | NB cell 81 (extract_mitbih_beats) | ✅ exact |

## C. RAG corpus claims (§3.5–3.6, Tables 2–3)

| # | Claim | Source | Status |
|---|---|---|---|
| C1 | Tier-1: 4 guideline PDFs; 2,133,668 chars; 679 chunks | NB cells 37–39 (rag) | ✅ exact |
| C2 | Tier-2: 200 articles; buckets 50/25/45/35/25/20; 12,918,008 chars; 4,053 chunks | NB cells 38–39 (rag) | ✅ exact |
| C3 | 4,732 total chunks; 500 words / 50 overlap; mean 489 words | NB cell 39 (rag) | ✅ exact |
| C4 | Tier-2 licensing 149 CC BY / 51 CC BY-NC/NC-ND; metadata audit (title 200/200, journal 200/200, authors 191/200, DOI 184/200) | DESC §8.2 (manifest-derived) | ⚠ DESC-only — spot-check `manifest.csv` before submission (license column exists; quick `value_counts` reconfirms) |
| C5 | MiniLM 384-dim; ChromaDB persistent; pool 20, max 1 chunk/source, top-5 | NB cells 40–41 (rag) | ✅ exact |
| C6 | Tier-1 ~305k words; Tier-2 ~1.78M words; median 8,450 w/article | DESC §8 | ⚠ DESC-only (derived from corpus build scripts; recompute from Tier-2 .md files if cited in final paper) |

## D. Query construction & retrieval diversity (§3.7, Table 6)

| # | Claim | Source | Status |
|---|---|---|---|
| D1 | Query = anomaly character + top-2 |z| channels (direction, z, mean) + topic phrases + other means | NB `build_query` cell | ✅ exact |
| D2 | 53 unique corpus documents used across 398 alerts | `rag_explanations.csv` recomputed: 53 | ✅ exact |
| D3 | Top-source context share 11.8% (235/1,990 slots, PMC12736534) | CSV recomputed | ✅ exact |
| D4 | Naive templated queries → ~11 unique documents | NB `build_query` docstring + DESC §11 | ⚠ DESC/dev-history (ablation not preserved as artifact) — keep "~11" hedged, or re-run the 3-line ablation to produce an artifact before submission |

## E. Generation & citation claims (§3.8, Table 7)

| # | Claim | Source | Status |
|---|---|---|---|
| E1 | Qwen3.5:9b via Ollama; temp 0.1; think=False; num_ctx 10000 | NB config + `generate_explanation` | ✅ exact |
| E2 | 4-field output format; <150 words; strict no-hallucination prompt | NB `SYSTEM_PROMPT` | ✅ exact |
| E3 | Canonicalizer: valid keep / near-miss snap (cutoff 0.75) / unresolvable drop | NB `canonicalize_citations` | ✅ exact |
| E4 | 1,359 citations; 1,359 valid; 0 hallucinated; 398/398 rows clean; avg 3.4 | NB citation cells + CSV recomputed: 1,359 / 3.4 | ✅ exact |
| E5 | Pre-repair raw accuracy 99.3% (9 near-miss rows) → 100% | NB repair cell history + DESC §11 (final re-run shows 0/398 to fix — idempotent) | ⚠ DESC/dev-history for the 99.3% baseline figure; final 100% is artifact-verified |
| E6 | Example alert (Fig. 5) verbatim | NB evaluation-section sample (df_results.iloc[0]) | ✅ verbatim |

## F. Judge claims (§3.9, Table 8)

| # | Claim | Source | Status |
|---|---|---|---|
| F1 | Local llama3.1:8b means 2.99 / 2.99 / 2.02 | `rag_evaluation_v2.csv` recomputed | ✅ exact |
| F2 | Local judge constant: faithfulness {3: 397, 0: 1}; the 0 is a parse failure | CSV recomputed | ✅ exact — paper's degeneracy claim precisely worded |
| F3 | API judge (deepseek/deepseek-v4-flash-0731, reasoning off) means 2.59 / 2.80 / 2.47 | `rag_evaluation_api.csv` recomputed | ✅ exact |
| F4 | Raw agreement 59.2 / 79.8 / 52.4 % (F/R/C) | CSV + NB frozen | ✅ exact |
| F5 | Within-1 agreement 100.0% all axes (1 masked parse-fail row per axis) | CSV recomputed with (a>0)&(b>0) mask | ✅ exact |
| F6 | 0/398 hallucination verdicts from BOTH judges; 0/398 API any-axis=1 | CSV recomputed | ✅ exact |
| F7 | κ degenerate (constant rater) — correctly reported instead of quoting κ≈0 | NB agreement cell note | ✅ exact |
| F8 | Checkpoint: 418 lines, 398 valid unique, 20 truncated dropped | `api_judge_checkpoint.jsonl` recomputed | ✅ exact |
| F9 | Local judge latency 5.6 s/judgment | DESC §11 only | ⚠ DESC-only (no latency column in v2 CSV; re-measure or drop before submission) |

## G. System / latency (§4.6, §5)

| # | Claim | Source | Status |
|---|---|---|---|
| G1 | Generation latency mean 10.9 s (σ 1.3, median 10.7) per alert | `rag_explanations.csv` recomputed: 10.9 / 1.28 / 10.7 | ✅ exact (DESC's "10.7 s" was the median; NB batch print 9.7 s was an earlier run — CSV is authoritative) |
| G2 | Retrieval sub-second | design fact (MiniLM 1-query encode + ChromaDB 20-candidate search); no artifact timer | ⚠ qualitative — add a timer if reviewers need a number |
| G3 | RTX 5060 Laptop GPU, fully local | DESC §9 (environment fact) | ✅ environmental |

## H. Reference verification (references.bib, 25 entries)

| Key | DOI | Resolution check |
|---|---|---|
| reiss2019deepppg | 10.3390/s19143079 | ✅ WEB (MDPI; Sensors 19(14):3079) |
| ppgdalia2023 | 10.24432/C53890 | ✅ WEB (UCI repo page) |
| schmidt2018wesad | 10.1145/3242969.3242985 | ✅ WEB (ACM DL, ICMI'18, pp. 400–408) |
| wesad2023 | 10.24432/C57K5T | ✅ WEB (UCI repo page) |
| moody2001mitbih | 10.1109/51.932724 | ✅ WEB (PubMed 11446209; IEEE EMBM 20(3):45–50) |
| mitbih2023 | 10.13026/C2F61Q | ✅ WEB (PhysioNet) |
| wagner2020ptbxl | 10.1038/s41597-020-0495-6 | ✅ WEB (Nature, Sci Data 7:154; PubMed 32451379) |
| goldberger2000physionet | 10.1161/01.CIR.101.23.e215 | ✅ STANDARD (canonical) |
| shen2017syncope | 10.1161/CIR.0000000000000499 | ✅ WEB (AHA journals; Circulation 136(5)) |
| alkhatib2017vascd | 10.1161/CIR.000000000000054 | ✅ WEB (PubMed 29097296; Circulation 138(13):e272–e391) |
| brignole2018syncope | **10.1093/eurheartj/ehy037** | ✅ WEB (PubMed 29562304; EHJ 39(21):1883–1948) — ⚠ commonly miscited as ehy036; bib has the correct suffix |
| zeppenfeld2022vascd | 10.1093/eurheartj/ehac262 | ✅ WEB (OUP; EHJ 43(40):3997–4126; PubMed 36017572) |
| liu2008isolationforest | 10.1109/ICDM.2008.17 | ✅ STANDARD |
| breunig2000lof | 10.1145/335191.335388 | ✅ STANDARD |
| pedregosa2011scikitlearn | (JMLR, no DOI) | ✅ STANDARD |
| lewis2020rag | arXiv:2005.11401 | ✅ STANDARD |
| reimers2019sbert | 10.18653/v1/D19-1410 | ✅ STANDARD (ACL Anthology) |
| wang2020minilm | arXiv:2002.10957 | ✅ STANDARD |
| qwen2025qwen3 | arXiv:2505.09388 | ⚠ verify qwen3.5-specific tech report exists before submission; else cite Qwen3 report (placeholder note in bib) |
| zheng2023llmjudge | arXiv:2306.05685 | ✅ STANDARD |
| cohen1960kappa | 10.1177/001316446002000104 | ✅ STANDARD |
| ollama2023 / chromadb2023 | (software URLs) | ✅ WEB (live products) |
| turja2026unsupervised | 10.1016/j.bspc.2026.110416 | ✅ WEB — DOI resolves to exact paper; ScienceDirect BSPC 123:110416; arXiv:2603.12278 |
| turja2026forecasting | 10.48550/arXiv.2602.22673 | ✅ WEB — arXiv abs page verified (v2, 2026-06-13); title/author match |

## I. Known open items before submission

1. **C4/C6** (Tier-2 license split, word counts): one-line recompute from `manifest.csv` / .md files — 5 min.
2. **D4** (~11 naive-query baseline): re-run tiny ablation to create an artifact (the current source is a dev-run note).
3. **E5** (99.3% pre-repair): reconstruct by disabling the canonicalizer on the same explanations (deterministic) if the number is kept.
4. **F9** (5.6 s local-judge latency): re-measure or delete from Table 8.
5. **G2**: add a retrieval timer (one `time.perf_counter` pair) if a number is wanted.
6. **qwen2025qwen3**: check for a qwen3.5 report; else keep Qwen3 report with note.
7. **Figure 1**: redraw ASCII pipeline as TikZ for the LaTeX build.
8. project_description.md §11 still carries the stale **0.758** PTB-XL IF precision and the "10.7 s avg" phrasing — fix at next doc update (paper already uses correct values).
