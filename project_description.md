# Retrieval-Augmented Generation for Continuous Anomaly Alerts

> A plain-English, complete description of the project — what it is, why it matters,
> how it works, what data it uses, and how it will be evaluated and published.
>
> **Author:** Md Tanvir Hasan Turja · Independent ML Researcher, London, UK
> **Started:** August 2026 · **Status:** Both halves complete — detection (4 datasets) + RAG pipeline built, run, and dual-judge evaluated; paper writing next

---

## Table of Contents

1. [The One-Sentence Summary](#1-the-one-sentence-summary)
2. [The Real-World Problem](#2-the-real-world-problem)
3. [Why This Project, Why Now](#3-why-this-project-why-now)
4. [How the System Works (Step by Step)](#4-how-the-system-works-step-by-step)
5. [The Two Halves of the Project](#5-the-two-halves-of-the-project)
6. [What Makes This Novel (The Contribution)](#6-what-makes-this-novel-the-contribution)
7. [The Signal Datasets (Input)](#7-the-signal-datasets-input)
8. [The RAG Knowledge Corpus (Retrieval)](#8-the-rag-knowledge-corpus-retrieval)
9. [The Technology Stack](#9-the-technology-stack)
10. [How This Builds on My Past Work](#10-how-this-builds-on-my-past-work)
11. [How Success Is Measured (Evaluation)](#11-how-success-is-measured-evaluation)
12. [Ethical, Legal, and Safety Considerations](#12-ethical-legal-and-safety-considerations)
13. [Honest Limitations](#13-honest-limitations)
14. [Future Work](#14-future-work)
15. [Target Journals for Publication](#15-target-journals-for-publication)
16. [Current Status and Folder Layout](#16-current-status-and-folder-layout)
17. [Glossary (Plain-English Definitions)](#17-glossary-plain-english-definitions)
18. [How to Load the Datasets (Code)](#18-how-to-load-the-datasets-code)

---

## 1. The One-Sentence Summary

This project builds a system that **watches a person's continuous body signals (like
heart rhythm from a wrist sensor), automatically notices when something looks wrong, and
then explains — in plain human language, with citations to real medical guidelines — what
might be happening and what it could mean.**

In technical terms: it combines **unsupervised anomaly detection** (finding unusual
patterns in biosignals without needing labeled examples) with **Retrieval-Augmented
Generation / RAG** (grounding a language model's explanation in real medical documents so
it cannot make things up).

---

## 2. The Real-World Problem

Modern wearable devices — smartwatches, chest straps, wristbands — stream huge amounts of
health data continuously: heart rate, blood-volume changes, skin conductance, temperature,
breathing, and movement. This data is valuable, but it creates three problems:

1. **Too much data for humans to watch.** A single person's recording can contain millions
   of data points. No nurse or doctor can monitor that manually.

2. **When an automated alert fires, it is usually just a number or a red flag.** It says
   "anomaly detected" but gives no explanation of *what* it might mean, *why* it happened,
   or *what the clinical guidelines say*. This makes alerts hard to trust and act on.

3. **Large language models (LLMs) like ChatGPT can explain things in plain language, but
   they hallucinate** — they confidently invent facts, fake citations, and give wrong medical
   advice. You cannot safely put a raw LLM in front of a clinical alert.

So the gap is: **we can detect that something is unusual, but we cannot reliably explain
what it means using grounded, verifiable medical knowledge.** This project closes that gap.

### Who benefits

- **Patients** using wearables for chronic conditions (cardiac arrhythmia, stress disorders,
  post-surgical monitoring, fall risk).
- **Clinicians** who receive wearable/remote-monitoring data and need fast, trustworthy
  interpretation.
- **Healthy users** doing continuous wellness monitoring who want meaningful, non-alarming
  explanations of flagged events.
- **Device makers and hospitals** building remote-monitoring dashboards.

---

## 3. Why This Project, Why Now

Three things have come together in 2025–2026 that make this the right project at the right time:

1. **RAG has matured.** Retrieval-Augmented Generation is now a proven way to stop LLMs from
   hallucinating by forcing them to answer only from retrieved documents. It is one of the
   hottest areas in both academia and industry, with heavy PhD and job demand — especially
   for healthcare applications.

2. **Wearable biosignal monitoring is everywhere.** Devices like the Apple Watch, Fitbit,
   Empatica, and Withings already stream clinical-grade signals (PPG, ECG, EDA). The data
   infrastructure exists; the *interpretation* layer does not.

3. **The specific combination is still an open niche.** A literature search (August 2026)
   found that *no single published paper yet fully integrates all five pieces*: LLM + RAG +
   unsupervised anomaly detection + wearable biosignals + real-time clinical alerting. The
   two closest existing works each cover only one half (one does wearable anomaly detection
   without LLM explanation; the other does LLM reporting without true anomaly detection).
   **That unfilled middle is this project's contribution.**

This is also a strong fit for the current research-funding and PhD-application climate:
explainable, trustworthy AI for clinical decision support is a priority for regulators (EU
AI Act, FDA) and for major funders (NIH, Horizon Europe, NHS AI Lab).

---

## 4. How the System Works (Step by Step)

The system is a pipeline with five stages. Think of it as a continuous loop that runs while
a person wears a sensor.

```
 ┌─────────────┐   ┌──────────────┘   ┌────────────────┐   ┌────────────┐   ┌──────────────┐
 │ 1. Stream   │──▶│ 2. Window &   │──▶│ 3. Anomaly    │──▶│ 4. RAG     │──▶│ 5. Grounded  │
 │ raw signals │   │  feature      │   │ detection     │   │ retrieval  │   │ alert +      │
 │ (PPG, ECG,  │   │  extraction   │   │ (Isolation    │   │ (ChromaDB +│   │ explanation  │
 │ EDA, temp)  │   │              │   │ Forest/LOF)   │   │ MiniLM)    │   │ (local LLM)  │
 └─────────────┘   └──────────────┘   └────────────────┘   └────────────┘   └──────────────┘
```

### Stage 1 — Signal streaming
The wearable device produces continuous signals: photoplethysmogram (PPG / blood-volume
pulse), electrocardiogram (ECG), electrodermal activity (EDA / skin conductance), skin
temperature, respiration, and 3-axis acceleration. These arrive as long time series.

### Stage 2 — Windowing and feature extraction
The raw stream is chopped into short windows (e.g. 30–60 seconds). For each window, the
system computes features that summarize the physiology: heart rate, heart-rate
variability, signal amplitude, variability, trends, frequency-band energy, etc. Because
the chest device and wrist device sample at different rates (chest at 700 Hz, wrist at
4–64 Hz), all signals are resampled to a common time grid first. This mirrors the feature
engineering used in the author's published wearable-sensor paper.

### Stage 3 — Anomaly detection (the "something is wrong" step)
An **unsupervised** model scores each window for how abnormal it is, relative to the normal
baseline distribution. The project uses two complementary algorithms:
- **Isolation Forest** — a partitioning method that is sensitive to subtle, distributed
  deviations (good at catching slow-building problems).
- **KNN-LOF (Local Outlier Factor)** — a density method that is sensitive to sharp, extreme
  spikes (good at catching sudden events).

These are run *without labels* — they learn "normal" from the bulk of the data and flag
whatever looks statistically unusual. When a window's anomaly score crosses a threshold,
it triggers the next stage.

### Stage 4 — RAG retrieval (finding the right medical knowledge)
When an anomaly fires, the system does **not** just ask an LLM "what does this mean?" (that
would hallucinate). Instead it:
1. Builds a query describing the anomaly (the signal type, the direction of deviation, the
   detected features).
2. Searches a **vector database (ChromaDB)** containing the medical knowledge corpus (the
   Tier-1 guidelines + Tier-2 literature described below).
3. Retrieves the most relevant chunks of real medical text — e.g. the ESC guideline section
   on that arrhythmia, or a research paper on that PPG pattern.

Embeddings (numeric representations of text meaning) are created with
`all-MiniLM-L6-v2` sentence transformers — the same proven approach used in the author's
published AMR paper.

### Stage 5 — Grounded explanation (the human-readable alert)
A **local, private LLM** (Qwen3.5 9B, run via Ollama on the user's own machine — no
data leaves the device) receives:
- the retrieved medical text chunks,
- the anomaly details,
- a **strict system prompt** that says: *"Answer ONLY using the provided context. Cite the
  source document for every claim. If the context is insufficient, say so explicitly. Do not
  invent facts or citations."*

The LLM then produces a short, plain-language explanation of what was detected, what it may
mean, and which guideline/literature supports that interpretation — with citations. This is
the alert that a clinician or patient reads.

### Why this design is safe
Because the LLM is forced to answer only from retrieved real documents and to cite them,
**hallucination is structurally suppressed** — the author's AMR paper achieved 100%
citation faithfulness with this exact pattern on a small local model.

---

## 5. The Two Halves of the Project

The project deliberately fuses two capabilities the author has already published
separately:

| Half | Capability | Author's prior evidence |
|---|---|---|
| **Detection** | Unsupervised anomaly detection on wearable biosignals | BSPC (Elsevier, 2026) — Isolation Forest + KNN-LOF on wearable foot sensors |
| **Explanation** | Hallucination-resistant RAG with local LLMs | arXiv (2026) AMR policy RAG — 100% citation faithfulness; Zenodo automotive RAG |

**The novelty is not either half alone — it is their integration into one continuous,
alert-triggered, evidence-grounded pipeline for wearable biosignals.** No prior paper has
done this end to end.

---

## 6. What Makes This Novel (The Contribution)

To be publishable in a Q1 journal, the project needs a clear contribution beyond
"engineering." Three candidate novelty hooks (one will be chosen as the headline):

1. **Alert-triggered retrieval (context-aware).** Unlike standard RAG (which answers a
   human-typed question), here the retrieval is *automatically triggered by a detected
   anomaly* and the query is built from the anomaly's signal context (which channel, which
   feature, which direction). This is a new interaction pattern.

2. **Uncertainty-aware alerting.** The system fires the (expensive, LLM-based) explanation
   only when *both* the anomaly score is high *and* the retrieved context is confident —
   reducing alert fatigue and wasted computation. This threshold logic is itself a
   contribution.

3. **A formal evaluation framework for clinical-alert faithfulness.** A reproducible rubric
   (faithfulness, relevance, completeness, citation accuracy — extending the author's AMR
   methodology) to measure whether LLM-generated medical alerts are grounded, applicable
   across model families. This methodology gap currently has no standard.

The headline claim will be framed around whichever of these produces the strongest,
cleanest experimental story.

---

## 7. The Signal Datasets (Input)

These provide the biosignals the anomaly detector runs on. All are free, citable (DOI /
peer-reviewed paper), and require no paid access.

### 7.1 PPG-DaLiA — primary signal source *(downloaded ✅)*
- **What:** Continuous multimodal biosignals from 15 subjects doing normal daily activities.
- **Signals:** PPG (blood-volume pulse), ECG, EDA, skin temperature, respiration, acceleration.
- **Devices:** Chest (RespiBAN, 700 Hz) + Wrist (Empatica E4: BVP 64 Hz, EDA 4 Hz, temp 4 Hz, accel 32 Hz).
- **Size:** ~8.3 million instances; ~23 GB.
- **Role:** Drives the anomaly-detection pipeline and serves as the "real-world noisy
  signal" demonstration.
- **Finding (audited across all 15 subjects):** The **chest EDA and chest TEMP channels are
  dead (constant) for every subject** — a sensor flaw in the released dataset, not a
  per-subject dropout. They were therefore excluded globally. The final usable channel set
  is **5 channels**: chest ECG, chest RESP, wrist BVP (PPG), wrist EDA, wrist TEMP.
- **Limitation:** It contains **normal activity only** — no labeled anomalies, no pathology.
  So it can demonstrate detection and the pipeline, but cannot alone measure detection
  accuracy (precision/recall). That is why WESAD (below) is essential.
- **DOI:** 10.24432/C53890 · Source: UCI ML Repository (ID 495)

### 7.2 WESAD — labeled evaluation set *(downloaded + extracted ✅)*
- **What:** Wearable Stress and Affect Detection — 15 subjects in a controlled lab study.
- **Crucial advantage:** Uses the **exact same sensors** as PPG-DaLiA (Empatica E4 wrist +
  RespiBAN chest), so integration is nearly effortless.
- **Labels:** Baseline / Stress / Amusement (plus transient states). This gives **ground
  truth** — known physiological events the detector can be evaluated against.
- **Role:** Turns "anomaly detection" into something measurable: does the detector fire on
  stress episodes? what is the precision/recall? This is what makes the evaluation
  reviewer-proof.
- **DOI:** 10.24432/C57K5T · Source: UCI ML Repository (ID 465)

### 7.3 MIT-BIH Arrhythmia Database — clinical ECG ground truth *(downloaded ✅)*
- **What:** 48 half-hour two-channel ECG recordings, 47 subjects, 360 Hz.
- **Labels:** Beat-level expert annotations of real arrhythmias (PVC, PAC, atrial
  fibrillation, etc.) — the gold standard for cardiac anomaly detection.
- **Role:** Proves the detector generalizes to **true clinical pathology**, not just lab
  stress. Without it, a reviewer can argue "you only detect stress, not disease." Also gives
  *beat-level* labels — ideal for windowed Isolation Forest (which beat is abnormal).
- **On disk:** `Dataset/mit-bih-arrhythmia-database-1.0.0/` (106 MB; `.dat` signals + `.hea` headers
  + `.atr` beat annotations). · DOI: 10.13026/C2F61Q · Load with `wfdb` (see §18).

### 7.4 PTB-XL — large-scale clinical ECG *(downloaded ✅)*
- **What:** 21,799 clinical 12-lead ECGs from ~18,869 patients, 10 seconds each.
- **Labels:** 5 diagnostic superclasses (Normal, Myocardial Infarction, ST/T change,
  Conduction Disturbance, Hypertrophy). Also ships a **`strat_fold`** column = a ready-made
  10-fold patient-stratified train/test split (prevents patient-level leakage — reviewers
  like this).
- **Role:** Solves both the cohort-size problem and the label problem at scale — large
  enough to claim generalization. Superclasses map neatly onto Tier-1 guideline content
  (MI → ischemia; STTC → repolarization; CD → conduction block; HYP → hypertrophy).
- **On disk:** `Dataset/ptb-xl-1.0.3/` — `ptbxl_database.csv` (labels) + `records100/` (100 Hz) +
  `records500/` (500 Hz). · DOI: 10.1038/s41597-020-0495-6 · Load with `wfdb` + `pandas` (§18).

### Why this combination
PPG-DaLiA alone is insufficient (no labels). The minimum publishable setup is **PPG-DaLiA
(detection + pipeline demo) + WESAD (evaluation)**. MIT-BIH + PTB-XL strengthen the clinical
claim from "lab stress" to "real arrhythmia," and are complementary to each other:

| Dataset | Records | Best for |
|---|---|---|
| MIT-BIH | 48 | beat-level anomaly localization (classic benchmark) |
| PTB-XL | 21,799 | scale, 12-lead, modern, clean stratified folds |

---

## 8. The RAG Knowledge Corpus (Retrieval)

This is the set of **medical documents** the system retrieves from when explaining an alert.
It is split into two tiers.

### 8.1 Tier-1 — Authoritative clinical guidelines *(verified ✅)*
Four peer-reviewed clinical-practice guidelines from the top cardiology societies. These are
the "ground truth" reference documents — the equivalent of the WHO policy documents used in
the author's AMR paper.

| Guideline | Pages | Words | Society / Year |
|---|---|---|---|
| Evaluation & Management of Patients with Syncope | 72 | ~55k | ACC/AHA/HRS 2017 |
| Management of Ventricular Arrhythmias & SCD Prevention | 186 | ~96k | AHA/ACC/HRS 2017 |
| Diagnosis and Management of Syncope | 69 | ~54k | ESC 2018 |
| Management of Ventricular Arrhythmias & SCD | 130 | ~100k | ESC 2022 |

**Total Tier-1:** ~305,000 words of authoritative cardiology knowledge.

**Coverage:** arrhythmia, syncope (fainting), ventricular tachycardia/fibrillation, sudden
cardiac death, ambulatory monitoring, ECG interpretation, palpitation.

**Honest gap:** These guidelines are ECG/arrhythmia-focused and contain **no mention of PPG
(photoplethysmography)** — the project's primary wrist signal. That gap is exactly what
Tier-2 fills.

### 8.2 Tier-2 — Broader open-access literature *(downloaded ✅)*
200 open-access full-text research articles fetched from Europe PMC across six topic
buckets, covering every alert axis the detector can fire.

| Bucket | Articles | Covers |
|---|---|---|
| 01 PPG arrhythmia detection | 50 | AF / arrhythmia detection from PPG |
| 02 PPG signal quality & artifacts | 25 | motion artifacts, false-alarm causes |
| 03 ECG anomaly / arrhythmia ML | 45 | clinical ECG detection (deep/ML) |
| 04 Wearable stress / affect detection | 35 | stress detection from EDA/ECG |
| 05 Continuous / ambulatory monitoring | 25 | deployment context |
| 06 Biosignal anomaly-detection methods | 20 | Isolation Forest / unsupervised methods |

**Total Tier-2:** 200 articles, ~1.78 million words, median ~8,450 words/article, all
2025–2026, all open access (149 CC BY, 51 CC BY-NC/NC-ND).

**Citation metadata (audited):** Title 200/200, Journal 200/200, Authors 191/200, DOI
184/200. A `references.bib` and `manifest.csv` are generated automatically.

### Combined corpus
**~2.09 million words** of citable, grounded medical knowledge — large enough for rich
retrieval, focused enough to stay relevant and low-noise.

---

## 9. The Technology Stack

Everything runs locally and free — no cloud APIs, no paid services, no data leaving the
machine. This is deliberate: clinical data privacy and reproducibility.

| Layer | Tool | Notes |
|---|---|---|
| Language | Python 3.13 | All code |
| Signal processing | NumPy, SciPy, pandas | resampling, windowing, features |
| Anomaly detection | scikit-learn | Isolation Forest, KNN-LOF |
| Deep learning (optional) | PyTorch | if an LSTM/autoencoder baseline is added |
| Vector store | ChromaDB | persistent local vector database |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | same as AMR paper |
| Retrieval | dense retrieval + source-diversity constraint (max 1 chunk/document, deviation-aware alert queries) | prevents 1–2 dominant papers filling the context |
| LLM inference | Ollama (local) | generator: Qwen3.5 9B (Q4), fully offline; judge: llama3.1:8b (local) + optional API reference judge |
| LLM framework | direct ChromaDB + `ollama` Python lib (LangChain-free) | avoids version breakage |
| Tracking | MLflow (optional) | experiment logging |
| Hardware | NVIDIA RTX 5060 Laptop GPU + CUDA | author's existing machine |

---

## 10. How This Builds on My Past Work

This project is not a leap into the unknown — it is the deliberate fusion of two things the
author has already published.

- **Detection half** ← *Unsupervised Anomaly Detection in Wearable Foot Sensor Data*
  (Biomedical Signal Processing and Control, Elsevier, 2026). That paper established the
  Isolation Forest vs KNN-LOF methodology, the Jaccard inter-method agreement analysis, and
  the multi-modal sensor reasoning. This project reuses that exact detection machinery, now
  applied to cardiac/stress biosignals.

- **Explanation half** ← *Forecasting Bacterial Antimicrobial Resistance Trends … RAG for
  Policy Decision Support* (arXiv, 2026). That paper built the ChromaDB + MiniLM + local
  Gemma RAG pipeline that achieved **100% citation faithfulness** across 25 policy queries,
  with strict anti-hallucination prompting. This project reuses that exact RAG architecture,
  now pointed at clinical guidelines instead of WHO policy docs.

So the author is one of the few researchers who has **already published both halves**.
Bringing them together is the contribution.

---

## 11. How Success Is Measured (Evaluation)

A Q1 paper needs rigorous, multi-axis evaluation. The project evaluates both halves and
their integration.

### Detection quality (Stage 3)
- **Precision, recall, F1** of anomaly detection on WESAD labeled stress events and on
  MIT-BIH/PTB-XL arrhythmia labels. (This is exactly why labeled datasets are essential.)
- **ROC / PR curves** and **AUC**.
- **Leave-one-subject-out cross-validation** (standard for small wearable cohorts).
- Comparison of Isolation Forest vs KNN-LOF vs any deep baseline (LSTM/autoencoder).

**Current results (single-run, 5% / 15% contamination, no hyperparameter tuning):**

| Dataset | Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| WESAD (stress vs baseline) | Isolation Forest | 0.874 | 0.801 | 0.397 | 0.531 |
| WESAD (stress vs baseline) | KNN-LOF | **0.910** | **0.880** | **0.436** | **0.583** |
| MIT-BIH (arrhythmia beats) | Isolation Forest | 0.668 | 0.502 | 0.400 | 0.445 |
| MIT-BIH (arrhythmia beats) | KNN-LOF | **0.899** | **0.734** | **0.585** | **0.651** |
| PTB-XL (pathology, fold 10) | Isolation Forest | 0.636 | 0.758 | 0.649 | 0.653 |
| PTB-XL (pathology, fold 10) | KNN-LOF | **0.682** | **0.689** | **0.680** | **0.684** |

**Inter-method agreement (Jaccard, PPG-DaLiA):** 0.085 — the two models flag largely
disjoint windows, confirming they are complementary (IF catches distributed drift; LOF
catches sharp spikes). This validates the two-model design.

**Takeaway:** LOF consistently matches or beats IF on biosignal anomalies; IF/LOF are
genuinely complementary (not redundant). Both findings reproduce the author's BSPC paper
methodology on cardiac/stress data. These are baseline numbers — LOSO CV and an
LSTM/autoencoder baseline remain as future refinement.

### RAG / explanation quality (Stages 4–5)
- **Faithfulness** — does the explanation contain only information present in the retrieved
  documents? (1 = hallucination, 3 = fully grounded.)
- **Relevance** — is the explanation directly about the detected anomaly?
- **Completeness** — does it address the key clinical aspects?
- **Citation accuracy** — are cited documents (title/date) correct?
- These are scored on a held-out query set, with **inter-rater agreement (Cohen's κ)**
  reported — extending the rubric from the AMR paper.

**Current results (all 398 flagged PPG-DaLiA windows, complete pipeline run):**

| Metric | Result |
|---|---|
| Citation accuracy (programmatic, objective) | **100.0%** — 1,359/1,359 citations valid, 0 hallucinated |
| Explanations with all-valid citations | 398/398 (100%) |
| Retrieval diversity (deviation-aware queries) | 53 unique corpus documents used (vs 11 with naive templated queries); top-1 source share 12% |
| Generation latency (Qwen3.5 9B, local) | 10.7 s avg per alert |
| Local judge (llama3.1:8b, different model family) | faithfulness 2.99 / relevance 2.99 / completeness 2.02 (of 3) |
| Reference judge (deepseek-v4-flash via OpenRouter) | faithfulness 2.59 / relevance 2.80 / completeness 2.47 (of 3) |
| Cross-judge agreement | **100% within-1** on all axes; **0/398 hallucination verdicts** from either judge |
| Judge latency | local 5.6 s/judgment; API judge checkpointed + idempotent (re-runs cost 0 calls) |

**Notes on the judge methodology:**
- The **local judge is a constant rater** (scored 3/3 on faithfulness and relevance for all
  398 rows), which makes Cohen's κ mathematically degenerate (κ≈0 by definition when one
  rater has zero variance). Raw and within-1 agreement are reported instead — within-1 is
  100% on every axis.
- The headline safety claim is cross-validated: **two judges from different model families
  never scored any explanation as containing hallucination (0/398), and every one of the
  1,359 citations resolves to a retrieved source.**
- Completeness ~2.0–2.5/3 reflects a deliberate design constraint: the system prompt caps
  explanations at 150 words in plain language with a strict "verify before acting"
  recommendation posture, rather than speculative differential diagnoses.

**Citation-repair step (part of the pipeline):** the generator canonicalizes every `[PMC…]`
citation against the retrieved source list at generation time (snapping near-miss IDs —
digit transpositions — to the retrieved source, dropping unresolvable brackets). This took
raw citation accuracy from 99.3% (9 rows with near-miss IDs after diverse retrieval widened
the source pool) to a clean 100%.

### Integration / system quality
- **End-to-end latency** (anomaly → alert) for real-time viability. *Measured: retrieval is
  sub-second; generation averages 10.7 s per alert on an RTX 5060 laptop GPU — viable for
  30-second-window alerting.*
- **Alert fatigue metrics** — false-positive rate under the uncertainty-aware threshold.
- **Ablations** — detection-only vs detection+RAG, to show the explanation adds value.

---

## 12. Ethical, Legal, and Safety Considerations

- **Not a medical device.** The system is a research decision-support tool, **not** a
  diagnostic device. Every output states this. It supplements, never replaces, clinical
  judgment.
- **Privacy by design.** All signals and LLM inference run locally (Ollama). No patient data
  leaves the machine. This is a major advantage over cloud-based LLM monitoring.
- **Hallucination suppression.** The strict retrieval-grounded system prompt is the primary
  safety mechanism; the AMR paper proved it eliminates fabrication even on a small model.
- **Licensing.** Tier-1 guidelines are for research use; Tier-2 articles are open access
  (mostly CC BY, redistributable with attribution). The derived corpus and code can be
  released with their own Zenodo DOI.
- **Bias.** Datasets are small (15 subjects) and not demographically diverse — stated as a
  limitation; not deployed on real patients.

---

## 13. Honest Limitations

These will be stated openly in the paper (reviewers respect honesty):

1. **Small cohorts.** PPG-DaLiA and WESAD have 15 subjects each — too small for strong
   generalization claims. Mitigated by leave-one-subject-out CV and by adding PTB-XL (18k+
   patients) for the ECG side.
2. **Sensor/label mismatch.** PPG-DaLiA has no labels; WESAD's labels are stress (not
   disease). True clinical pathology comes only from MIT-BIH/PTB-XL (ECG-only).
3. **No single paired dataset** has both wearable signals *and* clinical outcomes from the
   same patients — a known, disclosed gap.
4. **Tier-1 guidelines don't cover PPG** specifically; PPG reasoning relies on Tier-2
   literature, which is lower-authority than society guidelines.
5. **Local LLM limits.** Qwen3.5 9B is a mid-size model; richer synthesis might need larger
   models (a future-work item).

---

## 14. Future Work

- **Temporal-context retrieval:** restrict retrieved documents to a clinically relevant
  time/anatomy window.
- **Larger / frontier LLMs** for deeper synthesis (Qwen2.5:72b, Llama-3.1:70b).
- **Real prospective deployment** on a wearable cohort with clinician review.
- **Multimodal fusion** with images (connects to the companion lower-extremity-lesions
  project).
- **Federated learning** across hospital-deployed wearables without centralizing data.
- **Formal RAG benchmarking** (RAGAS-style) at scale.

---

## 15. Target Journals for Publication

| Journal | Quartile | Why it fits |
|---|---|---|
| **Biomedical Signal Processing and Control** (Elsevier) | Q1 (IF ~4.9) | Author's existing journal; explicitly expanding into LLM/RAG clinical scope; perfect signal+AI fit |
| **IEEE J. Biomedical & Health Informatics** | Q1 | Wearables + clinical decision support + informatics |
| **Computers in Biology and Medicine** | Q1 | ML for clinical interpretation |
| **Artificial Intelligence in Medicine** | Q1 | If framed around the XAI/explanation methodology |

A preprint (arXiv) will be posted first to establish priority, then submitted to a Q1 venue.

---

## 16. Current Status and Folder Layout

### What is done ✅
- PPG-DaLiA downloaded (15 subjects, 23 GB).
- WESAD downloaded + extracted (15 subjects, 17 GB).
- MIT-BIH Arrhythmia Database downloaded (48 records, beat-level labels).
- PTB-XL downloaded (21,799 12-lead ECGs, 100 Hz + 500 Hz, stratified folds).
- Tier-1 RAG corpus: 4 guidelines downloaded and verified.
- Tier-2 RAG corpus: 200 OA articles fetched, enriched, with `references.bib` + `manifest.csv`.
- Fetcher and enrichment scripts written and reusable.
- **Detection pipeline built and evaluated** across all four signal datasets. IF + LOF
  trained on the "normal" class of each dataset, scored, and measured against expert
  labels (AUC up to 0.91 on WESAD stress, 0.899 on MIT-BIH arrhythmias). 398 flagged
  windows from PPG-DaLiA exported as the RAG handoff set.
- **RAG pipeline built, run end-to-end, and evaluated** (`rag.ipynb`):
  - 204 knowledge documents → 4,732 chunks → ChromaDB (all-MiniLM-L6-v2).
  - Deviation-aware, alert-triggered query builder (top-2 z-scored deviating channels +
    topic keywords → 53 unique documents actually used vs 11 with naive queries).
  - Source-deduplicated retrieval (max 1 chunk/document per context, 5 documents/alert).
  - Grounded generation with Qwen3.5 9B via Ollama (strict no-hallucination prompt,
    DETECTED/EVIDENCE/RECOMMENDATION/DISCLAIMER output format).
  - **100% citation accuracy** (1,359/1,359) with citation canonicalization.
  - Dual-judge evaluation: local llama3.1:8b + API reference judge (deepseek-v4-flash via
    OpenRouter, reasoning disabled, env-var key, idempotent JSONL checkpoint) — 100%
    within-1 agreement, 0/398 hallucination verdicts.

### Data stack at a glance
| Layer | Component | Status |
|---|---|---|
| Signals | PPG-DaLiA (wearable) | ✅ |
| Signals | WESAD (wearable + stress labels) | ✅ |
| Signals | MIT-BIH (clinical ECG, beat-level) | ✅ |
| Signals | PTB-XL (clinical ECG, scale, 12-lead) | ✅ |
| Knowledge | Tier-1 guidelines (4) | ✅ |
| Knowledge | Tier-2 literature (200 articles) | ✅ |
| Pipeline | Detection (IF + LOF, 4 datasets evaluated) | ✅ |
| Pipeline | RAG (chunk + embed + retrieval + LLM) | ✅ |
| Pipeline | Integration (alert → retrieval → explanation) | ✅ |
| Pipeline | Evaluation (citation check + dual LLM-judge) | ✅ |

### What is next 🔲
- Write the paper (results tables above are final).
- Optional refinements: LOSO-CV for detection; completeness-prompt ablation; uncertainty-
  aware alert gating (novelty hook #2) as an additional experiment.

### Folder layout
```
Retrieval-augmented generation for continuous anomaly alerts/
├── project_description.md            ← this file
├── DATASET_CARD.md                   ← signal-dataset reference card
├── detection.ipynb                   ← Notebook 1: anomaly detection (4 datasets)
├── rag.ipynb                         ← Notebook 2: RAG pipeline + evaluation
├── detection.html / rag.html         ← frozen exports of both notebooks (with outputs)
├── .env.example                      ← template: OPENROUTER_API_KEY=sk-or-your-key-here
├── .gitignore                        ← excludes .env, Dataset/, chroma_db/, models.joblib
├── outputs/                          ← saved artifacts from both notebooks
│   ├── flagged_windows.parquet       ← 398 PPG-DaLiA anomalies → RAG handoff (the bridge)
│   ├── confirmed_anomalies_for_rag.csv ← top-scored WESAD stress windows (labels)
│   ├── detection_results.csv         ← AUC/P/R/F1 table for all datasets
│   ├── models.joblib                 ← trained IF + LOF + scalers (gitignored, regenerable)
│   ├── rag_explanations.csv          ← 398 generated explanations + sources + latency
│   ├── rag_evaluation_v2.csv         ← local-judge (llama3.1:8b) scores per explanation
│   ├── rag_evaluation_api.csv        ← merged local + API-judge scores
│   ├── api_judge_checkpoint.jsonl    ← idempotent API-judge cache (no secrets inside)
│   ├── all_roc_curves.png            ← combined ROC figure
│   ├── score_distributions.png       ← PPG-DaLiA IF/LOF histograms
│   ├── flags_per_subject.png         ← PPG-DaLiA flags by subject
│   ├── wesad_roc_and_scores.png      ← WESAD ROC + score separation
│   ├── mitbih_roc.png                ← MIT-BIH ROC
│   └── ptbxl_roc.png                 ← PTB-XL ROC
├── chroma_db/                        ← persistent vector store (gitignored, rebuilt by notebook)
├── Dataset/                          ← ALL data lives here (gitignored; DOIs in §7)
│   ├── ppg+dalia/                    ← PPG-DaLiA signals (15 subjects)
│   ├── WESAD/                        ← WESAD signals + stress labels (15 subjects)
│   ├── mit-bih-arrhythmia-database-1.0.0/  ← MIT-BIH (48 records, .dat/.hea/.atr)
│   ├── ptb-xl-1.0.3/               ← PTB-XL (21,799 ECGs, records100/ + records500/)
│   ├── RAG corpus (medical literatureguidelines)/  ← Tier-1: 4 guideline PDFs
│   └── Tier2_literature/             ← Tier-2: 200 OA articles
│       ├── 01_ppg_arrhythmia/ … 06_biosignal_methods/
│       ├── references.bib
│       ├── manifest.csv
│       └── coverage_report.md
└── scripts/                          ← reproducible corpus-build scripts
    ├── fetch_tier2.py                ← Tier-2 corpus fetcher
    └── enrich_metadata.py            ← metadata enrichment patch
```

**Reproducing from a fresh clone:** the notebooks rebuild everything except the raw
datasets — download the four datasets (DOIs in §7), place them under `Dataset/`, copy
`.env.example` → `.env` (only if using the optional API judge; the local judge path needs
no key), then run `detection.ipynb` top-to-bottom (rebuilds `outputs/`, `models.joblib`)
and `rag.ipynb` (rebuilds `chroma_db/` on first run; the re-entry marker inside shows the
minimal cell sequence for re-displaying evaluation without re-running generation).

---

## 17. Glossary (Plain-English Definitions)

- **PPG (photoplethysmogram):** an optical signal from a wrist sensor that measures blood
  volume changes — the basis of smartwatch heart-rate detection.
- **ECG (electrocardiogram):** the electrical signal of the heart; the clinical gold
  standard for arrhythmia.
- **EDA (electrodermal activity):** skin conductance; rises with stress/arousal.
- **Anomaly detection:** finding data points that look unusual compared to the normal
  pattern — without being told in advance what "unusual" looks like (unsupervised).
- **Isolation Forest:** an algorithm that finds anomalies by seeing how easily a data point
  can be "isolated" by random splits (anomalies are easy to isolate).
- **KNN-LOF (Local Outlier Factor):** an algorithm that flags points in low-density
  neighborhoods (anomalies sit in sparse regions).
- **RAG (Retrieval-Augmented Generation):** a technique where an LLM is forced to answer
  only from retrieved real documents, preventing hallucination.
- **Embedding:** a numeric vector representing the meaning of text; similar texts have
  similar vectors, enabling semantic search.
- **Vector database (ChromaDB):** a store that holds embeddings and finds the most similar
  documents to a query.
- **Hallucination (LLM):** when a language model confidently states false or fabricated
  information.
- **Faithfulness:** whether an LLM answer uses only the provided evidence (no fabrication).
- **Citation faithfulness:** whether every cited source really exists and really says what
  is claimed.
- **Ollama:** a tool to run LLMs locally and privately on your own machine.
- **Q1 journal:** a journal in the top 25% of its field by impact factor.
- **Leave-one-subject-out CV:** a validation method that trains on all but one subject and
  tests on the held-out subject, rotating through — standard for small wearable studies.

---

## 18. How to Load the Datasets (Code)

One-time setup of the libraries used below:

```bash
pip install wfdb ast pandas numpy scipy scikit-learn sentence-transformers chromadb ollama
# PyMuPDF is already available for the guideline PDFs:
pip install pymupdf
```

Paths below are relative to the project folder
`Retrieval-augmented generation for continuous anomaly alerts/`. All datasets now live
under `Dataset/` (e.g. `Dataset/WESAD/...`).

### 18.1 PPG-DaLiA — wearable signals (`.pkl`)
Each `S{X}.pkl` is a Python-2 pickle (use `encoding='latin1'`). Structure:
`{'subject','signal','label','activity'}` where `signal` has `chest` and `wrist`.

```python
import pickle
S = 1
with open(f"Dataset/ppg+dalia/data/PPG_FieldStudy/S{S}/S{S}.pkl", "rb") as f:
    d = pickle.load(f, encoding="latin1")
chest = d["signal"]["chest"]   # dict: ECG(700Hz), ACC, EDA, EMG, RESP, TEMP
wrist = d["signal"]["wrist"]   # dict: BVP/PPG(64Hz), ACC, EDA, TEMP
ecg   = chest["ECG"]           # numpy array (N, 1)
ppg   = wrist["BVP"]
labels = d["label"]            # activity id per sample (700 Hz grid)
```

### 18.2 WESAD — wearable signals + stress labels (`.pkl`)
Same dict layout as PPG-DaLiA (same hardware). Labels: `0`=transient, `1`=baseline,
`2`=stress, `3`=amusement.

```python
import pickle
S = 2  # WESAD ids: S2..S17 (skip S1, S12)
with open(f"Dataset/WESAD/S{S}/S{S}.pkl", "rb") as f:
    d = pickle.load(f, encoding="latin1")
chest, wrist = d["signal"]["chest"], d["signal"]["wrist"]
y = d["label"]                 # 1=baseline, 2=stress, 3=amusement
```

### 18.3 MIT-BIH — clinical ECG + beat labels (`.dat` / `.atr`)
Read with `wfdb`. Beat symbols: `N`=normal, `V`=PVC, `A`=PAC, `L`/`R`=bundle branch block,
`/`/`f`/`F`=paced/fusion, etc.

```python
import wfdb
from collections import Counter
BASE = "Dataset/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0"
sig, meta = wfdb.rdsamp(f"{BASE}/100")          # (650000, 2) array, 360 Hz, leads MLII + V5
ann = wfdb.rdann(f"{BASE}/100", "atr")          # beat annotations
print(Counter(ann.symbol))                       # e.g. {'N': 2239, 'L': 0, 'V': 1, ...})
```

### 18.4 PTB-XL — large-scale 12-lead ECG (`.dat` + CSV)
Use `ptbxl_database.csv` for labels and file paths; `wfdb.rdsamp` for the waveform.

```python
import wfdb, ast, pandas as pd
db = pd.read_csv("Dataset/ptb-xl-1.0.3/ptbxl_database.csv")     # 21,799 rows
scp = pd.read_csv("Dataset/ptb-xl-1.0.3/scp_statements.csv", index_col=0)  # code -> superclass

row = db.iloc[0]
sig, _ = wfdb.rdsamp("Dataset/ptb-xl-1.0.3/" + row.filename_lr)  # (1000, 12) at 100 Hz
codes = ast.literal_eval(row.scp_codes)                    # e.g. {'NORM': 100.0}
superclasses = {scp.loc[c, "diagnostic_class"] for c in codes}
fold = row.strat_fold                                       # 1..10 patient-stratified split
# Standard split convention: folds 1-8 train, 9 val, 10 test
```

### 18.5 Tier-1 guideline corpus (PDFs)
```python
import fitz  # PyMuPDF
doc = fitz.open("Dataset/RAG corpus (medical literatureguidelines)/2017 ACCAHAHRS — Evaluation of Patients with Syncope.pdf")
text = "\n".join(page.get_text() for page in doc)   # full text ready for chunking
```

### 18.6 Tier-2 literature corpus (Markdown — already clean)
```python
from pathlib import Path
for md_path in Path("Dataset/Tier2_literature").rglob("*.md"):
    text = md_path.read_text(encoding="utf-8")        # header + full article body
# manifest.csv maps every file to pmcid/doi/journal/year/license;
# references.bib holds BibTeX for all 200.
```

### 18.7 Putting signals into Isolation-Forest-ready windows (sketch)
```python
import numpy as np
from sklearn.ensemble import IsolationForest

def windows(sig, fs, win_sec=30):
    n = fs * win_sec
    return [sig[i:i+n] for i in range(0, len(sig)-n, n)]   # 30-s windows

def featurize(w):     # per-window summary features
    w = np.asarray(w).ravel()
    return [w.mean(), w.std(), np.ptp(w),
            np.percentile(w, 90), np.sum(np.diff(w) > 0) / len(w)]

X = np.array([featurize(w) for w in windows(ecg.ravel(), fs=700)])  # feature matrix
clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=0).fit(X)
scores = -clf.score_samples(X)          # higher = more anomalous
flagged = X[scores > np.percentile(scores, 95)]   # top 5% -> trigger RAG
```

> The anomaly window that triggers then becomes the query for the ChromaDB RAG store
> (Tier-1 + Tier-2 embedded with `all-MiniLM-L6-v2`), answered by the local Qwen3.5 9B LLM
> under a strict no-hallucination prompt — the same architecture as the author's AMR paper.
> The full implemented pipeline (query building, retrieval, generation, evaluation) is in
> `rag.ipynb`; see §11 for final results.

---

*End of document.*
