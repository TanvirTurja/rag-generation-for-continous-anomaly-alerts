# Retrieval-Augmented Generation for Continuous Anomaly Alerts — v2

> Plain-English description of the project after the **v2 revision** (branch `fix/v2`).
> The v1 draft claimed strong results; a protocol audit showed three defects, and the
> corrected evaluation — pre-registered in `THRESHOLDS.md` before any v2 run — tells a
> different, honest story. This file summarizes the current state.
>
> **Author:** Md Tanvir Hasan Turja · Revised August 2026

---

## 1. One-sentence summary

A local pipeline that watches continuous wearable biosignals, flags unusual 30-second
windows with unsupervised detectors, and explains every flag in plain language using
only retrieved peer-reviewed literature — now evaluated under contamination-hardened
protocols, on labeled clinical events, with validated judges.

## 2. What the v2 revision fixed (and what it revealed)

| v1 claim | v2 reality |
|---|---|
| MIT-BIH LOF AUC 0.899 | **0.502 — chance** under the standard inter-patient protocol (IF holds 0.670) |
| WESAD LOF 0.910 | 0.827 [0.800–0.852] under leave-one-subject-out |
| "100% citation accuracy" | raw text: **99.01%** — 12 fabricated citations in 9 of 546 explanations |
| "Zero hallucination verdicts" | v1's judge detects **0 of 100** injected corruptions — a null instrument; the validated judge finds 2 hallucination verdicts and ~44% "slightly extrapolated" |
| (never evaluated on labeled events) | explanations name the true condition for **94%** of stress events but **12%** of arrhythmic windows and **6%** of pathology ECGs — attributing 42–56% of true pathology to "artifact" |
| 4 guidelines "anchoring" the corpus | guideline content reached 6.5% of v1 alerts; corpus expanded (EHRA 2022 digital-devices guide + 2023 ACC/AHA AF guideline) → 17.6% |

Additional honest findings: a trivial autoencoder beats both classic detectors on
WESAD (0.855); neither local nor API judges are cross-run stable (gemma4 means
2.21–2.69 across identical runs; DeepSeek FP 7–31%); 47.7% of atomic claims are
unverifiable from the retrieved context.

## 3. Repository layout (v2)

```
detection_v2.ipynb      ← code deliverable 1: clean-protocol detection (executed, 99 cells)
rag_v2.ipynb            ← code deliverable 2: full RAG pipeline + evaluation (executed, 123 cells)
draft_paper/            ← paper.md (v2), VERIFICATION.md (59/59 claims artifact-traced),
                           verify_claims_v2.py, references (+ v2 additions), figures_v2/
THRESHOLDS.md           ← pre-registered protocols (timestamped before any v2 run)
SEARCH_PROTOCOL.md      ← documented literature search behind the revised related work
scripts/fetch_tier2.py  ← Tier-2 corpus builder (Europe PMC) — required for a cold rebuild
scripts/enrich_metadata.py
outputs_v2/             ← all v2 artifacts (caches, generations, judge benchmarks, API checkpoint)
outputs_v1_archive/     ← preserved v1 outputs (used for the before/after comparisons)
chroma_db_v2/           ← v2 vector store (206 docs / 5,045 chunks; rebuildable)
clinician_eval/         ← ready-to-run human-evaluation kit (60-item sample pre-drawn)
Dataset/                ← raw data (gitignored; DOIs in the paper)
archive_v1/             ← original v1 notebooks/manuscript (see its README)
```

## 4. Reproducing

1. Place the four public datasets under `Dataset/` (DOIs in the paper).
2. Run `detection_v2.ipynb` top-to-bottom (cold ≈ 15 min; cached, seconds).
3. Run `rag_v2.ipynb` top-to-bottom (cached ≈ 10–15 min; cold rebuild ≈ 3 h GPU +
   OpenRouter key; **judge cells are cache-only — re-execution makes zero API calls**).
4. `python draft_paper/verify_claims_v2.py` → every paper number recomputed from artifacts.

Requirements: Python 3.13 (wfdb, sklearn, torch+cuda, sentence-transformers, chromadb,
ollama), Ollama with qwen3.5:9b / gemma4:e4b / llama3.1:8b, and an `OPENROUTER_API_KEY`
in `.env` (read at init; never called on re-runs).

## 5. What remains before submission

1. Clinician ratings — **moved to future work** (cannot recruit raters now). The
   kit in `clinician_eval/` is complete and released for any clinical group to run;
   the paper explicitly claims no clinical adequacy.
2. Four references in `draft_paper/references_v2_additions.bib` are marked
   `verify-before-submit` (indirect DOIs) — recheck at submission.
3. Optional: re-sign commits (GPG pinentry timed out mid-revision; later commits
   used `--no-gpg-sign`).

## 6. Honesty summary

The systems idea survived; the original numbers did not. The paper now leads with
protocol sensitivity: how intra-patient evaluation, test-derived thresholds, and
unvalidated judges together manufactured a publishable-looking result that the
community-standard protocols dissolve — and what an honest alert-explanation
system actually achieves today.
