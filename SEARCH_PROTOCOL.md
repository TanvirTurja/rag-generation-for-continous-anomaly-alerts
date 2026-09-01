# Literature Search Protocol — v2 Related Work (fix/v2 W8)

**Date executed:** 2026-08-28
**Engine:** Tavily web search (built-in Z.ai search unavailable: weekly quota exhausted)
**Recorder:** fix/v2 revision cycle. All queries verbatim below; result triage recorded inline.

## Scope and rationale

The v1 manuscript's priority claim ("no published system connects unsupervised
biosignal anomaly detection with LLM explanation") rested on an undocumented
"literature search, August 2026". This protocol documents the search behind the
revised, narrower claim, and populates the previously missing competing-paradigm
literature in six areas.

## Queries (verbatim)

| # | Area | Query string |
|---|---|---|
| Q1 | LLM ECG interpretation / report generation | `large language model ECG report generation interpretation 2024 2025 journal paper DOI` |
| Q2 | Clinical RAG / hallucination | `retrieval augmented generation clinical medicine survey hallucination medical question answering DOI 2024 2025` |
| Q3 | ICU false alarms / alert fatigue | `ICU false alarm reduction alert fatigue physionet challenge 2015 ECG arrhythmia suppression review DOI` |
| Q4 | RAG evaluation / faithfulness metrics | `RAGAS automated evaluation RAG faithfulness FActScore atomic factuality evaluation LLM arXiv DOI` |
| Q5 | LLM-as-judge biases | `large language models as judges bias survey leniency position self-preference verbiage NeurIPS ACL 2023 2024 arXiv` |
| Q6 | Wearable AF detection at scale | `Apple Heart Study smartwatch photoplethysmography atrial fibrillation detection Perez NEJM 2019 Lubitz Fitbit Circulation 2022 DOI` |

Inclusion criteria: peer-reviewed journal/conference paper or influential preprint;
2015–2026 (classic alert-fatigue and heartbeat-classification anchors excepted);
English. Exclusion: blogs, news coverage, non-peer-reviewed industry pages.

## Findings that change the paper's claims

1. **Q1 refutes the v1 framing of the gap.** LLM-based ECG interpretation and
   report generation is an active field, including surveys (Ansari et al. 2025),
   ECG-conditioned LLMs (ECG-Chat, ECG-LM), and RAG-based ECG report generation
   reviewed in 2026. The v2 paper therefore claims a narrower, defensible gap:
   *alert-triggered* explanation of *unsupervised detector flags* on wearables
   (label-free at inference), not "first LLM + ECG" or "first RAG + ECG".
2. **Q3 shows the motivating problem has a decade of prior art** (PhysioNet/CinC
   2015 challenge; alarm-fatigue reviews) that v1 cited zero times.
3. **Q4/Q5 provide the evaluation standards** (RAGAS, FActScore-style atomic
   verification, citation-generation work; judge bias literature incl.
   self-preference and leniency) that justify the v2 judge-validation protocol.
4. **Q6 grounds the alert space** (Apple Heart Study, Fitbit Heart Study) and the
   two added Tier-1 documents (EHRA 2022 digital-devices guide; 2023 ACC/AHA AF
   guideline) in the deployment literature.

## Compiled references

Merged into `draft_paper/references.bib` (section 'v2 additions'). Verification status per entry:
`verified` = DOI/URL resolved via search results or canonical knowledge;
`verify-before-submit` = DOI from secondary source, recheck at submission
(marked with a comment in the .bib).

Notable verification notes:
- de Chazal et al. 2004 is included to cite the DS1/DS2 inter-patient MIT-BIH
  protocol standard that the v2 detection evaluation adopts.
- Wataoka et al. 2024 (self-preference bias) and Panickssery et al. 2024
  (LLM evaluators favor own generations) anchor the judge-independence design.
- The "comprehensive review on ECG report generation" (Elsevier 2026, PII
  S235291482600047X) DOI requires final verification (page was cookie-gated).
