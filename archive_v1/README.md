# archive_v1 — original pre-revision artifacts

These are the v1 files superseded by the v2 revision (branch `fix/v2`):

| File | What it was |
|---|---|
| `detection.ipynb` / `detection.html` | original detection notebook (contaminated protocols) + frozen export |
| `rag.ipynb` / `rag.html` | original RAG notebook (v1 query semantics, unvalidated judges) + frozen export |
| `DATASET_CARD.md` | v1 signal-dataset reference card |
| `paper_v1.md` | the original manuscript draft (kept for diffing against `draft_paper/paper.md`) |

Successors: `detection_v2.ipynb`, `rag_v2.ipynb`, `draft_paper/paper.md`,
`draft_paper/VERIFICATION.md`. The v1 *outputs* (flagged windows, explanations,
judge scores) that the v2 evaluation compares against are in `outputs_v1_archive/`.
The v1 ChromaDB was a rebuildable cache and has been deleted (`rag.ipynb` rebuilds it).
| `figures/` | v1 manuscript figures (paper v2 uses `figures_v2/`) |
| `verify_claims.py` | v1 claim checker (superseded by `verify_claims_v2.py`) |
| `references_tier2_corpus.bib` | duplicate of `Dataset/Tier2_literature/references.bib` |

Everything here is also preserved in git history.
