"""
rag_analysis_v1.py — Analysis-side fixes on the ARCHIVED v1 artifacts (fix/v2 W2).

Produces, in outputs_v2/rag_analysis_v1/:
  ablation_2x2.csv            query scheme x diversity-constraint retrieval ablation
  near_duplicates.csv         per-alert nearest-neighbour similarity + cluster ids
  guideline_utilization.json  tier-1 vs tier-2 retrieval/citation utilization + coverage
  retrieval_latency.json      embed+search timing over the 398 v1 queries
  judge_latency.json          local-judge latency re-measurement (20 calls, llama3.1:8b)
  prerepair_note.json         documents that the v1 '99.3% pre-repair' figure is NOT
                              reconstructable from any preserved artifact (raw
                              pre-canonicalization text was never saved); v2 pipeline
                              saves raw text so this number becomes artifact-backed.

Read-only with respect to the v1 ChromaDB and archived CSVs.
"""

import ast
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "outputs_v1_archive"
OUT = ROOT / "outputs_v2" / "rag_analysis_v1"
OUT.mkdir(parents=True, exist_ok=True)

TOP_K = 5
POOL = 20

NAIVE_TEMPLATE = ("Anomaly detected in wearable biosignal monitoring. "
                  "What could explain this anomaly, and what does the medical "
                  "literature recommend?")
# NOTE: the exact dev-history naive template string from v1 was not preserved
# (VERIFICATION.md D4). This fixed template is the reconstruction used for the
# artifact; the manuscript reports it as a reconstructed baseline.

CIT_RE = re.compile(r"\[(PMC\d+)\]")


def load_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    return client.get_collection("medical_corpus")


def retrieve(collection, embedder, query, top_k=TOP_K, pool=POOL, max_per_source=1):
    qe = embedder.encode([query])
    res = collection.query(query_embeddings=qe.tolist(), n_results=pool,
                           include=["documents", "metadatas", "distances"])
    metas = res["metadatas"][0]
    if max_per_source is None:
        picked = list(range(min(top_k, len(metas))))
    else:
        picked, counts = [], {}
        for i, m in enumerate(metas):
            c = counts.get(m["source"], 0)
            if c >= max_per_source:
                continue
            picked.append(i)
            counts[m["source"]] = c + 1
            if len(picked) == top_k:
                break
        if len(picked) < top_k:
            for i in range(len(metas)):
                if i not in picked:
                    picked.append(i)
                    if len(picked) == top_k:
                        break
    return [metas[i]["source"] for i in picked], [metas[i].get("tier", "") for i in picked]


def ablation_metrics(rows):
    """rows: list of (alert_id, [sources], [tiers])"""
    all_src = [s for _, srcs, _ in rows for s in srcs]
    n_slots = len(all_src)
    vc = pd.Series(all_src).value_counts()
    tier1_alerts = sum(1 for _, _, tiers in rows if any(t == "tier1" for t in tiers))
    return {
        "unique_docs": int(vc.shape[0]),
        "top_source_share": round(float(vc.iloc[0] / n_slots), 4) if n_slots else None,
        "top_source": str(vc.index[0]) if vc.shape[0] else None,
        "slots": int(n_slots),
        "alerts_with_tier1": int(tier1_alerts),
        "tier1_alert_rate": round(tier1_alerts / max(1, len(rows)), 4),
    }


def main():
    df = pd.read_csv(ARCH / "rag_explanations.csv")
    df["sources_list"] = df["sources"].apply(ast.literal_eval)
    print(f"loaded {len(df)} archived explanations", flush=True)

    # ------------------------------------------------ 0. pre-repair note
    note = {
        "claim": "v1 paper: raw citation accuracy 99.3% (9 near-miss rows) before canonicalization",
        "reconstructable_from_artifacts": False,
        "reason": ("The canonicalizer ran in-line during generation and only post-repair text "
                   "was persisted to rag_explanations.csv; no raw pre-repair output exists in "
                   "outputs_v1_archive. Re-running the checker on archived text yields 0 repairs "
                   "(idempotent) and cannot recover the pre-repair figure."),
        "action": ("v2 pipeline (rag_pipeline_v2.py) saves raw pre-canonicalization text, so the "
                   "pre-repair number becomes deterministic and artifact-backed for the new run. "
                   "The 99.3% v1 figure is reported in the v2 manuscript only as dev-history, "
                   "explicitly flagged, or dropped."),
    }
    (OUT / "prerepair_note.json").write_text(json.dumps(note, indent=2))

    # ------------------------------------------------ 1. corpus + embedder
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    collection = load_collection()
    n_docs_corpus = len({m["source"] for m in collection.get(include=["metadatas"])["metadatas"]})
    print(f"corpus: {collection.count()} chunks, {n_docs_corpus} unique source docs", flush=True)

    # ------------------------------------------------ 2. 2x2 ablation + latency
    latencies = []
    conds = {
        "naive_diversity_on": [], "naive_diversity_off": [],
        "deviation_diversity_off": [],
    }
    for idx, row in df.iterrows():
        t0 = time.perf_counter()
        srcs, tiers = retrieve(collection, embedder, NAIVE_TEMPLATE, max_per_source=1)
        conds["naive_diversity_on"].append((idx, srcs, tiers))
        t1 = time.perf_counter()
        srcs, tiers = retrieve(collection, embedder, NAIVE_TEMPLATE, max_per_source=None)
        conds["naive_diversity_off"].append((idx, srcs, tiers))
        t2 = time.perf_counter()
        srcs, tiers = retrieve(collection, embedder, row["query"], max_per_source=None)
        conds["deviation_diversity_off"].append((idx, srcs, tiers))
        t3 = time.perf_counter()
        latencies.append(t3 - t0)
        if idx % 50 == 0:
            print(f"  ablation {idx}/398", flush=True)

    # actual v1 condition: archived sources (deviation-aware + diversity ON)
    v1_rows = [(i, srcs, ["tier2" if s.startswith("PMC") else "tier1" for s in srcs])
               for i, srcs in zip(df.index, df["sources_list"])]
    rows = {"deviation_diversity_on_ARCHIVED": ablation_metrics(v1_rows)}
    for name, r in conds.items():
        rows[name] = ablation_metrics(r)
    pd.DataFrame(rows).T.to_csv(OUT / "ablation_2x2.csv")
    lat = np.array(latencies) * 3  # 3 retrievals timed together; per-retrieval = /3
    lat_per = lat / 3
    (OUT / "retrieval_latency.json").write_text(json.dumps({
        "n_queries": int(len(lat_per)),
        "mean_s": round(float(np.mean(lat_per)), 4),
        "p50_s": round(float(np.percentile(lat_per, 50)), 4),
        "p95_s": round(float(np.percentile(lat_per, 95)), 4),
    }, indent=2))
    print("ablation:", json.dumps(rows, indent=2, default=str)[:800], flush=True)

    # ------------------------------------------------ 3. near-duplicate analysis
    def section(txt, start, end):
        m = re.search(rf"{start}:\s*(.*?)(?={end}:|$)", str(txt), re.S)
        return m.group(1).strip() if m else ""

    texts = [section(t, "DETECTED", "EVIDENCE") + "\n" +
             section(t, "EVIDENCE", "RECOMMENDATION") for t in df["explanation"]]
    emb = embedder.encode(texts, normalize_embeddings=True)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    nn_sim = sim.max(axis=1)
    # greedy clustering at cos>0.9
    unassigned = set(range(len(texts)))
    cluster = np.full(len(texts), -1)
    cid = 0
    while unassigned:
        seed = min(unassigned)
        members = [j for j in unassigned if sim[seed, j] > 0.9] + [seed]
        for j in members:
            cluster[j] = cid
            unassigned.discard(j)
        cid += 1
    nd = pd.DataFrame({
        "row": df.index, "subject": df["subject"],
        "nearest_neighbour_cos": np.round(nn_sim, 4),
        "cluster_09": cluster,
    })
    nd.to_csv(OUT / "near_duplicates.csv", index=False)
    nd_summary = {
        "n_texts": int(len(texts)),
        "n_clusters_at_0.9": int(cid),
        "mean_nn_cos": round(float(np.mean(nn_sim)), 4),
        "pct_rows_with_nn_gt_0.8": round(float((nn_sim > 0.8).mean() * 100), 2),
        "pct_rows_with_nn_gt_0.9": round(float((nn_sim > 0.9).mean() * 100), 2),
        "n_unique_detected_strings": int(len({t for t in texts})),
    }
    (OUT / "near_duplicate_summary.json").write_text(json.dumps(nd_summary, indent=2))
    print("near-dup:", nd_summary, flush=True)

    # ------------------------------------------------ 4. guideline utilization
    tier1_alerts = sum(1 for srcs in df["sources_list"] if any(not s.startswith("PMC") for s in srcs))
    cited = [c for txt in df["explanation"] for c in CIT_RE.findall(str(txt))]
    used_docs = {s for srcs in df["sources_list"] for s in srcs}
    util = {
        "corpus_unique_docs": int(n_docs_corpus),
        "unique_docs_used": int(len(used_docs)),
        "corpus_utilization": round(len(used_docs) / n_docs_corpus, 4),
        "alerts_with_tier1_source": int(tier1_alerts),
        "tier1_alert_rate": round(tier1_alerts / len(df), 4),
        "n_citations_total": int(len(cited)),
        "citations_to_tier1": 0,
        "citations_to_tier1_note": ("Tier-1 guideline chunks carry no PMC identifier, so the "
                                    "citation format [PMC...] cannot reference them by design; "
                                    "tier-1 usage must be measured at retrieval, not citation."),
        "retrieval_2x2": rows,
    }
    (OUT / "guideline_utilization.json").write_text(json.dumps(util, indent=2, default=str))
    print("utilization:", {k: v for k, v in util.items() if k != "retrieval_2x2"}, flush=True)

    # ------------------------------------------------ 5. local judge latency (20 calls)
    import ollama
    JUDGE_PROMPT = open(ROOT / "scripts" / "v2" / "judge_prompt_v1.txt", encoding="utf-8").read()
    sample_rows = df.sample(20, random_state=42)
    times = []
    for _, row in sample_rows.iterrows():
        srcs, tiers = retrieve(collection, embedder, row["query"], max_per_source=1)
        res = collection.query(query_embeddings=embedder.encode([row["query"]]).tolist(),
                               n_results=POOL, include=["documents", "metadatas"])
        # rebuild full-chunk sources_text exactly as v1 evaluation did
        metas, docs = res["metadatas"][0], res["documents"][0]
        by_source = {}
        for m, d in zip(metas, docs):
            by_source.setdefault(m["source"], d)
        src_set = set(row["sources_list"])
        sources_text = "\n\n".join(f"[{s}]\n{by_source[s]}" for s in row["sources_list"] if s in by_source)
        t0 = time.perf_counter()
        ollama.chat(model="llama3.1:8b",
                    messages=[{"role": "system", "content": JUDGE_PROMPT},
                              {"role": "user", "content":
                               f"QUERY: {row['query']}\nSOURCES:\n{sources_text}\nEXPLANATION:\n{row['explanation']}"}],
                    options={"temperature": 0.1, "num_predict": 300, "num_ctx": 10000, "num_gpu": 99})
        times.append(time.perf_counter() - t0)
    (OUT / "judge_latency.json").write_text(json.dumps({
        "model": "llama3.1:8b", "n": len(times),
        "mean_s": round(float(np.mean(times)), 2),
        "sd_s": round(float(np.std(times)), 2),
    }, indent=2))
    print(f"judge latency measured: mean {np.mean(times):.2f}s over {len(times)} calls", flush=True)

    print("W2 done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
