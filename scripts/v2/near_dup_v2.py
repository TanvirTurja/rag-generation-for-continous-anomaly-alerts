"""near_dup_v2.py — Near-duplicate analysis on the v2 explanations (before/after
query-fix + corpus expansion comparison). Writes outputs_v2/near_duplicate_v2.json."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "outputs_v2" / "generation_v2.jsonl"


def section(txt, start, end):
    m = re.search(rf"{start}:\s*(.*?)(?={end}:|$)", str(txt), re.S)
    return m.group(1).strip() if m else ""


def main():
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    rows = [json.loads(l) for l in open(GEN, encoding="utf-8")]
    dalia = [r for r in rows if r["group"] == "dalia" and r["subgroup"] == "main"]
    texts = [section(r["explanation"], "DETECTED", "EVIDENCE") + "\n" +
             section(r["explanation"], "EVIDENCE", "RECOMMENDATION") for r in dalia]
    emb = emb_model.encode(texts, normalize_embeddings=True)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    nn = sim.max(axis=1)
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
    # corpus utilization in v2
    used = {s for r in dalia for s in r["sources"]}
    tier1_alerts = sum(1 for r in dalia if any(t == "tier1" for t in r["tiers"]))
    out = {
        "n_texts": len(texts),
        "n_clusters_at_0.9": int(cid),
        "mean_nn_cos": round(float(np.mean(nn)), 4),
        "pct_rows_with_nn_gt_0.8": round(float((nn > 0.8).mean() * 100), 2),
        "pct_rows_with_nn_gt_0.9": round(float((nn > 0.9).mean() * 100), 2),
        "unique_docs_used": len(used),
        "alerts_with_tier1": tier1_alerts,
        "tier1_alert_rate": round(tier1_alerts / len(dalia), 4),
        "v1_comparison": {"n_clusters_at_0.9": 173, "mean_nn_cos": 0.9356,
                          "unique_docs_used": 53, "tier1_alert_rate": 0.0653},
    }
    (ROOT / "outputs_v2" / "near_duplicate_v2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
