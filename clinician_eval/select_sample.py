"""
select_sample.py — Build the stratified 60-item clinician-review sample (W7).

Reads outputs_v2/generation_v2.jsonl and writes:
  clinician_eval/sample_60.csv   (what raters see: id, query, explanation, sources)
  clinician_eval/sample_key.csv  (hidden mapping id -> group/true_label, for analysis)

Stratification: 20 labeled true events, 20 PPG-DaLiA in-the-wild flags,
10 word-cap ablation items, 10 random remaining.
"""
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "outputs_v2" / "generation_v2.jsonl"
OUT = Path(__file__).resolve().parent


def section(txt, start, end):
    m = re.search(rf"{start}:\s*(.*?)(?={end}:|$)", str(txt), re.S)
    return m.group(1).strip() if m else ""


def main():
    rng = random.Random(17)
    rows = [json.loads(l) for l in open(GEN, encoding="utf-8")]
    labeled = [r for r in rows if r["subgroup"] == "labeled"]
    dalia = [r for r in rows if r["group"] == "dalia" and r["subgroup"] == "main"]
    wordcap = [r for r in rows if r["subgroup"] == "wordcap"]

    # stratify labeled across groups/classes
    by_class = {}
    for r in labeled:
        by_class.setdefault((r["group"], r["true_label"]), []).append(r)
    labeled_pick = []
    classes = sorted(by_class.keys())
    while len(labeled_pick) < 20 and any(by_class.values()):
        for c in classes:
            if by_class[c] and len(labeled_pick) < 20:
                labeled_pick.append(by_class[c].pop(rng.randrange(len(by_class[c]))))
    dalia_pick = rng.sample(dalia, 20)
    wordcap_pick = rng.sample(wordcap, min(10, len(wordcap)))
    rest_pool = [r for r in dalia if r not in dalia_pick]
    rest_pick = rng.sample(rest_pool, min(10, len(rest_pool)))

    items = ([("labeled", r) for r in labeled_pick] +
             [("dalia", r) for r in dalia_pick] +
             [("wordcap", r) for r in wordcap_pick] +
             [("dalia", r) for r in rest_pick])
    rng.shuffle(items)

    vis, key = [], []
    for i, (g, r) in enumerate(items):
        vis.append({
            "item_id": f"A{i+1:02d}",
            "query": r["query"],
            "explanation": r["explanation"],
            "retrieved_sources": "\n".join(r["sources"]),
        })
        key.append({"item_id": f"A{i+1:02d}", "group": g,
                    "true_label": r.get("true_label"), "orig_key": r["key"],
                    "generator_model": r["model"], "prompt_cap": r["prompt"]})
    pd.DataFrame(vis).to_csv(OUT / "sample_60.csv", index=False)
    pd.DataFrame(key).to_csv(OUT / "sample_key.csv", index=False)
    print(f"wrote {len(vis)} items -> sample_60.csv / sample_key.csv")


if __name__ == "__main__":
    main()
