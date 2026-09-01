"""Verify every quantitative claim in the paper draft against output artifacts."""

import pandas as pd, numpy as np, json, ast, re
from collections import Counter

O = "outputs/"

# --- RAG explanations ---
r = pd.read_csv(O + "rag_explanations.csv")
print("EXPLANATIONS rows:", len(r))
print(
    "  latency mean/sd/median:",
    round(r.latency_sec.mean(), 2),
    round(r.latency_sec.std(), 2),
    round(r.latency_sec.median(), 2),
)

allsrc, n_cit = [], 0
for s, e in zip(r.sources, r.explanation):
    allsrc += [x.split("_")[0] for x in ast.literal_eval(s)]
    n_cit += len([c for c in re.findall(r"\[([A-Za-z0-9_]+)\]", e) if len(c) > 5])
c = Counter(allsrc)
print(
    "  unique source docs:",
    len(c),
    "| total retrieval slots:",
    len(allsrc),
    "| top-1 share %:",
    round(100 * c.most_common(1)[0][1] / len(allsrc), 1),
)
print("  total citations found:", n_cit, "| avg/explanation:", round(n_cit / len(r), 1))
print("  top-3 sources:", c.most_common(3))

# --- Local judge ---
v = pd.read_csv(O + "rag_evaluation_v2.csv")
print(
    "LOCAL JUDGE cols:",
    [x for x in v.columns if "faith" in x or "relev" in x or "compl" in x],
)
print(
    "  means:",
    [round(v[k].mean(), 2) for k in ["faithfulness", "relevance", "completeness"]],
)
print(
    "  rows with any score=0 (parse fail):",
    int((v[["faithfulness", "relevance", "completeness"]] == 0).any(axis=1).sum()),
)
print("  faithfulness==1 (hallucination):", int((v.faithfulness == 1).sum()))
print("  faithfulness value counts:", v.faithfulness.value_counts().to_dict())
print("  completeness value counts:", v.completeness.value_counts().to_dict())

# --- API judge + agreement ---
a = pd.read_csv(O + "rag_evaluation_api.csv")
print(
    "API JUDGE means:",
    [
        round(a[k].mean(), 2)
        for k in ["api_faithfulness", "api_relevance", "api_completeness"]
    ],
)
for ax in ["faithfulness", "relevance", "completeness"]:
    x, y = a["api_" + ax], a[ax]
    m = (x > 0) & (y > 0)
    print(
        f"  {ax}: raw agree {round((x[m] == y[m]).mean() * 100, 1)}% | within-1 {round(((x[m] - y[m]).abs() <= 1).mean() * 100, 1)}% | masked-out rows {int((~m).sum())}"
    )
print(
    "  api hallucination flags:",
    int((a.api_faithfulness == 1).sum()),
    "| api any-axis=1:",
    int(
        (a[["api_faithfulness", "api_relevance", "api_completeness"]] == 1)
        .any(axis=1)
        .sum()
    ),
)

# --- Checkpoint ---
recs = [
    json.loads(l)
    for l in open(O + "api_judge_checkpoint.jsonl", encoding="utf-8")
    if l.strip()
]
good = [x for x in recs if 0 not in x.get("scores", {}).values()]
print(
    "CHECKPOINT lines:",
    len(recs),
    "| valid:",
    len(good),
    "| unique rows:",
    len({x["i"] for x in good}),
    "| dropped garbage:",
    len(recs) - len(good),
)

# --- Flagged windows ---
f = pd.read_parquet(O + "flagged_windows.parquet")
print("FLAGGED parquet:", f.shape, "| subjects:", sorted(f.subject.unique()))
print(
    "  flag_if:",
    int(f.flag_if.sum()),
    "| flag_lof:",
    int(f.flag_lof.sum()),
    "| flag_both:",
    int(f.flag_both.sum()),
)

# --- Confirmed anomalies ---
conf = pd.read_csv(O + "confirmed_anomalies_for_rag.csv")
print(
    "CONFIRMED CSV rows:",
    len(conf),
    "| dataset:",
    conf.dataset.unique(),
    "| label:",
    conf.true_label.unique(),
)

# --- Detection results ---
d = pd.read_csv(O + "detection_results.csv")
print("DETECTION CSV:")
print(d.to_string(index=False))
