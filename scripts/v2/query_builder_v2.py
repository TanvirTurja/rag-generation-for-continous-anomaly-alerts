"""
query_builder_v2.py — Fixed deviation-aware query construction (fix/v2 W4).

v1 flaw: z-scores were computed across the 398 flagged windows themselves, so
"elevated/reduced" was measured against other anomalies (wrong reference class),
top-|z| could be a trivial 0.5, the character phrase ("abrupt isolated spike")
was a template keyed to detector identity, and the batch cross-subject
computation contradicted the online-deployment claim.

v2 fixes:
  1. Reference = each subject's NON-FLAGGED windows (the normal population the
     detectors learned). Implementable online via running stats; no cross-subject
     dependency.
  2. Character phrase states detector facts + evidence-tied shape description
     (kurtosis/peak-to-peak z of the named channels), not causal templates.
  3. Honesty guard: actual z values printed; "mildly unusual" wording when
     |z| < 1; no "confirmed" language.

Outputs:
  outputs_v2/alerts_queries_v2.csv          398 alerts with query_v2 + per-channel z
  outputs_v2/alerts_retrieval_v2.jsonl      per-alert top-5 diverse chunks from chroma_db_v2
  outputs_v2/wesad_zscore_sanity.json       WESAD stress-vs-baseline separation of the new metric
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v2"))
from detection_v2 import get_cache, WESAD_SUBJECTS, FEATURE_NAMES  # noqa: E402

OUT = ROOT / "outputs_v2"
CHANNELS = ["ecg", "resp", "bvp", "wrist_eda", "wrist_temp"]

TOPIC_PHRASES = {
    "ecg": "electrocardiogram rhythm irregularity heart rate variability arrhythmia ectopic beats atrial fibrillation",
    "resp": "respiration rate breathing pattern tachypnea bradypnea ventilation",
    "bvp": "photoplethysmography pulse waveform amplitude perfusion signal quality motion artifact atrial fibrillation screening",
    "wrist_eda": "electrodermal activity skin conductance sympathetic stress arousal sweat response",
    "wrist_temp": "skin temperature thermal perfusion vasomotor ambient temperature sensor effects",
}


def subject_reference(df_all, subject):
    """Running-stats-friendly reference: this subject's non-flagged windows."""
    sub = df_all[df_all["subject"] == subject]
    normal = sub[(~sub["flag_if"]) & (~sub["flag_lof"])]
    ref = {}
    for ch in CHANNELS:
        for feat in ["mean", "kurt", "ptp"]:
            col = f"{ch}__{feat}"
            ref[col] = (normal[col].mean(), normal[col].std(ddof=0) or 1e-9)
    return ref


def build_query_v2(row, ref):
    """Evidence-tied, correct-reference query for one flagged window."""
    zs, kzs, pzs = {}, {}, {}
    for ch in CHANNELS:
        m, s = ref[f"{ch}__mean"]
        zs[ch] = (row[f"{ch}__mean"] - m) / s
        km, ks = ref[f"{ch}__kurt"]
        kzs[ch] = (row[f"{ch}__kurt"] - km) / ks
        pm, ps = ref[f"{ch}__ptp"]
        pzs[ch] = (row[f"{ch}__ptp"] - pm) / ps

    top2 = sorted(CHANNELS, key=lambda c: -abs(zs[c]))[:2]
    both = bool(row["flag_if"]) and bool(row["flag_lof"])

    # factual detector statement
    if both:
        parts = ["Biosignal window flagged by both anomaly detectors (Isolation Forest and LOF)."]
    elif row["flag_if"]:
        parts = ["Biosignal window flagged by the Isolation Forest anomaly detector."]
    else:
        parts = ["Biosignal window flagged by the LOF anomaly detector."]

    # evidence-tied shape description from the named channels' kurtosis/ptp z
    shape_z = max(max(kzs[c], pzs[c]) for c in top2)
    shift_z = max(abs(zs[c]) for c in top2)
    if shape_z > 1.5:
        parts.append("Deviating channels show an abrupt, high-amplitude pattern "
                     "(elevated kurtosis/peak-to-peak vs this subject's baseline).")
    elif shift_z > 1.5:
        parts.append("Deviating channels show a sustained level shift from this subject's baseline.")
    elif shift_z > 1.0:
        parts.append("Deviating channels show a moderate shift from this subject's baseline.")
    else:
        parts.append("Deviating channels are only mildly unusual vs this subject's baseline.")

    for c in top2:
        direction = "elevated" if zs[c] > 0 else "reduced"
        parts.append(f"{direction} {c} (z={zs[c]:+.1f} vs subject baseline, mean={row[f'{c}__mean']:.2f})")
    parts.append("Relevant topics: " + " ".join(TOPIC_PHRASES[c] for c in top2) + ".")
    others = ", ".join(f"{c} mean={row[f'{c}__mean']:.2f}" for c in CHANNELS if c not in top2)
    parts.append(f"Other readings: {others}.")
    zrow = {f"z_{c}": round(zs[c], 3) for c in CHANNELS}
    zrow["top2"] = ",".join(top2)
    return " ".join(parts), zrow


def main():
    # ---- 1. full PPG-DaLiA feature matrix (all 4,308 windows) + flagged set
    cache = get_cache("ppgdalia")
    X_all, subj_all = cache["X"], cache["subject"]
    cols = [f"{ch}__{fn}" for ch in CHANNELS for fn in FEATURE_NAMES]
    df_all = pd.DataFrame(X_all, columns=cols)
    df_all["subject"] = subj_all
    df_all["window_idx"] = np.arange(len(df_all))

    flagged = pd.read_parquet(ROOT / "outputs_v1_archive" / "flagged_windows.parquet")
    df_all["flag_if"] = False
    df_all["flag_lof"] = False
    df_all["score_if"] = np.nan
    df_all["score_lof"] = np.nan
    # map archived flags onto the feature matrix via (subject, local window index)
    subj_offsets = {}
    off = 0
    for s in sorted(df_all["subject"].unique()):
        subj_offsets[s] = off
        off += int((df_all["subject"] == s).sum())
    matched = 0
    for _, r in flagged.iterrows():
        g = subj_offsets[int(r["subject"])] + int(r["window_idx"])
        df_all.loc[g, "flag_if"] = bool(r["flag_if"])
        df_all.loc[g, "flag_lof"] = bool(r["flag_lof"])
        df_all.loc[g, "score_if"] = r["score_if"]
        df_all.loc[g, "score_lof"] = r["score_lof"]
        matched += 1
    union = df_all[df_all["flag_if"] | df_all["flag_lof"]]
    both = int((df_all["flag_if"] & df_all["flag_lof"]).sum())
    print(f"archived flags mapped: {matched} rows; IF {int(df_all['flag_if'].sum())}, "
          f"LOF {int(df_all['flag_lof'].sum())}, both {both}, union {len(union)} "
          f"(v1 archived: 216/216/34/398)", flush=True)
    assert len(union) == 398 and int(df_all["flag_if"].sum()) == 216

    # ---- 2. build queries for the union alerts
    rows, zrows = [], []
    for _, row in union.iterrows():
        ref = subject_reference(df_all, row["subject"])
        q, z = build_query_v2(row, ref)
        rows.append({"subject": row["subject"], "window_idx": row["window_idx"],
                     "query_v2": q, "flag_if": row["flag_if"], "flag_lof": row["flag_lof"],
                     "score_if": row["score_if"], "score_lof": row["score_lof"], **z})
        zrows.append(z)
    qdf = pd.DataFrame(rows)
    qdf.to_csv(OUT / "alerts_queries_v2.csv", index=False)
    print(f"built {len(qdf)} v2 queries", flush=True)
    print(qdf.iloc[0]["query_v2"][:300], flush=True)

    # ---- 3. WESAD sanity check of the new z-metric
    w = get_cache("wesad")
    stress_top2, baseline_top2 = [], []
    for s in WESAD_SUBJECTS:
        X, y = w[f"X_{s}"], w[f"y_{s}"]
        dfw = pd.DataFrame(X, columns=cols)
        base = dfw[y == 1]
        ref = {}
        for ch in CHANNELS:
            col = f"{ch}__mean"
            ref[col] = (base[col].mean(), base[col].std(ddof=0) or 1e-9)
        for mask, acc in [(y == 2, stress_top2), (y == 1, baseline_top2)]:
            for _, r in dfw[mask].iterrows():
                zz = [abs((r[f"{ch}__mean"] - ref[f"{ch}__mean"][0]) / ref[f"{ch}__mean"][1])
                      for ch in CHANNELS]
                acc.append(sorted(zz)[-2:])
    st = np.array(stress_top2).max(axis=1)
    bt = np.array(baseline_top2).max(axis=1)
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(st, bt, alternative="greater")
    sanity = {
        "metric": "max |z| of channel means vs subject's own baseline windows",
        "mean_stress": round(float(st.mean()), 3), "mean_baseline": round(float(bt.mean()), 3),
        "mannwhitney_u": float(u), "p_value": float(p),
        "interpretation": "stress windows deviate more than baseline windows under the new reference"
                          if st.mean() > bt.mean() else "WARNING: metric does not discriminate",
    }
    (OUT / "wesad_zscore_sanity.json").write_text(json.dumps(sanity, indent=2))
    print("WESAD sanity:", sanity, flush=True)

    # ---- 4. retrieval against chroma_db_v2 (top-5, source-diverse)
    import chromadb
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    col = chromadb.PersistentClient(path=str(ROOT / "chroma_db_v2")).get_collection("medical_corpus_v2")
    with open(OUT / "alerts_retrieval_v2.jsonl", "w", encoding="utf-8") as f:
        t0 = time.time()
        for _, r in qdf.iterrows():
            qe = embedder.encode([r["query_v2"]])
            res = col.query(query_embeddings=qe.tolist(), n_results=20,
                            include=["documents", "metadatas", "distances"])
            metas, docs, dists = res["metadatas"][0], res["documents"][0], res["distances"][0]
            picked, counts = [], {}
            for i, m in enumerate(metas):
                c = counts.get(m["source"], 0)
                if c >= 1:
                    continue
                picked.append(i)
                counts[m["source"]] = c + 1
                if len(picked) == 5:
                    break
            rec = {"subject": int(r["subject"]), "window_idx": int(r["window_idx"]),
                   "query": r["query_v2"], "flag_if": bool(r["flag_if"]), "flag_lof": bool(r["flag_lof"]),
                   "sources": [metas[i]["source"] for i in picked],
                   "tiers": [metas[i]["tier"] for i in picked],
                   "context": "\n\n".join(f"[{metas[i]['source']}]\n{docs[i]}" for i in picked)}
            f.write(json.dumps(rec) + "\n")
    print(f"retrieval done in {time.time()-t0:.0f}s -> alerts_retrieval_v2.jsonl", flush=True)


if __name__ == "__main__":
    main()
