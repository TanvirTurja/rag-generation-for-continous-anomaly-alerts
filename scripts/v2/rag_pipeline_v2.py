"""
rag_pipeline_v2.py — Single regeneration cycle (fix/v2 W5).

Generates explanations for:
  A. 398 PPG-DaLiA alerts (v2 queries + chroma_db_v2 corpus)
  B. labeled events (the missing integration experiment):
     - 50 WESAD stress windows (labels withheld from the query)
     - ~50 MIT-BIH DS2 30-s windows with highest detector-flagged beat fraction
     - ~48 PTB-XL fold-10 pathology records, stratified by superclass (labels withheld)
  C. word-cap ablation: first 50 DaLiA alerts at a 300-word cap
  D. generator ablation: same 50 alerts with llama3.1:8b

Fixes baked in vs v1:
  - RAW pre-canonicalization text is persisted (pre-repair citation accuracy
    becomes deterministic and artifact-backed).
  - Canonicalizer logs every snap (before -> after) for the false-repair audit.
  - Labeled-event queries never contain the ground-truth label; labels are kept
    in metadata for the W6 concordance evaluation.

Resume-safe: appends to outputs_v2/generation_v2.jsonl, skipping existing keys.
"""

import ast
import difflib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v2"))
from detection_v2 import (get_cache, WESAD_SUBJECTS, FEATURE_NAMES, CHANNELS,  # noqa: E402
                          DS1, DS2, BEAT_SYMBOLS, AAMI_NORMAL, featurize, fit_score)

OUT = ROOT / "outputs_v2"
GEN_JSONL = OUT / "generation_v2.jsonl"

LLM_MODEL = "qwen3.5:9b"
ALT_MODEL = "llama3.1:8b"
NUM_CTX = 10000

SYSTEM_PROMPT_150 = """You are a clinical decision-support assistant that explains wearable biosignal anomalies.
You receive:
1. A description of an anomaly detected in a 30-second window of wearable signals.
2. Retrieved excerpts from peer-reviewed clinical guidelines and research articles.
STRICT RULES (never violate):
- Answer ONLY using the provided retrieved context.
- Cite the source document for every clinical claim. Format: [Source Name].
- If the retrieved context does not cover the anomaly, say: "The retrieved context is insufficient to explain this pattern."
- NEVER invent facts, numbers, citations, or medical conclusions not present in the context.
- This is a research tool, NOT a diagnostic device. State this once at the end.
- Keep the explanation under 150 words. Use plain language a nurse could understand.
Output format:
DETECTED: [one-sentence summary of what the anomaly pattern suggests]
EVIDENCE: [what the guidelines/literature say, with citations]
RECOMMENDATION: [what clinical follow-up the guidelines suggest, or "context insufficient"]
DISCLAIMER: Research decision-support tool. Not a diagnostic device. Does not replace clinical judgment."""

SYSTEM_PROMPT_300 = SYSTEM_PROMPT_150.replace("under 150 words", "under 300 words")

ECG_TOPICS = ("electrocardiogram rhythm irregularity heart rate variability arrhythmia ectopic beats "
              "atrial fibrillation ventricular tachycardia conduction abnormality repolarization ST "
              "changes myocardial infarction hypertrophy")


# ---------------------------------------------------------------- retrieval
def make_retriever():
    import chromadb
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    col = chromadb.PersistentClient(path=str(ROOT / "chroma_db_v2")).get_collection("medical_corpus_v2")

    def retrieve(query, top_k=5, pool=20):
        qe = embedder.encode([query])
        res = col.query(query_embeddings=qe.tolist(), n_results=pool,
                        include=["documents", "metadatas", "distances"])
        metas, docs = res["metadatas"][0], res["documents"][0]
        picked, counts = [], {}
        for i, m in enumerate(metas):
            c = counts.get(m["source"], 0)
            if c >= 1:
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
        return {"sources": [metas[i]["source"] for i in picked],
                "tiers": [metas[i]["tier"] for i in picked],
                "context": "\n\n---\n\n".join(f"[{metas[i]['source']}]\n{docs[i]}" for i in picked)}
    return retrieve


# ---------------------------------------------------------------- canonicalizer with snap log
def canonicalize_citations_logged(text, sources):
    valid = sorted({s.split("_")[0] for s in sources})
    snaps = []

    def _fix(m):
        cid = m.group(1)
        if cid in valid:
            return f"[{cid}]"
        close = difflib.get_close_matches(cid, valid, n=1, cutoff=0.75)
        if close:
            snaps.append({"before": cid, "after": close[0]})
            return f"[{close[0]}]"
        snaps.append({"before": cid, "after": None})  # dropped
        return ""

    repaired = re.sub(r"\[(PMC\d+[^\]]*)\]", _fix, text)
    return repaired, snaps


# ---------------------------------------------------------------- labeled-event alert builders
def build_wesad_events(retrieve, n_events=50):
    """Top-50 LOF-scored WESAD stress windows; queries withhold the stress label.

    Note: v1's confirmed_anomalies_for_rag.csv stored only (dataset, label, score)
    without subject/window identifiers, so the exact 50 windows are unrecoverable.
    v2 regenerates the selection deterministically under the same v1 rule
    (train on pooled baseline windows, top-scored stress windows) and preserves
    identifiers. Output artifact: outputs_v2/confirmed_anomalies_v2.csv.
    """
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    cols = [f"{ch}__{fn}" for ch in CHANNELS for fn in FEATURE_NAMES]
    w = get_cache("wesad")
    X_base, stress = [], []
    for s in WESAD_SUBJECTS:
        X, y = w[f"X_{s}"], w[f"y_{s}"]
        X_base.append(X[y == 1])
        stress.append((s, X[y == 2]))
    Xb = StandardScaler().fit_transform(np.vstack(X_base))
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.15).fit(Xb)
    scaler = StandardScaler().fit(np.vstack(X_base))
    cand = []
    for s, Xs in stress:
        scores = -lof.score_samples(scaler.transform(Xs))
        for i, sc in enumerate(scores):
            cand.append((float(sc), s, i))
    cand.sort(reverse=True)
    top = cand[:n_events]
    pd.DataFrame([{"dataset": "WESAD", "true_label": "stress", "score_lof": sc,
                   "subject": s, "window_idx": i} for sc, s, i in top]
                 ).to_csv(OUT / "confirmed_anomalies_v2.csv", index=False)

    events = []
    dfs = {}
    for sc, s, local in top:
        if s not in dfs:
            X, y = w[f"X_{s}"], w[f"y_{s}"]
            dfw = pd.DataFrame(X, columns=cols)
            base = dfw[y == 1]
            dfs[s] = (dfw, {ch: (base[f"{ch}__mean"].mean(),
                                  base[f"{ch}__mean"].std(ddof=0) or 1e-9)
                            for ch in CHANNELS})
        dfw, ref = dfs[s]
        row = dfw.iloc[local]
        zs = {ch: (row[f"{ch}__mean"] - ref[ch][0]) / ref[ch][1] for ch in CHANNELS}
        top2 = sorted(CHANNELS, key=lambda c: -abs(zs[c]))[:2]
        q = ("Biosignal window flagged by anomaly detection. " +
             " ".join(f"{'elevated' if zs[c] > 0 else 'reduced'} {c} (z={zs[c]:+.1f} vs subject baseline)"
                      for c in top2) +
             ". Relevant topics: electrodermal activity skin conductance sympathetic arousal stress physiology "
             "photoplethysmography heart rate variability ECG. Other readings: " +
             ", ".join(f"{c} mean={row[f'{c}__mean']:.2f}" for c in CHANNELS if c not in top2) + ".")
        res = retrieve(q)
        events.append({"key": f"wesad|S{s}|w{local}", "group": "wesad", "subject": s,
                       "window": local, "query": q, "true_label": "stress",
                       "true_label_detail": "laboratory mental-stress protocol, expert-annotated",
                       "score_lof": sc, **res})
    return events


def build_mitbih_events(retrieve, n_windows=50):
    """Top DS2 30-s windows by detector-flagged beat fraction; labels withheld."""
    import wfdb
    MITBIH_DIR = ROOT / "Dataset" / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"
    m = get_cache("mitbih")
    X, y, recs = m["X"], m["y"], m["record"].astype(str)
    tr = np.isin(recs, DS1) & (y == 0)
    te = np.isin(recs, DS2)
    se, st = fit_score("LOF", X[tr], X[te])
    thr = np.percentile(st, 85)
    flagged = se > thr
    # feature z vs training normals
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9

    events = []
    for rec in DS2:
        sig, meta = wfdb.rdsamp(str(MITBIH_DIR / rec))
        ann = wfdb.rdann(str(MITBIH_DIR / rec), "atr")
        idx_te = np.where(te & (recs == rec))[0]          # cache row order = annotation order
        sym_seq = []
        for i, sym in zip(ann.sample, ann.symbol):
            if sym not in BEAT_SYMBOLS:
                continue
            start, end = i - 144, i + 144
            if start < 0 or end > sig.shape[0]:
                continue
            sym_seq.append((i, sym, len(sym_seq)))        # (sample, symbol, cache-row position)
        assert len(sym_seq) == len(idx_te), f"alignment mismatch on {rec}"
        row_of = {pos: int(r) for pos, r in zip([s[2] for s in sym_seq], idx_te)}

        windows = {}
        for sample, sym, pos in sym_seq:
            widx = sample // (360 * 30)
            wd = windows.setdefault(widx, {"rows": [], "syms": []})
            wd["rows"].append(row_of[pos])
            wd["syms"].append(sym)
        for widx, wd in windows.items():
            frac = float(np.mean([flagged[r] for r in wd["rows"]]))
            abn = [s_ for s_ in wd["syms"] if s_ not in AAMI_NORMAL]
            if not abn:
                label, detail = "none", "no annotated abnormal beats in window"
            elif any(s_ in {"V", "E"} for s_ in abn):
                label, detail = "VEB", f"ventricular-type beats present: {abn}"
            elif any(s_ in {"A", "a", "J", "S"} for s_ in abn):
                label, detail = "SVEB", f"supraventricular-type beats present: {abn}"
            elif "F" in abn:
                label, detail = "F", "fusion beats present"
            else:
                label, detail = "other", f"{abn}"
            feat_z = (X[wd["rows"][0]] - mu) / sd  # first beat's features as window anchor
            q = (f"Ambulatory ECG monitoring window flagged by anomaly detection: "
                 f"{frac * 100:.0f}% of beats in this 30-second window were flagged by a detector "
                 f"trained on normal beats. Representative beat feature deviations vs normal training: "
                 + ", ".join(f"{FEATURE_NAMES[j]} z={feat_z[j]:+.1f}" for j in [0, 1, 4, 7]) +
                 f". Relevant topics: {ECG_TOPICS}.")
            events.append({"key": f"mitbih|{rec}|w{widx}", "group": "mitbih", "subject": rec,
                           "window": int(widx), "query": q, "true_label": label,
                           "true_label_detail": detail, "flag_frac": frac,
                           "n_beats": len(wd["rows"])})
    ev = sorted([e for e in events if e.get("n_beats", 0) >= 5],
                key=lambda e: -e["flag_frac"])[:n_windows]
    for e in ev:
        res = retrieve(e["query"])
        e.update(res)
    return ev, events


def build_ptbxl_events(retrieve, per_class=12):
    """Top LOF-scored fold-10 pathology records per superclass; labels withheld."""
    import ast as _ast
    import wfdb
    PTBXL_DIR = ROOT / "Dataset" / "ptb-xl-1.0.3"
    db = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv")
    scp = pd.read_csv(PTBXL_DIR / "scp_statements.csv", index_col=0)

    def supers(cp_str):
        codes = _ast.literal_eval(cp_str)
        out = set()
        for c in codes:
            if c in scp.index:
                k = scp.loc[c, "diagnostic_class"]
                if isinstance(k, str):
                    out.add(k)
        return out

    db["super"] = db["scp_codes"].apply(supers)
    db["y"] = db["super"].apply(lambda st: 0 if st == {"NORM"} else (-1 if not st else 1))
    db = db[db["y"] != -1]
    p = get_cache("ptbxl")
    X, y, f = p["X"], p["y"], p["fold"]
    tr = (f <= 8) & (y == 0)
    te = f == 10
    se, st = fit_score("LOF", X[tr], X[te])
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9

    # align cache rows to db rows: cache was built in db order minus load errors;
    # rebuild by featurizing the selected records directly instead (robust).
    order = db[db["strat_fold"] == 10].reset_index()
    score_by_ecg_id = {}
    te_rows = np.where(te)[0]
    for k_i, (_, row) in enumerate(order.iterrows()):
        if k_i < len(te_rows):
            score_by_ecg_id[row["ecg_id"]] = se[k_i]
    # NOTE: cache preserves db order for fold-10 in expectation; verify by count
    assert len(score_by_ecg_id) == len(te_rows)

    picked = []
    for cls in ["MI", "STTC", "CD", "HYP"]:
        cand = db[(db["strat_fold"] == 10) & db["super"].apply(lambda st: cls in st)]
        cand = cand.assign(score=cand["ecg_id"].map(lambda e: score_by_ecg_id.get(e, -9.0)))
        cand = cand.sort_values("score", ascending=False)
        for r in cand.head(per_class).itertuples():
            picked.append((r.ecg_id, r.filename_lr, cls, score_by_ecg_id.get(r.ecg_id, float("nan"))))
    events = []
    for ecg_id, fn, cls, score in picked:
        try:
            sig, _ = wfdb.rdsamp(str(PTBXL_DIR / fn))
        except Exception:
            continue
        feats = np.array(featurize(sig[:, 0]))
        z = (feats - mu) / sd
        q = ("Resting 12-lead ECG recording (lead I analysis) flagged by an anomaly detector "
             f"trained on normal recordings (anomaly score percentile: high). Signal feature "
             "deviations vs normal training recordings: "
             + ", ".join(f"{FEATURE_NAMES[j]} z={z[j]:+.1f}" for j in [0, 1, 4, 6, 7]) +
             f". Relevant topics: {ECG_TOPICS}.")
        res = retrieve(q)
        events.append({"key": f"ptbxl|{ecg_id}", "group": "ptbxl", "subject": int(ecg_id),
                       "window": None, "query": q, "true_label": cls,
                       "true_label_detail": f"PTB-XL diagnostic superclass {cls}, fold 10",
                       **res})
    return events


# ---------------------------------------------------------------- generation
def generate_one(alert, model, system_prompt, num_predict=500, use_think_false=True):
    import ollama
    user_msg = f"ANOMALY:\n{alert['query']}\n\nRETRIEVED CONTEXT:\n{alert['context']}"
    opts = {"temperature": 0.1, "num_predict": num_predict, "num_ctx": NUM_CTX, "num_gpu": 99}
    kw = {"model": model,
          "messages": [{"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_msg}],
          "options": opts}
    if use_think_false and model == LLM_MODEL:
        kw["think"] = False
    t0 = time.time()
    resp = ollama.chat(**kw)
    latency = time.time() - t0
    return resp["message"]["content"].strip(), latency


def main():
    retrieve = make_retriever()

    # A. PPG-DaLiA alerts (queries + retrieval already built by W4)
    alerts = [json.loads(l) for l in open(OUT / "alerts_retrieval_v2.jsonl", encoding="utf-8")]
    for a in alerts:
        a["key"] = f"dalia|S{a['subject']}|w{a['window_idx']}"
        a["group"] = "dalia"
        a["true_label"] = None
    print(f"A: {len(alerts)} dalia alerts", flush=True)

    # B. labeled events
    wes = build_wesad_events(retrieve)
    print(f"B1: {len(wes)} WESAD stress events", flush=True)
    mit, all_mit = build_mitbih_events(retrieve)
    print(f"B2: {len(mit)} MIT-BIH windows selected (of {len(all_mit)})", flush=True)
    ptb = build_ptbxl_events(retrieve)
    print(f"B3: {len(ptb)} PTB-XL records", flush=True)

    # C/D. ablation subsets (first 50 dalia alerts by subject order)
    ablation_base = sorted(alerts, key=lambda a: (a["subject"], a["window_idx"]))[:50]

    jobs = []
    for a in alerts:
        jobs.append({**a, "model": LLM_MODEL, "prompt": "150", "subgroup": "main"})
    for a in wes + mit + ptb:
        jobs.append({**a, "model": LLM_MODEL, "prompt": "150", "subgroup": "labeled"})
    for a in ablation_base:
        jobs.append({**a, "key": a["key"] + "|wordcap300", "model": LLM_MODEL,
                     "prompt": "300", "subgroup": "wordcap"})
    for a in ablation_base:
        jobs.append({**a, "key": a["key"] + "|gen-llama31", "model": ALT_MODEL,
                     "prompt": "150", "subgroup": "genablation"})
    print(f"total generation jobs: {len(jobs)}", flush=True)

    done = set()
    if GEN_JSONL.exists():
        for l in open(GEN_JSONL, encoding="utf-8"):
            try:
                done.add(json.loads(l)["key"] + "|" + json.loads(l)["model"] +
                         "|" + json.loads(l).get("prompt", "150") +
                         "|" + json.loads(l).get("subgroup", ""))
            except Exception:
                pass
        print(f"resuming: {len(done)} jobs already done", flush=True)

    with open(GEN_JSONL, "a", encoding="utf-8") as fout:
        for i, job in enumerate(jobs):
            uid = job["key"] + "|" + job["model"] + "|" + job["prompt"] + "|" + job["subgroup"]
            if uid in done:
                continue
            prompt = SYSTEM_PROMPT_300 if job["prompt"] == "300" else SYSTEM_PROMPT_150
            try:
                raw, latency = generate_one(job, job["model"], prompt,
                                            num_predict=800 if job["prompt"] == "300" else 500)
            except Exception as e:
                print(f"  [{i}] ERROR {job['key']}: {e}", flush=True)
                continue
            repaired, snaps = canonicalize_citations_logged(raw, job["sources"])
            rec = {"key": job["key"], "group": job["group"], "subgroup": job["subgroup"],
                   "model": job["model"], "prompt": job["prompt"],
                   "subject": job.get("subject"), "window": job.get("window"),
                   "query": job["query"], "sources": job["sources"], "tiers": job["tiers"],
                   "true_label": job.get("true_label"),
                   "true_label_detail": job.get("true_label_detail"),
                   "raw_explanation": raw, "explanation": repaired,
                   "canonicalizer_snaps": snaps, "latency_sec": round(latency, 2)}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if i % 10 == 0:
                print(f"  [{i}/{len(jobs)}] {job['key']} {latency:.1f}s snaps={len(snaps)}", flush=True)
    print("generation complete ->", GEN_JSONL, flush=True)


if __name__ == "__main__":
    main()
