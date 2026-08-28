"""
judge_eval_v2.py — Judge validation + full evaluation suite (fix/v2 W6).

Order of operations:
 1. Citation audit v2 on RAW vs canonicalized text (deterministic, artifact-backed
    pre-repair number + canonicalizer snap/drop log analysis).
 2. Judge validation on a corruption benchmark: 100 corrupted explanations
    (4 corruption types) + their 100 clean originals. Judge candidates:
    llama3.1:8b, gemma4:e4b, gpt-oss:20b (local, free) + DeepSeek-V4-Flash (API).
    Per-judge detection rate (faithfulness=1 on corrupted) and false-positive rate
    on clean rows; the local judge for the main run is SELECTED by this benchmark.
 3. Main judging run: selected local judge (free) on all groups; API judge
    (checkpointed, hard budget cap $10 est.) on all groups.
 4. Agreement: score distributions, raw + within-1, Gwet's AC1 (degeneracy-aware).
 5. Labeled-events concordance: keyword-lexicon check whether DETECTED matches the
    true label + artifact-conclusion rate on true pathology (safety metric).
 6. FActScore-lite: atomic claim decomposition + context entailment on a 60-row
    sample, verifier = gemma4:e4b (different family from generator).

Outputs -> outputs_v2/:
  citation_audit_v2.json, judge_validation.json, rag_evaluation_v2.csv,
  agreement_v2.json, concordance_v2.json, factscore_lite.json,
  api_judge_checkpoint_v2.jsonl, api_budget.json
"""

import difflib
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v2"))
from rag_pipeline_v2 import make_retriever  # noqa: E402

OUT = ROOT / "outputs_v2"
GEN = OUT / "generation_v2.jsonl"
CKPT = OUT / "api_judge_checkpoint_v2.jsonl"

API_MODEL = "deepseek/deepseek-v4-flash-0731"
LOCAL_CANDIDATES = ["llama3.1:8b", "gemma4:e4b"]  # gpt-oss excluded per author decision (2026-08-28)
BUDGET_CAP_USD = 10.0
PRICE_IN, PRICE_OUT = 0.068 / 1e6, 0.18 / 1e6  # OpenRouter listed 2026-08-28

JUDGE_PROMPT_V2 = open(ROOT / "scripts" / "v2" / "judge_prompt_v2.txt", encoding="utf-8").read()


def canonicalize_fixed(raw_text, sources):
    """v1-intended canonicalizer with the ID capture fixed: the citation ID is the
    PMC token only; trailing slug text is display noise, not part of the ID.
    Returns (repaired_text, snaps) where snaps record before/after."""
    valid = sorted({s.split("_")[0] for s in sources if s.startswith("PMC")})
    snaps = []

    def _fix(m):
        cid = m.group(1)
        if cid in valid:
            return f"[{cid}]"
        close = difflib.get_close_matches(cid, valid, n=1, cutoff=0.75)
        if close:
            snaps.append({"before": cid, "after": close[0]})
            return f"[{close[0]}]"
        snaps.append({"before": cid, "after": None})
        return ""

    repaired = re.sub(r"\[(PMC\d+)[^\]]*\]", _fix, raw_text)
    return repaired, snaps

LEXICONS = {
    "stress": ["stress", "arousal", "sympathetic", "anxiety", "mental load",
               "psychological", "emotional"],
    "VEB": ["ventricular", "pvc", "premature ventricular", "ventricular tachycard"],
    "SVEB": ["supraventricular", "atrial premature", "pac", "atrial ectopy",
             "premature atrial", "atrial fibrillation", "atrial tachyarrhythm"],
    "F": ["fusion beat"],
    "MI": ["infarct", "ischemi", "stemi", "coronary occlusion", "st-elevation",
           "st elevation"],
    "STTC": ["repolarization", "st depression", "st-segment", "st segment", "t-wave",
             "t wave inversion"],
    "CD": ["conduction", "bundle branch", "heart block", "av block", "pr interval"],
    "HYP": ["hypertroph", "chamber enlargement", "left ventricular mass"],
}
ARTIFACT_TERMS = ["artifact", "motion", "sensor displacement", "sensor contact",
                  "signal quality", "electrode", "noise", "poor contact", "device"]


def section(txt, start, end):
    m = re.search(rf"{start}:\s*(.*?)(?={end}:|$)", str(txt), re.S)
    return m.group(1).strip() if m else ""


# -------------------------------------------------------------- citation audit
def citation_audit(rows):
    """Audit on raw text and on text re-canonicalized with the FIXED capture rule.
    Also counts name-style citations to (non-PMC) guideline sources."""
    cit_re = re.compile(r"\[(PMC\d+)[^\]]*\]")
    name_re = re.compile(r"\[([^]\[]+)\]")
    res = {"n_rows": len(rows), "raw": {"citations": 0, "valid": 0, "rows_all_valid": 0},
           "repaired": {"citations": 0, "valid": 0, "rows_all_valid": 0},
           "snaps": 0, "drops": 0,
           "tier1_name_citations": 0, "tier1_name_unmatched": 0}
    for r in rows:
        valid_ids = {s.split("_")[0] for s in r["sources"] if s.startswith("PMC")}
        tier1_sources = [s for s in r["sources"] if not s.startswith("PMC")]
        fixed, snaps = canonicalize_fixed(r["raw_explanation"], r["sources"])
        res["snaps"] += sum(1 for s in snaps if s["after"])
        res["drops"] += sum(1 for s in snaps if not s["after"])
        for which, txt in [("raw", r["raw_explanation"]), ("repaired", fixed)]:
            cits = [m.group(1) for m in cit_re.finditer(txt)]
            res[which]["citations"] += len(cits)
            ok = sum(1 for c in cits if c in valid_ids)
            res[which]["valid"] += ok
            res[which]["rows_all_valid"] += int(ok == len(cits))
        # name-style citations (guideline sources etc.)
        for m in name_re.finditer(fixed):
            tok = m.group(1).strip()
            if tok.startswith("PMC"):
                continue
            if any(s.startswith(tok) or tok.startswith(s) for s in tier1_sources):
                res["tier1_name_citations"] += 1
            elif any(s.split("_")[0].lower() in tok.lower() for s in tier1_sources):
                res["tier1_name_citations"] += 1
            else:
                res["tier1_name_unmatched"] += 1
    for which in ("raw", "repaired"):
        c, v = res[which]["citations"], res[which]["valid"]
        res[which]["accuracy"] = round(v / c, 4) if c else None
    return res


# -------------------------------------------------------------- corruption benchmark
FABRICATED_FACTS = [
    "This pattern is associated with a 3-fold increase in 30-day mortality according to the sources.",
    "The literature states that patients with this finding require immediate hospital admission.",
    "According to the retrieved context, this signal pattern predicts stroke with 92% probability.",
    "The sources confirm that this finding always progresses to cardiac arrest if untreated.",
    "Retrieved guidelines recommend starting anticoagulation immediately for this pattern.",
]


def build_corruption_set(rows, n=100, seed=7):
    rng = random.Random(seed)
    pool = [r for r in rows if r["group"] in ("dalia", "wesad", "mitbih", "ptbxl")
            and r["subgroup"] == "main" and len(re.findall(r"\[PMC\d+", r["explanation"])) >= 2]
    sample = rng.sample(pool, n)
    out = []
    for i, r in enumerate(sample):
        txt = r["explanation"]
        cits = re.findall(r"PMC\d+", txt)
        ctype = i % 4
        corrupted = txt
        applied = ctype
        if ctype == 0 and len(cits) >= 1:  # citation swap to wrong retrieved doc
            others = [s.split("_")[0] for s in r["sources"]
                      if s.split("_")[0] not in cits[:1]]
            if others:
                corrupted = txt.replace(f"[{cits[0]}]", f"[{others[0]}]", 1)
            else:  # no distinct doc to swap to -> fall through to fabricated fact
                applied = 1
        if applied == 1:  # fabricated fact
            corrupted = txt.replace(
                "EVIDENCE:",
                f"EVIDENCE: {rng.choice(FABRICATED_FACTS)} ", 1)
        elif applied == 2:  # fabricated citation (non-retrieved id)
            corrupted = txt.replace(f"[{cits[0]}]", f"[PMC99999999]", 1)
        elif applied == 3:  # diagnostic exaggeration
            corrupted = txt.replace("DETECTED:", "DETECTED: This finding is diagnostic of "
                                    "acute myocardial infarction and requires emergency "
                                    "treatment. ", 1)
        ctype_name = ["citation_swap", "fabricated_fact", "fabricated_citation",
                      "diagnostic_exaggeration"][applied]
        out.append({"row": r, "clean": txt, "corrupted": corrupted,
                    "corruption_type": ctype_name})
    return out


# -------------------------------------------------------------- judges
def parse_scores(text):
    scores = {}
    for line in str(text).split("\n"):
        line = line.strip()
        m = re.match(r"(FAITHFULNESS|RELEVANCE|COMPLETENESS):\s*([123])", line, re.I)
        if m:
            scores[m.group(1).lower()] = int(m.group(2))
    return scores


def local_judge(model, query, explanation, context):
    import ollama
    resp = ollama.chat(model=model,
                       messages=[{"role": "system", "content": JUDGE_PROMPT_V2},
                                 {"role": "user", "content":
                                  f"QUERY: {query}\nSOURCES:\n{context}\nEXPLANATION:\n{explanation}"}],
                       options={"temperature": 0.1, "num_predict": 300,
                                "num_ctx": 6000, "num_gpu": 99})
    return parse_scores(resp["message"]["content"])


class ApiJudge:
    def __init__(self):
        key = None
        for line in open(ROOT / ".env", encoding="utf-8"):
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
        self.key = key
        self.spent = 0.0
        self.calls = 0
        self.cache = {}
        if CKPT.exists():
            for l in open(CKPT, encoding="utf-8"):
                rec = json.loads(l)
                self.cache[rec["uid"]] = rec
                self.spent += rec.get("cost_usd", 0)
                self.calls += 1

    def est_cost(self, n_in, n_out):
        return n_in * PRICE_IN + n_out * PRICE_OUT

    def judge(self, uid, query, explanation, context):
        if uid in self.cache:
            return self.cache[uid]["scores"]
        est = self.est_cost((len(query) + len(context) + len(explanation)) // 3 + 600, 300)
        if self.spent + est > BUDGET_CAP_USD:
            raise RuntimeError(f"API budget cap reached (${self.spent:.2f})")
        body = json.dumps({
            "model": API_MODEL,
            "messages": [{"role": "system", "content": JUDGE_PROMPT_V2},
                         {"role": "user", "content":
                          f"QUERY: {query}\nSOURCES:\n{context}\nEXPLANATION:\n{explanation}"}],
            "temperature": 0.1, "max_tokens": 400,
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        usage = d.get("usage", {})
        cost = self.est_cost(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        text = d["choices"][0]["message"]["content"]
        scores = parse_scores(text)
        rec = {"uid": uid, "scores": scores, "cost_usd": round(cost, 6),
               "usage": usage, "raw_head": text[:200]}
        with open(CKPT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.cache[uid] = rec
        self.spent += cost
        self.calls += 1
        return scores


def gwet_ac1(r1, r2):
    """Gwet's AC1 for two raters on ordinal-as-nominal scores (1..3)."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    ok = (r1 > 0) & (r2 > 0)
    r1, r2 = r1[ok], r2[ok]
    n = len(r1)
    if n == 0:
        return None
    cats = [1, 2, 3]
    p_c = np.mean([np.mean(r1 == c) for c in cats])
    pe = (2 * p_c * (1 - p_c)) / (len(cats) - 1) if len(cats) > 1 else 0.0
    po = np.mean(r1 == r2)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


# -------------------------------------------------------------- main
def main():
    rows = [json.loads(l) for l in open(GEN, encoding="utf-8")]
    sup = OUT / "superseded_keys.json"
    if sup.exists():
        drop = set(json.loads(sup.read_text())["superseded_mitbih_keys"])
        rows = [r for r in rows if r["key"] not in drop]
        print(f"dropped {len(drop)} superseded mitbih rows", flush=True)
    print(f"loaded {len(rows)} generated explanations", flush=True)

    # Enrich rows: re-canonicalized text with the FIXED capture rule + re-retrieved
    # context (deterministic: same corpus, same query, same embedder). The stored
    # v1-style canonicalized text dropped slug-form citations of valid sources.
    retrieve = make_retriever()
    ctx_cache = {}
    for r in rows:
        fixed, _ = canonicalize_fixed(r["raw_explanation"], r["sources"])
        r["explanation"] = fixed
        ck = r["key"]
        if ck not in ctx_cache:
            ctx_cache[ck] = retrieve(r["query"])["context"]
        r["context"] = ctx_cache[ck]
    main_rows = [r for r in rows if r["subgroup"] == "main"]

    # 1. citation audit
    audit = citation_audit(main_rows)
    (OUT / "citation_audit_v2.json").write_text(json.dumps(audit, indent=2))
    print("citation audit:", {k: audit[k] for k in ("raw", "repaired", "snaps", "drops")}, flush=True)

    # 2. corruption benchmark
    corrupt = build_corruption_set(main_rows)
    api = ApiJudge()
    val = {"per_judge": {}}
    bench_rows = []
    for r in corrupt:
        bench_rows.append({"uid": f"val|clean|{r['row']['key']}", "query": r["row"]["query"],
                           "expl": r["clean"], "context": r["row"]["context"],
                           "is_corrupted": 0, "ctype": None})
        bench_rows.append({"uid": f"val|corr|{r['row']['key']}", "query": r["row"]["query"],
                           "expl": r["corrupted"], "context": r["row"]["context"],
                           "is_corrupted": 1, "ctype": r["corruption_type"]})

    for model in LOCAL_CANDIDATES:
        csv_path = OUT / f"judge_validation_{model.replace(':', '_').replace('/', '_')}.csv"
        df = None
        if csv_path.exists():
            try:
                cached = pd.read_csv(csv_path)
                if len(cached) == len(bench_rows) and {"is_corrupted", "faithfulness"} <= set(cached.columns):
                    df = cached
                    print(f"  {model}: loaded cached validation ({len(df)} rows)", flush=True)
            except Exception:
                df = None
        if df is None:
            t0 = time.time()
            results = []
            for b in bench_rows:
                try:
                    sc = local_judge(model, b["query"], b["expl"], b["context"])
                except Exception as e:
                    print(f"  {model} error {b['uid'][:24]}: {e}", flush=True)
                    sc = {}
                results.append({**b, "faithfulness": sc.get("faithfulness", 0),
                                "relevance": sc.get("relevance", 0),
                                "completeness": sc.get("completeness", 0)})
                if len(results) % 50 == 0:
                    print(f"  {model}: {len(results)}/{len(bench_rows)} ({time.time()-t0:.0f}s)", flush=True)
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
        det = df[(df.is_corrupted == 1) & (df.faithfulness == 1)].shape[0] / max(1, (df.is_corrupted == 1).sum())
        fp = df[(df.is_corrupted == 0) & (df.faithfulness == 1)].shape[0] / max(1, (df.is_corrupted == 0).sum())
        per_type = df[df.is_corrupted == 1].groupby("ctype").apply(
            lambda g: (g.faithfulness == 1).mean(), include_groups=False).round(3).to_dict()
        val["per_judge"][model] = {"detection_rate": round(det, 3), "false_positive_rate": round(fp, 3),
                                   "detection_by_type": per_type, "n": len(df)}
        print(f"{model}: detection {det:.2f}, FP {fp:.2f}, by type {per_type}", flush=True)

    # API judge on the benchmark
    try:
        results = []
        for b in bench_rows:
            sc = api.judge(b["uid"], b["query"], b["expl"], b["context"])
            results.append({**b, "faithfulness": sc.get("faithfulness", 0),
                            "relevance": sc.get("relevance", 0),
                            "completeness": sc.get("completeness", 0)})
        df = pd.DataFrame(results)
        det = df[(df.is_corrupted == 1) & (df.faithfulness == 1)].shape[0] / max(1, (df.is_corrupted == 1).sum())
        fp = df[(df.is_corrupted == 0) & (df.faithfulness == 1)].shape[0] / max(1, (df.is_corrupted == 0).sum())
        per_type = df[df.is_corrupted == 1].groupby("ctype").apply(
            lambda g: (g.faithfulness == 1).mean(), include_groups=False).round(3).to_dict()
        val["per_judge"]["api_deepseek_v4_flash"] = {
            "detection_rate": round(det, 3), "false_positive_rate": round(fp, 3),
            "detection_by_type": per_type, "n": len(df)}
        df.to_csv(OUT / "judge_validation_api.csv", index=False)
        print(f"api: detection {det:.2f}, FP {fp:.2f}, by type {per_type}", flush=True)
    except Exception as e:
        print("API validation failed:", e, flush=True)
        val["per_judge"]["api_deepseek_v4_flash"] = {"error": str(e)}

    # select local judge ONLY if it validated: detection >= 0.3 on the benchmark.
    # Both llama3.1:8b and gemma4:e4b scored 0.00 detection -> no valid local judge;
    # per pre-registered fallback, evaluation relies on API judge + atomic
    # verification + clinician kit.
    local_scores = {m: (v.get("detection_rate", 0) - v.get("false_positive_rate", 0))
                    for m, v in val["per_judge"].items() if m in LOCAL_CANDIDATES}
    qualified = {m: s for m, s in local_scores.items()
                 if val["per_judge"][m].get("detection_rate", 0) >= 0.3}
    best_local = max(qualified, key=qualified.get) if qualified else None
    val["selected_local_judge"] = {
        "model": best_local,
        "criterion": "detection >= 0.3 on corruption benchmark, then max (detection - FP)",
        "note": ("no local judge qualified (llama3.1:8b detection 0.00, gemma4:e4b detection "
                 "0.00); falling back to API judge + atomic verification + clinician kit")
        if not best_local else ""}
    (OUT / "judge_validation.json").write_text(json.dumps(val, indent=2))
    print("selected local judge:", best_local, flush=True)

    # 3. main judging (all rows incl. ablation subgroups)
    eval_rows = []
    for r in rows:
        eval_rows.append(r)
    out_records = []
    for r in eval_rows:
        rec = {"key": r["key"], "group": r["group"], "subgroup": r["subgroup"],
               "model": r["model"], "prompt": r["prompt"], "subject": r.get("subject"),
               "window": r.get("window"), "true_label": r.get("true_label"),
               "n_citations_raw": len(re.findall(r"\[PMC\d+", r["raw_explanation"])),
               "latency_sec": r.get("latency_sec")}
        if best_local:
            try:
                sc = local_judge(best_local, r["query"], r["explanation"], r["context"])
            except Exception:
                sc = {}
            rec.update({f"local_{k}": sc.get(k, 0) for k in ("faithfulness", "relevance", "completeness")})
        try:
            sc = api.judge(f"main|{r['key']}|{r['model']}|{r['prompt']}|{r['subgroup']}",
                           r["query"], r["explanation"], r["context"])
            rec.update({f"api_{k}": sc.get(k, 0) for k in ("faithfulness", "relevance", "completeness")})
        except Exception as e:
            print("api main-loop stop:", e, flush=True)
            break
        out_records.append(rec)
        if len(out_records) % 25 == 0:
            print(f"  judged {len(out_records)}/{len(eval_rows)} (api spent ${api.spent:.2f})", flush=True)
    ev = pd.DataFrame(out_records)
    ev.to_csv(OUT / "rag_evaluation_v2.csv", index=False)
    (OUT / "api_budget.json").write_text(json.dumps(
        {"calls": api.calls, "est_spent_usd": round(api.spent, 4),
         "cap_usd": BUDGET_CAP_USD}, indent=2))
    print(f"judging done: {len(out_records)} rows; api ${api.spent:.2f}", flush=True)

    # 4. agreement (main dalia group; requires a valid local judge — else API-only)
    m = ev[(ev.group == "dalia") & (ev.subgroup == "main")] if len(ev) and "group" in ev.columns else ev
    if len(m) and {"local_faithfulness", "api_faithfulness"} <= set(m.columns):
        agr = {}
        for ax in ("faithfulness", "relevance", "completeness"):
            a = m[f"local_{ax}"].values
            b = m[f"api_{ax}"].values
            ok = (a > 0) & (b > 0)
            a, b = a[ok], b[ok]
            agr[ax] = {
                "local_dist": {int(k): int(v) for k, v in pd.Series(a).value_counts().items()},
                "api_dist": {int(k): int(v) for k, v in pd.Series(b).value_counts().items()},
                "raw_agreement": round(float(np.mean(a == b)), 4),
                "within1_agreement": round(float(np.mean(np.abs(a - b) <= 1)), 4),
                "gwet_ac1": gwet_ac1(a, b),
            }
        (OUT / "agreement_v2.json").write_text(json.dumps(agr, indent=2))
        print("agreement:", json.dumps(agr, indent=2)[:600], flush=True)

    # 5. labeled-events concordance
    conc = {}
    # concordance depends only on explanations + labels, not on judge success
    labeled_rows = [r for r in rows if r["subgroup"] == "labeled"]
    for grp in ("wesad", "mitbih", "ptbxl"):
        sel = [r for r in labeled_rows if r["group"] == grp]
        if not sel:
            continue
        stats = {"n": len(sel), "concordant": 0, "artifact_conclusion": 0,
                 "insufficient": 0, "other": 0, "by_label": {}}
        for r in sel:
            det = section(r["explanation"], "DETECTED", "EVIDENCE").lower()
            lab = r["true_label"]
            stats["by_label"].setdefault(lab, {"n": 0, "concordant": 0})
            stats["by_label"][lab]["n"] += 1
            if any(t in det for t in LEXICONS.get(lab, [])):
                stats["concordant"] += 1
                stats["by_label"][lab]["concordant"] += 1
                if any(t in det for t in ARTIFACT_TERMS):
                    stats["artifact_conclusion"] += 1
            elif "insufficient" in det:
                stats["insufficient"] += 1
            elif any(t in det for t in ARTIFACT_TERMS):
                stats["artifact_conclusion"] += 1
            else:
                stats["other"] += 1
        conc[grp] = stats
    (OUT / "concordance_v2.json").write_text(json.dumps(conc, indent=2))
    print("concordance:", json.dumps(conc, indent=2)[:600], flush=True)


if __name__ == "__main__":
    main()
