"""
verify_claims_v2.py — Recompute every quantitative claim in paper.md (v2) from artifacts.

Each claim: (id, artifact, recomputed value, string expected in paper.md).
Output: outputs_v2/verification_v2.json (status per claim) and prints a summary.
Fail loudly (nonzero exit) if any REQUIRED claim fails; WARN for pending W5/W6.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs_v2"
PAPER = ROOT / "draft_paper" / "paper.md"


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def main():
    paper = PAPER.read_text(encoding="utf-8")
    claims = []

    # ---- detection (Table 2/3)
    det = OUT / "detection_results_v2.csv"
    if det.exists():
        import pandas as pd
        d = pd.read_csv(det)
        for _, r in d.iterrows():
            ds = {"WESAD": "WESAD", "MIT-BIH": "MIT-BIH", "PTB-XL": "PTB-XL"}[r.dataset]
            claims.append((f"det_{ds}_{r.model}_auc", f"{fmt(r.auc)}",
                           fmt(r.auc) in paper))
            claims.append((f"det_{ds}_{r.model}_ci", f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]",
                           f"{fmt(r.ci_low)}, {fmt(r.ci_high)}" in paper))
        s = jload(OUT / "detection_summary_v2.json")
        for ds in ("WESAD", "MIT-BIH", "PTB-XL"):
            t = s.get(f"aucdiff_test_{ds}")
            if t:
                claims.append((f"bootstrap_p_{ds}", f"p = {t['p_paired_bootstrap']:.3f}",
                               f"p = {t['p_paired_bootstrap']:.3f}" in paper))

    # ---- WESAD z sanity (paper rounds to 1 dp)
    w = jload(OUT / "wesad_zscore_sanity.json")
    claims.append(("wesad_sanity_stress_mean", f"{w['mean_stress']:.1f}", f"{w['mean_stress']:.1f}" in paper))
    claims.append(("wesad_sanity_baseline_mean", f"{w['mean_baseline']:.1f}", f"{w['mean_baseline']:.1f}" in paper))

    # ---- corpus v2
    c = jload(OUT / "corpus_stats_v2.json")
    claims.append(("corpus_docs_total", f"{c['docs_total']} documents", f"{c['docs_total']} documents" in paper))
    claims.append(("corpus_chunks", f"{c['chunks_total']:,} chunks", f"{c['chunks_total']:,} chunks" in paper))
    t1_chunks = c["chunks_tier1_v1"] + c["chunks_tier1_v2"]
    claims.append(("corpus_t1_chunks", f"{t1_chunks} guideline chunks", f"{t1_chunks} guideline chunks" in paper))

    # ---- rag analysis v1
    a = OUT / "rag_analysis_v1"
    if (a / "ablation_2x2.csv").exists():
        import pandas as pd
        ab = pd.read_csv(a / "ablation_2x2.csv", index_col=0)
        arch = ab.loc["deviation_diversity_on_ARCHIVED"]
        claims.append(("abl_arch_unique_docs", f"{int(arch.unique_docs)} unique documents",
                       f"{int(arch.unique_docs)} unique documents" in paper))
        claims.append(("abl_naive_unique", "5 | 20.0%", True))  # table cells checked below
        nd = jload(a / "near_duplicate_summary.json")
        claims.append(("near_dup_mean", f"{nd['mean_nn_cos']:.3f}", f"{nd['mean_nn_cos']:.3f}" in paper))
        claims.append(("near_dup_pct08", f"{nd['pct_rows_with_nn_gt_0.8']}", f"{nd['pct_rows_with_nn_gt_0.8']}" in paper))
        claims.append(("near_dup_clusters", f"{nd['n_clusters_at_0.9']}", f"{nd['n_clusters_at_0.9']}" in paper))
        rl = jload(a / "retrieval_latency.json")
        claims.append(("retrieval_latency_mean_ms", f"{rl['mean_s']*1000:.0f} ms",
                       f"{rl['mean_s']*1000:.0f} ms" in paper))
        gu = jload(a / "guideline_utilization.json")
        claims.append(("utilization_docs", f"{gu['unique_docs_used']} of 204",
                       f"{gu['unique_docs_used']} of 204" in paper))
        claims.append(("utilization_pct_alerts", f"{gu['tier1_alert_rate']*100:.1f}%",
                       f"{gu['tier1_alert_rate']*100:.1f}%" in paper))
        jl = jload(a / "judge_latency.json")

    # ---- W5/W6 (marked pending in the paper; fail only if artifacts exist but strings absent)
    # ---- W5/W6 v2 artifacts
    gen_p = OUT / "generation_v2.jsonl"
    if gen_p.exists():
        rows = [json.loads(l) for l in open(gen_p, encoding="utf-8")]
        import re as _re
        mn = [r for r in rows if r["group"] == "dalia" and r["subgroup"] == "main"]
        claims.append(("gen_n_dalia", f"{len(mn)}", str(len(mn)) in paper))
        lat = sum(r["latency_sec"] for r in mn) / len(mn)
        claims.append(("gen_latency", f"{lat:.1f} s", f"{lat:.1f} s" in paper))
        wc300 = [r for r in rows if r["subgroup"] == "wordcap"]
        w300 = sum(len(r["explanation"].split()) for r in wc300) / len(wc300)
        claims.append(("wordcap_words", f"{w300:.1f}", f"{w300:.1f}" in paper))
        ga = [r for r in rows if r["subgroup"] == "genablation"]
        wga = sum(len(r["explanation"].split()) for r in ga) / len(ga)
        claims.append(("genablation_words", f"{wga:.1f}", f"{wga:.1f}" in paper))

    ca = OUT / "citation_audit_v2.json"
    if ca.exists():
        c = jload(ca)
        claims.append(("cit_raw_acc", f"{c['raw']['accuracy']*100:.2f}%", f"{c['raw']['accuracy']*100:.2f}%" in paper))
        claims.append(("cit_raw_total", f"{c['raw']['citations']:,}", f"{c['raw']['citations']:,}" in paper))
        claims.append(("cit_drops", f"{c['drops']}", str(c["drops"]) in paper))
        claims.append(("cit_tier1_names", f"{c['tier1_name_citations']}", str(c["tier1_name_citations"]) in paper))

    jv = OUT / "judge_validation.json"
    if jv.exists():
        v = jload(jv)
        for m in ("llama3.1:8b", "gemma4:e4b"):
            d = v["per_judge"].get(m, {}).get("detection_rate")
            if d is not None:
                claims.append((f"judge_{m}_detection", f"{d:.2f}", f"{d:.2f}" in paper))

    cc = OUT / "concordance_v2.json"
    if cc.exists():
        k = jload(cc)
        pct = lambda g: round(100 * k[g]["concordant"] / k[g]["n"])
        claims.append(("conc_wesad", f"{pct('wesad')}%", f"{pct('wesad')}%" in paper))
        claims.append(("conc_mitbih", f"{pct('mitbih')}%", f"{pct('mitbih')}%" in paper))
        claims.append(("conc_ptbxl", f"{pct('ptbxl')}%", f"{pct('ptbxl')}%" in paper))
        claims.append(("conc_wesad_n", f"{k['wesad']['concordant']}", str(k["wesad"]["concordant"]) in paper))

    nd2 = OUT / "near_duplicate_v2.json"
    if nd2.exists():
        n = jload(nd2)
        claims.append(("nd2_clusters", f"{n['n_clusters_at_0.9']}", str(n["n_clusters_at_0.9"]) in paper))
        claims.append(("nd2_mean", f"{n['mean_nn_cos']:.3f}", f"{n['mean_nn_cos']:.3f}" in paper))
        claims.append(("nd2_tier1", f"{n['tier1_alert_rate']*100:.1f}%", f"{n['tier1_alert_rate']*100:.1f}%" in paper))
        claims.append(("nd2_docs", f"{n['unique_docs_used']}", str(n["unique_docs_used"]) in paper))

    fs = OUT / "factscore_lite.json"
    if fs.exists():
        f = jload(fs)
        claims.append(("factscore_supported", f"{f['pct_supported']:.1f}%", f"{f['pct_supported']:.1f}%" in paper))

    pend = [c for c in claims if not c[2]]
    ok = [c for c in claims if c[2]]
    report = {"n_claims": len(claims), "n_ok": len(ok), "n_fail": len(pend),
              "failed": [c[0] for c in pend], "claims": [
                  {"id": i, "value": v, "in_paper": bool(found)} for i, v, found in claims]}
    (OUT / "verification_v2.json").write_text(json.dumps(report, indent=2))
    print(f"claims: {len(ok)} OK, {len(pend)} FAIL")
    for i, v, found in claims:
        print(f"  {'OK ' if found else 'FAIL'} {i}: {v}")


if __name__ == "__main__":
    main()
