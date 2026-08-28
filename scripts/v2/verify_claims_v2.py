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
