"""
make_notebooks.py — Build and EXECUTE detection_v2.ipynb and rag_v2.ipynb.

The notebooks are the reproducibility interface (matching the v1 project's
detection.ipynb / rag.ipynb convention): each rebuilds everything from scratch
on a clean machine (delegating to the v2 scripts) and loads cached artifacts
instantly where they exist. Execution happens here via nbclient so the .ipynb
files ship with frozen outputs.
"""
import nbformat as nbf
from nbclient import NotebookClient

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)


def build_detection_nb():
    c = [
        md("""# Detection v2 — Contamination-Hardened Evaluation

Rebuilds and presents the clean-protocol detection results of the revised paper
(§4.1–4.3). Protocols were **pre-registered** in `THRESHOLDS.md` before any v2 run:

- **WESAD**: leave-one-subject-out (fixes v1's train-inside-eval + intra-subject leakage)
- **MIT-BIH**: inter-patient DS1→DS2, paced records 102/104/107/217 excluded (AAMI convention)
- **PTB-XL**: operating threshold selected on validation fold 9, frozen onto fold 10
  (fixes v1's test-prevalence circularity)
- Isolation Forest over **10 seeds**; bootstrap 95% CIs; paired-bootstrap IF-vs-LOF tests
- Baselines: one-class SVM + dense autoencoder (both new in v2)

Running this notebook top-to-bottom regenerates `outputs_v2/` (feature caches are
reused if present; ~15 min cold)."""),
        code("""import subprocess, sys
from pathlib import Path
import pandas as pd, json

ROOT = Path.cwd()
OUT = ROOT / "outputs_v2"
if not (OUT / "detection_results_v2.csv").exists():
    print("outputs missing -> running scripts/v2/detection_v2.py (cold start)...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "v2" / "detection_v2.py")], check=True)
print("run log tail:")
print("\\n".join((OUT / "detection_v2_run.log").read_text().splitlines()[-6:]))"""),
        md("## Table 2 — clean-protocol results (paper §4.1)"),
        code("""det = pd.read_csv(OUT / "detection_results_v2.csv")
det.style.format({"auc": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}",
                  "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}",
                  "auc_sd_over_seeds": "{:.3f}", "auc_seed0": "{:.3f}"}, na_rep="")
det[["dataset", "protocol", "model", "auc", "ci_low", "ci_high", "precision", "recall", "f1"]]"""),
        md("## Table 3 — what the protocol fixes change (same features, protocol only)"),
        code("""v1 = pd.read_csv(ROOT / "outputs_v1_archive" / "detection_results.csv")
v1["protocol"] = "v1 (contaminated)"
cmp = pd.concat([
    v1[["Dataset", "Model", "AUC", "protocol"]].rename(columns=str.lower),
    det.assign(protocol="v2 (clean)")[["dataset", "model", "auc", "protocol"]],
], ignore_index=True)
cmp.pivot_table(index=["dataset", "model"], columns="protocol", values="auc").round(3)"""),
        md("## Significance: paired bootstrap IF vs LOF (THRESHOLDS.md deviation 1)"),
        code("""s = json.load(open(OUT / "detection_summary_v2.json"))
for ds in ("WESAD", "MIT-BIH", "PTB-XL"):
    t = s[f"aucdiff_test_{ds}"]
    print(f"{ds}: IF {t['auc_if']:.3f} vs LOF {t['auc_lof']:.3f}, paired-bootstrap p = {t['p_paired_bootstrap']:.4g}")"""),
        md("## Feature-group ablation (PTB-XL, LOF) — replaces v1's 'feature budget ceiling' assertion"),
        code("""pd.read_csv(OUT / "feature_ablation_ptbxl.csv").style.format({"auc": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"})
pd.read_csv(OUT / "feature_ablation_ptbxl.csv")"""),
        md("## Figures"),
        code("""from IPython.display import Image, display
for f in ("roc_wesad", "roc_mitbih", "roc_ptbxl", "ptbxl_threshold_curve"):
    print(f"--- {f}")
    display(Image(filename=str(OUT / "figures_v2" / f"{f}.png"), width=520))"""),
        md("## WESAD per-subject LOSO AUCs (spread behind the pooled number)"),
        code("""ps = pd.read_csv(OUT / "wesad_loso_per_subject.csv")
ps.pivot(index="subject", columns="model", values="auc").round(3)"""),
        md("""## Reading

The v1 headline (MIT-BIH LOF 0.899) does not survive the community-standard
inter-patient protocol — LOF falls to chance (0.502) while IF holds 0.670; the
autoencoder wins WESAD. Detector rankings are protocol-dependent, and the
pre-registered file + this notebook reproduce every number above from raw data."""),
    ]
    return nbf.v4.new_notebook(cells=c, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def build_rag_nb():
    c = [
        md("""# RAG v2 — Alert-Triggered Grounded Explanations (revised evaluation)

Presents the v2 explanation pipeline results (paper §4.4–4.10). v2 changes vs v1:

- **Query semantics fixed**: z-scores vs each subject's *non-flagged* windows (the
  detector's normal population), evidence-tied character phrases, online-capable.
- **Corpus expanded** with wearable-relevant guidance (EHRA 2022 digital-devices
  guide; 2023 ACC/AHA AF guideline).
- **Raw pre-canonicalization text preserved** → citation accuracy reported before
  AND after repair (v1's "99.3% → 100%" was not reconstructable).
- **Judges validated** on a 200-item corruption benchmark before use; v1's judge
  (llama3.1:8b) detects 0/100 corruptions.
- **Labeled-event evaluation**: 148 true events (WESAD stress, MIT-BIH ectopy,
  PTB-XL pathology) with labels never entering the queries.
- Atomic-claim verification + released clinician kit.

Cold-start rebuild: this notebook delegates to `scripts/v2/rag_pipeline_v2.py`
and `scripts/v2/judge_eval_v2.py` when artifacts are missing (GPU generation
~2 h + judging ~1 h + OpenRouter key required for API columns)."""),
        code("""import json, re
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path.cwd()
OUT = ROOT / "outputs_v2"
GEN = OUT / "generation_v2.jsonl"
rows = [json.loads(l) for l in open(GEN, encoding="utf-8")]
sup = json.loads((OUT / "superseded_keys.json").read_text())["superseded_mitbih_keys"]
rows = [r for r in rows if r["key"] not in sup]
df = pd.DataFrame([{k: r.get(k) for k in ("key", "group", "subgroup", "model", "prompt",
                                           "true_label", "latency_sec")} for r in rows])
df.groupby(["group", "subgroup"]).size().to_frame("n")"""),
        md("## Corpus v2 (206 documents / 5,045 chunks)"),
        code("""cs = json.load(open(OUT / "corpus_stats_v2.json"))
print({k: cs[k] for k in ("docs_total", "chunks_total", "chunks_tier1_v1", "chunks_tier1_v2",
                          "chunks_tier2", "mean_chunk_words")})
pd.DataFrame([m for m in cs["tier1_v2_manifest"]])""",
         ),
        md("## Citation audit — raw vs repaired (paper §4.4)"),
        code("""ca = json.load(open(OUT / "citation_audit_v2.json"))
print(json.dumps(ca, indent=2)[:900])"""),
        md("## Judge validation on the corruption benchmark (paper §4.5)"),
        code("""jv = json.load(open(OUT / "judge_validation.json"))
pd.DataFrame(jv["per_judge"]).T
print("selected local judge:", jv["selected_local_judge"]["model"],
      "| criterion:", jv["selected_local_judge"]["criterion"])"""),
        md("## Main judge scores (validated local judge; paper §4.6)"),
        code("""ev = pd.read_csv(OUT / "rag_evaluation_v2.csv")
agg = ev.groupby("subgroup").agg(n=("local_faithfulness", "size"),
                                 faith=("local_faithfulness", "mean"),
                                 relev=("local_relevance", "mean"),
                                 compl=("local_completeness", "mean")).round(2)
agg
m = ev[(ev.group == "dalia") & (ev.subgroup == "main")]
print("dalia faithfulness dist:", m.local_faithfulness.value_counts().sort_index().to_dict())"""),
        md("## Labeled-event concordance — the integration result (paper §4.7)"),
        code("""cc = json.load(open(OUT / "concordance_v2.json"))
tab = pd.DataFrame({g: {"n": v["n"], "concordant": v["concordant"],
                        "concordant_%": round(100 * v["concordant"] / v["n"]),
                        "artifact_language": v["artifact_conclusion"]}
                    for g, v in cc.items()}).T
tab"""),
        md("## Before/after query-and-corpus fix (398 wearable alerts, same flags)"),
        code("""nd2 = json.load(open(OUT / "near_duplicate_v2.json"))
nd1 = json.load(open(OUT / "rag_analysis_v1" / "near_duplicate_summary.json"))
pd.DataFrame({"v1": {"clusters@0.9": nd1["n_clusters_at_0.9"], "mean_NN_cos": nd1["mean_nn_cos"],
                     "pct_twins>0.9": nd1["pct_rows_with_nn_gt_0.9"], "guideline_reach_%": 6.5},
              "v2": {"clusters@0.9": nd2["n_clusters_at_0.9"], "mean_NN_cos": nd2["mean_nn_cos"],
                     "pct_twins>0.9": nd2["pct_rows_with_nn_gt_0.9"],
                     "guideline_reach_%": round(100 * nd2["tier1_alert_rate"], 1)}}).T"""),
        md("## Atomic-claim verification (paper §4.9)"),
        code("""fs = json.load(open(OUT / "factscore_lite.json"))
print(json.dumps(fs, indent=2))"""),
        md("## Ablations (paper §4.8) + latency"),
        code("""for sg in ("main", "wordcap", "genablation"):
    s = ev[(ev.subgroup == sg) & (ev.group == "dalia")] if sg == "main" else ev[ev.subgroup == sg]
    words = np.mean([len(r["explanation"].split()) for r in rows
                     if (r["subgroup"] == sg and (sg != "main" or r["group"] == "dalia"))])
    print(f"{sg:12s} n={len(s):3d} words={words:6.1f} faith={s.local_faithfulness.mean():.2f} compl={s.local_completeness.mean():.2f}")
mn = [r for r in rows if r["group"] == "dalia" and r["subgroup"] == "main"]
print(f"generation latency: mean {np.mean([r['latency_sec'] for r in mn]):.1f} s/alert")"""),
        md("## Example alert (paper §4.10)"),
        code("""def sec(t, a, b):
    m = re.search(rf"{a}:\\s*(.*?)(?={b}:|$)", t, re.S)
    return m.group(1).strip() if m else ""

ex = next(r for r in rows if r["group"] == "wesad"
          and "stress" in sec(r["explanation"], "DETECTED", "EVIDENCE").lower())
print("TRUE LABEL:", ex["true_label"])
print("QUERY:", ex["query"])
print()
print(ex["explanation"])"""),
        md("""## Reading

Under a validated judge the system's faithfulness averages 2.2/3 with ~44% of
explanations containing extrapolated claims; 47.7% of atomic claims are
unverifiable from the retrieved context; and explanations name the true
condition for 94% of stress events but 12% / 6% of ECG pathology, attributing
42–56% of true pathology to artifact. The clinician kit in `clinician_eval/`
adjudicates these findings with human raters."""),
    ]
    return nbf.v4.new_notebook(cells=c, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def execute_and_save(nb, path):
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    nbf.write(nb, path)
    print("executed + saved", path)


if __name__ == "__main__":
    execute_and_save(build_detection_nb(), "detection_v2.ipynb")
    execute_and_save(build_rag_nb(), "rag_v2.ipynb")
