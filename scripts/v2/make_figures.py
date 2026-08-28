"""make_figures.py — Figure 1 (pipeline diagram) for paper v2 + copy result figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "draft_paper" / "figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12.5, 3.2))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 3.2)
ax.axis("off")

stages = [
    ("1. Stream\nPPG ECG EDA\nTEMP RESP", "#eef3fb"),
    ("2. Window +\nfeaturize\n30 s, 12 feats/ch", "#eef3fb"),
    ("3. Anomaly\ndetection\nIF + LOF\n(train-normal)", "#eef3fb"),
    ("4. Deviation-aware\nretrieval\n(ChromaDB v2 +\nMiniLM)", "#fdf3e7"),
    ("5. Grounded\nexplanation\n(Qwen3.5 9B, raw\ntext preserved)", "#eaf6ee"),
]
W, H, X0, Y0, GAP = 2.1, 1.7, 0.35, 0.85, 0.45
for i, (label, color) in enumerate(stages):
    x = X0 + i * (W + GAP)
    box = FancyBboxPatch((x, Y0), W, H, boxstyle="round,pad=0.08",
                         fc=color, ec="#33507a", lw=1.2)
    ax.add_patch(box)
    ax.text(x + W / 2, Y0 + H / 2, label, ha="center", va="center", fontsize=8.6)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + W + 0.03, Y0 + H / 2),
                                     (x + W + GAP - 0.03, Y0 + H / 2),
                                     arrowstyle="-|>", mutation_scale=14, color="#33507a"))

ax.text(0.35, 2.85, "continuous (on-device)", fontsize=8, color="#33507a", style="italic")
ax.text(7.2, 2.85, "alert-triggered (v2: per-subject baseline-reference queries · validated judges · snap-logged canonicalization)",
        fontsize=8, color="#8a5a00", style="italic")
ax.text(3.4, 0.42, "evaluation: LOSO / inter-patient / val-fold thresholds · 10-seed IF · CIs",
        fontsize=7.5, color="#33507a", ha="center")
ax.text(9.6, 0.42, "evaluation: citation audit (raw vs repaired) · corruption-validated judges\nlabeled-event concordance · FActScore-lite · clinician kit (pending)",
        fontsize=7.5, color="#2f6b45", ha="center")

fig.tight_layout()
fig.savefig(OUT / "pipeline_v2.png", dpi=220)
plt.close(fig)
print("wrote", OUT / "pipeline_v2.png")

# copy result figures from outputs_v2
src = ROOT / "outputs_v2" / "figures_v2"
for f in src.glob("*.png"):
    (OUT / f.name).write_bytes(f.read_bytes())
    print("copied", f.name)
