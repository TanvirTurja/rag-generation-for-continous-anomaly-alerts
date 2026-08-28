"""
analyze_ratings.py — Analyze filled clinician rating forms (W7).

Reads all ratings_<rater>.csv in this folder (columns: item_id, faithfulness,
actionability, potential_harm, harm_note, overall) plus sample_key.csv.
Writes clinician_results.json: per-axis means (per rater + pooled),
Fleiss' kappa on faithfulness/actionability/overall, harm-flag counts,
per-group (labeled vs dalia vs wordcap) summaries, and items needing
adjudication (any axis spread >= 2).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def fleiss_kappa(df, col):
    """Fleiss' kappa for m raters over n items, categories = observed values."""
    table = df.pivot_table(index="item_id", columns="rater", values=col)
    table = table.dropna()
    if len(table) < 2:
        return None
    cats = sorted(pd.unique(df[col].dropna()))
    n_raters = table.shape[1]
    counts = np.zeros((len(table), len(cats)))
    for j, c in enumerate(cats):
        counts[:, j] = (table == c).sum(axis=1)
    p_j = counts.sum(axis=0) / counts.sum()
    P_i = (counts ** 2).sum(axis=1) - n_raters
    P_i /= n_raters * (n_raters - 1)
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    return round(float((P_bar - P_e) / (1 - P_e)), 4) if P_e < 1 else None


def main():
    frames = []
    for f in sorted(HERE.glob("ratings_*.csv")):
        rater = f.stem.replace("ratings_", "")
        df = pd.read_csv(f)
        df["rater"] = rater
        frames.append(df)
    if not frames:
        print("No ratings_*.csv files found.")
        return
    df = pd.concat(frames)
    key = pd.read_csv(HERE / "sample_key.csv")
    df = df.merge(key, on="item_id")

    res = {
        "n_raters": df.rater.nunique(),
        "n_items": df.item_id.nunique(),
        "pooled_means": {c: round(float(df[c].mean()), 3)
                         for c in ("faithfulness", "actionability", "overall")},
        "per_rater_means": {r: {c: round(float(g[c].mean()), 3)
                                for c in ("faithfulness", "actionability", "overall")}
                            for r, g in df.groupby("rater")},
        "fleiss_kappa": {c: fleiss_kappa(df, c)
                         for c in ("faithfulness", "actionability", "overall")},
        "potential_harm": {
            "n_yes": int((df.potential_harm.astype(str).str.lower() == "yes").sum()),
            "n_total": int(len(df)),
            "harm_items": df.loc[df.potential_harm.astype(str).str.lower() == "yes",
                                 ["item_id", "rater", "harm_note"]].to_dict("records"),
        },
        "by_group": {g: {c: round(float(d[c].mean()), 3)
                         for c in ("faithfulness", "actionability", "overall")}
                     for g, d in df.groupby("group")},
    }
    # items needing adjudication: spread >= 2 on any scored axis
    spread = df.groupby("item_id")[["faithfulness", "actionability", "overall"]].agg(
        lambda s: s.max() - s.min())
    res["adjudication_items"] = spread[(spread >= 2).any(axis=1)].reset_index().to_dict("records")

    (HERE / "clinician_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
