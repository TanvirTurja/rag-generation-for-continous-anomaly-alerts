"""
factscore_lite.py — Atomic-claim verification against retrieved context (fix/v2 W6b).

FActScore-style (Min et al. 2023) decomposition + entailment, fully local:
  - Decomposer: qwen3.5:9b (the generator) splits an explanation into atomic claims.
  - Verifier:   gemma4:e4b (DIFFERENT family from the generator, to avoid
                self-preference) judges each claim SUPPORTED / NOT SUPPORTED /
                UNVERIFIABLE from the context alone.
Sample: 60 explanations (30 PPG-DaLiA main + 30 labeled events).
Output: outputs_v2/factscore_lite.json (+ per-claim CSV).
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs_v2"
GEN = OUT / "generation_v2.jsonl"

DECOMP_PROMPT = """Split the following clinical alert explanation into a numbered list of atomic factual claims (one fact per line, no interpretation). Output ONLY the numbered list.
EXPLANATION:
"""
VERIFY_PROMPT = """You are a strict fact-checker. Given ONLY the context below, decide whether the claim is supported.
Answer with exactly one word: SUPPORTED, UNSUPPORTED, or UNVERIFIABLE.
- SUPPORTED: the context explicitly states or directly implies the claim.
- UNSUPPORTED: the context contradicts the claim.
- UNVERIFIABLE: the context does not address the claim.

CONTEXT:
{context}

CLAIM: {claim}"""


def main():
    import ollama
    import random
    import sys
    random.seed(11)
    sys.path.insert(0, str(ROOT / "scripts" / "v2"))
    from rag_pipeline_v2 import make_retriever
    retrieve = make_retriever()
    rows = [json.loads(l) for l in open(GEN, encoding="utf-8")]
    dalia = [r for r in rows if r["group"] == "dalia" and r["subgroup"] == "main"]
    labeled = [r for r in rows if r["subgroup"] == "labeled"
               and not r["key"].startswith("mitbih|")]
    sample = random.sample(dalia, min(30, len(dalia))) + random.sample(labeled, min(30, len(labeled)))
    for r in sample:
        r["context"] = retrieve(r["query"])["context"]

    claims_rows = []
    for r in sample:
        resp = ollama.chat(model="qwen3.5:9b",
                           messages=[{"role": "user", "content": DECOMP_PROMPT + r["explanation"]}],
                           think=False,
                           options={"temperature": 0.0, "num_predict": 400, "num_ctx": 4000})
        lines = [l.strip() for l in resp["message"]["content"].split("\n")
                 if re.match(r"^\d+[\.\)]", l.strip())]
        for cl in lines:
            cl = re.sub(r"^\d+[\.\)]\s*", "", cl).strip()
            if not cl:
                continue
            v = ollama.chat(model="gemma4:e4b",
                            messages=[{"role": "user", "content":
                                       VERIFY_PROMPT.format(context=r["context"][:9000], claim=cl)}],
                            options={"temperature": 0.0, "num_predict": 10, "num_ctx": 6000})
            verdict = "UNKNOWN"
            for w in ("SUPPORTED", "UNVERIFIABLE", "UNSUPPORTED"):
                if w in v["message"]["content"].upper():
                    verdict = w
                    break
            claims_rows.append({"key": r["key"], "group": r["group"],
                                "claim": cl, "verdict": verdict})
        print(f"  {r['key']}: {len(lines)} claims", flush=True)

    df = pd.DataFrame(claims_rows)
    df.to_csv(OUT / "factscore_lite_claims.csv", index=False)
    summary = {
        "n_explanations": len(sample),
        "n_claims": len(df),
        "pct_supported": round(float((df.verdict == "SUPPORTED").mean() * 100), 2) if len(df) else None,
        "pct_unsupported": round(float((df.verdict == "UNSUPPORTED").mean() * 100), 2) if len(df) else None,
        "pct_unverifiable": round(float((df.verdict == "UNVERIFIABLE").mean() * 100), 2) if len(df) else None,
        "verifier": "gemma4:e4b (different family from generator)",
        "by_group": {g: {"n_claims": int((df.group == g).sum()),
                         "pct_supported": round(float((df[df.group == g].verdict == "SUPPORTED").mean() * 100), 2)}
                     for g in df.group.unique()} if len(df) else {},
    }
    (OUT / "factscore_lite.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
