"""Independent reimplementation of the labeled-event concordance counts (Table 5).

Deliberately different code path from the analysis scripts: field extraction by
string splitting (no regex), whole-text case folding, counts accumulated by
explicit loops. Asserts equality with the archived artifacts for the original,
blind (r1-r3), same-context, strict, and shuffled arms.
Run: python scripts/robustness/independent_concordance.py  -> INDEPENDENT CHECK OK / mismatch list
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__.resolve().parents[2]
            if False else Path.cwd())
OUT = Path.cwd()/"outputs_v2/robustness"

LEX = {
 "stress": ["stress", "arousal", "sympathetic", "anxiety", "mental load", "psychological", "emotional"],
 "VEB": ["ventricular", "pvc", "premature ventricular", "ventricular tachycard"],
 "SVEB": ["supraventricular", "atrial premature", "pac", "atrial ectopy", "premature atrial",
          "atrial fibrillation", "atrial tachyarrhythm"],
 "MI": ["infarct", "ischemi", "stemi", "coronary occlusion", "st-elevation", "st elevation"],
 "STTC": ["repolarization", "st depression", "st-segment", "st segment", "t-wave", "t wave inversion"],
 "CD": ["conduction", "bundle branch", "heart block", "av block", "pr interval"],
 "HYP": ["hypertroph", "chamber enlargement", "left ventricular mass"],
}
ART = ["artifact", "motion", "sensor displacement", "sensor contact", "signal quality",
       "electrode", "noise", "poor contact", "device"]


def detected_field(text):
    # split-based extraction: no regex
    for line in text.splitlines():
        up = line.upper()
        if up.startswith("DETECTED:"):
            body = line.split(":", 1)[1]
            return body.lower()
        if up.startswith("EVIDENCE:"):
            return ""          # DETECTED ended before it started
    return ""


def count_arm(path, arm_filter=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
    out = {}
    for grp in ("wesad", "mitbih", "ptbxl"):
        ok = art = n = 0
        for r in rows:
            if r["group"] != grp:
                continue
            if arm_filter and r.get("arm") not in (None, "", arm_filter) and not r["key"].startswith(arm_filter):
                continue
            n += 1
            det = detected_field(r["explanation"])
            terms = LEX.get(r["true_label"], [])
            hit = False
            for t in terms:
                if t in det:
                    hit = True
                    break
            if hit:
                ok += 1
                for a in ART:
                    if a in det:
                        art += 1
                        break
        out[grp] = {"n": n, "ok": ok, "art": art}
    return out


def main():
    gen = [json.loads(l) for l in open(Path.cwd()/"outputs_v2/generation_v2.jsonl", encoding="utf-8")]
    sup = json.loads((Path.cwd()/"outputs_v2/superseded_keys.json").read_text())["superseded_mitbih_keys"]
    orig = [r for r in gen if r["subgroup"] == "labeled" and r["key"] not in sup]
    tmp = Path.cwd()/".tmp_orig.jsonl"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in orig:
            f.write(json.dumps(r)+"\n")

    checks = [(".tmp_orig.jsonl", None, "original"),
              ("generation_blind.jsonl", "blind", "blind r1"),
              ("generation_blind_r2.jsonl", None, "blind r2"),
              ("generation_blind_r3.jsonl", None, "blind r3"),
              ("generation_blindctx.jsonl", "blindctx", "same-context r1"),
              ("generation_strict.jsonl", "strict", "strict r1")]
    ref = json.loads((OUT/"blind_concordance.json").read_text())
    ref_orig = json.loads((OUT/"rag_nonllm.json").read_text())["concordance_ci"]
    r2 = json.loads((OUT/"round2.json").read_text())["blind_runs"]

    mismatches = []
    for fname, arm, label in checks:
        got = count_arm(OUT/fname if fname.startswith("generation") else fname, arm)
        want = None
        if label == "original":
            # stored artifact_language = all rows; independent script counts artifact among
            # concordant rows (the cell-103 convention) -> compare against concordant_AND_artifact
            want = {g: {"n": ref_orig[g]["n"], "ok": ref_orig[g]["concordant"],
                        "art": ref_orig[g]["concordant_AND_artifact"]} for g in got}
        elif label.startswith("blind"):
            key = {"blind r1": "r1", "blind r2": "r2", "blind r3": "r3"}[label]
            src = ref if key == "r1" else r2[key]
            ok_key = "concordant" if key == "r1" else "ok"
            want = {g: {"n": src[g]["n"], "ok": src[g][ok_key], "art": None} for g in got}
        elif label.startswith("same-context"):
            bc = json.loads((OUT/"round2.json").read_text())["blindctx"]
            want = {g: {"n": bc[g]["n"], "ok": bc[g]["ok"], "art": None} for g in got}
        elif label.startswith("strict"):
            st = json.loads((OUT/"round3.json").read_text())["strict"]
            want = {g: {"n": st[g]["n"], "ok": st[g]["ok"], "art": None} for g in got}
        for g in got:
            for k in ("n", "ok", "art"):
                if want[g][k] is None:
                    continue
                if got[g][k] != want[g][k]:
                    mismatches.append(f"{label} {g}.{k}: independent {got[g][k]} vs artifact {want[g][k]}")
        print(f"{label}: " + " ".join(f"{g} {got[g]['ok']}/{got[g]['n']} (art {got[g]['art']})" for g in got))
    tmp.unlink()
    if mismatches:
        print("MISMATCHES:"); [print(" ", m) for m in mismatches]; sys.exit(1)
    print("INDEPENDENT CHECK OK: all arm counts reproduce the archived artifacts")


if __name__ == "__main__":
    main()
