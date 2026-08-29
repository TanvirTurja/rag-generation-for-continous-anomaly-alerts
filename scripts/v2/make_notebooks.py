"""
make_notebooks.py — Build + execute detection_v2.ipynb / rag_v2.ipynb in the
exact style of the v1 notebooks: `# === SECTION ===` / `### === sub-step ===`
markdown rhythm, ONE small code cell per step, and a verification/inspection
cell after every load or transformation. Caches only speed re-runs.
"""
import nbformat as nbf
from nbclient import NotebookClient

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)
KS = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}


def cells_to_nb(cells):
    return nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": KS["kernelspec"]})


def execute_and_save(nb, path):
    client = NotebookClient(nb, timeout=1800, kernel_name="python3")
    client.execute()
    nbf.write(nb, path)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    errs = [o for c in nb.cells if c.cell_type == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error"]
    print(f"executed + saved {path} ({n_code} code cells, {len(errs)} errors)")


# ============================================================== detection_v2
def detection_cells():
    c = []
    A = c.append

    A(md("""# Detection v2 — Contamination-Hardened Evaluation

Rebuilds §4.1–4.3 of the revised paper, in the style of the original `detection.ipynb`.
All protocols were **pre-registered** in `THRESHOLDS.md` before any v2 run:

| v1 defect | v2 protocol |
|---|---|
| WESAD train ⊂ eval, intra-subject | leave-one-subject-out (LOSO) |
| MIT-BIH intra-patient, train ⊂ eval | inter-patient DS1→DS2, paced (102/104/107/217) excluded |
| PTB-XL threshold = test-score pct matched to test prevalence | pct chosen on fold 9, frozen onto fold 10 |
| single run, no CIs, no baselines | 10-seed IF, bootstrap CIs, paired-bootstrap, OC-SVM + AE baselines |

Features are v1-identical, so every difference below is protocol only."""))

    A(md("# === Imports ==="))
    A(md("### === Core scientific stack ==="))
    A(code("import json, pickle, time, warnings\nfrom collections import Counter\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd"))
    A(md("### === Signal processing ==="))
    A(code("from scipy.stats import skew, kurtosis, mannwhitneyu"))
    A(md("### === Anomaly detection + evaluation ==="))
    A(code("""from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
warnings.filterwarnings("ignore")"""))
    A(md("### === Plotting ==="))
    A(code("import matplotlib.pyplot as plt"))
    A(md("### === Deep-learning baseline (autoencoder) ==="))
    A(code("import torch, torch.nn as nn\nDEV = \"cuda\" if torch.cuda.is_available() else \"cpu\"\nprint('torch device:', DEV)"))

    A(md("# === Project root & paths ==="))
    A(code("""PROJECT_ROOT = Path.cwd()
OUTPUT_DIR   = PROJECT_ROOT / "outputs_v2"
CACHE_DIR    = OUTPUT_DIR / "cache"
FIG_DIR      = OUTPUT_DIR / "figures_v2"
for d in (CACHE_DIR, FIG_DIR): d.mkdir(parents=True, exist_ok=True)

PPGDALIA_DIR = PROJECT_ROOT / "Dataset" / "ppg+dalia" / "data" / "PPG_FieldStudy"
WESAD_DIR    = PROJECT_ROOT / "Dataset" / "WESAD"
MITBIH_DIR   = PROJECT_ROOT / "Dataset" / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"
PTBXL_DIR    = PROJECT_ROOT / "Dataset" / "ptb-xl-1.0.3"
print("paths OK — outputs:", OUTPUT_DIR.name)"""))

    A(md("# === Pre-registered protocol constants (THRESHOLDS.md) ==="))
    A(md("### === Channels & windowing (v1-identical) ==="))
    A(code("""CHANNELS = ["ecg", "resp", "bvp", "wrist_eda", "wrist_temp"]
FS_MAP = {"ecg": 700, "resp": 700, "bvp": 64, "wrist_eda": 4, "wrist_temp": 4}
WIN_SEC = 30
WESAD_SUBJECTS = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]   # S1/S12 missing in dataset"""))
    A(md("### === MIT-BIH inter-patient split (de Chazal standard; paced excluded) ==="))
    A(code("""PACED = {"102","104","107","217"}
DS1 = ["101","106","108","109","112","114","115","116","118","119",
       "122","124","201","203","205","207","208","209","215","220","223","230"]
DS2 = ["100","103","105","111","113","117","121","123","200","202",
       "210","212","213","214","219","221","222","228","231","232","233","234"]
assert len(DS1)==22 and len(DS2)==22
assert not (set(DS1) & set(DS2)) and not ((set(DS1)|set(DS2)) & PACED)
assert len(set(DS1)|set(DS2)|PACED) == 48
print("inter-patient split OK: 22 train + 22 test, paced excluded:", sorted(PACED))"""))
    A(md("### === AAMI labels (v2 adds 'j' to the normal set) ==="))
    A(code("""AAMI_NORMAL = {"N","L","R","e","j"}
BEAT_SYMBOLS = AAMI_NORMAL | {"A","a","J","S","V","E","F","f","Q","/","!"}
MITBIH_HALF, FS_MITBIH = 144, 360
print("normal beats:", sorted(AAMI_NORMAL), "| all beat symbols:", len(BEAT_SYMBOLS))"""))
    A(md("### === Reproducibility & pre-registered settings ==="))
    A(code("""IF_SEEDS = list(range(10))     # no seed selection; mean ± SD reported
N_BOOT  = 1000
RNG = np.random.default_rng(42)
MODELS  = ["IF", "LOF", "OCSVM", "AE"]
print("10 IF seeds, 1000 bootstrap resamples, models:", MODELS)"""))

    A(md("# === Feature pipeline (v1-identical) ==="))
    A(md("### === featurize ==="))
    A(code("""def featurize(window):
    \"\"\"12 summary stats per 1-D window — identical to v1.\"\"\"
    w = np.asarray(window).ravel().astype(float)
    return [w.mean(), w.std(), w.min(), w.max(), np.ptp(w),
            np.median(w), skew(w), kurtosis(w),
            np.percentile(w,25), np.percentile(w,75),
            np.sum(np.diff(w)>0)/len(w), np.sqrt(np.mean(np.diff(w)**2))]

FEATURE_NAMES = ["mean","std","min","max","ptp","median","skew","kurt","p25","p75","up_ratio","roughness"]
FEATURE_GROUPS = {"location":["mean","median","p25","p75"], "spread":["std","min","max","ptp"],
                  "shape":["skew","kurt"], "dynamics":["up_ratio","roughness"]}
print(len(FEATURE_NAMES), "features:", FEATURE_NAMES)"""))
    A(md("### === make_windows ==="))
    A(code("def make_windows(sig, fs, win_sec=WIN_SEC):\n    n = fs*win_sec\n    return [sig[i:i+n] for i in range(0, len(sig)-n, n)]"))
    A(md("### === clean_signal ==="))
    A(code("""def clean_signal(sig):
    \"\"\"Kill Inf, interpolate short NaN gaps (v1 logic).\"\"\"
    s = np.asarray(sig, dtype=float).copy()
    s[np.isinf(s)] = np.nan
    if np.isnan(s).any():
        idx = np.arange(len(s)); good = ~np.isnan(s)
        if good.sum() < 10: return s
        s = np.interp(idx, idx[good], s[good])
    return s"""))
    A(md("### === featurize_subject ==="))
    A(code("""def featurize_subject(data, channels=CHANNELS):
    feats_list, n_windows = [], None
    for ch in channels:
        feats = np.array([featurize(w) for w in make_windows(clean_signal(data[ch]), fs=FS_MAP[ch])])
        if n_windows is None: n_windows = feats.shape[0]
        feats_list.append(feats[:n_windows])
    return np.hstack(feats_list), n_windows"""))
    A(md("### === window_labels ==="))
    A(code("""def window_labels(label_sig, fs, win_sec, n_windows):
    # Dominant label per window (v1 logic)
    n = fs*win_sec; labels = []
    for i in range(n_windows):
        chunk = label_sig[i*n:(i+1)*n]
        labels.append(int(Counter(chunk).most_common(1)[0][0]) if len(chunk) else 0)
    return np.array(labels)"""))
    A(md("### === pkl loader (PPG-DaLiA & WESAD share the format) ==="))
    A(code("""def load_pkl_subject(base, subject):
    with open(base/f"S{subject}"/f"S{subject}.pkl", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    chest, wrist = d["signal"]["chest"], d["signal"]["wrist"]
    return {"subject": subject,
            "ecg": chest["ECG"].ravel(), "resp": chest["Resp"].ravel(),
            "bvp": wrist["BVP"].ravel(), "wrist_eda": wrist["EDA"].ravel(),
            "wrist_temp": wrist["TEMP"].ravel(), "label": d["label"].ravel()}"""))

    A(md("# === WESAD: Load & featurize all 15 subjects ==="))
    A(md("### === Verify loader on S2 ==="))
    A(code("""d2 = load_pkl_subject(WESAD_DIR, 2)
for k, v in d2.items():
    print(f"{k:10s}", v.shape if hasattr(v, "shape") else v)"""))
    A(md("### === Featurize all 15 subjects (cached after first build) ==="))
    A(code("""def build_wesad():
    per = {}
    for s in WESAD_SUBJECTS:
        d = load_pkl_subject(WESAD_DIR, s)
        X, n = featurize_subject(d)
        y = window_labels(d["label"], fs=700, win_sec=WIN_SEC, n_windows=n)
        per[s] = (X, y)
        print(f"  S{s}: {n} windows ({int((y==2).sum())} stress, {int((y==3).sum())} amusement)")
    return per

t0 = time.time(); p = CACHE_DIR/"wesad.npz"
if p.exists():
    z = np.load(p, allow_pickle=True)
    per_subject = {int(s): (z[f"X_{s}"], z[f"y_{s}"]) for s in z["subjects"]}
    print(f"loaded cache: {len(per_subject)} subjects")
else:
    per_subject = build_wesad()
    np.savez_compressed(p, subjects=np.array(WESAD_SUBJECTS),
        **{f"{k}_{s}": v for s,(X,y) in per_subject.items() for k,v in (("X",X),("y",y))})
    print(f"built + cached in {time.time()-t0:.0f}s")"""))
    A(md("### === Overall label distribution ==="))
    A(code("""y_all = np.concatenate([per_subject[s][1] for s in WESAD_SUBJECTS])
print("total windows:", len(y_all), "| label counts:", dict(Counter(y_all)))
print("baseline windows (train pool for LOSO):", int((y_all==1).sum()))"""))

    A(md("# === MIT-BIH: inter-patient beat extraction ==="))
    A(md("### === Load one record to inspect ==="))
    A(code("""import wfdb
sig100, meta100 = wfdb.rdsamp(str(MITBIH_DIR/"100"))
ann100 = wfdb.rdann(str(MITBIH_DIR/"100"), "atr")
print("record 100:", sig100.shape, meta100["fs"], "Hz | leads:", meta100["sig_name"])
print("annotation symbols:", Counter(ann100.symbol).most_common(8))"""))
    A(md("### === Beat extraction function (v2 labels: full AAMI normal set) ==="))
    A(code("""def extract_beats(record, ann):
    # 0.8 s beat segments around annotated R-peaks; AAMI labels
    beats, labels, syms, samples = [], [], [], []
    for i, sym in zip(ann.sample, ann.symbol):
        if sym not in BEAT_SYMBOLS: continue
        start, end = i-MITBIH_HALF, i+MITBIH_HALF
        if start < 0 or end > record.shape[0]: continue
        beats.append(featurize(record[start:end, 0]))
        labels.append(0 if sym in AAMI_NORMAL else 1)
        syms.append(sym); samples.append(i)
    return beats, labels, syms, samples

b, l, s_, smp = extract_beats(sig100, ann100)
print(f"record 100: {len(b)} beats ({sum(l)} anomalous) | symbols: {Counter(s_).most_common(5)}")"""))
    A(md("### === Extract + featurize all 44 records (cached after first build) ==="))
    A(code("""def build_mitbih():
    Xs, ys, recs, syms = [], [], [], Counter()
    for rec in sorted(set(DS1)|set(DS2)|PACED):
        sig, _ = wfdb.rdsamp(str(MITBIH_DIR/rec))
        ann = wfdb.rdann(str(MITBIH_DIR/rec), "atr")
        bts, labs, sy, _ = extract_beats(sig, ann)
        Xs += bts; ys += labs; recs += [rec]*len(bts); syms += sy
        print(f"  {rec}: {len(bts)} beats ({sum(labs)} anomalous)")
    return np.array(Xs), np.array(ys), np.array(recs), syms

t0 = time.time(); p = CACHE_DIR/"mitbih.npz"
if p.exists():
    z = np.load(p, allow_pickle=True)
    X_m, y_m, rec_m = z["X"], z["y"], z["record"].astype(str)
    symbols = Counter(json.loads(str(z["symbols"])))
    print("loaded cache")
else:
    X_m, y_m, rec_m, symbols = build_mitbih()
    np.savez_compressed(p, X=X_m, y=y_m, record=rec_m, symbols=np.array(json.dumps(dict(symbols))))
    print(f"built + cached in {time.time()-t0:.0f}s")
print("total beats:", len(y_m), "| symbol census:", symbols.most_common(10))"""))
    A(md("### === Inter-patient split stats ==="))
    A(code("""tr_m = np.isin(rec_m, DS1) & (y_m == 0)      # train: DS1 normal beats ONLY
te_m = np.isin(rec_m, DS2)                   # test: DS2 beats ONLY (no DS1 scored)
print(f"train: {int(tr_m.sum()):,} DS1-normal beats")
print(f"test:  {int(te_m.sum()):,} DS2 beats ({int(y_m[te_m].sum()):,} anomalous = {y_m[te_m].mean()*100:.1f}%)")"""))

    A(md("# === PTB-XL: labels & featurization ==="))
    A(md("### === Load + parse superclasses ==="))
    A(code("""import ast
db = pd.read_csv(PTBXL_DIR/"ptbxl_database.csv")
scp = pd.read_csv(PTBXL_DIR/"scp_statements.csv", index_col=0)

def supers(cp_str):
    out = set()
    for c in ast.literal_eval(cp_str):
        if c in scp.index and isinstance(scp.loc[c,"diagnostic_class"], str):
            out.add(scp.loc[c,"diagnostic_class"])
    return out

db["super"] = db.scp_codes.apply(supers)
db["y"] = db.super.apply(lambda st: 0 if st=={"NORM"} else (-1 if not st else 1))
db_clean = db[db.y != -1].reset_index(drop=True)
print(f"{len(db)} -> {len(db_clean)} usable | normal {int((db_clean.y==0).sum()):,} | pathology {int((db_clean.y==1).sum()):,}")
print("superclass counts:", Counter(s for st in db_clean.super for s in st))"""))
    A(md("### === Featurize all ECGs (cached after first build) ==="))
    A(code("""def build_ptbxl():
    Xs, ys, folds = [], [], []
    for idx, row in db_clean.iterrows():
        if idx % 4000 == 0: print(f"  PTB-XL {idx}/{len(db_clean)}")
        try:
            sig, _ = wfdb.rdsamp(str(PTBXL_DIR/row["filename_lr"]))
            Xs.append(featurize(sig[:,0])); ys.append(row["y"]); folds.append(row["strat_fold"])
        except Exception: continue
    return np.array(Xs), np.array(ys), np.array(folds)

t0 = time.time(); p = CACHE_DIR/"ptbxl.npz"
if p.exists():
    z = np.load(p, allow_pickle=True); X_p, y_p, f_p = z["X"], z["y"], z["fold"]
    print("loaded cache")
else:
    X_p, y_p, f_p = build_ptbxl()
    np.savez_compressed(p, X=X_p, y=y_p, fold=f_p)
    print(f"built + cached in {time.time()-t0:.0f}s")
print(f"n={len(y_p):,} | fold9 (val) {int((f_p==9).sum())} | fold10 (test) {int((f_p==10).sum())} "
      f"({int(y_p[f_p==10].sum())} pathology)")"""))

    A(md("# === Models ==="))
    A(md("### === Autoencoder (new v2 baseline) ==="))
    A(code("""def train_autoencoder(X_train):
    # dense AE with early stop on 10% of train normals
    torch.manual_seed(0)
    sc = StandardScaler().fit(X_train)
    Xt = torch.tensor(sc.transform(X_train), dtype=torch.float32, device=DEV)
    d = Xt.shape[1]
    net = nn.Sequential(nn.Linear(d,d//2), nn.ReLU(), nn.Linear(d//2,d//4), nn.ReLU(),
                        nn.Linear(d//4,d//2), nn.ReLU(), nn.Linear(d//2,d)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n_val = max(1, int(0.1*len(Xt))); perm = torch.randperm(len(Xt), device=DEV)
    Xtr, Xva = Xt[perm[n_val:]], Xt[perm[:n_val]]
    best, patience, state = np.inf, 0, None
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr), batch_size=256, shuffle=True)
    for epoch in range(100):
        for (b,) in dl:
            opt.zero_grad(); ((net(b)-b)**2).mean().backward(); opt.step()
        with torch.no_grad(): v = ((net(Xva)-Xva)**2).mean().item()
        if v < best-1e-6: best, patience, state = v, 0, {k:t.detach().clone() for k,t in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 10: break
    if state: net.load_state_dict(state)
    return sc, net"""))
    A(md("### === Unified fit_score ==="))
    A(code("""def fit_score(model_name, X_train, X_eval, seed=0, contamination=0.15):
    \"\"\"Fit on training (normal) data; return (eval_scores, train_scores).\"\"\"
    if model_name == "IF":
        m = make_pipeline(StandardScaler(),
                          IsolationForest(n_estimators=100, contamination=contamination, random_state=seed))
        m.fit(X_train); return -m.score_samples(X_eval), -m.score_samples(X_train)
    if model_name == "LOF":
        m = make_pipeline(StandardScaler(),
                          LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contamination))
        m.fit(X_train); return -m.score_samples(X_eval), -m.score_samples(X_train)
    if model_name == "OCSVM":
        m = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=contamination))
        m.fit(X_train); return -m.decision_function(X_eval), -m.decision_function(X_train)
    if model_name == "AE":
        sc, net = train_autoencoder(X_train)
        with torch.no_grad():
            Xe = torch.tensor(sc.transform(X_eval), dtype=torch.float32, device=DEV)
            Xt = torch.tensor(sc.transform(X_train), dtype=torch.float32, device=DEV)
            return (((net(Xe)-Xe)**2).mean(dim=1).cpu().numpy(),
                    ((net(Xt)-Xt)**2).mean(dim=1).cpu().numpy())
    raise KeyError(model_name)"""))
    A(md("### === Statistics helpers ==="))
    A(code("""def prf(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return p, r, f

def boot_ci(y, s, n=N_BOOT):
    aucs = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max(): continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(aucs,2.5)), float(np.percentile(aucs,97.5))

def paired_boot(y, s1, s2, n=N_BOOT):
    diffs = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max(): continue
        diffs.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    d = np.array(diffs)
    return float(max(2*min((d<=0).mean(), (d>=0).mean()), 1.0/len(d)))"""))
    A(md("### === Verify fit_score on one WESAD fold ==="))
    A(code("""Xtr = np.vstack([per_subject[s][0][per_subject[s][1]==1] for s in WESAD_SUBJECTS if s != 2])
Xte, yte = per_subject[2]
keep = np.isin(yte, [1,2,3]); Xte, yte2 = Xte[keep], (yte[keep]==2).astype(int)
for m in MODELS:
    se, st = fit_score(m, Xtr, Xte)
    print(f"  {m:5s} held-out S2: AUC {roc_auc_score(yte2, se):.3f}")"""))

    A(md("# === WESAD: leave-one-subject-out ==="))
    A(md("### === Full LOSO loop (all 4 models) ==="))
    A(code("""wesad_rows, wesad_ps = [], []
pooled = {}
for model in MODELS:
    S, Y, P = [], [], []
    for held in WESAD_SUBJECTS:
        Xtr = np.vstack([per_subject[s][0][per_subject[s][1]==1] for s in WESAD_SUBJECTS if s != held])
        Xte, yte = per_subject[held]
        keep = np.isin(yte, [1,2,3]); Xte, yte = Xte[keep], (yte[keep]==2).astype(int)
        seeds = IF_SEEDS if model=="IF" else [0]
        for seed in seeds:
            se, st = fit_score(model, Xtr, Xte, seed=seed)
            if seed == 0: se0, pred0 = se, (se > np.percentile(st, 85)).astype(int)
        S.append(se0); Y.append(yte); P.append(pred0)
        wesad_ps.append({"subject": held, "model": model, "auc": roc_auc_score(yte, se0)})
        print(f"  {model:5s} held-out S{held}: AUC {roc_auc_score(yte, se0):.3f}")
    s_all, y_all_w, p_all = np.concatenate(S), np.concatenate(Y), np.concatenate(P)
    auc = roc_auc_score(y_all_w, s_all); lo, hi = boot_ci(y_all_w, s_all)
    prec, rec, f1 = prf(y_all_w, p_all)
    pooled[model] = (s_all, y_all_w)
    wesad_rows.append({"dataset":"WESAD","protocol":"LOSO","model":model,"auc":auc,
                       "ci_low":lo,"ci_high":hi,"precision":prec,"recall":rec,"f1":f1})
    print(f"==> {model}: pooled AUC {auc:.3f} [{lo:.3f},{hi:.3f}] F1 {f1:.3f}\\n")"""))
    A(md("### === Per-subject AUC spread ==="))
    A(code("pd.DataFrame(wesad_ps).pivot(index='subject', columns='model', values='auc').round(3)"))

    A(md("# === MIT-BIH: inter-patient DS1 → DS2 ==="))
    A(md("### === Train on DS1 normals, evaluate on DS2 only ==="))
    A(code("""mit_rows = {}
for model in MODELS:
    seeds = IF_SEEDS if model=="IF" else [0]
    aucs = []
    for seed in seeds:
        se, st = fit_score(model, X_m[tr_m], X_m[te_m], seed=seed)
        aucs.append(roc_auc_score(y_m[te_m], se))
        if seed == 0: s0, st0 = se, st
    prec, rec, f1 = prf(y_m[te_m], (s0 > np.percentile(st0, 85)).astype(int))
    lo, hi = boot_ci(y_m[te_m], s0)
    mit_rows[model] = {"auc": float(np.mean(aucs)), "sd": float(np.std(aucs)),
                       "ci": (lo,hi), "P": prec, "R": rec, "F1": f1}
    print(f"{model}: AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f} [{lo:.3f},{hi:.3f}] F1 {f1:.3f}")"""))

    A(md("# === PTB-XL: validation-fold threshold, frozen onto fold 10 ==="))
    A(md("### === Threshold grid on fold 9 (pre-registered F1-max) ==="))
    A(code("""tr_p, va_p, te_p = (f_p<=8)&(y_p==0), f_p==9, f_p==10
ptbxl_rows = {}
for model in MODELS:
    seeds = IF_SEEDS if model=="IF" else [0]
    aucs = []
    for seed in seeds:
        se_te, st_tr = fit_score(model, X_p[tr_p], X_p[te_p], seed=seed)
        aucs.append(roc_auc_score(y_p[te_p], se_te))
        if seed == 0: s0, st0 = se_te, st_tr
    se_va, _ = fit_score(model, X_p[tr_p], X_p[va_p])
    best_p, best_f = None, -1
    for pct in range(1, 100):
        _, _, f = prf(y_p[va_p], (se_va > np.percentile(se_va, pct)).astype(int))
        if f > best_f: best_f, best_p = f, pct
    prec, rec, f1 = prf(y_p[te_p], (s0 > np.percentile(se_va, best_p)).astype(int))
    lo, hi = boot_ci(y_p[te_p], s0)
    ptbxl_rows[model] = {"auc": float(np.mean(aucs)), "ci": (lo,hi), "P": prec, "R": rec, "F1": f1, "pct": best_p}
    print(f"{model}: AUC {np.mean(aucs):.3f} [{lo:.3f},{hi:.3f}] F1 {f1:.3f} (frozen val pct {best_p}, val F1 {best_f:.3f})")"""))
    A(md("### === Threshold-sensitivity curve on fold 10 (LOF) ==="))
    A(code("""se_te, _ = fit_score("LOF", X_p[tr_p], X_p[te_p])
rows = []
for pct in range(1, 100):
    prec, rec, f1 = prf(y_p[te_p], (se_te > np.percentile(se_te, pct)).astype(int))
    rows.append((pct, prec, rec, f1))
cur = pd.DataFrame(rows, columns=["pct","P","R","F1"])
fig, ax = plt.subplots(figsize=(5,3.4))
ax.plot(cur.pct, cur.P, label="Precision"); ax.plot(cur.pct, cur.R, label="Recall"); ax.plot(cur.pct, cur.F1, label="F1")
ax.axvline(ptbxl_rows["LOF"]["pct"], ls="--", c="gray", label="frozen val choice")
ax.set_xlabel("score percentile threshold"); ax.set_title("PTB-XL threshold sensitivity (LOF, fold 10)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()"""))

    A(md("# === Master results: v2 clean vs v1 contaminated ==="))
    A(md("### === v2 table ==="))
    A(code("""table = []
for r in wesad_rows:
    table.append({"dataset":"WESAD (LOSO)","model":r["model"],"AUC":round(r["auc"],3),
                  "CI":f'[{r["ci_low"]:.3f},{r["ci_high"]:.3f}]',"F1":round(r["f1"],3)})
for m,v in mit_rows.items():
    table.append({"dataset":"MIT-BIH (inter-patient)","model":m,"AUC":round(v["auc"],3),
                  "CI":f'[{v["ci"][0]:.3f},{v["ci"][1]:.3f}]',"F1":round(v["F1"],3)})
for m,v in ptbxl_rows.items():
    table.append({"dataset":"PTB-XL (fold9→10)","model":m,"AUC":round(v["auc"],3),
                  "CI":f'[{v["ci"][0]:.3f},{v["ci"][1]:.3f}]',"F1":round(v["F1"],3)})
det_df = pd.DataFrame(table)
det_df"""))
    A(md("### === Side-by-side with the v1 archive (same features, protocol only) ==="))
    A(code("""v1 = pd.read_csv(PROJECT_ROOT/"outputs_v1_archive/detection_results.csv")
cmp = pd.DataFrame({"v1 (contaminated)": v1.set_index(["Dataset","Model"]).AUC,
                    "v2 (clean)": det_df.set_index(["dataset","model"]).AUC}).dropna()
cmp.assign(delta=(cmp["v2 (clean)"]-cmp["v1 (contaminated)"]).round(3))"""))
    A(md("### === Significance: paired bootstrap IF vs LOF ==="))
    A(code("""s_if, yy = pooled["IF"]; s_lof, _ = pooled["LOF"]
print(f"WESAD:    IF {roc_auc_score(yy,s_if):.3f} vs LOF {roc_auc_score(yy,s_lof):.3f} — p = {paired_boot(yy,s_if,s_lof):.4g}")
for nm, Xtr, Xte, yte in [("MIT-BIH", X_m[tr_m], X_m[te_m], y_m[te_m]),
                          ("PTB-XL",  X_p[tr_p], X_p[te_p], y_p[te_p])]:
    s1,_ = fit_score("IF", Xtr, Xte); s2,_ = fit_score("LOF", Xtr, Xte)
    print(f"{nm:8s} IF {roc_auc_score(yte,s1):.3f} vs LOF {roc_auc_score(yte,s2):.3f} — p = {paired_boot(yte,s1,s2):.4g}")"""))

    A(md("# === Feature-group ablation (PTB-XL, LOF) ==="))
    A(code("""abl = []
for g, feats in FEATURE_GROUPS.items():
    cols = [i for i,n in enumerate(FEATURE_NAMES) if n not in feats]
    Xa = X_p[:, cols]
    se_va,_ = fit_score("LOF", Xa[tr_p], Xa[va_p]); se_te,_ = fit_score("LOF", Xa[tr_p], Xa[te_p])
    bp, bf = None, -1
    for pct in range(1,100):
        _,_,f = prf(y_p[va_p], (se_va>np.percentile(se_va,pct)).astype(int))
        if f > bf: bf, bp = f, pct
    prec, rec, f1 = prf(y_p[te_p], (se_te>np.percentile(se_va,bp)).astype(int))
    abl.append({"dropped": g, "AUC": round(roc_auc_score(y_p[te_p], se_te),3), "F1": round(f1,3)})
    print(f"  -{g}: AUC {abl[-1]['AUC']}")
pd.DataFrame(abl)"""))

    A(md("# === Figures ==="))
    A(md("### === WESAD LOSO ROC ==="))
    A(code("""fig, ax = plt.subplots(figsize=(4.6,4.2))
for model, color in zip(MODELS, ["steelblue","darkorange","seagreen","purple"]):
    s, y = pooled[model]
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=color, label=f"{model} ({roc_auc_score(y,s):.3f})")
ax.plot([0,1],[0,1],"k--",lw=.8); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("WESAD (LOSO, pooled)"); ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()"""))
    A(md("### === MIT-BIH + PTB-XL ROC ==="))
    A(code("""fig, axes = plt.subplots(1, 2, figsize=(10,4))
for ax, (Xtr, Xte, yte, ttl) in zip(axes, [(X_m[tr_m],X_m[te_m],y_m[te_m],"MIT-BIH (inter-patient DS2)"),
                                           (X_p[tr_p],X_p[te_p],y_p[te_p],"PTB-XL (fold 10)")]):
    for model, color in zip(MODELS, ["steelblue","darkorange","seagreen","purple"]):
        s,_ = fit_score(model, Xtr, Xte)
        fpr, tpr, _ = roc_curve(yte, s)
        ax.plot(fpr, tpr, color=color, label=f"{model} ({roc_auc_score(yte,s):.3f})")
    ax.plot([0,1],[0,1],"k--",lw=.8); ax.set_title(ttl); ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()"""))

    A(md("# === Save ==="))
    A(code("""det_df.to_csv(OUTPUT_DIR/"detection_results_v2_notebook.csv", index=False)
print("saved", OUTPUT_DIR/"detection_results_v2_notebook.csv")"""))

    A(md("""# Reading

- The v1 headline (MIT-BIH LOF 0.899) does not survive the inter-patient protocol: **LOF falls to chance (0.502)** while IF holds 0.670.
- Detector rankings are protocol-dependent: LOF>IF on WESAD/PTB-XL, IF≫LOF inter-patient on MIT-BIH; the autoencoder wins WESAD (0.855).
- The pre-registered validation rule honestly selects a near-saturating PTB-XL operating point — the sensitivity curve above is the real trade-off.
- `scripts/v2/detection_v2.py` is the batch equivalent; `THRESHOLDS.md` is the pre-registration."""))
    return c


# ==================================================================== rag_v2
def rag_cells():
    c = []
    A = c.append

    A(md("""# RAG v2 — Alert-Triggered Grounded Explanations

Rebuilds §4.4–4.10 of the revised paper, in the style of the original `rag.ipynb`.
v2 changes walked through below: corpus expansion (wearable-relevant guidelines),
corrected query semantics (subject-baseline reference), raw-text citation audit,
judge validation on a corruption benchmark, and the labeled-event evaluation."""))

    A(md("# === Imports ==="))
    A(md("### === Core ==="))
    A(code("import difflib, json, re, sys, time\nfrom collections import Counter\nfrom pathlib import Path\nimport numpy as np, pandas as pd"))
    A(md("### === PDF parsing ==="))
    A(code("import fitz"))
    A(md("### === Embeddings + vector store ==="))
    A(code("import chromadb\nfrom sentence_transformers import SentenceTransformer"))
    A(md("### === Local LLM (Ollama) ==="))
    A(code("import ollama"))
    A(md("### === Detection-side helpers (from the detection notebook's caches) ==="))
    A(code("import wfdb\nfrom sklearn.neighbors import LocalOutlierFactor\nfrom sklearn.preprocessing import StandardScaler\nimport ast"))

    A(md("# === Paths + constants ==="))
    A(md("### === Project root & corpus paths ==="))
    A(code("""PROJECT_ROOT = Path.cwd()
OUTPUT_DIR   = PROJECT_ROOT / "outputs_v2"
TIER1_DIR    = PROJECT_ROOT / "Dataset" / "RAG corpus (medical literatureguidelines)"
TIER1_V2_DIR = PROJECT_ROOT / "Dataset" / "Tier1_v2"
TIER2_DIR    = PROJECT_ROOT / "Dataset" / "Tier2_literature"
CHROMA_DIR   = PROJECT_ROOT / "chroma_db_v2"
GEN_JSONL    = OUTPUT_DIR / "generation_v2.jsonl"
"""))
    A(md("### === Config ==="))
    A(code("""EMBED_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL    = "qwen3.5:9b"
ALT_MODEL    = "llama3.1:8b"
JUDGE_CANDIDATES = ["llama3.1:8b", "gemma4:e4b"]   # gpt-oss excluded per author decision
CHUNK_WORDS, OVERLAP_WORDS, TOP_K, POOL = 500, 50, 5, 20
CHANNELS = ["ecg","resp","bvp","wrist_eda","wrist_temp"]"""))
    A(md("### === Verify the paths ==="))
    A(code("""print("Tier-1 v1 PDFs:", len(list(TIER1_DIR.glob('*.pdf'))))
print("Tier-1 v2 texts:", list(TIER1_V2_DIR.glob('*.txt')))
print("Tier-2 buckets:", sorted(d.name for d in TIER2_DIR.iterdir() if d.is_dir()))
print("generation jsonl exists:", GEN_JSONL.exists())"""))

    A(md("# === Load Tier-1 v1 corpus ==="))
    A(md("### === PDF loader ==="))
    A(code("""def load_tier1():
    docs = []
    for pdf in sorted(TIER1_DIR.glob("*.pdf")):
        d = fitz.open(pdf); text = "\\n".join(pg.get_text() for pg in d); d.close()
        docs.append({"source": pdf.stem, "tier": "tier1", "tier1_v": "v1", "text": text})
        print(f"  {pdf.stem[:58]:58s} {len(text):>9,} chars")
    return docs
tier1_docs = load_tier1()"""))
    A(md("# === Load Tier-1 v2 (new wearable-relevant guidelines) ==="))
    A(md("### === Text loader + provenance manifest ==="))
    A(code("""tier1_v2_docs = []
for t in sorted(TIER1_V2_DIR.glob("*.txt")):
    tier1_v2_docs.append({"source": t.stem, "tier": "tier1", "tier1_v": "v2",
                          "text": t.read_text(encoding="utf-8")})
    print(f"  {t.stem[:58]:58s} {len(tier1_v2_docs[-1]['text']):>9,} chars")
pd.read_csv(TIER1_V2_DIR/"manifest.csv")[["slug","status","provenance","license"]]"""))
    A(md("# === Load Tier-2 corpus ==="))
    A(md("### === Markdown loader ==="))
    A(code("""def load_tier2():
    docs = []
    for bdir in sorted(TIER2_DIR.iterdir()):
        if not bdir.is_dir(): continue
        mds = sorted(bdir.glob("*.md"))
        for m in mds:
            docs.append({"source": m.stem, "tier": "tier2", "bucket": bdir.name,
                         "text": m.read_text(encoding="utf-8")})
        print(f"  {bdir.name:28s} {len(mds):>3} articles")
    return docs
tier2_docs = load_tier2()"""))

    A(md("# === Chunk corpus ==="))
    A(md("### === Chunking function (v1-identical) ==="))
    A(code("""def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    if len(words) <= chunk_words: return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start+chunk_words])); start += chunk_words - overlap
    return chunks"""))
    A(md("### === Chunk all documents ==="))
    A(code("""all_chunks = []
for doc in tier1_docs + tier1_v2_docs + tier2_docs:
    for i, ch in enumerate(chunk_text(doc["text"])):
        all_chunks.append({"text": ch, "source": doc["source"], "tier": doc["tier"],
                           "tier1_v": doc.get("tier1_v",""), "bucket": doc.get("bucket",""), "chunk_idx": i})
print(f"{len(all_chunks):,} chunks (mean {np.mean([len(c['text'].split()) for c in all_chunks]):.0f} words)")
print("  guideline chunks:", sum(1 for c in all_chunks if c["tier"]=="tier1"))"""))

    A(md("# === Embed + Index (ChromaDB) ==="))
    A(code("""embedder = SentenceTransformer(EMBED_MODEL)
col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection("medical_corpus_v2")
if col.count() == 0:
    t0 = time.time(); embs = []
    for i in range(0, len(all_chunks), 64):
        embs.extend(embedder.encode([c["text"] for c in all_chunks[i:i+64]]).tolist())
    col.add(embeddings=embs, documents=[c["text"] for c in all_chunks],
           metadatas=[{k:v for k,v in c.items() if k!="text"} for c in all_chunks],
           ids=[f"chunk_{i}" for i in range(len(all_chunks))])
    print(f"embedded {len(all_chunks):,} chunks in {time.time()-t0:.0f}s")
else:
    print(f"collection already populated ({col.count():,} chunks) — skipping embed (v1 convention)")"""))

    A(md("# === Sanity retrieval test ==="))
    A(md("### === retrieve (dense + source-diversity) ==="))
    A(code('''def retrieve(query, top_k=TOP_K, pool=POOL, max_per_source=1):
    res = col.query(query_embeddings=embedder.encode([query]).tolist(), n_results=pool,
                    include=["documents","metadatas","distances"])
    metas = res["metadatas"][0]
    picked, counts = [], {}
    for i, m in enumerate(metas):
        cnt = counts.get(m["source"], 0)
        if cnt >= max_per_source: continue
        picked.append(i); counts[m["source"]] = cnt+1
        if len(picked) == top_k: break
    if len(picked) < top_k:
        for i in range(len(metas)):
            if i not in picked: picked.append(i)
            if len(picked) == top_k: break
    return {"sources": [metas[i]["source"] for i in picked],
            "tiers":   [metas[i]["tier"] for i in picked],
            "docs":    [res["documents"][0][i] for i in picked],
            "dists":   [res["distances"][0][i] for i in picked]}'''))
    A(md("### === 3 test queries ==="))
    A(code("""for q in ["How does PPG detect atrial fibrillation?",
              "What causes false alarms in wearable heart rate monitors?",
              "How is stress detected from electrodermal activity?"]:
    r = retrieve(q, top_k=3)
    print("QUERY:", q)
    for s, t, d in zip(r["sources"], r["tiers"], r["dists"]):
        print(f"   [{t}] {s[:70]} (d={d:.3f})")
    print()"""))

    A(md("# === Load flagged windows + build v2 queries ==="))
    A(md("### === Load the handoff (v1 archive) ==="))
    A(code("""flagged = pd.read_parquet(PROJECT_ROOT/"outputs_v1_archive/flagged_windows.parquet")
print(f"{len(flagged)} flagged windows | IF {int(flagged.flag_if.sum())}, "
      f"LOF {int(flagged.flag_lof.sum())}, both {int(flagged.flag_both.sum())}")"""))
    A(md("### === Load the full PPG-DaLiA feature matrix ==="))
    A(code("""z = np.load(OUTPUT_DIR/"cache/ppgdalia.npz", allow_pickle=True)
X_all, subj_all = z["X"], z["subject"]
cols = [f"{ch}__{fn}" for ch in CHANNELS
        for fn in ["mean","std","min","max","ptp","median","skew","kurt","p25","p75","up_ratio","roughness"]]
df_all = pd.DataFrame(X_all, columns=cols); df_all["subject"] = subj_all
df_all["window_idx"] = np.arange(len(df_all))
print(f"{len(df_all)} windows x {len(cols)} features, {df_all.subject.nunique()} subjects")"""))
    A(md("### === Map archived flags exactly onto the matrix ==="))
    A(code("""df_all["flag_if"] = False; df_all["flag_lof"] = False
offsets, off = {}, 0
for s in sorted(df_all.subject.unique()):
    offsets[s] = off; off += int((df_all.subject==s).sum())
for _, r in flagged.iterrows():
    g = offsets[int(r["subject"])] + int(r["window_idx"])
    df_all.loc[g, "flag_if"] = bool(r["flag_if"]); df_all.loc[g, "flag_lof"] = bool(r["flag_lof"])
print(f"IF {int(df_all.flag_if.sum())}, LOF {int(df_all.flag_lof.sum())}, "
      f"both {int((df_all.flag_if & df_all.flag_lof).sum())}, "
      f"union {int((df_all.flag_if | df_all.flag_lof).sum())}  (archive: 216/216/34/398)")"""))
    A(md("### === Query builder v2 (correct reference class) ==="))
    A(code('''TOPIC_PHRASES = {
    "ecg": "electrocardiogram rhythm irregularity heart rate variability arrhythmia ectopic beats atrial fibrillation",
    "resp": "respiration rate breathing pattern tachypnea bradypnea ventilation",
    "bvp": "photoplethysmography pulse waveform amplitude perfusion signal quality motion artifact atrial fibrillation screening",
    "wrist_eda": "electrodermal activity skin conductance sympathetic stress arousal sweat response",
    "wrist_temp": "skin temperature thermal perfusion vasomotor ambient temperature sensor effects"}

def subject_reference(df, subject):
    """This subject's NON-FLAGGED windows — the normal population the detector
    learned, implementable online via running stats (fixes v1's batch, cross-subject
    flagged-window reference)."""
    sub = df[df.subject == subject]
    normal = sub[(~sub.flag_if) & (~sub.flag_lof)]
    return {f"{ch}__{feat}": (normal[f"{ch}__{feat}"].mean(), normal[f"{ch}__{feat}"].std(ddof=0) or 1e-9)
            for ch in CHANNELS for feat in ["mean","kurt","ptp"]}

def build_query_v2(row, ref):
    zs  = {ch: (row[f"{ch}__mean"]-ref[f"{ch}__mean"][0]) / ref[f"{ch}__mean"][1] for ch in CHANNELS}
    kzs = {ch: (row[f"{ch}__kurt"]-ref[f"{ch}__kurt"][0]) / ref[f"{ch}__kurt"][1] for ch in CHANNELS}
    pzs = {ch: (row[f"{ch}__ptp"] -ref[f"{ch}__ptp"][0])  / ref[f"{ch}__ptp"][1]  for ch in CHANNELS}
    top2 = sorted(CHANNELS, key=lambda ch: -abs(zs[ch]))[:2]
    if row.flag_if and row.flag_lof: parts = ["Biosignal window flagged by both anomaly detectors (Isolation Forest and LOF)."]
    elif row.flag_if:                parts = ["Biosignal window flagged by the Isolation Forest anomaly detector."]
    else:                            parts = ["Biosignal window flagged by the LOF anomaly detector."]
    shape = max(max(kzs[ch], pzs[ch]) for ch in top2); shift = max(abs(zs[ch]) for ch in top2)
    if shape > 1.5:   parts.append("Deviating channels show an abrupt, high-amplitude pattern (elevated kurtosis/peak-to-peak vs this subject's baseline).")
    elif shift > 1.5: parts.append("Deviating channels show a sustained level shift from this subject's baseline.")
    elif shift > 1.0: parts.append("Deviating channels show a moderate shift from this subject's baseline.")
    else:             parts.append("Deviating channels are only mildly unusual vs this subject's baseline.")
    for ch in top2:
        parts.append(f"{'elevated' if zs[ch]>0 else 'reduced'} {ch} (z={zs[ch]:+.1f} vs subject baseline, mean={row[f'{ch}__mean']:.2f})")
    parts.append("Relevant topics: " + " ".join(TOPIC_PHRASES[ch] for ch in top2) + ".")
    parts.append("Other readings: " + ", ".join(f"{ch} mean={row[f'{ch}__mean']:.2f}" for ch in CHANNELS if ch not in top2) + ".")
    return " ".join(parts)'''))
    A(md("### === Show the first 3 v2 queries ==="))
    A(code("""union = df_all[df_all.flag_if | df_all.flag_lof]
for _, row in union.head(3).iterrows():
    q = build_query_v2(row, subject_reference(df_all, row.subject))
    print(f"S{int(row.subject)} w{int(row.window_idx)}: {q[:210]}...\\n")"""))
    A(md("### === Sanity check: the corrected z-metric separates stress on WESAD ==="))
    A(code("""from scipy.stats import mannwhitneyu
w = np.load(OUTPUT_DIR/"cache/wesad.npz", allow_pickle=True)
stress_nn, baseline_nn = [], []
for s in w["subjects"]:
    s = int(s); X, y = w[f"X_{s}"], w[f"y_{s}"]
    dfw = pd.DataFrame(X, columns=cols); base = dfw[y==1]
    ref = {ch: (base[f"{ch}__mean"].mean(), base[f"{ch}__mean"].std(ddof=0) or 1e-9) for ch in CHANNELS}
    for mask, acc in [(y==2, stress_nn), (y==1, baseline_nn)]:
        for _, r in dfw[mask].iterrows():
            zz = [abs((r[f"{ch}__mean"]-ref[ch][0])/ref[ch][1]) for ch in CHANNELS]
            acc.append(sorted(zz)[-2:])
st, bt = np.array(stress_nn).max(axis=1), np.array(baseline_nn).max(axis=1)
print(f"stress max|z| mean {st.mean():.1f} vs baseline {bt.mean():.1f} — p = {mannwhitneyu(st, bt, alternative='greater').pvalue:.3g}")"""))
    A(md("### === Build all 398 queries + retrieval ==="))
    A(code("""p = OUTPUT_DIR/"alerts_retrieval_v2.jsonl"
if p.exists():
    alerts = [json.loads(l) for l in open(p, encoding="utf-8")]
    print(f"loaded {len(alerts)} cached retrievals")
else:
    alerts = []
    for _, row in union.iterrows():
        q = build_query_v2(row, subject_reference(df_all, row.subject))
        r = retrieve(q)
        alerts.append({"subject": int(row.subject), "window_idx": int(row.window_idx), "query": q,
                       "flag_if": bool(row.flag_if), "flag_lof": bool(row.flag_lof),
                       "sources": r["sources"], "tiers": r["tiers"],
                       "context": "\\n\\n---\\n\\n".join(f"[{s}]\\n{d}" for s, d in zip(r["sources"], r["docs"]))})
    with open(p, "w", encoding="utf-8") as f:
        for a in alerts: f.write(json.dumps(a)+"\\n")
    print(f"built + saved {len(alerts)} retrievals")
t1 = sum(1 for a in alerts if any(t=="tier1" for t in a["tiers"]))
print(f"unique docs used: {len({s for a in alerts for s in a['sources']})} | "
      f"guideline reach: {t1}/{len(alerts)} ({100*t1/len(alerts):.1f}%)")"""))

    A(md("# === Labeled events (labels NEVER enter the queries) ==="))
    A(md("### === WESAD: top-50 LOF-scored stress windows ==="))
    A(code("""DS1 = ["101","106","108","109","112","114","115","116","118","119","122","124",
       "201","203","205","207","208","209","215","220","223","230"]
DS2 = ["100","103","105","111","113","117","121","123","200","202","210","212",
       "213","214","219","221","222","228","231","232","233","234"]
AAMI2 = {"N","L","R","e","j"}; BEAT_SYMBOLS2 = AAMI2 | {"A","a","J","S","V","E","F","f","Q","/","!"}
MITBIH_DIR = PROJECT_ROOT/"Dataset/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0"

Xb = np.vstack([w[f"X_{s}"][w[f"y_{s}"]==1] for s in w["subjects"]])
sc_w = StandardScaler().fit(Xb)
lof_w = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.15).fit(sc_w.transform(Xb))
cand = []
for s in w["subjects"]:
    s = int(s); Xs = w[f"X_{s}"][w[f"y_{s}"]==2]
    for i, scr in enumerate(-lof_w.score_samples(sc_w.transform(Xs))): cand.append((float(scr), s, i))
cand.sort(reverse=True)
print(f"WESAD: picked 50 stress windows (top LOF scores {cand[0][0]:.2f}..{cand[49][0]:.2f})")"""))
    A(md("### === MIT-BIH: annotation-driven window selection ==="))
    A(code("""# NOTE: a first-pass selection ranked windows by DETECTOR flags — but the inter-patient
# detector is near chance, so 49/50 picked windows had NO annotated ectopy. Superseded
# (see superseded_keys.json); v2 selects by expert annotations (query stays detector-side).
z_m = np.load(OUTPUT_DIR/"cache/mitbih.npz", allow_pickle=True)
X_m2, y_m2, rec_m2 = z_m["X"], z_m["y"], z_m["record"].astype(str)
trm = np.isin(rec_m2, DS1) & (y_m2==0); tem = np.isin(rec_m2, DS2)
sc_m = StandardScaler().fit(X_m2[trm])
lof_m = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.15).fit(sc_m.transform(X_m2[trm]))
se_te = -lof_m.score_samples(sc_m.transform(X_m2[tem]))
thr_m = np.percentile(-lof_m.score_samples(sc_m.transform(X_m2[trm])), 85)
flag_abs = np.zeros(len(X_m2), bool); flag_abs[np.where(tem)[0][se_te > thr_m]] = True

cands = {"VEB": [], "SVEB": []}
for rec in DS2:
    sig, _ = wfdb.rdsamp(str(MITBIH_DIR/rec))
    ann = wfdb.rdann(str(MITBIH_DIR/rec), "atr")
    idx_te = np.where(tem & (rec_m2==rec))[0]
    seq = []
    for i, sym in zip(ann.sample, ann.symbol):
        if sym not in BEAT_SYMBOLS2: continue
        st_, en_ = i-144, i+144
        if st_ < 0 or en_ > sig.shape[0]: continue
        seq.append((i, sym, len(seq)))
    row_of = {pos: int(r) for pos, r in zip([x[2] for x in seq], idx_te)}
    wins = {}
    for sample, sym, pos in seq:
        wd = wins.setdefault(sample//(360*30), {"rows": [], "syms": []})
        wd["rows"].append(row_of[pos]); wd["syms"].append(sym)
    for widx, wd in wins.items():
        abn = [s_ for s_ in wd["syms"] if s_ not in AAMI2]
        if len(abn) < 3 or len(wd["rows"]) < 5: continue
        cls = "VEB" if any(s_ in {"V","E"} for s_ in abn) else "SVEB"
        cands[cls].append((len(abn), rec, widx, wd, abn))
print(f"MIT-BIH candidates (>=3 annotated abnormal beats): VEB {len(cands['VEB'])}, SVEB {len(cands['SVEB'])} → picked 25+25")"""))
    A(md("### === PTB-XL: stratified top-scored pathology per superclass ==="))
    A(code("""zp = np.load(OUTPUT_DIR/"cache/ptbxl.npz", allow_pickle=True)
Xp2, yp2, fp2 = zp["X"], zp["y"], zp["fold"]
trp = (fp2<=8) & (yp2==0)
sc_p = StandardScaler().fit(Xp2[trp])
lof_p = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.15).fit(sc_p.transform(Xp2[trp]))
se_fold10 = -lof_p.score_samples(sc_p.transform(Xp2[fp2==10]))

db = pd.read_csv(PROJECT_ROOT/"Dataset/ptb-xl-1.0.3/ptbxl_database.csv")
scp = pd.read_csv(PROJECT_ROOT/"Dataset/ptb-xl-1.0.3/scp_statements.csv", index_col=0)
db["super"] = db.scp_codes.apply(lambda cp: {scp.loc[c,"diagnostic_class"] for c in ast.literal_eval(cp)
                                             if c in scp.index and isinstance(scp.loc[c,"diagnostic_class"], str)})
db["y"] = db.super.apply(lambda st: 0 if st=={"NORM"} else (-1 if not st else 1))
dbf = db[db.y != -1]                       # same filter as the cache — keeps alignment
f10 = dbf[dbf.strat_fold==10].reset_index()
assert len(f10) == len(se_fold10)
score_by_ecg = {int(f10.iloc[k].ecg_id): float(se_fold10[k]) for k in range(len(f10))}
picked_ptbxl = []
for cls in ["MI","STTC","CD","HYP"]:
    cdb = f10[f10.super.apply(lambda st: cls in st)].copy()
    cdb["score"] = cdb.ecg_id.map(lambda e: score_by_ecg.get(e, -9.0))
    for r in cdb.sort_values("score", ascending=False).head(12).itertuples():
        picked_ptbxl.append((r.ecg_id, r.filename_lr, cls))
print(f"PTB-XL: {len(picked_ptbxl)} records:", dict(Counter(c for _,_,c in picked_ptbxl)))"""))

    A(md("# === Grounded explanation generator ==="))
    A(md("### === System prompt (v1-identical rules) ==="))
    A(code('''SYSTEM_PROMPT_150 = """You are a clinical decision-support assistant that explains wearable biosignal anomalies.
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
DISCLAIMER: Research decision-support tool. Not a diagnostic device. Does not replace clinical judgment."""'''))
    A(md("### === Canonicalizer (capture rule FIXED vs v1) ==="))
    A(code('''def canonicalize_fixed(raw_text, sources):
    """v1-intended canonicalizer with the ID capture fixed: the citation ID is the
    PMC token; trailing slug text is display noise (v1's greedy regex dropped valid
    citations written as [PMC123_full_title]). Every snap is logged."""
    valid = sorted({s.split("_")[0] for s in sources if s.startswith("PMC")})
    snaps = []
    def _fix(m):
        cid = m.group(1)
        if cid in valid: return f"[{cid}]"
        close = difflib.get_close_matches(cid, valid, n=1, cutoff=0.75)
        if close:
            snaps.append({"before": cid, "after": close[0]}); return f"[{close[0]}]"
        snaps.append({"before": cid, "after": None}); return ""
    return re.sub(r"\\[(PMC\\d+)[^\\]]*\\]", _fix, raw_text), snaps'''))
    A(md("### === generate_one ==="))
    A(code("""def generate_one(query, context, model=LLM_MODEL, prompt=SYSTEM_PROMPT_150, num_predict=500):
    resp = ollama.chat(model=model, think=False,
        messages=[{"role":"system","content":prompt},
                  {"role":"user","content":f"ANOMALY:\\n{query}\\n\\nRETRIEVED CONTEXT:\\n{context}"}],
        options={"temperature":0.1,"num_predict":num_predict,"num_ctx":10000,"num_gpu":99})
    return resp["message"]["content"].strip()"""))
    A(md("### === Test on 3 flagged windows ==="))
    A(code("""for a in alerts[:3]:
    raw = generate_one(a["query"], a["context"])
    fixed, snaps = canonicalize_fixed(raw, a["sources"])
    print(f"--- S{a['subject']} w{a['window_idx']} ({len(snaps)} citation repairs/drops) ---")
    print(fixed[:520], "\\n")"""))

    A(md("# === Batch generation (resume-safe) ==="))
    A(md("### === Load the completed run (the cold-start loop is in scripts/v2/rag_pipeline_v2.py) ==="))
    A(code("""rows = [json.loads(l) for l in open(GEN_JSONL, encoding="utf-8")]
sup = json.loads((OUTPUT_DIR/"superseded_keys.json").read_text())["superseded_mitbih_keys"]
rows = [r for r in rows if r["key"] not in sup]
print(f"{len(rows)} explanations (50 superseded detector-ranked MIT-BIH keys excluded)")
pd.DataFrame([{k: r.get(k) for k in ("group","subgroup","model","prompt")} for r in rows]
            ).value_counts(["group","subgroup","model","prompt"]).to_frame("n")"""))
    A(md("### === Generation latency ==="))
    A(code("""mn = [r for r in rows if r["group"]=="dalia" and r["subgroup"]=="main"]
lat = np.array([r["latency_sec"] for r in mn])
print(f"{len(lat)} alerts | mean {lat.mean():.1f}s | median {np.median(lat):.1f}s | max {lat.max():.1f}s (RTX 5060 laptop)")"""))

    A(md("# === Citation accuracy (programmatic, raw AND repaired) ==="))
    A(md("### === Audit loop ==="))
    A(code("""cit_re = re.compile(r"\\[(PMC\\d+)[^\\]]*\\]")
name_re = re.compile(r"\\[([^]\\[]+)\\]")
stats = {"raw_cit":0, "raw_ok":0, "rep_cit":0, "rep_ok":0, "snaps":0, "drops":0, "t1_ok":0, "t1_bad":0}
for r in [x for x in rows if x["subgroup"]=="main"]:
    valid = {s.split("_")[0] for s in r["sources"] if s.startswith("PMC")}
    t1 = [s for s in r["sources"] if not s.startswith("PMC")]
    fixed, snaps = canonicalize_fixed(r["raw_explanation"], r["sources"])
    stats["snaps"] += sum(1 for s in snaps if s["after"])
    stats["drops"] += sum(1 for s in snaps if not s["after"])
    for which, txt in [("raw", r["raw_explanation"]), ("rep", fixed)]:
        cits = [m.group(1) for m in cit_re.finditer(txt)]
        stats[f"{which[:3]}_cit"] += len(cits)
        stats[f"{which[:3]}_ok"]  += sum(1 for c in cits if c in valid)
    for m in name_re.finditer(fixed):
        tok = m.group(1).strip()
        if tok.startswith("PMC"): continue
        if any(s.startswith(tok) or tok.startswith(s) for s in t1): stats["t1_ok"] += 1
        else: stats["t1_bad"] += 1
print(f"RAW:      {stats['raw_cit']} citations, {stats['raw_ok']} valid = {100*stats['raw_ok']/stats['raw_cit']:.2f}%")
print(f"REPAIRED: {stats['rep_cit']} citations, {stats['rep_ok']} valid = {100*stats['rep_ok']/stats['rep_cit']:.2f}%")
print(f"canonicalizer: {stats['snaps']} snapped, {stats['drops']} dropped (genuine fabrications)")
print(f"guideline name-citations: {stats['t1_ok']} valid, {stats['t1_bad']} unmatched")"""))

    A(md("# === Judge validation (corruption benchmark) ==="))
    A(md("### === Build the corruption set ==="))
    A(code("""import random as _rnd; _rnd.seed(7)
FABRICATED_FACTS = [
    "This pattern is associated with a 3-fold increase in 30-day mortality according to the sources.",
    "The literature states that patients with this finding require immediate hospital admission.",
    "According to the retrieved context, this signal pattern predicts stroke with 92% probability.",
    "The sources confirm that this finding always progresses to cardiac arrest if untreated.",
    "Retrieved guidelines recommend starting anticoagulation immediately for this pattern."]
main_rows = [r for r in rows if r["subgroup"]=="main"
             and len(re.findall(r"\\[PMC\\d+", r["explanation"])) >= 2]
sample = _rnd.sample(main_rows, 100)
bench = []
for i, r in enumerate(sample):
    txt = r["explanation"]; cits = re.findall(r"PMC\\d+", txt)
    ctype = i % 4; corrupted = txt
    if ctype == 0 and len(cits) >= 1:
        others = [s.split("_")[0] for s in r["sources"] if s.split("_")[0] not in cits[:1]]
        if others: corrupted = txt.replace(f"[{cits[0]}]", f"[{others[0]}]", 1)
        else: ctype = 1
    if ctype == 1:   corrupted = txt.replace("EVIDENCE:", f"EVIDENCE: {_rnd.choice(FABRICATED_FACTS)} ", 1)
    elif ctype == 2: corrupted = txt.replace(f"[{cits[0]}]", "[PMC99999999]", 1)
    elif ctype == 3: corrupted = txt.replace("DETECTED:", "DETECTED: This finding is diagnostic of acute myocardial infarction and requires emergency treatment. ", 1)
    bench.append({"row": r, "clean": txt, "corrupted": corrupted,
                  "ctype": ["citation_swap","fabricated_fact","fabricated_citation","diagnostic_exaggeration"][ctype]})
print("100 corrupted + 100 clean controls | example (", bench[0]["ctype"], "):")
print(bench[0]["corrupted"][:260], "...")"""))
    A(md("### === Judge prompt + local judge ==="))
    A(code('''JUDGE_PROMPT = open(PROJECT_ROOT/"scripts/v2/judge_prompt_v2.txt", encoding="utf-8").read()

def local_judge(model, query, explanation, context):
    # think=False is essential for gemma4: default thinking mode consumes the
    # token budget and returns EMPTY content (see *_INVALID_empty_thinking.csv)
    resp = ollama.chat(model=model, think=False,
        messages=[{"role":"system","content":JUDGE_PROMPT},
                  {"role":"user","content":f"QUERY: {query}\\nSOURCES:\\n{context}\\nEXPLANATION:\\n{explanation}"}],
        options={"temperature":0.1,"num_predict":300,"num_ctx":6000,"num_gpu":99})
    scores = {}
    for line in str(resp["message"]["content"]).split("\\n"):
        m = re.match(r"(FAITHFULNESS|RELEVANCE|COMPLETENESS):\\s*([123])", line.strip(), re.I)
        if m: scores[m.group(1).lower()] = int(m.group(2))
    return scores'''))
    A(md("### === Spot-check: gemma4 on 2 corrupted + 2 clean ==="))
    A(code("""_alert_ctx = {f"S{a['subject']}|w{a['window_idx']}": a["context"] for a in alerts}
def ctx_of(row):
    if row["key"].startswith("dalia|"):
        return _alert_ctx[row["key"].split("|",1)[1]]
    return retrieve(row["query"])["context"]

for b in bench[:2]:
    print("corrupted:", b["ctype"], "->", local_judge("gemma4:e4b", b["row"]["query"], b["corrupted"], ctx_of(b["row"])))
for b in bench[:2]:
    print("clean    :", local_judge("gemma4:e4b", b["row"]["query"], b["clean"], ctx_of(b["row"])))"""))
    A(md("### === Full benchmark results (200 calls per judge, cached CSVs) ==="))
    A(code("""for model in JUDGE_CANDIDATES:
    d = pd.read_csv(OUTPUT_DIR/f"judge_validation_{model.replace(':','_').replace('/','_')}.csv")
    det = (d[(d.is_corrupted==1) & (d.faithfulness==1)].shape[0]) / (d.is_corrupted==1).sum()
    fp  = (d[(d.is_corrupted==0) & (d.faithfulness==1)].shape[0]) / (d.is_corrupted==0).sum()
    by_type = d[d.is_corrupted==1].groupby("ctype").apply(lambda g: (g.faithfulness==1).mean(), include_groups=False).round(2).to_dict()
    print(f"{model}: detection {det:.2f}, FP {fp:.2f}, by type {by_type}")"""))

    A(md("# === Main judging (validated local judge) ==="))
    A(md("### === Scores by group ==="))
    A(code("""ev = pd.read_csv(OUTPUT_DIR/"rag_evaluation_v2.csv")
ev.groupby("subgroup").agg(n=("local_faithfulness","size"),
                           faith=("local_faithfulness","mean"),
                           relev=("local_relevance","mean"),
                           compl=("local_completeness","mean")).round(2)"""))
    A(md("### === Faithfulness distribution on the 398 wearable alerts ==="))
    A(code("""m = ev[(ev.group=="dalia") & (ev.subgroup=="main")]
print("score -> count:", m.local_faithfulness.value_counts().sort_index().to_dict(), "(0 = parse failure)")"""))

    A(md("# === Labeled-event concordance ==="))
    A(md("### === Lexicons + concordance loop ==="))
    A(code('''LEXICONS = {
 "stress":["stress","arousal","sympathetic","anxiety","mental load","psychological","emotional"],
 "VEB":["ventricular","pvc","premature ventricular","ventricular tachycard"],
 "SVEB":["supraventricular","atrial premature","pac","atrial ectopy","premature atrial","atrial fibrillation","atrial tachyarrhythm"],
 "MI":["infarct","ischemi","stemi","coronary occlusion","st-elevation","st elevation"],
 "STTC":["repolarization","st depression","st-segment","st segment","t-wave","t wave inversion"],
 "CD":["conduction","bundle branch","heart block","av block","pr interval"],
 "HYP":["hypertroph","chamber enlargement","left ventricular mass"]}
ARTIFACT_TERMS = ["artifact","motion","sensor displacement","sensor contact","signal quality",
                  "electrode","noise","poor contact","device"]

def sec(t, a, b):
    mm = re.search(rf"{a}:\\s*(.*?)(?={b}:|$)", t, re.S)
    return mm.group(1).strip().lower() if mm else ""

table = {}
for grp in ("wesad","mitbih","ptbxl"):
    sel = [r for r in rows if r["subgroup"]=="labeled" and r["group"]==grp]
    st = {"n":len(sel),"concordant":0,"artifact":0,"insufficient":0,"other":0,"by_label":{}}
    for r in sel:
        det = sec(r["explanation"], "DETECTED", "EVIDENCE"); lab = r["true_label"]
        st["by_label"].setdefault(lab, {"n":0,"ok":0}); st["by_label"][lab]["n"] += 1
        if any(t in det for t in LEXICONS.get(lab, [])):
            st["concordant"] += 1; st["by_label"][lab]["ok"] += 1
            if any(t in det for t in ARTIFACT_TERMS): st["artifact"] += 1
        elif "insufficient" in det: st["insufficient"] += 1
        elif any(t in det for t in ARTIFACT_TERMS): st["artifact"] += 1
        else: st["other"] += 1
    table[grp] = st

pd.DataFrame({g: {"n":v["n"], "concordant":f'{v["concordant"]} ({100*v["concordant"]/v["n"]:.0f}%)',
                  "artifact language":v["artifact"], "other":v["other"]} for g,v in table.items()}).T'''))

    A(md("# === Near-duplicates: before/after the v2 fixes ==="))
    A(md("### === Embed explanations + cluster at cosine 0.9 ==="))
    A(code("""texts = [sec(r["explanation"],"DETECTED","EVIDENCE") + "\\n" +
         sec(r["explanation"],"EVIDENCE","RECOMMENDATION")
         for r in rows if r["group"]=="dalia" and r["subgroup"]=="main"]
emb = embedder.encode(texts, normalize_embeddings=True)
sim = emb @ emb.T; np.fill_diagonal(sim, -1)
nn = sim.max(axis=1)
unassigned, cluster, cid = set(range(len(texts))), np.full(len(texts), -1), 0
while unassigned:
    seed = min(unassigned)
    members = [j for j in unassigned if sim[seed, j] > 0.9] + [seed]
    for j in members: cluster[j] = cid; unassigned.discard(j)
    cid += 1
nd1 = json.load(open(OUTPUT_DIR/"rag_analysis_v1/near_duplicate_summary.json"))
pd.DataFrame({
  "v1": {"clusters@0.9": nd1["n_clusters_at_0.9"], "mean_NN_cos": nd1["mean_nn_cos"],
         "pct_twins>0.9": nd1["pct_rows_with_nn_gt_0.9"], "guideline_reach_%": 6.5},
  "v2": {"clusters@0.9": cid, "mean_NN_cos": round(float(nn.mean()),4),
         "pct_twins>0.9": round(float((nn>0.9).mean()*100),2), "guideline_reach_%": 17.6}}).T"""))

    A(md("# === Atomic-claim verification (FActScore-lite) ==="))
    A(md("### === Results (verifier: gemma4, different family; method in scripts/v2/factscore_lite.py) ==="))
    A(code("""fs = json.load(open(OUTPUT_DIR/"factscore_lite.json"))
print({k: fs[k] for k in ("n_explanations","n_claims","pct_supported","pct_unsupported","pct_unverifiable")})
claims = pd.read_csv(OUTPUT_DIR/"factscore_lite_claims.csv")
print("\\nrandom claims:")
for _, r in claims.sample(5, random_state=3).iterrows():
    print(f"  [{r.verdict:14s}] {r.claim[:100]}")"""))

    A(md("# === Ablations ==="))
    A(code("""for sg in ("main","wordcap","genablation"):
    sel = [r for r in rows if r["subgroup"]==sg and (sg!="main" or r["group"]=="dalia")]
    words = np.mean([len(r["explanation"].split()) for r in sel])
    s = ev[ev.subgroup==sg] if sg!="main" else ev[(ev.subgroup=="main") & (ev.group=="dalia")]
    print(f"{sg:12s} n={len(sel):3d} words={words:6.1f} faith={s.local_faithfulness.mean():.2f} compl={s.local_completeness.mean():.2f}")"""))

    A(md("# === Example alerts ==="))
    A(md("### === One concordant explanation ==="))
    A(code("""ex = next(r for r in rows if r["group"]=="wesad"
          and "stress" in sec(r["explanation"],"DETECTED","EVIDENCE")
          and "artifact" not in sec(r["explanation"],"DETECTED","EVIDENCE"))
print("TRUE LABEL:", ex["true_label"], "| QUERY:", ex["query"][:180])
print()
print(ex["explanation"][:800])"""))
    A(md("### === One typical pathology failure ==="))
    A(code("""ptb = next(r for r in rows if r["group"]=="ptbxl")
print("TRUE LABEL:", ptb["true_label"])
print("DETECTED:", sec(ptb["explanation"], "DETECTED", "EVIDENCE")[:400])"""))

    A(md("""# Reading

- Validated judge (gemma4): faithfulness 2.21/3 on wearable alerts, 44% fully faithful, 2 hallucination verdicts — a distribution, not a "zero hallucination" headline.
- Raw citation accuracy 99.01% (12 fabrications in 9/546 explanations); repair drops them.
- Concordance: 94% stress / 12% ectopy / 6% pathology, with 42–56% of true pathology attributed to artifact — the safety-critical finding.
- 47.7% of atomic claims unverifiable from the retrieved context.
- v2 fixes raised guideline reach (6.5→17.6%) and cut duplication (173→237 clusters) at the cost of narrower document spread (53→44).
- API-judge columns await OpenRouter key renewal; clinician ratings via `clinician_eval/`."""))
    return c


if __name__ == "__main__":
    execute_and_save(cells_to_nb(detection_cells()), "detection_v2.ipynb")
    execute_and_save(cells_to_nb(rag_cells()), "rag_v2.ipynb")
