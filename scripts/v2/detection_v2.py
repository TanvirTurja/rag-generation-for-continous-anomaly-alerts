"""
detection_v2.py — Clean-protocol detection evaluation (fix/v2).

Implements THRESHOLDS.md exactly:
  - WESAD: leave-one-subject-out CV (fixes v1 train-in-eval contamination + intra-subject leak)
  - MIT-BIH: inter-patient DS1->DS2, paced records 102/104/107/217 excluded, AAMI N={N,L,R,e,j}
  - PTB-XL: threshold selected on fold 9 (validation), frozen onto fold 10 (fixes v1 test-prevalence circularity)
  - IF 10 seeds, LOF, OC-SVM, dense autoencoder baselines
  - bootstrap 95% AUC CIs + DeLong IF-vs-LOF tests
  - PTB-XL feature-group ablation
Features are IDENTICAL to v1 (same 12 stats, same loaders) so differences are protocol-only.

Outputs -> outputs_v2/:
  detection_results_v2.csv      main table (AUC [CI], P/R/F1 per dataset/model)
  wesad_loso_per_subject.csv    per-subject AUCs
  ptbxl_threshold_curve.csv     P/R/F1 vs percentile on fold 10
  feature_ablation_ptbxl.csv    LOF AUC when dropping each feature group
  detection_summary_v2.json     all numbers incl. DeLong p-values, dataset stats
  figures_v2/{roc_wesad,roc_mitbih,roc_ptbxl,ptbxl_threshold_curve}.png
  cache/*.npz                   featurized matrices (reused by W4 query builder)
"""

import json
import pickle
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs_v2"
CACHE = OUT / "cache"
FIGS = OUT / "figures_v2"
for d in (OUT, CACHE, FIGS):
    d.mkdir(parents=True, exist_ok=True)

PPGDALIA_DIR = ROOT / "Dataset" / "ppg+dalia" / "data" / "PPG_FieldStudy"
WESAD_DIR = ROOT / "Dataset" / "WESAD"
MITBIH_DIR = ROOT / "Dataset" / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"
PTBXL_DIR = ROOT / "Dataset" / "ptb-xl-1.0.3"

WESAD_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
CHANNELS = ["ecg", "resp", "bvp", "wrist_eda", "wrist_temp"]
FS_MAP = {"ecg": 700, "resp": 700, "bvp": 64, "wrist_eda": 4, "wrist_temp": 4}
WIN_SEC = 30

# MIT-BIH inter-patient split (de Chazal standard), paced records excluded
PACED = {"102", "104", "107", "217"}
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
assert len(DS1) == 22 and len(DS2) == 22
assert not (set(DS1) & set(DS2)) and not ((set(DS1) | set(DS2)) & PACED)
assert len(set(DS1) | set(DS2) | PACED) == 48

AAMI_NORMAL = {"N", "L", "R", "e", "j"}          # v2: adds 'j' per AAMI
BEAT_SYMBOLS = AAMI_NORMAL | {"A", "a", "J", "S", "V", "E", "F", "f", "Q", "/", "!"}
MITBIH_HALF = 144
FS_MITBIH = 360

IF_SEEDS = list(range(10))
N_BOOT = 1000
RNG = np.random.default_rng(42)

DEV = None  # set in main for torch


# ---------------------------------------------------------------- features (v1-identical)
def featurize(window):
    w = np.asarray(window).ravel().astype(float)
    return [
        w.mean(), w.std(), w.min(), w.max(), np.ptp(w),
        np.median(w), skew(w), kurtosis(w),
        np.percentile(w, 25), np.percentile(w, 75),
        np.sum(np.diff(w) > 0) / len(w),
        np.sqrt(np.mean(np.diff(w) ** 2)),
    ]


FEATURE_NAMES = ["mean", "std", "min", "max", "ptp", "median", "skew", "kurt",
                 "p25", "p75", "up_ratio", "roughness"]
FEATURE_GROUPS = {
    "location": ["mean", "median", "p25", "p75"],
    "spread": ["std", "min", "max", "ptp"],
    "shape": ["skew", "kurt"],
    "dynamics": ["up_ratio", "roughness"],
}


def make_windows(sig, fs, win_sec=WIN_SEC):
    n = fs * win_sec
    return [sig[i:i + n] for i in range(0, len(sig) - n, n)]


def clean_signal(sig):
    s = np.asarray(sig, dtype=float).copy()
    s[np.isinf(s)] = np.nan
    if np.isnan(s).any():
        idx = np.arange(len(s))
        good = ~np.isnan(s)
        if good.sum() < 10:
            return s
        s = np.interp(idx, idx[good], s[good])
    return s


def featurize_subject(data, channels=CHANNELS):
    feats_list, n_windows = [], None
    for ch in channels:
        wins = make_windows(clean_signal(data[ch]), fs=FS_MAP[ch])
        feats = np.array([featurize(w) for w in wins])
        if n_windows is None:
            n_windows = feats.shape[0]
        feats_list.append(feats[:n_windows])
    return np.hstack(feats_list), n_windows


def window_labels(label_sig, fs, win_sec, n_windows):
    labels = []
    n = fs * win_sec
    for i in range(n_windows):
        chunk = label_sig[i * n:(i + 1) * n]
        labels.append(int(Counter(chunk).most_common(1)[0][0]) if len(chunk) else 0)
    return np.array(labels)


def load_pkl_subject(base, subject):
    path = base / f"S{subject}" / f"S{subject}.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    chest, wrist = d["signal"]["chest"], d["signal"]["wrist"]
    return {
        "subject": subject,
        "ecg": chest["ECG"].ravel(), "resp": chest["Resp"].ravel(),
        "bvp": wrist["BVP"].ravel(), "wrist_eda": wrist["EDA"].ravel(),
        "wrist_temp": wrist["TEMP"].ravel(),
        "label": d["label"].ravel(),
    }


# ---------------------------------------------------------------- build caches
def build_wesad():
    per_subject = {}
    for s in WESAD_SUBJECTS:
        d = load_pkl_subject(WESAD_DIR, s)
        X, n = featurize_subject(d)
        y = window_labels(d["label"], fs=700, win_sec=WIN_SEC, n_windows=n)
        per_subject[s] = (X, y)
        print(f"  WESAD S{s}: {n} windows {Counter(y).get(2, 0)} stress", flush=True)
    return per_subject


def build_ppgdalia():
    all_X, all_subj = [], []
    for s in range(1, 16):
        d = load_pkl_subject(PPGDALIA_DIR, s)
        X, n = featurize_subject(d)
        all_X.append(X)
        all_subj.extend([s] * n)
        print(f"  DaLiA S{s}: {n} windows", flush=True)
    return np.vstack(all_X), np.array(all_subj)


def build_mitbih():
    import wfdb
    Xs, ys, recs, syms_all = [], [], [], Counter()
    for rec in sorted(set(DS1) | set(DS2) | PACED):
        sig, _ = wfdb.rdsamp(str(MITBIH_DIR / rec))
        ann = wfdb.rdann(str(MITBIH_DIR / rec), "atr")
        for i, sym in zip(ann.sample, ann.symbol):
            if sym not in BEAT_SYMBOLS:
                continue
            start, end = i - MITBIH_HALF, i + MITBIH_HALF
            if start < 0 or end > sig.shape[0]:
                continue
            Xs.append(featurize(sig[start:end, 0]))
            ys.append(0 if sym in AAMI_NORMAL else 1)
            recs.append(rec)
            syms_all[sym] += 1
        print(f"  MIT-BIH {rec}: done", flush=True)
    return np.array(Xs), np.array(ys), np.array(recs), syms_all


def build_ptbxl():
    import ast
    import wfdb
    db = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv")
    scp = pd.read_csv(PTBXL_DIR / "scp_statements.csv", index_col=0)

    def supers(cp_str):
        codes = ast.literal_eval(cp_str)
        out = set()
        for c in codes:
            if c in scp.index:
                k = scp.loc[c, "diagnostic_class"]
                if isinstance(k, str):
                    out.add(k)
        return out

    db["super"] = db["scp_codes"].apply(supers)
    db["y"] = db["super"].apply(lambda st: 0 if st == {"NORM"} else (-1 if not st else 1))
    db = db[db["y"] != -1].reset_index(drop=True)
    Xs, ys, folds = [], [], []
    for idx, row in db.iterrows():
        if idx % 4000 == 0:
            print(f"  PTB-XL {idx}/{len(db)}", flush=True)
        try:
            sig, _ = wfdb.rdsamp(str(PTBXL_DIR / row["filename_lr"]))
            Xs.append(featurize(sig[:, 0]))
            ys.append(row["y"])
            folds.append(row["strat_fold"])
        except Exception:
            continue
    return np.array(Xs), np.array(ys), np.array(folds)


def get_cache(name):
    p = CACHE / f"{name}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return {k: z[k] for k in z.files}
    print(f"Featurizing {name} ...", flush=True)
    t0 = time.time()
    if name == "wesad":
        per_subject = build_wesad()
        data = {"subjects": np.array(WESAD_SUBJECTS)}
        for s in WESAD_SUBJECTS:
            X, y = per_subject[s]
            data[f"X_{s}"], data[f"y_{s}"] = X, y
    elif name == "ppgdalia":
        X, subj = build_ppgdalia()
        data = {"X": X, "subject": subj}
    elif name == "mitbih":
        X, y, recs, syms = build_mitbih()
        data = {"X": X, "y": y, "record": recs,
                "symbols": np.array(json.dumps(dict(syms)))}
    elif name == "ptbxl":
        X, y, folds = build_ptbxl()
        data = {"X": X, "y": y, "fold": folds}
    else:
        raise KeyError(name)
    np.savez_compressed(p, **data)
    print(f"  cached {name} in {time.time()-t0:.0f}s", flush=True)
    return data


# ---------------------------------------------------------------- models
def make_if(seed, contamination):
    return IsolationForest(n_estimators=100, contamination=contamination, random_state=seed)


def fit_score(model_name, X_train, X_eval, seed=0, contamination=0.15):
    """Fit on training (normal) data, return (eval_scores, train_scores)."""
    if model_name == "IF":
        m = make_pipeline(StandardScaler(), make_if(seed, contamination))
        m.fit(X_train)
        return -m.score_samples(X_eval), -m.score_samples(X_train)
    if model_name == "LOF":
        m = make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=20, novelty=True,
                                                                contamination=contamination))
        m.fit(X_train)
        return -m.score_samples(X_eval), -m.score_samples(X_train)
    if model_name == "OCSVM":
        m = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=contamination))
        m.fit(X_train)
        return -m.decision_function(X_eval), -m.decision_function(X_train)
    if model_name == "AE":
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        scaler = StandardScaler().fit(X_train)
        Xt = torch.tensor(scaler.transform(X_train), dtype=torch.float32, device=DEV)
        Xe = torch.tensor(scaler.transform(X_eval), dtype=torch.float32, device=DEV)
        d = Xt.shape[1]
        net = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(),
            nn.Linear(d // 2, d // 4), nn.ReLU(),
            nn.Linear(d // 4, d // 2), nn.ReLU(),
            nn.Linear(d // 2, d)).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        n_val = max(1, int(0.1 * len(Xt)))
        perm = torch.randperm(len(Xt), device=DEV)
        Xtr, Xva = Xt[perm[n_val:]], Xt[perm[:n_val]]
        best, patience, best_state = np.inf, 0, None
        ds = torch.utils.data.TensorDataset(Xtr)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        for epoch in range(100):
            for (b,) in dl:
                opt.zero_grad()
                loss = ((net(b) - b) ** 2).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                vloss = ((net(Xva) - Xva) ** 2).mean().item()
            if vloss < best - 1e-6:
                best, patience = vloss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 10:
                    break
        if best_state:
            net.load_state_dict(best_state)
        with torch.no_grad():
            se = ((net(Xe) - Xe) ** 2).mean(dim=1).cpu().numpy()
            st = ((net(Xt) - Xt) ** 2).mean(dim=1).cpu().numpy()
        return se, st
    raise KeyError(model_name)


# ---------------------------------------------------------------- stats
def bootstrap_auc_ci(y, s, n=N_BOOT):
    y = np.asarray(y)
    aucs = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, s[idx]))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (np.nan, np.nan)


def _delong_auc_var_parts(y, s):
    """Standard DeLong components for paired AUC comparison."""
    pos = s[y == 1]
    neg = s[y == 0]
    m, n = len(pos), len(neg)
    tx = np.searchsorted(np.sort(neg), pos, side="right") / n
    ty = np.searchsorted(np.sort(pos), neg, side="left") / m
    return tx, ty


def delong_test(y, s1, s2):
    """DeLong test for two correlated AUCs (same cases). Returns (auc1, auc2, z, p)."""
    from scipy.stats import norm
    y = np.asarray(y)
    auc1, auc2 = roc_auc_score(y, s1), roc_auc_score(y, s2)
    t1, t2 = _delong_auc_var_parts(y, s1)
    u1, u2 = _delong_auc_var_parts(y, s2)
    m, n = (y == 1).sum(), (y == 0).sum()
    v1 = t1 - t1.mean()
    v2 = t2 - t2.mean()
    s_e1 = np.var(np.concatenate([v1 / m, -u1 / n]), ddof=1)
    s_e2 = np.var(np.concatenate([v2 / m, -u2 / n]), ddof=1)
    # covariance term
    cov = np.cov(np.concatenate([v1 / m, -u1 / n]), np.concatenate([v2 / m, -u2 / n]), ddof=1)[0, 1]
    var_d = s_e1 + s_e2 - 2 * cov
    z = (auc1 - auc2) / np.sqrt(var_d) if var_d > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return auc1, auc2, z, p


def prf(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return p, r, f


# ---------------------------------------------------------------- protocols
def run_wesad(per_subject):
    rows, per_subj_rows = [], []
    pooled = {}  # model -> (scores, y)
    seed_scores = {s: [] for s in WESAD_SUBJECTS}
    for model in ["IF", "LOF", "OCSVM", "AE"]:
        pooled_scores_seed0, pooled_y, pooled_pred_seed0 = [], [], []
        for held in WESAD_SUBJECTS:
            Xtr = np.vstack([per_subject[s][0][per_subject[s][1] == 1] for s in WESAD_SUBJECTS if s != held])
            Xte, yte = per_subject[held]
            keep = np.isin(yte, [1, 2, 3])
            Xte, yte = Xte[keep], (yte[keep] == 2).astype(int)
            seeds = IF_SEEDS if model == "IF" else [0]
            fold_seed_preds = []
            se0 = None
            for seed in seeds:
                se, st = fit_score(model, Xtr, Xte, seed=seed)
                thr = np.percentile(st, 85)  # training-derived, pre-registered
                pred = (se > thr).astype(int)
                fold_seed_preds.append(pred)
                if model == "IF":
                    seed_scores[held].append(roc_auc_score(yte, se) if yte.min() != yte.max() else np.nan)
                if seed == 0:
                    se0 = se
                    pooled_scores_seed0.append(se)
                    pooled_pred_seed0.append(pred)
            pooled_y.append(yte)
            per_subj_rows.append({"subject": held, "model": model,
                                  "auc": roc_auc_score(yte, se0),
                                  "n_pos": int(yte.sum()), "n_neg": int((yte == 0).sum())})
        s_all = np.concatenate(pooled_scores_seed0)
        y_all = np.concatenate(pooled_y)
        p_all = np.concatenate(pooled_pred_seed0)
        auc = roc_auc_score(y_all, s_all)
        lo, hi = bootstrap_auc_ci(y_all, s_all)
        prec, rec, f1 = prf(y_all, p_all)
        rows.append({"dataset": "WESAD", "protocol": "LOSO", "model": model,
                     "auc": auc, "ci_low": lo, "ci_high": hi,
                     "precision": prec, "recall": rec, "f1": f1,
                     "threshold": "85th pct train"})
        pooled[model] = (s_all, y_all)
        print(f"  WESAD LOSO {model}: AUC {auc:.3f} [{lo:.3f},{hi:.3f}] P {prec:.3f} R {rec:.3f} F1 {f1:.3f}", flush=True)
    # IF seed variability (per-subject AUCs across seeds)
    seed_stats = {}
    arr = np.array([[seed_scores[s][i] for s in WESAD_SUBJECTS] for i in range(len(IF_SEEDS))])
    seed_stats["if_pooled_auc_mean"] = float(np.nanmean(arr.mean(axis=1)))
    seed_stats["if_pooled_auc_sd"] = float(np.nanstd(arr.mean(axis=1)))
    return rows, per_subj_rows, pooled, seed_stats


def run_mitbih(X, y, recs):
    tr = np.isin(recs, DS1) & (y == 0)
    te = np.isin(recs, DS2)
    Xtr, Xte, yte = X[tr], X[te], y[te]
    rows, pooled = [], {}
    for model in ["IF", "LOF", "OCSVM", "AE"]:
        seeds = IF_SEEDS if model == "IF" else [0]
        aucs = []
        for seed in seeds:
            se, st = fit_score(model, Xtr, Xte, seed=seed)
            aucs.append(roc_auc_score(yte, se))
            if seed == 0:
                s0, st0, thr0 = se, st, np.percentile(st, 85)
                pred0 = (se > thr0).astype(int)
        auc = float(np.mean(aucs))
        auc_sd = float(np.std(aucs))
        auc0 = roc_auc_score(yte, s0)
        lo, hi = bootstrap_auc_ci(yte, s0)
        prec, rec, f1 = prf(yte, pred0)
        rows.append({"dataset": "MIT-BIH", "protocol": "inter-patient DS1->DS2 (paced excl.)", "model": model,
                     "auc": auc, "auc_sd_over_seeds": auc_sd,
                     "auc_seed0": auc0, "ci_low": lo, "ci_high": hi,
                     "precision": prec, "recall": rec, "f1": f1,
                     "threshold": "85th pct train"})
        pooled[model] = (s0, yte)
        print(f"  MIT-BIH {model}: AUC {auc:.3f}±{auc_sd:.3f} [{lo:.3f},{hi:.3f}] P {prec:.3f} R {rec:.3f} F1 {f1:.3f}", flush=True)
    return rows, pooled


def run_ptbxl(X, y, folds, drop_cols=None):
    cols = None if drop_cols is None else [i for i, n in enumerate(FEATURE_NAMES) if n not in drop_cols]
    Xx = X if cols is None else X[:, cols]
    tr = (folds <= 8) & (y == 0)
    va = folds == 9
    te = folds == 10
    Xtr, Xva, Xte, yva, yte = Xx[tr], Xx[va], Xx[te], y[va], y[te]
    rows, curves, pooled = [], {}, {}
    for model in ["IF", "LOF", "OCSVM", "AE"]:
        seeds = IF_SEEDS if model == "IF" else [0]
        aucs = []
        for seed in seeds:
            se_te, st_tr = fit_score(model, Xtr, Xte, seed=seed)
            aucs.append(roc_auc_score(yte, se_te))
            if seed == 0:
                se_va, _ = fit_score(model, Xtr, Xva, seed=seed)
                s0, thr_train = se_te, np.percentile(st_tr, 85)
        # pre-registered: percentile maximizing F1 on fold 9, frozen to fold 10
        pcts = np.arange(1, 100)
        best_p, best_f = None, -1
        for p in pcts:
            thr = np.percentile(se_va, p)
            _, _, f = prf(yva, (se_va > thr).astype(int))
            if f > best_f:
                best_f, best_p = f, p
        thr_val = np.percentile(se_va, best_p)
        pred = (s0 > thr_val).astype(int)
        auc = float(np.mean(aucs))
        lo, hi = bootstrap_auc_ci(yte, s0)
        prec, rec, f1 = prf(yte, pred)
        rows.append({"dataset": "PTB-XL", "protocol": "thr on fold9 -> fold10", "model": model,
                     "auc": auc, "auc_sd_over_seeds": float(np.std(aucs)),
                     "ci_low": lo, "ci_high": hi,
                     "precision": prec, "recall": rec, "f1": f1,
                     "threshold": f"pct {best_p} on fold 9 (F1={best_f:.3f})"})
        pooled[model] = (s0, yte)
        curves[model] = {"best_pct": int(best_p)}
        print(f"  PTB-XL {model}: AUC {auc:.3f} [{lo:.3f},{hi:.3f}] P {prec:.3f} R {rec:.3f} F1 {f1:.3f} (pct {best_p})", flush=True)
    # threshold-sensitivity curve for LOF (figure)
    curve_rows = []
    se_te, _ = fit_score("LOF", Xtr, Xte)
    for p in range(1, 100):
        thr = np.percentile(se_te, p)
        prec, rec, f1 = prf(yte, (se_te > thr).astype(int))
        curve_rows.append({"percentile": p, "precision": prec, "recall": rec, "f1": f1})
    return rows, pd.DataFrame(curve_rows), pooled


def run_feature_ablation(X, y, folds):
    rows = []
    for gname, feats in FEATURE_GROUPS.items():
        cols = [i for i, n in enumerate(FEATURE_NAMES) if n not in feats]
        Xa = X[:, cols]
        tr = (folds <= 8) & (y == 0)
        te = folds == 10
        va = folds == 9
        se_va, _ = fit_score("LOF", Xa[tr], Xa[va])
        se_te, _ = fit_score("LOF", Xa[tr], Xa[te])
        best_p, best_f = None, -1
        for p in range(1, 100):
            thr = np.percentile(se_va, p)
            _, _, f = prf(y[va], (se_va > thr).astype(int))
            if f > best_f:
                best_f, best_p = f, p
        pred = (se_te > np.percentile(se_va, best_p)).astype(int)
        prec, rec, f1 = prf(y[te], pred)
        rows.append({"dropped_group": gname, "features_dropped": ",".join(feats),
                     "auc": roc_auc_score(y[te], se_te), "precision": prec, "recall": rec, "f1": f1})
        print(f"  ablation -{gname}: AUC {rows[-1]['auc']:.3f} F1 {f1:.3f}", flush=True)
    return rows


def make_figures(wesad_pooled, mitbih_pooled, ptbxl_pooled, curve_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, pooled, fname in [("WESAD (LOSO, pooled)", wesad_pooled, "roc_wesad"),
                                ("MIT-BIH (inter-patient DS2)", mitbih_pooled, "roc_mitbih"),
                                ("PTB-XL (fold 10)", ptbxl_pooled, "roc_ptbxl")]:
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        for model, color in [("IF", "steelblue"), ("LOF", "darkorange"), ("OCSVM", "seagreen"), ("AE", "purple")]:
            s, y = pooled[model]
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=color, label=f"{model} ({roc_auc_score(y, s):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(name); ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(FIGS / f"{fname}.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(curve_df.percentile, curve_df.precision, label="Precision")
    ax.plot(curve_df.percentile, curve_df.recall, label="Recall")
    ax.plot(curve_df.percentile, curve_df.f1, label="F1")
    ax.set_xlabel("Score percentile threshold (fold 10, LOF)")
    ax.set_title("PTB-XL threshold sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "ptbxl_threshold_curve.png", dpi=200); plt.close(fig)


def main():
    global DEV
    import torch
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"device": DEV, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    print("== WESAD ==", flush=True)
    w = get_cache("wesad")
    per_subject = {int(s): (w[f"X_{s}"], w[f"y_{s}"]) for s in w["subjects"]}
    wesad_rows, wesad_ps, wesad_pooled, wesad_seed_stats = run_wesad(per_subject)
    pd.DataFrame(wesad_ps).to_csv(OUT / "wesad_loso_per_subject.csv", index=False)
    summary["wesad"] = wesad_seed_stats

    print("== MIT-BIH ==", flush=True)
    m = get_cache("mitbih")
    X_m, y_m, rec_m = m["X"], m["y"], m["record"].astype(str)
    summary["mitbih"] = {
        "train_beats": int(((np.isin(rec_m, DS1)) & (y_m == 0)).sum()),
        "test_beats": int(np.isin(rec_m, DS2).sum()),
        "test_pos": int(y_m[np.isin(rec_m, DS2)].sum()),
        "symbols": json.loads(str(m["symbols"])),
    }
    mitbih_rows, mitbih_pooled = run_mitbih(X_m, y_m, rec_m)

    print("== PTB-XL ==", flush=True)
    p = get_cache("ptbxl")
    X_p, y_p, f_p = p["X"], p["y"], p["fold"]
    summary["ptbxl"] = {"n": int(len(y_p)), "normal": int((y_p == 0).sum()),
                        "fold10": int((f_p == 10).sum()), "fold10_pos": int(y_p[f_p == 10].sum())}
    ptbxl_rows, curve_df, ptbxl_pooled = run_ptbxl(X_p, y_p, f_p)
    curve_df.to_csv(OUT / "ptbxl_threshold_curve.csv", index=False)

    print("== DeLong IF vs LOF ==", flush=True)
    for ds, pooled in [("WESAD", wesad_pooled), ("MIT-BIH", mitbih_pooled), ("PTB-XL", ptbxl_pooled)]:
        s_if, yy = pooled["IF"]
        s_lof, _ = pooled["LOF"]
        _, _, z, pv = delong_test(yy, s_if, s_lof)
        summary[f"delong_{ds}"] = {"z": float(z), "p": float(pv)}
        print(f"  {ds}: IF vs LOF z={z:.3f} p={pv:.4g}", flush=True)

    print("== PTB-XL feature ablation ==", flush=True)
    abl = run_feature_ablation(X_p, y_p, f_p)
    pd.DataFrame(abl).to_csv(OUT / "feature_ablation_ptbxl.csv", index=False)

    all_rows = wesad_rows + mitbih_rows + ptbxl_rows
    pd.DataFrame(all_rows).to_csv(OUT / "detection_results_v2.csv", index=False)
    make_figures(wesad_pooled, mitbih_pooled, ptbxl_pooled, curve_df)
    summary["rows"] = all_rows
    (OUT / "detection_summary_v2.json").write_text(json.dumps(summary, indent=2, default=str))
    print("Saved outputs_v2/detection_results_v2.csv + summary json + figures", flush=True)


if __name__ == "__main__":
    main()
