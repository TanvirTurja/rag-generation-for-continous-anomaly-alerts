# Dataset Card — Retrieval-Augmented Generation for Continuous Anomaly Alerts

Datasets for the anomaly-detection + RAG-alert pipeline. All entries below are
**free, citable (DOI / peer-reviewed paper), and accessible without paid access
or human-subjects training**. Access difficulty is flagged per dataset.

> **Inclusion rule for this card:** a dataset is listed only if it is
> (a) free to download, (b) citable with a persistent DOI / publication, and
> (c) reachable in a few steps with **no** credentialed training required.
> Datasets that failed this test are listed under
> [Explicitly excluded](#explicitly-excluded).

---

## 1. Primary dataset — PPG-DaLiA

| Field | Value |
|---|---|
| **Full name** | PPG-DaLiA (PPG Domain Adaptation for Living database) |
| **Source** | UCI Machine Learning Repository (ID 495) |
| **Download** | https://archive.ics.uci.edu/dataset/495/ppg+dalia |
| **Access** | ✅ **Easy** — direct download (2.7 GB `data.zip` + readme) **or** `ucimlrepo` pip package. No account. |
| **Size** | ~8.3 M instances, 15 subjects |
| **Sampling** | Chest (RespiBAN): ECG, respiration, 3-axis accel @ 700 Hz. Wrist (Empatica E4): BVP/PPG @ 64 Hz, EDA @ 4 Hz, temp @ 4 Hz, accel @ 32 Hz |
| **Signals of interest** | **PPG (BVP)** and **ECG** — matches the project scope exactly |
| **Labels** | Activity labels (daily activities) — **no labeled anomalies / no pathology** |
| **License** | CC BY 4.0 |
| **Role in project** | Unsupervised anomaly detection (Isolation Forest) + pipeline/RAG demonstration |

**Python loader**
```python
from ucimlrepo import fetch_ucirepo
ppg_dalia = fetch_ucirepo(id=495)
X, y = ppg_dalia.data.features, ppg_dalia.data.targets
```

---

## 2. Recommended supplementary datasets

PPG-DaLiA has **no labeled anomalies**, so detection quality cannot be measured
on it alone. The datasets below fill that gap and are all free + citable.

### 2a. WESAD — *highest-priority complement*

| Field | Value |
|---|---|
| **Full name** | WESAD (Wearable Stress and Affect Detection) |
| **Source** | UCI Machine Learning Repository (ID 465) |
| **Download** | https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection |
| **Access** | ✅ **Easy** — direct download. No account. |
| **Size** | 15 subjects, wrist + chest devices |
| **Signals** | ECG, EDA, EMG, respiration, temp, accel. **Same Empatica E4 + RespiBAN sensors as PPG-DaLiA** |
| **Labels** | ✅ **Baseline / Amusement / Stress** (+ transient states) |
| **License** | Cite + attribute (see UCI page) |
| **Why it fits** | Identical modalities to PPG-DaLiA **plus** labeled physiological events → lets you evaluate whether your detector fires on real deviations. Minimal integration cost. |

```python
from ucimlrepo import fetch_ucirepo
wesad = fetch_ucirepo(id=465)
```

### 2b. MIT-BIH Arrhythmia Database — *ECG clinical ground truth*

| Field | Value |
|---|---|
| **Full name** | MIT-BIH Arrhythmia Database v1.0.0 |
| **Source** | PhysioNet |
| **Download** | https://physionet.org/content/mitdb/ |
| **Access** | ⚠️ **Easy-ish** — free, but requires a **free PhysioNet account** (email + accept license). Then `wfdb` pip package. |
| **Size** | 48 half-hour two-channel ECG recordings, 47 subjects, 360 Hz |
| **Labels** | ✅ Beat-level arrhythmia labels (PVC, PAC, AF, etc.) — the gold standard for cardiac anomaly detection |
| **License** | ODC-By 1.0 |
| **Why it fits** | Provides real, expert-annotated arrhythmias to test the ECG branch of your detector and to report precision/recall. (ECG only — no PPG.) |

```python
import wfdb
wfdb.dl_database("mitdb", dl_dir="./mitdb")   # after PhysioNet login config
```

### 2c. PTB-XL — *scale + diagnostic labels on ECG*

| Field | Value |
|---|---|
| **Full name** | PTB-XL, a large publicly available electrocardiography dataset |
| **Source** | PhysioNet (v1.0.3) |
| **Download** | https://physionet.org/content/ptb-xl/ |
| **Access** | ⚠️ **Easy-ish** — free PhysioNet account required. |
| **Size** | 21,801 clinical 12-lead ECGs, 18,869 patients, 10 s each |
| **Labels** | ✅ 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP) with subclasses |
| **License** | Open (see PhysioNet page) |
| **Why it fits** | Solves both the cohort-size and label problems for the ECG side; large enough to claim generalization. (ECG only — no PPG.) |

```bash
# direct download of the PTB-XL zip after login:
wget -r -N -c -np --user <user> --ask-password https://physionet.org/files/ptb-xl/1.0.3/
```

---

## 3. Explicitly excluded

Listed so you know what was considered and *why* it didn't meet the
"easy + 1-click" bar.

| Dataset | Reason excluded |
|---|---|
| **MIMIC-III / MIMIC-IV / MIMIC-III Waveform (MIMIC-WDB)** | Requires CITI human-subjects training + credentialed PhysioNet access. Free but **not** 1-click / easy access. |
| **Kaggle / Figshare / GitHub mirrors** (e.g., WESAD, MIT-BIH on Kaggle) | Not authoritative sources — fine for quick use, but **do not cite them**. Always cite the official repository/paper. |
| **PPG-BP, BIDMC, GUDB, etc.** | Either niche, inconsistently licensed, or lacking a clean persistent citation — not added to keep the card lean. |

---

## 4. Suggested setup for this project

- **Minimum defensible:** PPG-DaLiA (unsupervised detection + pipeline demo) **+ WESAD** (labeled events for evaluation).
- **Stronger paper:** add **PTB-XL** and/or **MIT-BIH** for clinical ECG anomaly ground truth at scale.

---

## 5. BibTeX citations

```bibtex
% =====================================================
% 1. PPG-DaLiA  (PRIMARY)
% =====================================================

@misc{reiss2019ppgdalia,
  author    = {Reiss, Attila and Indlekofer, Ina and Schmidt, Philip},
  title     = {{PPG-DaLiA}},
  year      = {2019},
  publisher = {UCI Machine Learning Repository},
  doi       = {10.24432/C53890},
  url       = {https://archive.ics.uci.edu/dataset/495/ppg+dalia}
}

@article{reiss2019deepppg,
  author  = {Reiss, Attila and Indlekofer, Ina and Schmidt, Philip and
             Van Laerhoven, Kristof},
  title   = {Deep {PPG}: Large-Scale Heart Rate Estimation with
             Convolutional Neural Networks},
  journal = {Sensors},
  volume  = {19},
  number  = {14},
  pages   = {3079},
  year    = {2019},
  doi     = {10.3390/s19143079}
}

% =====================================================
% 2a. WESAD
% =====================================================

@inproceedings{schmidt2018wesad,
  author    = {Schmidt, Philip and Reiss, Attila and Duerichen, Robert and
               Van Laerhoven, Kristof},
  title     = {Introducing {WESAD}, a Multimodal Dataset for Wearable
               Stress and Affect Detection},
  booktitle = {Proceedings of the 2018 ACM International Symposium on
               Wearable Computers (ISWC)},
  pages     = {400--408},
  year      = {2018},
  doi       = {10.1145/3242969.3242985}
}

% =====================================================
% 2b. MIT-BIH Arrhythmia Database
% =====================================================

@article{moody2001mitbih,
  author  = {Moody, George B. and Mark, Roger G.},
  title   = {The impact of the {MIT-BIH} Arrhythmia Database},
  journal = {IEEE Engineering in Medicine and Biology Magazine},
  volume  = {20},
  number  = {3},
  pages   = {45--50},
  year    = {2001},
  doi     = {10.1109/51.932724}
}

@misc{mitdb_dataset,
  author    = {Moody, George B. and Mark, Roger G.},
  title     = {{MIT-BIH} Arrhythmia Database v1.0.0},
  year      = {2001},
  publisher = {PhysioNet},
  doi       = {10.13026/C2F61Q},
  url       = {https://physionet.org/content/mitdb/}
}

@article{goldberger2000physionet,
  author  = {Goldberger, A. L. and Amaral, L. A. N. and Glass, L. and
             Hausdorff, J. M. and Ivanov, P. Ch. and Mark, R. G. and
             Mietus, J. E. and Moody, G. B. and Peng, C.-K. and
             Stanley, H. E.},
  title   = {{PhysioBank}, {PhysioToolkit}, and {PhysioNet}: Components of a
             New Research Resource for Complex Physiologic Signals},
  journal = {Circulation},
  volume  = {101},
  number  = {23},
  pages   = {e215--e220},
  year    = {2000},
  doi     = {10.1161/01.CIR.101.23.e215}
}

% =====================================================
% 2c. PTB-XL
% =====================================================

@article{wagner2020ptbxl,
  author  = {Wagner, Patrick and Strodthoff, Tobias and
             Bousseljot, Rasmus-Daniel and Kreiseler, Dieter and
             Lunze, Fatima I. and Samek, Wojciech and Schaeffter, Thomas},
  title   = {{PTB-XL}, a large publicly available electrocardiography dataset},
  journal = {Scientific Data},
  volume  = {7},
  pages   = {154},
  year    = {2020},
  doi     = {10.1038/s41597-020-0495-6}
}
```

---

### Quick reference — access difficulty

| Dataset | Download | Account? | Citable? |
|---|---|---|---|
| PPG-DaLiA | direct / `ucimlrepo` | none | ✅ DOI 10.24432/C53890 |
| WESAD | direct / `ucimlrepo` | none | ✅ DOI 10.1145/3242969.3242985 |
| MIT-BIH | `wfdb` / PhysioNet | free account | ✅ DOI 10.13026/C2F61Q |
| PTB-XL | PhysioNet | free account | ✅ DOI 10.1038/s41597-020-0495-6 |
| ~~MIMIC~~ | credentialed + CITI training | **excluded** | — |
