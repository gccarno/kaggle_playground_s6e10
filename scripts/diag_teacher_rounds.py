#!/usr/bin/env python3
"""Phase 3 diagnostic: how teacher quality depends on boosting rounds.

The selection criterion is the ORIGINAL dataset's own 5-fold CV AUC, so competition labels
play no part in choosing teacher settings. The competition AUC is printed for comparison
only. Run it on the VM: 6 fits of 5,000 rounds are too heavy for the laptop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from dataset import RAW_CAT, RAW_NUM, load_competition, load_origin  # noqa: E402

tr, te, _, y = load_competition()
og, oy = load_origin()
cols = RAW_NUM + RAW_CAT
A, T = og[cols].copy(), tr[cols].copy()
for c in RAW_CAT:
    v = pd.api.types.CategoricalDtype(sorted(set(A[c]) | set(T[c])))
    A[c], T[c] = A[c].astype(v), T[c].astype(v)
P = dict(learning_rate=0.03, num_leaves=63, min_child_samples=20, colsample_bytree=0.8,
         subsample=0.8, subsample_freq=1, verbose=-1, random_state=42, n_jobs=2)
R = [250, 500, 1000, 1500, 3000, 5000]
cvs = {r: [] for r in R}
for i, j in StratifiedKFold(5, shuffle=True, random_state=0).split(A, oy):
    m = lgb.LGBMClassifier(n_estimators=max(R), **P).fit(A.iloc[i], oy[i])
    for r in R:
        cvs[r].append(roc_auc_score(oy[j], m.predict_proba(A.iloc[j], num_iteration=r)[:, 1]))
    print("fold done", flush=True)
m = lgb.LGBMClassifier(n_estimators=max(R), **P).fit(A, oy)
for r in R:
    comp = roc_auc_score(y, m.predict_proba(T, num_iteration=r)[:, 1])
    print(f"rounds {r:5d}: original CV AUC {np.mean(cvs[r]):.5f} | vs competition labels {comp:.5f}",
          flush=True)
