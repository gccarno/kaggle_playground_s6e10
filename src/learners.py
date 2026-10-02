"""Learners. Each returns (val_proba, test_proba, best_iteration); AUC is the eval metric.

Ported from S6E9 `fit_predict`, with one deliberate change: each learner reads its OWN
param dict (cfg["lgb_params"], cfg["xgb_params"], cfg["cat_params"], cfg["glm_params"]).
In S6E9 xgb and cat read LightGBM-named keys (num_leaves, colsample_bytree...) and mapped
them, so tuning one learner silently retuned the others.

The neural/EBM/TabICL learners were not ported: nine architectures were tested in S6E9 and
none reached the strong-and-decorrelated quadrant (FINDINGS section 4). Port one only when
a probe with a written mechanism needs it.
"""
import numpy as np
import pandas as pd


def fit_predict(cfg, Xtr, ytr, Xva, yva, Xte, feats_cat):
    name = cfg["learner"]
    esr = cfg["early_stopping_rounds"]

    if name == "lgb":
        import lightgbm as lgb
        p = cfg["lgb_params"]
        mono = None
        if cfg["monotone"]:
            mono = [int(cfg["monotone"].get(c, 0)) for c in Xtr.columns]
        # One group per column = every tree splits on one feature: a boosted GAM.
        inter = [[i] for i in range(Xtr.shape[1])] if cfg["additive_only"] else None
        m = lgb.LGBMClassifier(random_state=cfg["model_seed"], n_jobs=-1, verbose=-1,
                               monotone_constraints=mono, interaction_constraints=inter, **p)
        # eval_set=, not eval_X=/eval_y=: the latter only exists in LightGBM 4.7+ and broke
        # on Kaggle's pinned image in S6E9 (X3).
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        return (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1],
                int(m.best_iteration_ or p["n_estimators"]))

    if name == "xgb":
        import xgboost as xgb
        p = cfg["xgb_params"]
        m = xgb.XGBClassifier(random_state=cfg["model_seed"], n_jobs=-1, tree_method="hist",
                              enable_categorical=True, eval_metric="auc",
                              early_stopping_rounds=esr, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1],
                int(m.best_iteration or p["n_estimators"]))

    if name == "cat":
        from catboost import CatBoostClassifier, Pool
        p = cfg["cat_params"]
        cat_idx = [Xtr.columns.get_loc(c) for c in feats_cat if c in Xtr.columns]
        A, B, C = Xtr.copy(), Xva.copy(), Xte.copy()
        for c in feats_cat:
            if c in A.columns:
                A[c], B[c], C[c] = A[c].astype(str), B[c].astype(str), C[c].astype(str)
        m = CatBoostClassifier(random_seed=cfg["model_seed"], eval_metric="AUC", verbose=0,
                               early_stopping_rounds=esr, **p)
        m.fit(Pool(A, ytr, cat_features=cat_idx), eval_set=Pool(B, yva, cat_features=cat_idx))
        return (m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1],
                int(m.get_best_iteration() or p["iterations"]))

    if name == "glm":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        cats = [c for c in feats_cat if c in Xtr.columns]

        def design(X, cols=None):
            D = pd.get_dummies(X, columns=cats, drop_first=True).astype(float)
            D = D.fillna(0.0)
            return D if cols is None else D.reindex(columns=cols, fill_value=0)
        A = design(Xtr)
        B, C = design(Xva, A.columns), design(Xte, A.columns)
        sc = StandardScaler().fit(A)          # fit on the training fold only
        m = LogisticRegression(max_iter=3000, **cfg["glm_params"]).fit(sc.transform(A), ytr)
        return (m.predict_proba(sc.transform(B))[:, 1],
                m.predict_proba(sc.transform(C))[:, 1], 0)

    raise ValueError(f"unknown learner {name!r}")
