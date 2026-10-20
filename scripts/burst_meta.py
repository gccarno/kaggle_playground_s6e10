#!/usr/bin/env python3
"""Batch 29's non-linear meta-stackers on a packed pool (scripts/pack_pool.py), for oci-burst.

LightGBM over the pool's logit columns, optionally plus the raw competition columns, so
the meta-model can learn where each member is trustworthy (the L2 stack of batch 28 gives
every member one weight everywhere). Nested like burst_blends.py: for each outer fold the
model trains on three of the other four folds, early-stops on the fourth, and predicts the
outer fold, which therefore never touched training or stopping. The test prediction is the
mean of the five outer models' test probabilities.

    python burst_meta.py POOLDIR OUTDIR TAG {logits|logits+raw|resid+raw} [TRAIN_CSV TEST_CSV]

resid+raw (F13): the L2 logit stack (C = 0.01) is fit on the three training folds and its
logit becomes LightGBM's init_score, so the trees (lr 0.01, 7 leaves) only learn where the
linear stack is systematically wrong, from the logits plus the raw columns. The stopping
fold is outside the stack's fit, so early stopping stays honest.
"""
import json, sys, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PARAMS = {"objective": "binary", "metric": "auc", "learning_rate": 0.03, "num_leaves": 15,
          "min_data_in_leaf": 200, "feature_fraction": 0.5, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 10.0, "num_threads": 16, "verbose": -1, "seed": 42}


def auc(s, y):
    r = rankdata(s)
    n1 = int(y.sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def raw(train_csv, test_csv, ids, test_ids):
    tr, te = pd.read_csv(train_csv), pd.read_csv(test_csv)
    assert (tr["id"].to_numpy() == ids).all() and (te["id"].to_numpy() == test_ids).all()
    cols = [c for c in te.columns if c != "id"]
    both = pd.concat([tr[cols], te[cols]], ignore_index=True)
    for c in cols:
        if not pd.api.types.is_numeric_dtype(both[c]):
            both[c] = both[c].astype("category").cat.codes
    X = both.to_numpy(np.float32)
    return X[:len(tr)], X[len(tr):], cols


def main():
    pool, out, tag, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    O = np.load(pool / "oof.npy")
    T = np.load(pool / "test.npy")
    t = np.load(pool / "truth.npz")
    y, fold, ids, test_ids = t["y"].astype(int), t["fold"], t["ids"], t["test_ids"]
    names = json.loads((pool / "meta.json").read_text())["tags"]
    n_pool = O.shape[1]
    if mode in ("logits+raw", "resid+raw"):
        Rtr, Rte, rc = raw(sys.argv[5], sys.argv[6], ids, test_ids)
        O, T, names = np.hstack([O, Rtr]), np.hstack([T, Rte]), names + [f"raw:{c}" for c in rc]
    folds = np.unique(fold)
    oof, test, info = np.zeros(len(y)), np.zeros(len(T)), {}
    gain = np.zeros(O.shape[1])
    for i, f in enumerate(folds):
        stop = folds[(i + 1) % len(folds)]
        tr, es, va = (fold != f) & (fold != stop), fold == stop, fold == f
        if mode == "resid+raw":
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=0.01, max_iter=2000).fit(O[tr, :n_pool], y[tr])
            b = {k: lr.decision_function(X[:, :n_pool]) for k, X in
                 (("tr", O[tr]), ("es", O[es]), ("va", O[va]), ("te", T))}
            prm = {**PARAMS, "learning_rate": 0.01, "num_leaves": 7}
            m = lgb.train(prm, lgb.Dataset(O[tr], y[tr], init_score=b["tr"]), 3000,
                          valid_sets=[lgb.Dataset(O[es], y[es], init_score=b["es"])],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            it = m.best_iteration
            oof[va] = b["va"] + m.predict(O[va], num_iteration=it, raw_score=True)
            test += 1 / (1 + np.exp(-(b["te"] + m.predict(T, num_iteration=it, raw_score=True)))) / len(folds)
            info[int(f)] = {"stack_only": auc(b["va"], y[va])}
        else:
            m = lgb.train(PARAMS, lgb.Dataset(O[tr], y[tr]), 3000,
                          valid_sets=[lgb.Dataset(O[es], y[es])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va] = m.predict(O[va], num_iteration=m.best_iteration)
            test += m.predict(T, num_iteration=m.best_iteration) / len(folds)
        gain += m.feature_importance("gain")
        info[int(f)] = {**info.get(int(f), {}), "best_iter": m.best_iteration,
                        "held_out": auc(oof[va], y[va])}
        print(f"{tag} fold {f}: {info[int(f)]}", flush=True)
    top = np.argsort(-gain)[:25]
    res = {"tag": tag, "method": f"nested_lgb_meta[{mode}]", "nested_oof": auc(oof, y),
           "pool": json.loads((pool / "meta.json").read_text())["tags"], "params": PARAMS,
           "folds": info, "top_gain": {names[j]: round(float(gain[j] / gain.sum()), 4) for j in top},
           "minutes": (time.time() - t0) / 60}
    np.savez(out / f"{tag}.npz", oof=oof, test=test)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"DONE {tag} nested {res['nested_oof']:.6f} ({res['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
