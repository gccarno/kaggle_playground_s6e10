#!/usr/bin/env python3
"""Batch 28's Final B blends on a packed pool (scripts/pack_pool.py), off the laptop.

Self-contained (numpy, scipy, scikit-learn only), so it runs on a bare VM without the repo.
The methods are blend.py's, line for line:
  climb: Caruana forward selection with replacement over OOF ranks, nested (select on 4
         folds, score the 5th); the full-data multiset gives the test prediction as the mean
         of within-test ranks (blend.blend_test);
  stack: L2 logistic regression on the logit-scale columns, nested; the full-data fit
         gives the test prediction (blend.cmd_stack).
Jobs run in parallel processes. Each writes <out>/<tag>.npz (oof, test) and <tag>.json.

    python burst_blends.py POOLDIR OUTDIR
"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

POOL = OUT = None


def load():
    global O, T, y, fold, tags
    O = np.load(POOL / "oof.npy", mmap_mode="r")
    T = np.load(POOL / "test.npy", mmap_mode="r")
    t = np.load(POOL / "truth.npz")
    y, fold = t["y"].astype(np.int8), t["fold"]
    tags = json.loads((POOL / "meta.json").read_text())["tags"]


def nrank(a):
    return (rankdata(a) / len(a)).astype(np.float32)


def auc(s, yy):
    r = rankdata(s)
    n1 = int(yy.sum())
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1)))


def auc_fast(s, yy):
    r = np.empty(len(s))
    r[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    n1 = yy.sum()
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))


def caruana(R, yy, max_steps, tol=1e-7):
    S = np.zeros(len(yy))
    counts, best = {}, -1.0
    for _ in range(max_steps):
        sc = [auc_fast(S + R[:, j], yy) for j in range(R.shape[1])]
        j = int(np.argmax(sc))
        if sc[j] - best < tol:
            break
        best = sc[j]
        S += R[:, j]
        counts[j] = counts.get(j, 0) + 1
    return counts


def job_climb(tag, cols, max_steps):
    t0 = time.time()
    R = np.column_stack([nrank(O[:, j]) for j in cols])
    out = np.zeros(len(y))
    folds = {}
    for f in np.unique(fold):
        tr, va = fold != f, fold == f
        c = caruana(R[tr], y[tr], max_steps)
        out[va] = sum(n * R[va, j] for j, n in c.items()) / sum(c.values())
        folds[int(f)] = {"picks": sum(c.values()), "held_out": auc(out[va], y[va])}
        print(f"{tag} fold {f}: {folds[int(f)]}", flush=True)
    full = caruana(R, y, max_steps)
    n = sum(full.values())
    insample = auc(sum(k * R[:, j] for j, k in full.items()) / n, y)
    test = sum(k * nrank(T[:, cols[j]]) for j, k in full.items()) / n
    counts = {tags[cols[j]]: k for j, k in sorted(full.items(), key=lambda t: -t[1])}
    res = {"tag": tag, "method": "caruana_nested", "nested_oof": auc(out, y),
           "in_sample_oof": insample, "counts": counts, "max_steps": max_steps,
           "pool": [tags[j] for j in cols], "folds": folds, "minutes": (time.time() - t0) / 60}
    return res, out, test


def job_stack(tag, cols, C):
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    X = np.asarray(O[:, cols], dtype=np.float64)
    out = np.zeros(len(y))
    for f in np.unique(fold):
        tr, va = fold != f, fold == f
        m = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        out[va] = m.decision_function(X[va])
        print(f"{tag} fold {f}: held-out {auc(out[va], y[va]):.6f}", flush=True)
    full = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    test = full.decision_function(np.asarray(T[:, cols], dtype=np.float64))
    res = {"tag": tag, "method": "nested_logit_stack", "nested_oof": auc(out, y), "C": C,
           "full_data_weights": {tags[j]: round(float(w), 6) for j, w in zip(cols, full.coef_[0])},
           "intercept": float(full.intercept_[0]), "pool": [tags[j] for j in cols],
           "minutes": (time.time() - t0) / 60}
    return res, 1 / (1 + np.exp(-out)), 1 / (1 + np.exp(-test))


def run(spec):
    kind, tag, cols, arg = spec
    load()
    res, oof, test = (job_climb if kind == "climb" else job_stack)(tag, cols, arg)
    np.savez(OUT / f"{tag}.npz", oof=oof, test=test)
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"DONE {tag} nested {res['nested_oof']:.6f} ({res['minutes']:.1f} min)", flush=True)
    return tag


def main():
    global POOL, OUT
    POOL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    load()
    allc = list(range(len(tags)))
    pub = [j for j, t in enumerate(tags) if t.startswith("P_")]
    specs = [("climb", "F1", allc, 60), ("stack", "F2", allc, 1.0), ("climb", "F3", pub, 60),
             ("climb", "F6", allc, 25), ("stack", "F7", allc, 0.01)]
    with Pool(len(specs)) as p:
        for tag in p.imap_unordered(run, specs):
            pass


if __name__ == "__main__":
    main()
