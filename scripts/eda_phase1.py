#!/usr/bin/env python3
"""Phase 1 data facts for README section 1 / 6. Read-only; prints a report.

    python scripts/eda_phase1.py

Sections:
  1. duplicates and train/test row overlap
  2. adversarial validation (can a model tell train from test?)
  3. cardinality, NaN, rating-zero rates
  4. relation to the ORIGINAL dataset: exact-row and key-tuple matches, and label agreement
  5. LOOKUP DIAGNOSTIC per numeric column -- the S6E9 income test. After removing the smooth
     trend, do per-value positive rates vary MORE than binomial noise allows? Excess
     dispersion means the column carries value-specific information a tree reading it as a
     magnitude cannot see; ratio ~1 means it is a plain magnitude.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from config import ID, TARGET  # noqa: E402
from dataset import RAW_CAT, RAW_NUM, RATINGS, load_competition, load_origin  # noqa: E402

FEATS = RAW_NUM + RAW_CAT


def h(t):
    print(f"\n{'=' * 8} {t}")


def main():
    train, test, _, y = load_competition()
    og, oy = load_origin()
    print(f"train {train.shape}  test {test.shape}  original {og.shape}")
    print(f"positive rate: train {y.mean():.6f}  original {oy.mean():.6f}")

    h("1. duplicates")
    print(f"train dup feature rows: {train.duplicated(FEATS).sum():,}")
    print(f"  of which label-conflicting: "
          f"{(train.assign(y=y).groupby(FEATS, dropna=False)['y'].nunique() > 1).sum():,}")
    print(f"test dup feature rows : {test.duplicated(FEATS).sum():,}")
    tr_keys = pd.MultiIndex.from_frame(train[FEATS].astype(str))
    te_keys = pd.MultiIndex.from_frame(test[FEATS].astype(str))
    print(f"test rows whose exact feature row is in train: {te_keys.isin(tr_keys).sum():,}")

    h("2. adversarial validation (lgb, 3-fold, 200k sample each)")
    import lightgbm as lgb
    a = pd.concat([train[FEATS].sample(200_000, random_state=0),
                   test[FEATS].sample(200_000, random_state=0)], ignore_index=True)
    for c in RAW_CAT:
        a[c] = a[c].astype("category")
    lab = np.r_[np.zeros(200_000), np.ones(200_000)]
    oof = np.zeros(len(a))
    for i, j in StratifiedKFold(3, shuffle=True, random_state=0).split(a, lab):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                               n_jobs=4, verbose=-1).fit(a.iloc[i], lab[i])
        oof[j] = m.predict_proba(a.iloc[j])[:, 1]
    print(f"train-vs-test AUC = {roc_auc_score(lab, oof):.4f}   (0.5 = no drift)")

    h("3. cardinality / NaN / rating zeros")
    rows = []
    for c in FEATS:
        rows.append({"col": c, "nuniq_train": train[c].nunique(), "nuniq_test": test[c].nunique(),
                     "nuniq_orig": og[c].nunique(), "nan_train": int(train[c].isna().sum()),
                     "nan_test": int(test[c].isna().sum()),
                     "test_vals_unseen_in_train": int((~test[c].dropna().isin(
                         set(train[c].dropna()))).sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nshare of rating == 0:")
    print(pd.Series({c: (train[c] == 0).mean() for c in RATINGS}).round(4).to_string())

    h("4. relation to the ORIGINAL dataset")
    og_keys = pd.MultiIndex.from_frame(og[FEATS].astype(str))
    print(f"original dup feature rows: {og.duplicated(FEATS).sum():,}")
    print(f"train rows whose EXACT 21-feature row exists in original: "
          f"{tr_keys.isin(og_keys).sum():,} ({tr_keys.isin(og_keys).mean():.4%})")
    print(f"test  rows whose EXACT 21-feature row exists in original: "
          f"{te_keys.isin(og_keys).sum():,}")
    for key in (["flight_distance"], ["flight_distance", "age"],
                ["flight_distance", "age", "dep_delay"],
                ["flight_distance", "age", "dep_delay", "arr_delay"],
                ["flight_distance", "age", "customer_type", "travel_type", "travel_class"]):
        ok = og.assign(_y=oy).groupby(key, dropna=False)["_y"].agg(["mean", "count"])
        uniq = ok[ok["count"] == 1]
        t = train[key].copy()
        t["_y"] = y
        mt = t.merge(uniq.reset_index(), on=key, how="inner")
        agree = ((mt["mean"] > 0.5).astype(int) == mt["_y"]).mean() if len(mt) else float("nan")
        print(f"key {key}: {len(ok):,} original groups, {len(uniq):,} unique; train rows "
              f"hitting a UNIQUE original key: {len(mt):,} ({len(mt)/len(train):.2%}); "
              f"label agreement with that original row: {agree:.4f}")

    h("5. lookup diagnostic (per-value excess dispersion after a smooth trend)")
    for c in ["flight_distance", "age", "dep_delay", "arr_delay"]:
        d = pd.DataFrame({"v": train[c], "y": y}).dropna()
        st = d.groupby("v")["y"].agg(["mean", "count"]).sort_index()
        st = st[st["count"] >= 30]
        if len(st) < 20:
            print(f"{c}: too few populated values ({len(st)})")
            continue
        # smooth trend = count-weighted rolling mean over neighbouring values
        w = max(5, len(st) // 50) | 1
        trend = ((st["mean"] * st["count"]).rolling(w, center=True, min_periods=1).sum() /
                 st["count"].rolling(w, center=True, min_periods=1).sum())
        resid = st["mean"] - trend
        binom_var = (trend * (1 - trend) / st["count"]).mean()
        ratio = resid.var() / binom_var
        print(f"{c:<16} values(n>=30)={len(st):>5}  resid SD={resid.std():.4f}  "
              f"binomial SD={np.sqrt(binom_var):.4f}  dispersion ratio={ratio:.2f}"
              f"   {'<-- LOOKUP-LIKE' if ratio > 2 else ''}")


if __name__ == "__main__":
    main()
