#!/usr/bin/env python3
"""Residual lookup test: does a run's OOF still miss value-specific signal in some column?

For each key (a column, or a pair of columns), group the OOF residuals r = y - p by the key's
value and compare their per-group means with what calibrated noise allows:

    chi2 = sum_g (sum_{i in g} r_i)^2 / sum_{i in g} p_i (1 - p_i),   df = number of groups

chi2/df near 1 means the model has already used the column's values. Well above 1 means
the values carry label information the model misses: it's a lookup the model can't see.
Groups with fewer than --min-n rows are skipped. This is the CONDITIONED test, so it asks
about what is left after the model, not about raw variance (the T20 lesson, README
Phase 3).

    python scripts/residual_lookup.py experiments/preds/<run_id> [--min-n 30]
"""
import argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import numpy as np, pandas as pd  # noqa: E402
from dataset import RAW_CAT, RAW_NUM, load_competition  # noqa: E402

PAIRS = [("flight_distance", c) for c in RAW_CAT] + [("flight_distance", "age")]


def chi2_df(key, r, v, min_n):
    d = pd.DataFrame({"k": key, "r": r, "v": v})
    g = d.groupby("k", observed=True).agg(n=("r", "size"), s=("r", "sum"), var=("v", "sum"))
    g = g[g["n"] >= min_n]
    if len(g) == 0:
        return 0, float("nan")
    return len(g), float((g["s"] ** 2 / g["var"]).sum() / len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds_dir")
    ap.add_argument("--min-n", type=int, default=30)
    a = ap.parse_args()
    tr, _, _, y = load_competition()
    oof = pd.read_csv(Path(a.preds_dir) / "oof_proba_lgb.csv") if \
        (Path(a.preds_dir) / "oof_proba_lgb.csv").exists() else \
        pd.read_csv(next(Path(a.preds_dir).glob("oof_proba_*.csv")))
    p = np.clip(oof.iloc[:, -1].to_numpy(np.float64), 1e-6, 1 - 1e-6)
    assert len(p) == len(y), (len(p), len(y))
    r, v = y - p, p * (1 - p)
    print(f"{a.preds_dir}: residual chi2/df by key (groups with n>={a.min_n})")
    rows = []
    for c in RAW_NUM + RAW_CAT:
        rows.append((c, *chi2_df(tr[c].to_numpy(), r, v, a.min_n)))
    for k1, k2 in PAIRS:
        key = tr[k1].astype(str) + "|" + tr[k2].astype(str)
        rows.append((f"{k1} x {k2}", *chi2_df(key.to_numpy(), r, v, a.min_n)))
    for name, n, x in sorted(rows, key=lambda t: -np.nan_to_num(t[2])):
        print(f"  {name:<34} groups {n:>6}  chi2/df {x:6.2f}")


if __name__ == "__main__":
    main()
