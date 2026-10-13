#!/usr/bin/env python
"""Small EDA aggregates for the dashboard's EDA tab -> reports/eda.json. Read-only on data/.

    python scripts/eda_report.py

The data is static; re-run only when the champion changes (the residual panel reads its OOF).
Contents: data facts, positive rate by every column, the FD-as-route-ID scatter, train vs
original, and residual chi2/df per column at A0 and at the champion.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from dataset import RATINGS, RAW_CAT, RAW_NUM, load_competition, load_origin  # noqa: E402
from residual_lookup import PAIRS, chi2_df  # noqa: E402

OUT = ROOT / "reports" / "eda.json"
DELAY_EDGES = [0, 1, 5, 15, 30, 60, 120, 180, 1e9]
DELAY_LABELS = ["0", "1-4", "5-14", "15-29", "30-59", "60-119", "120-179", "180+"]


def rate_by(key, y, min_n=1):
    g = pd.DataFrame({"k": key, "y": y}).groupby("k", observed=True)["y"].agg(["size", "mean"])
    g = g[g["size"] >= min_n]
    return {"x": [str(i) if not isinstance(i, (int, np.integer, float)) else float(i) for i in g.index],
            "n": g["size"].astype(int).tolist(), "rate": g["mean"].round(5).tolist()}


def delay_bins(s):
    return pd.cut(s, DELAY_EDGES, right=False, labels=DELAY_LABELS).astype(object).where(s.notna(), "NaN")


def champion_oof():
    runs = pd.read_csv(ROOT / "experiments/runs.csv").set_index("run_tag")
    led = pd.read_csv(ROOT / "experiments/ledger.csv", dtype=str)
    champ = led.loc[led.verdict == "CHAMPION", "run_tag"].iloc[0]
    out = {}
    for tag in ["A0", champ]:
        d = ROOT / runs.loc[tag, "preds_dir"]
        f = d / "oof_proba_lgb.csv" if (d / "oof_proba_lgb.csv").exists() else next(d.glob("oof_proba_*.csv"))
        out[tag] = pd.read_csv(f).iloc[:, -1].to_numpy(np.float64)
    return champ, out


def main():
    tr, te, _, y = load_competition()
    og, oy = load_origin()
    y = np.asarray(y, dtype=np.float64)
    e = {}

    e["facts"] = {
        "train_rows": len(tr), "test_rows": len(te), "original_rows": len(og),
        "pos_rate_train": round(float(y.mean()), 6), "pos_rate_original": round(float(np.mean(oy)), 6),
        "cardinality": {c: int(tr[c].nunique()) for c in RAW_NUM + RAW_CAT},
        "nan": {c: int(tr[c].isna().sum()) for c in RAW_NUM + RAW_CAT if tr[c].isna().any()},
        "rating_zero_pct": {c: round(100 * float((tr[c] == 0).mean()), 2) for c in RATINGS},
        "fd_unseen_in_test": int((~te.flight_distance.isin(set(tr.flight_distance))).sum()),
        "adversarial_auc": 0.5010,  # README section 1 (eda_phase1.py); not refit here
    }

    e["age"] = rate_by(tr.age, y, 30)
    fd_bin = (tr.flight_distance // 50 * 50).astype(int)
    e["fd_binned"] = rate_by(fd_bin, y, 30)
    # Per-route rate vs the smooth (50-wide bin) trend for the most common routes: if FD were a
    # plain magnitude, these dots would hug the trend within binomial noise.
    g = pd.DataFrame({"fd": tr.flight_distance, "b": fd_bin, "y": y})
    trend = g.groupby("b")["y"].mean()
    top = g.groupby("fd").agg(n=("y", "size"), rate=("y", "mean"), b=("b", "first")).nlargest(400, "n")
    top["trend"] = trend.reindex(top.b).to_numpy()
    top["sd"] = np.sqrt(top.trend * (1 - top.trend) / top.n)
    e["fd_routes"] = {"fd": top.index.astype(int).tolist(), "n": top.n.astype(int).tolist(),
                      "rate": top.rate.round(4).tolist(), "trend": top.trend.round(4).tolist(),
                      "z": ((top.rate - top.trend) / top.sd).round(2).tolist()}
    e["delays"] = {c: rate_by(delay_bins(tr[c]), y) for c in ["dep_delay", "arr_delay"]}
    for d in e["delays"].values():  # keep bin order
        order = [l for l in DELAY_LABELS + ["NaN"] if l in d["x"]]
        idx = [d["x"].index(l) for l in order]
        for k in d:
            d[k] = [d[k][i] for i in idx]
    e["ratings"] = {c: rate_by(tr[c], y) for c in RATINGS}
    e["cats"] = {c: rate_by(tr[c], y) for c in RAW_CAT}
    oyy = np.asarray(oy, dtype=np.float64)
    e["original_cats"] = {c: rate_by(og[c], oyy) for c in RAW_CAT}

    champ, oofs = champion_oof()
    keys = [[c] for c in RAW_NUM + RAW_CAT] + [list(p) for p in PAIRS]
    chi = {"keys": [" x ".join(k) for k in keys], "models": {}}
    for tag, p in oofs.items():
        if tag != "A0":
            # The champion is a rank-mean blend: map its ranks to probabilities with an isotonic
            # fit on the OOF (in-sample, so slightly optimistic -- a diagnostic, not a score).
            p = IsotonicRegression(out_of_bounds="clip").fit(p, y).predict(p)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        r, v = y - p, p * (1 - p)
        vals = []
        for k in keys:
            key = tr[k[0]].map(str)
            for c in k[1:]:
                key = key + "|" + tr[c].map(str)
            vals.append(round(chi2_df(key.to_numpy(), r, v, 30)[1], 3))
        chi["models"][tag] = vals
    chi["champion"] = champ
    e["chi2"] = chi

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(e, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB); champion {champ}")
    f = e["facts"]
    print(f"rows {f['train_rows']}/{f['test_rows']}  pos {f['pos_rate_train']}  FD values {f['cardinality']['flight_distance']}  "
          f"NaN {f['nan']}  FD chi2/df A0 {chi['models']['A0'][1]} -> {champ} {chi['models'][champ][1]}")


if __name__ == "__main__":
    main()
