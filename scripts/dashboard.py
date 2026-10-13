#!/usr/bin/env python
"""Regenerate reports/dashboard.html from runs.csv + the curated ledger.

    python scripts/dashboard.py            # writes reports/dashboard.html
    python scripts/dashboard.py --check    # only verify the ledger covers runs.csv

Inputs:
  experiments/runs.csv     machine-written by runlog.py (scores, learner, timestamps)
  experiments/ledger.csv   hand-curated: phase, axis, twin, role, verdict, one_liner per run_tag
  experiments/axes.csv     hand-curated: status and summary per axis
  experiments/preds/<id>/manifest.json   blend legs (written by blend.py archive)

Update ritual (CLAUDE.md): after every batch collection, fill the ledger verdicts, re-run this,
rewrite STATUS.md, republish the artifact, commit.
"""
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "dashboard.html"
DEADLINE = datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc)
FIRST_DAY = datetime(2026, 10, 1, tzinfo=timezone.utc)
# README section 4 -- the instrument. Frozen; change only if README section 4 changes.
GATE1 = 0.00014          # shipping gate, OOF delta vs strict twin
PUBLIC_SD = 0.000152     # paired public dAUC SD, seed-twin pair
SEED_FLOOR = 0.000039    # |OOF(A0) - OOF(A0s)|


def load():
    runs = pd.read_csv(ROOT / "experiments/runs.csv")
    led = pd.read_csv(ROOT / "experiments/ledger.csv", dtype=str).fillna("")
    axes = pd.read_csv(ROOT / "experiments/axes.csv", dtype=str).fillna("")
    return runs, led, axes


def check(runs, led):
    missing = sorted(set(runs.run_tag) - set(led.run_tag))
    dup = led.run_tag[led.run_tag.duplicated()].tolist()
    if missing or dup:
        sys.exit(f"ledger.csv out of date -- missing: {missing}  duplicated: {dup}")


def num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def build(runs, led, axes):
    r = runs.drop_duplicates("run_tag", keep="last").set_index("run_tag")
    oof = r.final_oof_auc.to_dict()
    rows = []
    for _, l in led.iterrows():
        t = l.run_tag
        have = t in r.index
        o = num(oof.get(t)) if have else None
        tw = num(oof.get(l.twin)) if l.twin else None
        rows.append({
            "tag": t, "phase": l.phase, "axis": l.axis, "twin": l.twin, "role": l.role,
            "verdict": l.verdict, "note": l.one_liner,
            "learner": r.loc[t, "learner"] if have else "",
            "date": r.loc[t, "timestamp_utc"][:10] if have else "",
            "ts": r.loc[t, "timestamp_utc"] if have else "",
            "oof": o, "pub": num(r.loc[t, "public_lb_score"]) if have else None,
            "delta": (o - tw) if (o is not None and tw is not None) else None,
            "full": bool(have and r.loc[t, "sample_frac"] == 1),
        })

    champ = next(x for x in rows if x["verdict"] == "CHAMPION")
    legs = {}
    man = ROOT / r.loc[champ["tag"], "preds_dir"] / "manifest.json"
    if man.exists():
        legs = json.loads(man.read_text())["legs"]
    by_tag = {x["tag"]: x for x in rows}
    comp = [{"tag": k, "learner": by_tag.get(k, {}).get("learner", ""),
             "axis": by_tag.get(k, {}).get("axis", ""), "oof": by_tag.get(k, {}).get("oof"),
             "note": by_tag.get(k, {}).get("note", "")} for k in legs]

    phases = {}
    for x in rows:
        if x["ts"] and x["phase"] not in phases and x["phase"] != "0":
            phases[x["phase"]] = x["ts"]

    ax = []
    for _, a in axes.iterrows():
        mine = [x for x in rows if x["axis"] == a.axis]
        best = max((x for x in mine if x["delta"] is not None), key=lambda x: x["delta"], default=None)
        ax.append({"axis": a.axis, "status": a.status, "findings": a.findings, "summary": a.summary,
                   "n": len(mine), "best": best["tag"] if best else "",
                   "bestd": best["delta"] if best else None,
                   "tags": [x["tag"] for x in mine]})

    now = datetime.now(timezone.utc)
    sub = runs[runs.public_lb_score.notna()]
    return {
        "generated": now.strftime("%Y-%m-%d %H:%M UTC"),
        "days_left": max(0, (DEADLINE - now).days),
        "subs_used": int(len(sub)), "subs_total": 10 * 31,
        "champ": champ, "comp": comp, "rows": rows, "axes": ax, "phases": phases,
        "gate1": GATE1, "public_sd": PUBLIC_SD, "seed_floor": SEED_FLOOR,
        "n_runs": len(runs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    runs, led, axes = load()
    check(runs, led)
    if a.check:
        print(f"ledger OK: {len(led)} rows cover {runs.run_tag.nunique()} run tags")
        return
    data = build(runs, led, axes)
    tpl = (ROOT / "scripts" / "dashboard_template.html").read_text(encoding="utf-8")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(data['rows'])} ledger rows, champion {data['champ']['tag']})")


if __name__ == "__main__":
    main()
