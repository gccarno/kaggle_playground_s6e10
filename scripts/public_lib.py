#!/usr/bin/env python3
"""Public OOF/test libraries as pseudo-runs, for Final B only (README Checkpoint 4, Phase 5).

Checkpoint 4 (Oct 18): public OOF/test pairs may enter Final B, fold-aligned and with a
manifest. Final A stays ours-only. Every public member becomes an ordinary archived run
(learner "public", tag P_<owner>_<name>, notes NOSUBMIT PUBLIC), so blend.py works on it
unchanged and the nightly scheduler never submits it on its own.

    python scripts/public_lib.py fetch cdeotte/s6e10-tfm-oof-and-test-predictions ...
    python scripts/public_lib.py audit            # -> experiments/public_manifest.csv
    python scripts/public_lib.py import           # kept members -> runs.csv + preds + ledger

Downloads are untrusted data: each dataset gets its own directory under data/public/, .npy
files load with allow_pickle=False, and nothing in them is executed.

The audit, per oof_<name>.npy / test_<name>.npy pair:
  - shape 699,635 / 299,844 and finite values; logits and probabilities both work, since
    every blend here is rank-based;
  - solo OOF AUC and per-fold AUC on our frozen folds. An honest, aligned OOF has fold AUCs
    inside our legs' spread;
  - LEAK when solo OOF > 0.9625, above every honest member we know of (cdeotte's best TFM
    is about 0.9612 and the honest public stack CV is 0.96208), so a single member that
    high is fitting its own validation rows;
  - META when the name says stack, blend, ensemble, anchor, twin or submission: a
    meta-model refit over other members is not a member, and stacking it double-counts;
  - near-copy clusters: rank correlation > 0.999 with a stronger member already kept
    (karttikjangid: 181 of 340 public OOFs are near-copies).
A dataset that ships its fold file (train_ids_folds.csv) is checked partition-for-partition
against ours.
"""
import argparse, hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from config import ID, PREDS_DIR, TARGET, target_to_int  # noqa: E402
import runlog  # noqa: E402

PUB = REPO_ROOT / "data" / "public"
MANIFEST = REPO_ROOT / "experiments" / "public_manifest.csv"
LEDGER = REPO_ROOT / "experiments" / "ledger.csv"
N_TRAIN, N_TEST = 699_635, 299_844
LEAK_AUC = 0.9625
DUP_RHO = 0.999
META = re.compile(r"stack|blend|ens(emble)?\b|anchor|twin|submission|meta|hill", re.I)
# Known blend outputs re-hosted under member-like names. cdeotte's 124-library nj_01 is
# najiama's "01" blend (najiama/s6e10-oof ships 01_blend_oof): its solo 0.961744 is above
# every single model in every library.
META_NAMES = {("cdeotte/s6e10-124-public-oof-and-test-predictions", "nj_01")}


def folds_ids():
    """Our frozen fold array and the train/test id orders, from an archived base run."""
    ref = next(r for r in runlog.read_rows() if r.get("run_tag") == "T24")
    d = REPO_ROOT / ref["preds_dir"]
    o = pd.read_csv(next(d.glob("oof_proba_*.csv")))
    t = pd.read_csv(next(d.glob("test_proba_*.csv")))
    tr = pd.read_csv(REPO_ROOT / "data" / "train.csv", usecols=[ID, TARGET])
    assert (o["id"].to_numpy() == tr[ID].to_numpy()).all()
    return o["fold"].to_numpy(), tr[ID].to_numpy(), t["id"].to_numpy(), \
        target_to_int(tr[TARGET]).astype(np.int8)


def auc(s, y):
    r = rankdata(s)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cmd_fetch(datasets):
    for ds in datasets:
        out = PUB / ds.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["kaggle", "datasets", "download", ds, "-p", str(out), "--unzip", "-q"])
        if r.returncode != 0:
            if not any(out.iterdir()):
                out.rmdir()
            print(f"{ds}: download FAILED (exit {r.returncode}), skipped")
            continue
        print(f"{ds}: {len(list(out.iterdir()))} files -> {out.relative_to(REPO_ROOT)}")


def members():
    """(dataset dir, name, oof ref, test ref) for every oof_/test_ .npy pair, and for every
    .npz holding `oof` and `test` arrays (ref "file.npz:key")."""
    for d in sorted(p for p in PUB.iterdir() if p.is_dir()):
        for o in sorted(d.rglob("oof_*.npy")):
            name = o.stem[4:]
            t = o.with_name(f"test_{name}.npy")
            yield d, name, o, (t if t.exists() else None)
        for z in sorted(d.rglob("*.npz")):
            with np.load(z, allow_pickle=False) as f:
                keys = set(f.files)
            if {"oof", "test"} <= keys:
                yield d, z.stem, f"{z}:oof", f"{z}:test"


def rel(ref):
    p, _, key = str(ref).partition(".npz:")
    p = Path(p + ".npz") if key else Path(p)
    return p.relative_to(PUB).as_posix() + (f":{key}" if key else "")


def load(ref, n):
    p, _, key = str(ref).partition(".npz:")
    if key:
        with np.load(p + ".npz", allow_pickle=False) as f:
            a = f[key]
    else:
        a = np.load(p, allow_pickle=False)
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 2:
        a = a[:, 1]
    return a.reshape(-1) if a.size == n else None


def check_fold_file(d, fold, ids):
    f = d / "train_ids_folds.csv"
    if not f.exists():
        return "no fold file"
    g = pd.read_csv(f)
    if len(g) != len(ids) or not (g["id"].to_numpy() == ids).all():
        return "FOLD FILE id order differs"
    # Same partition, up to fold labels: every one of their folds maps to one of ours.
    m = pd.crosstab(g["fold"].to_numpy(), fold)
    return "aligned (fold file)" if ((m > 0).sum(axis=1) == 1).all() else "FOLDS DIFFER"


def cmd_audit():
    fold, ids, _, y = folds_ids()
    rows, kept_ranks = [], []
    cand = []
    for d, name, o, t in members():
        rec = {"dataset": d.name.replace("__", "/", 1), "name": name,
               "oof_file": rel(o), "test_file": rel(t) if t else ""}
        oof = load(o, N_TRAIN)
        te = load(t, N_TEST) if t else None
        flags = []
        if oof is None or te is None:
            flags.append("SHAPE")
        elif not (np.isfinite(oof).all() and np.isfinite(te).all()):
            flags.append("NONFINITE")
        if META.search(name) or (rec["dataset"], name) in META_NAMES:
            flags.append("META")
        if not flags:
            rec["oof_auc"] = round(auc(oof, y), 6)
            fa = [auc(oof[fold == f], y[fold == f]) for f in np.unique(fold)]
            rec["fold_aucs"] = " ".join(f"{a:.6f}" for a in fa)
            rec["fold_std"] = round(float(np.std(fa)), 6)
            if rec["oof_auc"] > LEAK_AUC:
                flags.append("LEAK")
            rec["folds"] = check_fold_file(d, fold, ids)
            if rec["folds"].startswith("FOLD"):
                flags.append("FOLDS")
            rec["oof_sha"] = hashlib.sha256(oof.tobytes()).hexdigest()[:16]
        rec["flags"] = " ".join(flags)
        rows.append(rec)
        if not flags:
            cand.append((rec["oof_auc"], len(rows) - 1, oof))
    # Near-copy clustering, strongest first: a member is kept unless its rank correlation
    # with an already-kept member exceeds DUP_RHO.
    kept = []
    for a, i, oof in sorted(cand, key=lambda c: -c[0]):
        r = rankdata(oof)
        r = (r - r.mean()) / r.std()
        dup = next((rows[j]["name"] + "@" + rows[j]["dataset"] for j, rk in kept
                    if float(np.mean(r * rk)) > DUP_RHO), None)
        if dup:
            rows[i]["flags"] = f"DUP of {dup}"
        else:
            kept.append((i, r))
            rows[i]["keep"] = True
    df = pd.DataFrame(rows)
    df["keep"] = df.get("keep", False).fillna(False).astype(bool)
    df["tag"] = [tag_for(r.dataset, r.name) for r in df.itertuples()]
    df.to_csv(MANIFEST, index=False)
    print(df[["tag", "oof_auc", "fold_std", "folds", "flags", "keep"]].to_string())
    print(f"\n{df['keep'].sum()} kept of {len(df)} -> {MANIFEST.relative_to(REPO_ROOT)}")


def tag_for(dataset, name):
    owner = dataset.split("/")[0]
    return f"P_{owner[:10]}_{re.sub(r'[^0-9A-Za-z]+', '_', name)}"


def cmd_import():
    fold, ids, test_ids, _ = folds_ids()
    man = pd.read_csv(MANIFEST)
    have = {r.get("run_tag") for r in runlog.read_rows()}
    led = pd.read_csv(LEDGER, dtype=str, keep_default_na=False)  # "NULL" is a verdict, not NaN
    new_led = []
    for r in man[man["keep"]].itertuples():
        if r.tag in have:
            continue
        oof, te = load(PUB / r.oof_file, N_TRAIN), load(PUB / r.test_file, N_TEST)
        run_id = "p" + hashlib.sha256(f"{r.dataset}/{r.name}".encode()).hexdigest()[:7]
        out = PREDS_DIR / run_id
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": ids, "fold": fold, "proba": oof}).to_csv(
            out / "oof_proba_public.csv", index=False)
        pd.DataFrame({"id": test_ids, "proba": te}).to_csv(out / "test_proba_public.csv",
                                                           index=False)
        (out / "manifest.json").write_text(json.dumps(
            {"source": r.dataset, "oof_file": r.oof_file, "test_file": r.test_file,
             "oof_sha": r.oof_sha, "folds": r.folds}, indent=2), encoding="utf-8")
        fa = [float(x) for x in str(r.fold_aucs).split()]
        metrics = {"run_tag": r.tag, "learner": "public", "n_folds": len(fa),
                   "sample_frac": 1.0, "final_oof_auc": r.oof_auc,
                   "fold_auc_mean": round(float(np.mean(fa)), 6), "fold_auc_std": r.fold_std,
                   "fold_aucs": fa}
        row = runlog.build_row(
            metrics, run_id=run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit="", git_dirty=False, kernel_ref=f"public:{r.dataset}",
            kernel_version="", description=f"{r.tag}: public member {r.name} ({r.dataset})",
            public_lb_score="", private_lb_score="", submission_msg="",
            preds_dir=out.relative_to(REPO_ROOT).as_posix(),
            notes=f"NOSUBMIT PUBLIC source={r.dataset}/{r.oof_file}. Final B only.")
        runlog.append_run_row(row)
        new_led.append({"tag": r.tag, "phase": "5", "axis": "public", "twin": "",
                        "role": "member", "verdict": "INFO",
                        "summary": f"Public member {r.name} from {r.dataset}: OOF {r.oof_auc:.6f}; Final B only"})
        print(f"{r.tag} -> {run_id}  OOF {r.oof_auc:.6f}")
    if new_led:
        cols = list(led.columns)
        add = pd.DataFrame(new_led).rename(columns=dict(zip(
            ["tag", "phase", "axis", "twin", "role", "verdict", "summary"], cols[:7])))
        pd.concat([led, add[cols[:7]]], ignore_index=True).to_csv(LEDGER, index=False)
    print(f"imported {len(new_led)} member(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch"); p.add_argument("datasets", nargs="+")
    sub.add_parser("audit")
    sub.add_parser("import")
    a = ap.parse_args()
    {"fetch": lambda: cmd_fetch(a.datasets), "audit": cmd_audit, "import": cmd_import}[a.cmd]()


if __name__ == "__main__":
    main()
