#!/usr/bin/env python3
"""Pack a blend pool into two float32 .npy matrices for an off-laptop blend run (Phase 5).

Two 195-leg blend jobs (F1/F2) ran the 16 GB laptop out of memory on Oct 19, because
blend.leg() holds every leg's OOF and test arrays in float64. This streams one leg at a
time into memory-mapped matrices instead, so the laptop never holds more than one leg.

Each column is the leg's score on a logit scale: logit(p) when the leg's values are
probabilities, or the raw values when they already are logits. Ranks, the only input of
the rank-mean and Caruana blends, are unchanged by that monotone map, and the logit stack
gets the same feature blend.cmd_stack builds.

    python scripts/pack_pool.py OUTDIR KEY [KEY ...]
    -> OUTDIR/oof.npy (n_train x m), test.npy (n_test x m), meta.json (tags, run_ids),
       truth.npz (y, fold, train ids, test ids)

A public member (P_ tag) whose imported CSVs are not on this machine is read from its raw
download under data/public/ instead (public_lib.py fetch), and must hash to the oof_sha
the audit recorded, so a VM re-download of a newer dataset version cannot slip in.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from blend import logit, resolve, truth  # noqa: E402


def score(p):
    p = p.astype(np.float64)
    return (logit(p) if p.min() >= 0 and p.max() <= 1 else p).astype(np.float32)


def raw_public(tag):
    import hashlib
    from public_lib import MANIFEST, N_TEST, N_TRAIN, PUB, load
    man = pd.read_csv(MANIFEST)
    m = man[(man["tag"] == tag) & man["keep"]].iloc[0]
    o, t = load(PUB / m.oof_file, N_TRAIN), load(PUB / m.test_file, N_TEST)
    if hashlib.sha256(o.tobytes()).hexdigest()[:16] != m.oof_sha:
        raise SystemExit(f"{tag}: raw OOF does not hash to the audited oof_sha (new version?)")
    return o, t


def main():
    out, keys = Path(sys.argv[1]), sys.argv[2:]
    out.mkdir(parents=True, exist_ok=True)
    ids, y = truth()
    rows = [resolve(k) for k in keys]
    fold = test_ids = None
    O = T = None
    for j, r in enumerate(rows):
        d = REPO_ROOT / r["preds_dir"]
        if r["run_tag"].startswith("P_") and not d.exists():
            assert fold is not None, "list an own leg first: it supplies the fold and id order"
            po, pt = raw_public(r["run_tag"])
            O[:, j], T[:, j] = score(po), score(pt)
            print(f"{j + 1}/{len(rows)} {r['run_tag']} (raw)", flush=True)
            continue
        o = pd.read_csv(next(d.glob("oof_proba_*.csv")))
        t = pd.read_csv(next(d.glob("test_proba_*.csv")))
        if fold is None:
            fold, test_ids = o["fold"].to_numpy(), t["id"].to_numpy()
            O = np.lib.format.open_memmap(out / "oof.npy", "w+", np.float32, (len(ids), len(rows)))
            T = np.lib.format.open_memmap(out / "test.npy", "w+", np.float32,
                                          (len(test_ids), len(rows)))
        if not ((o["id"].to_numpy() == ids).all() and (o["fold"].to_numpy() == fold).all()
                and (t["id"].to_numpy() == test_ids).all()):
            raise SystemExit(f"{r['run_tag']}: ids or folds not aligned")
        O[:, j] = score(o["proba"].to_numpy())
        T[:, j] = score(t["proba"].to_numpy())
        print(f"{j + 1}/{len(rows)} {r['run_tag']}", flush=True)
    O.flush(); T.flush()
    np.savez(out / "truth.npz", y=y, fold=fold, ids=ids, test_ids=test_ids)
    (out / "meta.json").write_text(json.dumps(
        {"tags": [r["run_tag"] for r in rows], "run_ids": [r["run_id"] for r in rows]},
        indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
