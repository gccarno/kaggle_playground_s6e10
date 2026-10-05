#!/usr/bin/env python3
"""Fixed rank-mean blends over archived runs, and Gate 2 (playbook section 3, section 6).

Gate 1 is a probe's solo OOF against its strict twin. Gate 2 is its CONTRIBUTION to the
blend, and only Gate 2 is a shipping decision for a leg: in S6E8, 10 of 17 probes cleared
gate 1 and 0 cleared gate 2.

The blend is fixed and unfitted: every leg's OOF is turned into ranks over the whole OOF,
scaled to (0, 1], and the ranks are averaged with equal weight. AUC only sees order, so
rank-averaging removes each learner's calibration scale (cat, xgb and lgb probabilities
are not on the same scale) without fitting anything. Test predictions are ranked within
test the same way.

    python scripts/blend.py score T8 T14 T15
    python scripts/blend.py gate2 --pool T8 T14 T15 --cand T27 T24
    python scripts/blend.py curve T15 T14 T8 T27 T26 T16      # AUC vs pool size, in order
    python scripts/blend.py stack T8 T14 T15                  # nested-CV logit control
    python scripts/blend.py disagree --ref T15 --all          # strength vs disagreement
    python scripts/blend.py write T8 T14 T15 --tag B1 --description "..." --notes "..."

Legs are run_tags or run_ids from experiments/runs.csv.

ADD(c | P)  = AUC(P + c) - AUC(P)
SWAP(c | P) = AUC(P - w + c) - AUC(P), where w is P's weakest contributor (the leg whose
              removal costs least). Pool size stays constant, so a gain can't come from
              just having one more average member.
Each comes with a paired-bootstrap SD (rows resampled once, both blends scored on the
same resample). A leg passes Gate 2 when ADD > 2 SD and SWAP > 0.
"""
import argparse, json, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from config import ID, PREDS_DIR, TARGET, target_to_int  # noqa: E402
import runlog  # noqa: E402

N_BOOT = 200


# ------------------------------------------------------------------ loading
_TRUTH = None


def truth():
    global _TRUTH
    if _TRUTH is None:
        tr = pd.read_csv(REPO_ROOT / "data" / "train.csv", usecols=[ID, TARGET])
        _TRUTH = tr[ID].to_numpy(), target_to_int(tr[TARGET]).astype(np.int8)
    return _TRUTH


def resolve(key):
    """run_tag or run_id -> runs.csv row. Exactly one match, or stop."""
    hit = [r for r in runlog.read_rows() if key in (r["run_id"], r.get("run_tag", ""))]
    if len(hit) != 1:
        raise SystemExit(f"{key!r} matches {len(hit)} runs in runs.csv")
    return hit[0]


_LEGS = {}


def leg(key):
    """(tag, run_id, oof proba, fold array, test ids, test proba), cached."""
    if key in _LEGS:
        return _LEGS[key]
    row = resolve(key)
    d = REPO_ROOT / row["preds_dir"]
    oof_f = sorted(d.glob("oof_proba_*.csv"))
    te_f = sorted(d.glob("test_proba_*.csv"))
    if len(oof_f) != 1 or len(te_f) != 1:
        raise SystemExit(f"{key}: expected one oof_proba_*/test_proba_* in {d}, "
                         f"found {[p.name for p in oof_f + te_f]}")
    ids, _ = truth()
    o = pd.read_csv(oof_f[0])
    if not (o["id"].to_numpy() == ids).all():
        raise SystemExit(f"{key}: OOF ids are not aligned to train.csv order")
    t = pd.read_csv(te_f[0])
    rec = (row.get("run_tag") or row["run_id"], row["run_id"], o["proba"].to_numpy(),
           o["fold"].to_numpy(), t["id"].to_numpy(), t["proba"].to_numpy())
    _LEGS[key] = rec
    return rec


def check_aligned(keys):
    """Every leg must share the frozen fold assignment and the test id order."""
    ref = leg(keys[0])
    for k in keys[1:]:
        L = leg(k)
        if not (L[3] == ref[3]).all():
            raise SystemExit(f"{k}: fold assignment differs from {keys[0]} -- not blendable")
        if not (L[4] == ref[4]).all():
            raise SystemExit(f"{k}: test id order differs from {keys[0]}")


def nrank(p):
    return rankdata(p) / len(p)


_RANKS = {}


def oof_rank(key):
    if key not in _RANKS:
        _RANKS[key] = nrank(leg(key)[2])
    return _RANKS[key]


def blend_oof(keys):
    return np.mean([oof_rank(k) for k in keys], axis=0)


def blend_test(keys):
    return np.mean([nrank(leg(k)[5]) for k in keys], axis=0)


# ------------------------------------------------------------------ fast weighted AUC
class AUC:
    """AUC with arbitrary row weights in O(n), after one sort per score vector.
    A bootstrap resample is a multinomial weight vector, so 200 resamples cost 200
    bincounts instead of 200 sorts. Ties get half credit, as in roc_auc_score."""

    def __init__(self, score, y):
        self.inv = np.unique(score, return_inverse=True)[1]
        self.k = self.inv.max() + 1
        self.y = y

    def __call__(self, w=None):
        w = np.ones(len(self.y)) if w is None else w
        pos = np.bincount(self.inv, weights=w * self.y, minlength=self.k)
        neg = np.bincount(self.inv, weights=w * (1 - self.y), minlength=self.k)
        below = np.cumsum(neg) - neg
        return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def boot_weights(n, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def paired_delta(score_a, score_b, n_boot=N_BOOT):
    """(AUC(b) - AUC(a), paired bootstrap SD of that delta)."""
    _, y = truth()
    fa, fb = AUC(score_a, y), AUC(score_b, y)
    d = fb() - fa()
    ds = [fb(w) - fa(w) for w in boot_weights(len(y), n_boot)]
    return d, float(np.std(ds))


def auc(score):
    return AUC(score, truth()[1])()


# ------------------------------------------------------------------ commands
def cmd_score(keys):
    check_aligned(keys)
    for k in keys:
        print(f"  {leg(k)[0]:<8} {leg(k)[1]}  solo {auc(leg(k)[2]):.6f}")
    print(f"rank_mean of {len(keys)}: {auc(blend_oof(keys)):.6f}")


def contributions(pool):
    """Leave-one-out cost of each leg: AUC(P) - AUC(P - leg)."""
    full = auc(blend_oof(pool))
    return {k: full - auc(blend_oof([j for j in pool if j != k])) for k in pool}


def cmd_gate2(pool, cands, n_boot):
    check_aligned(pool + cands)
    base = blend_oof(pool)
    base_auc = auc(base)
    loo = contributions(pool) if len(pool) > 1 else {pool[0]: np.nan}
    weakest = min(loo, key=loo.get)
    print(f"pool ({len(pool)}): {' '.join(leg(k)[0] for k in pool)}   rank_mean {base_auc:.6f}")
    print("leave-one-out contribution: " +
          "  ".join(f"{leg(k)[0]} {v:+.6f}" for k, v in loo.items()))
    print(f"weakest contributor (swapped out for SWAP): {leg(weakest)[0]}\n")
    print(f"{'cand':<8} {'solo':>9} {'ADD':>10} {'sd':>9} {'SWAP':>10} {'sd':>9}  verdict")
    rest = [k for k in pool if k != weakest]
    for c in cands:
        add, add_sd = paired_delta(base, blend_oof(pool + [c]), n_boot)
        if len(pool) > 1:
            swp, swp_sd = paired_delta(base, blend_oof(rest + [c]), n_boot)
        else:
            swp, swp_sd = paired_delta(base, oof_rank(c), n_boot)
        ok = add > 2 * add_sd and swp > 0
        print(f"{leg(c)[0]:<8} {auc(leg(c)[2]):.6f} {add:+.6f} {add_sd:.6f} "
              f"{swp:+.6f} {swp_sd:.6f}  {'PASS' if ok else 'fail'}")


def cmd_curve(keys):
    check_aligned(keys)
    prev = None
    print(f"{'n':>3} {'added':<8} {'rank_mean':>10} {'step':>10}")
    for i in range(1, len(keys) + 1):
        a = auc(blend_oof(keys[:i]))
        step = "" if prev is None else f"{a - prev:+.6f}"
        print(f"{i:>3} {leg(keys[i-1])[0]:<8} {a:.6f} {step:>10}")
        prev = a


def cmd_stack(keys):
    """Control: a logistic stack on leg logits, fit on 4 folds and scored on the 5th, so
    its OOF is honest. Compared against the unfitted rank_mean on the same rows."""
    from sklearn.linear_model import LogisticRegression
    check_aligned(keys)
    _, y = truth()
    fold = leg(keys[0])[3]
    X = np.column_stack([np.log(np.clip(leg(k)[2], 1e-7, 1 - 1e-7)
                                / np.clip(1 - leg(k)[2], 1e-7, 1)) for k in keys])
    out = np.zeros(len(y))
    for f in np.unique(fold):
        tr, va = fold != f, fold == f
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
        out[va] = m.decision_function(X[va])
    w = LogisticRegression(C=1.0, max_iter=1000).fit(X, y).coef_[0]
    rm = blend_oof(keys)
    d, sd = paired_delta(rm, out)
    print(f"rank_mean       {auc(rm):.6f}")
    print(f"nested logit    {auc(out):.6f}   delta {d:+.6f}  (paired sd {sd:.6f})")
    print("full-data weights: " + "  ".join(f"{leg(k)[0]} {v:+.3f}" for k, v in zip(keys, w)))


def cmd_disagree(ref, keys):
    """Strength vs disagreement (playbook section 7). Disagreement = 1 - Spearman of OOF
    ranks against the reference. The ensemble axis is open only if some leg is both
    strong and decorrelated."""
    r = oof_rank(ref)
    rows = []
    for k in keys:
        if k == ref:
            continue
        try:
            check_aligned([ref, k])
        except SystemExit as e:
            print(f"  skip {k}: {e}", file=sys.stderr)
            continue
        rows.append((leg(k)[0], auc(leg(k)[2]), 1 - np.corrcoef(r, oof_rank(k))[0, 1]))
    df = pd.DataFrame(rows, columns=["leg", "solo", "disagree"]).sort_values("solo",
                                                                           ascending=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    rho, p = spearmanr(df["solo"], df["disagree"])
    print(f"\nSpearman(solo, disagreement) = {rho:+.3f} (p={p:.2g}, n={len(df)})")


def cmd_write(keys, tag, description, notes):
    check_aligned(keys)
    if any(r.get("run_tag") == tag for r in runlog.read_rows()):
        raise SystemExit(f"run_tag {tag!r} already exists")
    ids, y = truth()
    oof = blend_oof(keys)
    fold = leg(keys[0])[3]
    test_ids, test = leg(keys[0])[4], blend_test(keys)
    fold_aucs = [AUC(oof[fold == f], y[fold == f])() for f in np.unique(fold)]

    run_id = uuid.uuid4().hex[:8]
    out = PREDS_DIR / run_id
    out.mkdir(parents=True)
    pd.DataFrame({"id": ids, "fold": fold, "proba": oof}).to_csv(out / "oof_proba_blend.csv",
                                                                 index=False)
    pd.DataFrame({"id": test_ids, "proba": test}).to_csv(out / "test_proba_blend.csv",
                                                         index=False)
    pd.DataFrame({ID: test_ids, TARGET: test}).to_csv(out / "submission.csv", index=False)
    manifest = {"method": "rank_mean", "legs": {leg(k)[0]: leg(k)[1] for k in keys}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    git = lambda *a: __import__("subprocess").run(["git", *a], capture_output=True,
                                                  text=True, cwd=REPO_ROOT).stdout.strip()
    metrics = {"run_tag": tag, "learner": "blend", "n_features": len(keys), "n_folds":
               len(fold_aucs), "sample_frac": 1.0, "final_oof_auc": round(auc(oof), 6),
               "fold_auc_mean": round(float(np.mean(fold_aucs)), 6),
               "fold_auc_std": round(float(np.std(fold_aucs)), 6),
               "fold_aucs": [round(a, 6) for a in fold_aucs],
               "engineered": [leg(k)[0] for k in keys]}
    row = runlog.build_row(
        metrics, run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
        kernel_ref="blend", kernel_version="", description=description,
        public_lb_score="", private_lb_score="", submission_msg="",
        preds_dir=out.relative_to(REPO_ROOT).as_posix(), notes=notes)
    runlog.append_run_row(row)
    print(f"{tag} -> {run_id}: rank_mean of {len(keys)} legs, OOF {metrics['final_oof_auc']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score"); p.add_argument("legs", nargs="+")
    p = sub.add_parser("gate2")
    p.add_argument("--pool", nargs="+", required=True)
    p.add_argument("--cand", nargs="+", required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p = sub.add_parser("curve"); p.add_argument("legs", nargs="+")
    p = sub.add_parser("stack"); p.add_argument("legs", nargs="+")
    p = sub.add_parser("disagree")
    p.add_argument("--ref", required=True)
    p.add_argument("legs", nargs="*")
    p.add_argument("--all", action="store_true", help="every full-data run in runs.csv")
    p = sub.add_parser("write")
    p.add_argument("legs", nargs="+")
    p.add_argument("--tag", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--notes", default="")
    a = ap.parse_args()

    if a.cmd == "score":
        cmd_score(a.legs)
    elif a.cmd == "gate2":
        cmd_gate2(a.pool, a.cand, a.n_boot)
    elif a.cmd == "curve":
        cmd_curve(a.legs)
    elif a.cmd == "stack":
        cmd_stack(a.legs)
    elif a.cmd == "disagree":
        keys = a.legs
        if a.all:
            keys = [r["run_id"] for r in runlog.read_rows()
                    if float(r.get("sample_frac") or 1) == 1.0 and r.get("learner") != "blend"]
        cmd_disagree(a.ref, keys)
    elif a.cmd == "write":
        cmd_write(a.legs, a.tag, a.description, a.notes)


if __name__ == "__main__":
    main()
