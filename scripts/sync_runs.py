#!/usr/bin/env python3
"""Keep the laptop's and the VM's runs.csv in step, merging by run_id, never overwriting.

Both copies gain rows on their own: VM batches on the VM, kernel runs and
`submit_run.py --fill-missing` scores on the laptop. Copying one file over the other
silently deletes the other side's rows. A merge appends only the run_ids that are missing
and fills blank score columns, and the preds dirs travel with their rows.

    python scripts/sync_runs.py pull                 # VM -> laptop: new rows + their preds
    python scripts/sync_runs.py push T16 T17         # laptop -> VM: these runs' rows + preds
                                                     #   (so the slot scheduler can submit them)
    python scripts/sync_runs.py merge other.csv [--only ID ...]   # the primitive both use
"""
import argparse, csv, re, subprocess, sys, tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import runlog  # noqa: E402
from submit_run import find_run  # noqa: E402

HOST = "oci-free"
REMOTE_REPO = "/data/kaggle_playground_s6e10"
# Columns that start blank and get filled in later. A blank here is "not known yet", so
# the other side's value wins. Every other column is set once, when the run is logged.
FILL = ("public_lb_score", "private_lb_score", "submission_msg")


def well_formed(r):
    """Only 8-hex run_ids with the canonical preds_dir cross machines. Both values end up
    in scp paths and a remote shell command, so a malformed row must not be able to steer
    either one."""
    rid = r.get("run_id", "")
    return bool(re.fullmatch(r"[0-9a-f]{8}", rid)) and r.get("preds_dir") == f"experiments/preds/{rid}"


def plan_merge(local, other, only=None):
    """Returns (rows to append, {run_id: {col: value}} fills). Pure, so it can be tested."""
    have = {r["run_id"]: r for r in local}
    append, fills = [], {}
    for r in other:
        rid = r.get("run_id", "")
        if only is not None and rid not in only:
            continue
        if not well_formed(r):
            print(f"  skip malformed row {rid!r}")
            continue
        if rid not in have:
            append.append(r)
            continue
        f = {c: r[c] for c in FILL if (r.get(c) or "").strip() and not (have[rid].get(c) or "").strip()}
        if f:
            fills[rid] = f
    return append, fills


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge(path, only=None):
    append, fills = plan_merge(runlog.read_rows(), read_csv(path), only)
    for r in append:
        runlog.append_run_row({k: v for k, v in r.items() if v is not None})
        print(f"  + {r['run_id']} {r.get('run_tag', '')}")
    for rid, f in fills.items():
        runlog.update_run(rid, **f)
        print(f"  ~ {rid}: {', '.join(f)}")
    print(f"merged {len(append)} new row(s), filled {len(fills)}")
    return append


def sh(cmd):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pull():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "vm_runs.csv"
        sh(["scp", "-q", f"{HOST}:{REMOTE_REPO}/experiments/runs.csv", str(tmp)])
        for r in merge(tmp):
            dest = REPO_ROOT / r["preds_dir"]
            if r.get("preds_dir") and not dest.exists():
                sh(["scp", "-q", "-r", f"{HOST}:{REMOTE_REPO}/{r['preds_dir']}",
                    str(dest.parent)])


def push(keys):
    rows = runlog.read_rows()
    hits = [find_run(rows, k) for k in keys]
    bad = [r["run_id"] for r in hits if not well_formed(r)]
    if bad:
        raise SystemExit(f"refusing malformed run_id/preds_dir: {bad}")
    ids = [r["run_id"] for r in hits]
    for r in hits:
        pd = r["preds_dir"]
        sh(["scp", "-q", "-r", str(REPO_ROOT / pd),
            f"{HOST}:{REMOTE_REPO}/{Path(pd).parent.as_posix()}/"])
    sh(["scp", "-q", str(runlog.RUNS_CSV), f"{HOST}:/tmp/laptop_runs.csv"])
    sh(["ssh", HOST, f"cd {REMOTE_REPO} && .venv/bin/python scripts/sync_runs.py merge "
                     f"/tmp/laptop_runs.csv --only {' '.join(ids)}"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pull")
    sub.add_parser("push").add_argument("runs", nargs="+")
    m = sub.add_parser("merge")
    m.add_argument("path")
    m.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    if a.cmd == "pull":
        pull()
    elif a.cmd == "push":
        push(a.runs)
    else:
        merge(a.path, set(a.only) if a.only else None)


if __name__ == "__main__":
    main()
