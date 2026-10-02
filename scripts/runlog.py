"""experiments/runs.csv -- the single most valuable file in the repo (append-only, in git).

Ported from S6E9 collect_run.py. The schema GROWS: a metrics key that isn't in the header
yet (e.g. a new learner's "<name>_oof_auc") rewrites the header with the column appended
instead of being silently dropped. `notes` stays last, free text, never truncated.
"""
import csv, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from config import RUNS_CSV  # noqa: E402

RUN_COLUMNS = [
    "run_id", "run_tag", "timestamp_utc", "git_commit", "git_dirty", "code_sha",
    "kernel_ref", "kernel_version", "description", "learner",
    "notebook_runtime_sec", "n_features", "n_folds", "cv_seed", "model_seed", "sample_frac",
    "fold_auc_mean", "fold_auc_std", "final_oof_auc", "public_lb_score", "private_lb_score",
    "submission_msg", "preds_dir", "best_iters", "fold_aucs", "te_cols", "engineered",
    "notes",
]

# Metrics copied straight into same-named columns. List values are space-joined:
# best_iters/fold_aucs exposed an early-stopping confound in S6E9 that was otherwise
# visible only in scrollback.
PASSTHROUGH = ["run_tag", "learner", "notebook_runtime_sec", "n_features", "n_folds",
               "cv_seed", "model_seed", "sample_frac", "fold_auc_mean", "fold_auc_std",
               "final_oof_auc", "best_iters", "fold_aucs", "te_cols", "engineered",
               "code_sha"]


def build_row(metrics, **fixed):
    row = dict(fixed)
    for k in PASSTHROUGH:
        if k in metrics:
            v = metrics[k]
            row[k] = " ".join(str(x) for x in v) if isinstance(v, list) else v
    for k, v in metrics.items():
        if k.endswith("_oof_auc"):
            row[k] = v
    return row


def _header():
    if RUNS_CSV.exists():
        with RUNS_CSV.open(newline="", encoding="utf-8") as f:
            return next(csv.reader(f), []) or list(RUN_COLUMNS)
    return list(RUN_COLUMNS)


def _rewrite(cols, rows):
    with RUNS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def read_rows():
    if not RUNS_CSV.exists():
        return []
    with RUNS_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_run_row(row):
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = _header()
    new = [c for c in row if c not in existing]
    if new:
        cols = [c for c in existing if c != "notes"] + new + ["notes"]
        print(f"  runs.csv schema grew by {new}")
        _rewrite(cols, read_rows())
    else:
        cols = existing
    if any(r.get("run_tag") == row.get("run_tag") and row.get("run_tag")
           for r in read_rows()):
        print(f"  WARNING: run_tag {row.get('run_tag')!r} already exists in runs.csv -- "
              "use a unique tag per run (S6E9 needed de-duplication passes).")
    is_new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if is_new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def update_run(run_id, **fields):
    rows = read_rows()
    hit = [r for r in rows if r["run_id"] == run_id]
    if len(hit) != 1:
        raise SystemExit(f"run_id {run_id!r} matches {len(hit)} rows in runs.csv")
    hit[0].update({k: str(v) for k, v in fields.items()})
    _rewrite(_header(), rows)
