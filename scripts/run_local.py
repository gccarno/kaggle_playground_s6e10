#!/usr/bin/env python3
"""Run src/pipeline.py LOCALLY under a config override, archive its artifacts, optionally
submit, and append a row to experiments/runs.csv (kernel_ref="local").

    python scripts/run_local.py --cfg '{"run_tag":"A0"}' --description "A0 baseline" \
        --notes "HYPOTHESIS ... MECHANISM ... GATE ..." [--submit]

    # strict twin: refuse to run unless EXACTLY ONE config key differs from the baseline
    python scripts/run_local.py --cfg '{"run_tag":"B1","freq_cols":["age"]}' \
        --diff-vs '{"run_tag":"A0"}' --description "..." --notes "..."

--cfg / --diff-vs accept inline JSON or a path to a .json file.
"""
import argparse, json, os, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from config import PREDS_DIR  # noqa: E402
from pipeline import DEFAULTS, load_cfg  # noqa: E402  (import is side-effect free)
import kaggle_io, runlog  # noqa: E402

PIPELINE = REPO_ROOT / "src" / "pipeline.py"


def read_json_arg(s):
    p = Path(s)
    return json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" and p.exists() \
        else json.loads(s)


def git(*args):
    out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT)
    return out.stdout.strip() if out.returncode == 0 else ""


def check_strict_twin(probe, baseline):
    """Playbook section 3: a probe changes exactly ONE thing, or its delta cannot be
    attributed. Compares fully-resolved configs (param dicts merged) over the key UNION."""
    a, b = load_cfg(baseline), load_cfg(probe)
    diff = [k for k in sorted(set(a) | set(b)) if k != "run_tag" and a.get(k) != b.get(k)]
    if len(diff) != 1:
        raise SystemExit(
            f"NOT A STRICT TWIN: {len(diff)} keys differ: {diff}\n"
            f"  baseline: { {k: a.get(k) for k in diff} }\n"
            f"  probe   : { {k: b.get(k) for k in diff} }\n"
            "Change exactly one thing, or drop --diff-vs if this is deliberate.")
    print(f"strict-twin OK: only {diff[0]!r} differs ({a[diff[0]]!r} -> {b[diff[0]]!r})")


def execute_pipeline(override, out_dir, run_id):
    """A subprocess, not an import: a clean namespace per run, and live streamed output."""
    env = dict(os.environ, KCFG=json.dumps(override), KOUT=str(out_dir), KRUN_ID=run_id,
               PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "-u", str(PIPELINE)], cwd=REPO_ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    captured = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured.append(line)
    if proc.wait() != 0:
        raise RuntimeError(f"pipeline failed with exit code {proc.returncode}")
    marker = "RUN_METRICS_JSON:"
    line = next((l for l in captured if l.startswith(marker)), None)
    if line is None:
        raise RuntimeError("RUN_METRICS_JSON not found in pipeline output")
    return json.loads(line[len(marker):])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default="{}", help="JSON override (inline or .json path)")
    ap.add_argument("--description", required=True)
    ap.add_argument("--notes", default="",
                    help="HYPOTHESIS / MECHANISM / GATE -- written BEFORE the result")
    ap.add_argument("--diff-vs", default=None, help="baseline override; enforces a strict twin")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="do not append to runs.csv")
    args = ap.parse_args()

    override = read_json_arg(args.cfg)
    cfg = load_cfg(override)                       # validates keys before any work
    if args.diff_vs is not None:
        check_strict_twin(override, read_json_arg(args.diff_vs))
    if cfg["sample_frac"] < 1.0 and args.submit:
        raise SystemExit("refusing to submit a sample_frac < 1 smoke run")

    run_id = uuid.uuid4().hex[:8]
    out_dir = PREDS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cfg.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"run_id={run_id}  override={json.dumps(override)}\n")

    metrics = execute_pipeline(override, out_dir, run_id)
    print(f"\n{'=' * 70}\nOOF AUC = {metrics['final_oof_auc']}   "
          f"({metrics['n_features']} features)")

    score, msg = "", ""
    if args.submit:
        msg = kaggle_io.submit(out_dir / "submission.csv", run_id, args.description)
        score = kaggle_io.poll_score(run_id)
        print(f"Public LB = {score}")

    if not args.no_log:
        row = runlog.build_row(
            metrics, run_id=run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
            kernel_ref="local", kernel_version="", description=args.description,
            public_lb_score=score, private_lb_score="", submission_msg=msg,
            preds_dir=out_dir.relative_to(REPO_ROOT).as_posix(), notes=args.notes)
        runlog.append_run_row(row)
        print(f"Appended run {run_id} to experiments/runs.csv")


if __name__ == "__main__":
    main()
