#!/usr/bin/env python3
"""Run one config on a Kaggle CPU kernel end to end: generate the kernel, push it, poll,
pull the output, verify it, archive the artifacts, log a runs.csv row, optionally submit.

    python scripts/push_code.py                     # once per code change (commit first)
    python scripts/collect_run.py --cfg '{"run_tag":"R1","te_cols":["age"]}' \
        --description "R1: ..." --notes "HYPOTHESIS ... GATE ..." [--submit]

Launch it as a harness BACKGROUND task (never shell `&`/disown -- in S6E9 racing pollers
wrote duplicate rows three times). Kaggle allows ~5 concurrent CPU sessions.

Guards, each one a past failure:
  * code: refuses unless Kaggle's code dataset is at git HEAD (push_code.py); the kernel
    itself re-checks the sha it finds against the one it was built for;
  * push: kaggle-cli exits 0 on a REJECTED push -- the output is regex-checked;
  * output: RUN_METRICS_JSON must echo this run's nonce run_id, or nothing is logged
    (a rejected push once let the PREVIOUS run's output be archived under a new id).
"""
import argparse, json, re, shutil, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from config import PREDS_DIR  # noqa: E402
from pipeline import load_cfg  # noqa: E402
import kaggle_io, runlog  # noqa: E402
from make_probe_kernel import make  # noqa: E402
from push_code import head_sha, pushed_sha  # noqa: E402
from run_local import check_strict_twin, read_json_arg  # noqa: E402


def push(push_dir, accelerator=None):
    cmd = [kaggle_io._kaggle_exe(), "kernels", "push", "-p", str(push_dir)]
    if accelerator:
        cmd += ["--accelerator", accelerator]   # only a push flag; e.g. NvidiaTeslaT4
    out = kaggle_io.run(cmd)
    blob = out.stdout + out.stderr
    print(blob.strip())
    if out.returncode != 0 or re.search(r"(?i)\bpush error\b|\berror:", blob):
        raise RuntimeError("kaggle kernels push was REJECTED: " + " ".join(blob.split()))
    m = re.search(r"[Vv]ersion\s+(\d+)", blob)
    return m.group(1) if m else ""


def poll(ref, interval, timeout_min):
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        out = kaggle_io.kaggle("kernels", "status", ref)
        text = (out.stdout + out.stderr).strip()
        m = re.search(r"KernelWorkerStatus\.(\w+)", text)
        status = m.group(1) if m else text
        print(f"  {datetime.now():%H:%M:%S} status: {status}", flush=True)
        if status in ("COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"):
            return status
        time.sleep(interval)
    raise TimeoutError(f"kernel did not finish within {timeout_min} min")


def _stdout_text(raw):
    """A kernel .log is a JSON array of {stream_name, data} entries, not plain text."""
    text = raw.decode("utf-8", errors="replace")
    try:
        entries = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(entries, list):
        return text
    return "".join(e.get("data", "") for e in entries if isinstance(e, dict))


def pull(ref, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    out = kaggle_io.kaggle("kernels", "output", ref, "-p", str(workdir), "--force")
    if out.returncode != 0:
        raise RuntimeError(f"kaggle kernels output failed: {out.stderr}")
    pat = re.compile(r"RUN_METRICS_JSON:(\{.*\})")
    log_text = ""
    for p in workdir.rglob("*"):
        if p.is_file() and p.suffix in (".log", ".txt", ".ipynb", ""):
            t = _stdout_text(p.read_bytes())
            log_text += t
            m = pat.search(t)
            if m:
                return json.loads(m.group(1)), log_text
    return None, log_text


def archive(workdir, run_id):
    dest = PREDS_DIR / run_id
    dest.mkdir(parents=True, exist_ok=True)
    names = ["submission.csv"] + sorted(
        {p.name for g in ("oof_proba_*.csv", "test_proba_*.csv") for p in workdir.rglob(g)})
    copied = []
    for n in names:
        hits = sorted(p for p in workdir.rglob(n) if p.is_file())
        if hits:
            shutil.copy2(hits[0], dest / n)
            copied.append(n)
    print(f"  archived {copied} -> {dest}")
    return dest


def adopt(override):
    """Recover the identity of an already-pushed kernel from the notebook make() wrote.
    The --cfg must equal the cfg baked into that notebook, and the same run_id/code_sha
    echo checks as a fresh run still apply after the pull."""
    from make_probe_kernel import slug_for
    from config import KAGGLE_USER
    nb_path = REPO_ROOT / ".kaggle_output" / f"_push_{slug_for(override['run_tag'])}" / "kernel.ipynb"
    src = "".join(json.loads(nb_path.read_text(encoding="utf-8"))["cells"][0]["source"])
    sha = re.search(r"^EXPECTED_SHA = '([0-9a-f]+)'", src, re.M).group(1)
    run_id = re.search(r"^RUN_ID = '([0-9a-f]+)'", src, re.M).group(1)
    baked = json.loads(eval(re.search(r"^cfg = json.loads\((.*)\)$", src, re.M).group(1)))
    if baked != override:
        raise SystemExit(f"--cfg differs from the kernel's baked cfg: {baked}")
    ref = f"{KAGGLE_USER}/{slug_for(override['run_tag'])}"
    out = kaggle_io.kaggle("kernels", "status", ref)
    m = re.search(r"KernelWorkerStatus\.(\w+)", out.stdout + out.stderr)
    status = m.group(1) if m else (out.stdout + out.stderr).strip()
    print(f"adopting run_id={run_id}  kernel={ref}  code={sha[:10]}  status={status}")
    return sha, run_id, ref, status


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True, help="JSON override (inline or .json path); "
                                                 "must contain a unique run_tag")
    ap.add_argument("--description", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--diff-vs", default=None, help="baseline override; enforces a strict twin")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--accelerator", default=None, choices=["NvidiaTeslaT4", "NvidiaTeslaP100"])
    ap.add_argument("--poll-interval", type=int, default=60)
    ap.add_argument("--timeout-min", type=int, default=240)
    ap.add_argument("--adopt", action="store_true",
                    help="log a kernel that already ran (its poller died): no push or poll; "
                         "run_id, sha and cfg come from the existing _push_<slug>/kernel.ipynb")
    args = ap.parse_args()

    override = read_json_arg(args.cfg)
    cfg = load_cfg(override)
    if not override.get("run_tag"):
        raise SystemExit("--cfg must set a run_tag (it names the kernel)")
    if args.diff_vs is not None:
        check_strict_twin(override, read_json_arg(args.diff_vs))
    if cfg["sample_frac"] < 1.0 and args.submit:
        raise SystemExit("refusing to submit a sample_frac < 1 smoke run")

    if args.adopt:
        sha, run_id, ref, status = adopt(override)
        version = ""
    else:
        sha = head_sha()
        if pushed_sha() != sha:
            raise SystemExit(f"Kaggle code dataset is at {pushed_sha() or '(never pushed)'}, "
                             f"HEAD is {sha}. Run scripts/push_code.py first.")

        run_id = uuid.uuid4().hex[:8]
        push_dir, ref = make(override["run_tag"], override, sha, run_id,
                             gpu=bool(args.accelerator))
        print(f"run_id={run_id}  kernel={ref}  code={sha[:10]}")
        version = push(push_dir, args.accelerator)
        time.sleep(15)       # let the new version register before the first status call
        status = poll(ref, args.poll_interval, args.timeout_min)

    workdir = REPO_ROOT / ".kaggle_output" / f"{run_id}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    metrics, log_text = pull(ref, workdir)
    if status != "COMPLETE" or metrics is None:
        tail = "\n".join(log_text.strip().splitlines()[-30:])
        raise SystemExit(f"kernel {ref} finished {status}; no metrics logged.\n--- log tail ---\n{tail}")
    if metrics.get("run_id") != run_id:
        raise SystemExit(f"STALE OUTPUT: metrics carry run_id {metrics.get('run_id')!r}, "
                         f"expected {run_id!r}. Nothing logged.")
    if metrics.get("code_sha") != sha:
        raise SystemExit(f"code_sha mismatch: ran {metrics.get('code_sha')}, expected {sha}")
    print("metrics:", json.dumps(metrics))

    dest = archive(workdir, run_id)
    (dest / "cfg.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    score, msg = "", ""
    if args.submit:
        msg = kaggle_io.submit(dest / "submission.csv", run_id, args.description)
        score = kaggle_io.poll_score(run_id)
        print(f"Public LB = {score}")

    git_dirty = bool(kaggle_io.run(["git", "status", "--porcelain"]).stdout.strip())
    row = runlog.build_row(
        metrics, run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=sha, git_dirty=git_dirty, kernel_ref=ref, kernel_version=version,
        description=args.description, public_lb_score=score, private_lb_score="",
        submission_msg=msg, preds_dir=dest.relative_to(REPO_ROOT).as_posix(),
        notes=args.notes)
    runlog.append_run_row(row)
    print(f"Appended run {run_id}  OOF {metrics['final_oof_auc']}  LB {score or '-'}")


if __name__ == "__main__":
    main()
