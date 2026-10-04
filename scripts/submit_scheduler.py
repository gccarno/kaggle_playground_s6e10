#!/usr/bin/env python3
"""Spend each day's leftover submission slots before the 00:00 UTC reset.

Playbook section 1: slots don't roll over, and an unspent slot is a lost paired OOF<->LB
point. S6E8 used 18 of ~290. This runs from cron on the always-on VM (oci-free) at about
23:15 UTC and submits, in order:
  1. the explicit queue (experiments/submit_queue.txt, one run_id per line);
  2. auto-fill: every other banked full-data run that hasn't been submitted, newest first.
     Fresh pre-registered probes are the points the current phase is reading.
     NOSUBMIT in a run's --notes keeps it out of auto-fill.

    python scripts/submit_scheduler.py enqueue T11 T14   # by run_tag or run_id
    python scripts/submit_scheduler.py list              # queue + what tonight would pick
    python scripts/submit_scheduler.py remove T14
    python scripts/submit_scheduler.py run [--dry-run] [--max N] [--no-autofill]

It never writes runs.csv. The VM's and the laptop's copies drift, so the Kaggle submissions
table is the record: every message starts with "[run_id]", and
`submit_run.py --fill-missing` on the laptop attributes the scores by that prefix. Its own
trail is experiments/submit_log.csv, which is VM-local and gitignored.
"""
import argparse, csv, sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from config import DEADLINE_UTC  # noqa: E402
import kaggle_io, runlog  # noqa: E402
from submit_run import DAILY_LIMIT, eligible, find_run, submitted_ids  # noqa: E402

QUEUE = REPO_ROOT / "experiments" / "submit_queue.txt"
LOG = REPO_ROOT / "experiments" / "submit_log.csv"
LOCK = REPO_ROOT / "experiments" / ".submit_scheduler.lock"
# Past this cutoff a submit can land after 00:00 UTC and eat tomorrow's slot.
CUTOFF_HHMM = (23, 50)
# The last days' slots belong to hand-picked finalists (playbook section 9), so auto-fill
# stops here; the explicit queue still runs until the deadline.
AUTOFILL_UNTIL = "2026-10-29"


def pick(queue, rows, submitted, free, root=REPO_ROOT, autofill=True):
    """The run_ids to submit tonight, in order: queue entries first (an explicit choice, so
    NOSUBMIT doesn't apply), then eligible banked runs newest first. Capped at `free`."""
    by_id = {r["run_id"]: r for r in rows}
    out = []
    for rid in queue:
        r = by_id.get(rid)
        if r and rid not in out and eligible(r, submitted, root, ignore_nosubmit=True)[0]:
            out.append(rid)
    if autofill:
        for r in sorted(rows, key=lambda r: r.get("timestamp_utc", ""), reverse=True):
            if r["run_id"] not in out and eligible(r, submitted, root)[0]:
                out.append(r["run_id"])
    return out[:max(free, 0)]


def read_queue():
    if not QUEUE.exists():
        return []
    return [l.strip() for l in QUEUE.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_queue(ids):
    QUEUE.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")


def log(run_id, run_tag, status, detail):
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp_utc", "run_id", "run_tag", "status", "detail"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    run_id, run_tag, status, detail])


def lock():
    """An exclusive lock, so a manual `run` and the cron one can't both spend slots.
    fcntl only exists on POSIX. The laptop only uses list/enqueue/--dry-run."""
    try:
        import fcntl
    except ImportError:
        return None
    fh = LOCK.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another submit_scheduler run holds the lock")
    return fh


def cmd_enqueue(keys):
    rows, q, done = runlog.read_rows(), read_queue(), submitted_ids()
    for k in keys:
        r = find_run(rows, k)
        ok, why = eligible(r, done, ignore_nosubmit=True)
        if not ok:
            print(f"  skip {k} ({r['run_id']}): {why}")
        elif r["run_id"] in q:
            print(f"  {k} ({r['run_id']}) already queued")
        else:
            q.append(r["run_id"])
            print(f"  queued {k} ({r['run_id']}) at position {len(q)}")
    write_queue(q)


def cmd_remove(keys):
    rows, q = runlog.read_rows(), read_queue()
    for k in keys:
        rid = find_run(rows, k)["run_id"]
        print(f"  removed {k} ({rid})" if rid in q else f"  {k} ({rid}) was not queued")
        q = [i for i in q if i != rid]
    write_queue(q)


def cmd_list():
    rows, q, done = runlog.read_rows(), read_queue(), submitted_ids()
    by_id = {r["run_id"]: r for r in rows}
    print(f"queue ({len(q)}):")
    for rid in q:
        r = by_id.get(rid)
        why = eligible(r, done, ignore_nosubmit=True)[1] if r else "not in runs.csv"
        print(f"  {rid} {r.get('run_tag', '') if r else ''} {'-- ' + why if why else ''}")
    free = DAILY_LIMIT - kaggle_io.submissions_today_utc()
    picks = pick(q, rows, done, DAILY_LIMIT, autofill=autofill_on())
    print(f"\n{free}/{DAILY_LIMIT} slots free today; run order "
          f"(first {free} would go tonight):")
    for i, rid in enumerate(picks, 1):
        r = by_id[rid]
        print(f"  {i:2d}. {rid} {r.get('run_tag', ''):6s} OOF {r.get('final_oof_auc', '')}"
              f"  {'queue' if rid in q else 'auto'}  {r.get('description', '')[:60]}")


def autofill_on(now=None):
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d") <= AUTOFILL_UNTIL


def cmd_run(dry_run, max_n, no_autofill):
    fh = lock() if not dry_run else None
    now = datetime.now(timezone.utc)
    if now >= datetime.fromisoformat(DEADLINE_UTC):
        raise SystemExit("contest deadline has passed -- remove the cron entry")
    if (now.hour, now.minute) >= CUTOFF_HHMM and not dry_run:
        raise SystemExit(f"{now:%H:%M} UTC is past the {CUTOFF_HHMM[0]}:{CUTOFF_HHMM[1]} "
                         "cutoff; a submit could land on tomorrow's quota")
    used = kaggle_io.submissions_today_utc()
    free = DAILY_LIMIT - used
    if max_n is not None:
        free = min(free, max_n)
    rows, q, done = runlog.read_rows(), read_queue(), submitted_ids()
    picks = pick(q, rows, done, free, autofill=autofill_on(now) and not no_autofill)
    by_id = {r["run_id"]: r for r in rows}
    sent = []
    for rid in picks:
        r = by_id[rid]
        if dry_run:
            print(f"  [dry-run] would submit {rid} ({r.get('run_tag')}): {r.get('description')}")
            continue
        try:
            msg = kaggle_io.submit(REPO_ROOT / r["preds_dir"] / "submission.csv", rid,
                                   r.get("description", ""))
        except RuntimeError as e:
            # Most likely the daily limit (a manual submit raced us). Stop, don't retry.
            log(rid, r.get("run_tag", ""), "error", str(e))
            print(f"  submit {rid} FAILED, stopping: {e}")
            break
        log(rid, r.get("run_tag", ""), "submitted", msg)
        sent.append(rid)
    # Drop queue entries that are now spent or can never go (already submitted elsewhere).
    if not dry_run:
        write_queue([i for i in q if i not in done and i not in sent])
    wasted = max(free - len(picks), 0)
    tags = [by_id[i].get("run_tag") or i for i in (picks if dry_run else sent)]
    print(f"SCHEDULER {now:%Y-%m-%d %H:%MZ}{' DRY-RUN' if dry_run else ''}: "
          f"used {used} before, {'would submit' if dry_run else 'submitted'} "
          f"{len(tags)} {tags}, {wasted} slot(s) unfilled"
          + (" -- bank more runs" if wasted else ""))
    if fh:
        fh.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enqueue").add_argument("runs", nargs="+")
    sub.add_parser("remove").add_argument("runs", nargs="+")
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--max", type=int, default=None)
    r.add_argument("--no-autofill", action="store_true")
    a = ap.parse_args()
    if a.cmd == "enqueue":
        cmd_enqueue(a.runs)
    elif a.cmd == "remove":
        cmd_remove(a.runs)
    elif a.cmd == "list":
        cmd_list()
    else:
        cmd_run(a.dry_run, a.max, a.no_autofill)


if __name__ == "__main__":
    main()
