#!/usr/bin/env python3
"""Submit an ALREADY-ARCHIVED run, poll its public score, and write it into runs.csv.

Slots are 10 a day and do not roll over; an unspent slot is a lost paired OOF<->LB point
(playbook section 1). This lets a run be submitted any time after it was trained.

    python scripts/submit_run.py A0                  # by run_tag or run_id
    python scripts/submit_run.py A0 --no-wait        # fill the score in later
    python scripts/submit_run.py --fill-missing      # backfill every pending score
    python scripts/submit_run.py --today             # how many slots are used today (UTC)

Every submission message starts with "[run_id]"; scores are matched ONLY by that prefix.
"""
import argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import kaggle_io, runlog  # noqa: E402

DAILY_LIMIT = 10


def find_run(rows, key):
    hit = [r for r in rows if key in (r["run_id"], r.get("run_tag", ""))]
    if len(hit) != 1:
        raise SystemExit(f"{key!r} matches {len(hit)} runs in runs.csv "
                         f"{[r['run_id'] for r in hit]}")
    return hit[0]


def submitted_ids():
    """run_ids that already carry a "[run_id]" submission on Kaggle."""
    d = (s.get("description", "") for s in kaggle_io.submissions_table())
    return {m[1:m.index("]")] for m in d if m.startswith("[") and "]" in m}


def eligible(row, submitted, root=REPO_ROOT, ignore_nosubmit=False):
    """The one set of submit guards, shared with submit_scheduler.py. Returns (ok, reason).
    NOSUBMIT in a run's notes keeps it out of automatic picks; a named, deliberate
    submission (this script, or an explicit queue entry) passes ignore_nosubmit=True."""
    if float(row.get("sample_frac") or 1.0) < 1.0:
        return False, "sample_frac < 1 smoke run"
    if row["run_id"] in submitted:
        return False, "already submitted"
    if not (root / row.get("preds_dir", "") / "submission.csv").is_file():
        return False, "submission.csv missing"
    if not ignore_nosubmit and "NOSUBMIT" in (row.get("notes") or ""):
        return False, "NOSUBMIT in notes"
    return True, ""


def fill_missing():
    by_tag = {}
    for s in kaggle_io.submissions_table():
        d = s.get("description", "")
        if d.startswith("[") and "]" in d:
            by_tag.setdefault(d[1:d.index("]")], []).append(s)
    n = 0
    for r in runlog.read_rows():
        if (r.get("public_lb_score") or "").strip():
            continue
        subs = by_tag.get(r["run_id"], [])
        if len(subs) > 1:
            print(f"  REFUSING {r['run_id']}: {len(subs)} submissions carry its prefix")
            continue
        if subs and subs[0].get("publicScore") not in (None, "", "None", "pending"):
            runlog.update_run(r["run_id"], public_lb_score=subs[0]["publicScore"],
                              submission_msg=subs[0]["description"])
            print(f"  {r['run_id']} ({r.get('run_tag')}): {subs[0]['publicScore']}")
            n += 1
    print(f"backfilled {n} score(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?")
    ap.add_argument("--message", default=None, help="defaults to the run's description")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--fill-missing", action="store_true")
    ap.add_argument("--today", action="store_true")
    args = ap.parse_args()

    if args.today:
        n = kaggle_io.submissions_today_utc()
        print(f"{n}/{DAILY_LIMIT} submissions used since 00:00 UTC")
        return
    if args.fill_missing:
        return fill_missing()
    if not args.run:
        raise SystemExit("give a run_id/run_tag, or --fill-missing / --today")

    row = find_run(runlog.read_rows(), args.run)
    done = {row["run_id"]} if kaggle_io.find_submissions(row["run_id"]) else set()
    ok, why = eligible(row, done, ignore_nosubmit=True)
    if not ok:
        raise SystemExit(f"refusing to submit {row['run_id']}: {why}")
    sub = REPO_ROOT / row["preds_dir"] / "submission.csv"
    msg = kaggle_io.submit(sub, row["run_id"], args.message or row["description"])
    runlog.update_run(row["run_id"], submission_msg=msg)
    print(f"submitted {row['run_id']} ({row.get('run_tag')}): {msg}")
    if args.no_wait:
        return
    score = kaggle_io.poll_score(row["run_id"])
    if score:
        runlog.update_run(row["run_id"], public_lb_score=score)
        print(f"  public LB = {score}")


if __name__ == "__main__":
    main()
