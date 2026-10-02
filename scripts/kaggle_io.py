"""The ONE place that talks to the Kaggle CLI for submissions.

S6E9 had four copy-pasted submit-then-poll loops that took `rows[0]` of the submissions
list as "ours" -- a race with any concurrent submission -- and a post-deadline backfill that
needed three matching rules plus 8 hand fixes (playbook section 12.7). The fix is a rule,
enforced here: every submission message STARTS WITH "[<run_id>]", and a score is only ever
attributed by that prefix.
"""
import csv, os, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from config import COMPETITION  # noqa: E402


def run(cmd, cwd=REPO_ROOT):
    """Run a CLI command with UTF-8 forced in the child process: kaggle-cli crashes on a
    non-UTF8 Windows console when it prints a character the codepage can't represent."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=cwd, env=env)


def kaggle(*args):
    return run(["kaggle", *args])


def submissions_table():
    out = kaggle("competitions", "submissions", COMPETITION, "--csv")
    return list(csv.DictReader(out.stdout.splitlines()))


def tag(run_id):
    return f"[{run_id}]"


def submit(sub_file, run_id, message):
    """Submit a file. The message always begins with "[run_id]". Returns the message."""
    msg = f"{tag(run_id)} {message}".strip()
    out = kaggle("competitions", "submit", COMPETITION, "-f", str(sub_file), "-m", msg)
    print(out.stdout.strip(), out.stderr.strip(), sep="\n")
    blob = out.stdout + out.stderr
    # Like `kernels push`, the CLI can exit 0 while reporting an error on stdout.
    if out.returncode != 0 or ("error" in blob.lower() and "successfully" not in blob.lower()):
        raise RuntimeError(f"kaggle competitions submit failed: {' '.join(blob.split())}")
    return msg


def find_submissions(run_id):
    return [r for r in submissions_table()
            if r.get("description", "").startswith(tag(run_id))]


def poll_score(run_id, timeout_min=20, poll_interval=20):
    """Wait for THIS run's submission (matched by its run_id prefix) to be scored.
    Refuses to attribute when more than one submission carries the prefix."""
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        hits = find_submissions(run_id)
        if len(hits) > 1:
            print(f"  WARNING: {len(hits)} submissions carry {tag(run_id)}; refusing to "
                  "attribute a score -- resolve by hand.", file=sys.stderr)
            return ""
        if hits:
            s = hits[0].get("publicScore") or ""
            status = hits[0].get("status", "")
            if "ERROR" in status.upper():
                print(f"  submission errored: {hits[0]}", file=sys.stderr)
                return ""
            if s not in ("", "None", "pending"):
                return s
        time.sleep(poll_interval)
    print("  WARNING: timed out waiting for a public score; backfill later with "
          "`python scripts/submit_run.py --fill-missing`", file=sys.stderr)
    return ""


def submissions_today_utc():
    """How many submissions were made since 00:00 UTC (the daily-limit reset)."""
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return sum(1 for r in submissions_table() if r.get("date", "").startswith(today))
