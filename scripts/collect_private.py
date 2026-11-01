"""Collect private leaderboard scores for every submission after the contest closes.

Pages through `kaggle competitions submissions` (JSON), attributes each row to a
run by its "[run_id]" message prefix (the only attribution rule we use), writes
experiments/private_scores.csv (one row per submission), and fills
runs.csv private_lb_score (identical files resubmitted give identical scores;
the script checks that and refuses to write a run whose copies disagree).
"""
import csv

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_io  # noqa: E402
from kaggle_io import COMPETITION, REPO_ROOT  # noqa: E402

OUT = Path(REPO_ROOT) / "experiments" / "private_scores.csv"
RUNS = Path(REPO_ROOT) / "experiments" / "runs.csv"
PREFIX = re.compile(r"^\[([0-9a-z]+)\]")


def fetch_all():
    """All submissions. The CLI (2.2.3) caps a page at 200 and never prints the
    next-page token, so page through the SDK client directly."""
    from kaggle.api.kaggle_api_extended import KaggleApi, SubmissionGroup, SubmissionSortBy
    from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
    api = KaggleApi()
    api.authenticate()
    rows, token = [], ""
    with api.build_kaggle_client() as client:
        while True:
            req = ApiListSubmissionsRequest()
            req.competition_name = COMPETITION
            req.page_token = token
            req.page_size = 200
            req.group = SubmissionGroup.SUBMISSION_GROUP_ALL
            req.sort_by = SubmissionSortBy.SUBMISSION_SORT_BY_DATE
            resp = client.competitions.competition_api_client.list_submissions(req)
            rows += resp.submissions
            token = getattr(resp, "next_page_token", "")
            if not token or not resp.submissions:
                return rows


def main():
    subs = fetch_all()
    recs = []
    for r in subs:
        desc = r.description or ""
        m = PREFIX.match(desc)
        recs.append({
            "ref": r.ref,
            "date": str(r.date),
            "run_id": m.group(1) if m else "",
            "status": str(r.status),
            "public": r.public_score or "",
            "private": r.private_score or "",
            "description": desc,
        })
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"{len(recs)} submissions -> {OUT}")

    by_run = {}
    for r in recs:
        if r["run_id"] and r["private"]:
            by_run.setdefault(r["run_id"], set()).add(r["private"])
    bad = {k: v for k, v in by_run.items() if len(v) > 1}
    if bad:
        print("WARNING: copies disagree, not writing:", bad)

    with RUNS.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, runs = reader.fieldnames, list(reader)
    n = 0
    for row in runs:
        v = by_run.get(row["run_id"])
        if v and len(v) == 1:
            row["private_lb_score"] = next(iter(v))
            n += 1
    with RUNS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(runs)
    print(f"runs.csv: private_lb_score filled for {n} runs")


if __name__ == "__main__":
    main()
