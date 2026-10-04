---
name: kaggle-cli
description: Use when touching Kaggle from this repo - submitting a run, checking slots left today, backfilling public scores, pushing code or a probe kernel, pulling kernel output, downloading competition data, or when a kaggle CLI command fails, hangs, or "succeeds" suspiciously.
---

# Kaggle CLI (S6E10 harness)

## Overview
All Kaggle traffic goes through the repo scripts, which wrap the CLI in `scripts/kaggle_io.py`. **Never call a bare `kaggle competitions submit` or `kaggle kernels push`.** Every guard in those scripts exists because a bare call once lost or misattributed a run.

## Quick reference

| Goal | Command |
|---|---|
| Slots used today (UTC) | `python scripts/submit_run.py --today` |
| Submit an archived run | `python scripts/submit_run.py <run_tag\|run_id>` (`--no-wait` skips the score poll) |
| Backfill pending public scores into runs.csv | `python scripts/submit_run.py --fill-missing` |
| Bank for tonight's automatic submit | REQUIRED SUB-SKILL: submission-scheduler |
| Upload `src/` to the private code dataset | commit first, then `python scripts/push_code.py` (`--check` reports only) |
| Run a probe on a Kaggle kernel | `python scripts/collect_run.py --cfg '{...}' --description ... --notes ... [--submit]`, as a **harness background task** (`run_in_background`), never shell `&`, at most ~5 at once |
| Run a probe locally or on the VM | `python scripts/run_local.py ...`; see the oci-vm skill for the VM |
| Data | `kaggle competitions download -c playground-series-s6e10 -p data`, then `kaggle datasets download arseniyshutko/binary-aviation-satisfaction-129k -p data/original --unzip` |

For ad-hoc CLI calls (`kernels status <ref>`, `kernels output`, `competitions submissions <slug> --csv`), use `kaggle_io.kaggle(...)` or `kaggle_io.run([...])` from Python. They force UTF-8 in the child process.

## Rules and the failure behind each one

| Rule | Why |
|---|---|
| A submission message **starts with `[run_id]`**, and a score is attributed **only** by that prefix | S6E9 read `rows[0]` of the submissions list, which raced with concurrent submits. Its backfill needed 8 hand fixes. |
| Treat exit code 0 as meaningless; check the output text for `error` | `kernels push` and `submit` both exit 0 when rejected. A rejected push once archived the *previous* run's output. |
| On Windows, call through `kaggle_io.run` (UTF-8 forced) | On a cp1252 console the CLI's progress-bar glyph crashed A0 *after* it had submitted, so the run was never logged. |
| Slots: **10 per day, reset at 00:00 UTC**, and they don't roll over | A submit at 00:05 UTC counts against the *next* day. Check `--today` before planning a batch. |
| Never submit `sample_frac < 1`; never resubmit a run_id | Smoke runs aren't comparable. A duplicate prefix makes `poll_score` refuse to attribute. |
| Commit `src/` before `push_code.py` | It refuses a dirty `src/`. Kernels assert `CODE_SHA` equals the pushed sha. |
| `collect_run.py` refuses output whose `run_id` or `code_sha` differs from what it pushed | That's a stale-output guard. Don't bypass it; re-run instead. |

## Common mistakes
- **"Submitted, but the score never came back."** Run `submit_run.py --fill-missing` later. Don't resubmit.
- **"Is the submission table up to date?"** Kaggle scoring takes a few minutes, and a status of `pending` is normal.
- **Platform parity:** OOF from Kaggle (x86), the laptop and `oci-free` (ARM) agree to within 4.4e-16. Deltas across platforms read directly.
