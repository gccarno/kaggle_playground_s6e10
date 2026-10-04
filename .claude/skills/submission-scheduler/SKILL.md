---
name: submission-scheduler
description: Use when banking a run for later submission, when today's Kaggle slots are spent or about to go unused, when queueing or prioritising tonight's submissions, keeping a run from ever being auto-submitted, reading the nightly scheduler log, or changing the cron job on oci-free.
---

# Nightly slot scheduler (oci-free cron)

## Overview
`scripts/submit_scheduler.py` runs from cron on `oci-free` at **23:15 UTC** and spends every slot still free that day:
1. the explicit **queue** (`experiments/submit_queue.txt` on the VM), in order;
2. then **auto-fill**: every other banked full-data run not yet submitted, newest first.

The scheduler **never writes runs.csv.** Every message starts with `[run_id]`, and the laptop's `submit_run.py --fill-missing` attributes the scores. The state (the queue, `submit_log.csv`) lives **on the VM**, so run the commands there.

## Quick reference
```bash
VM='ssh oci-free cd /data/kaggle_playground_s6e10 &&'
$VM .venv/bin/python scripts/submit_scheduler.py list            # queue + tonight's pick order
$VM .venv/bin/python scripts/submit_scheduler.py enqueue T16 T17  # jump the auto-fill order
$VM .venv/bin/python scripts/submit_scheduler.py remove T17
$VM .venv/bin/python scripts/submit_scheduler.py run --dry-run    # what would go now
$VM tail -20 logs/scheduler.log                                   # last nights' summaries
python scripts/submit_run.py --fill-missing                       # laptop, next morning
```

## Making a run visible to the scheduler
The VM can only submit a run whose **row is in the VM's runs.csv** and whose **`preds_dir/submission.csv` exists on the VM**.
- **Ran on the VM:** nothing to do.
- **Ran on the laptop or a Kaggle kernel:** append its row to the VM's runs.csv (merge by run_id; never copy the whole file over), and `scp -r experiments/preds/<run_id> oci-free:/data/kaggle_playground_s6e10/experiments/preds/`. Then `list` should show it.

## Keeping a run out
- Put `NOSUBMIT` in its `--notes` when you log it. Auto-fill skips it; an explicit `enqueue` still sends it.
- Typical candidates: parity replays and anything predicted to be identical to a run that's already submitted.

## Built-in guards
- It refuses after 23:50 UTC, so a late submit can't eat tomorrow's quota.
- It counts manual submits made earlier in the day (`free = 10 − used today`).
- An exclusive flock prevents two concurrent runs.
- It stops at the first submit error, e.g. the limit was hit.
- **Auto-fill stops after 2026-10-29.** After that only the queue goes, because the final days' slots are for hand-picked finalists (playbook §9). It refuses everything after the deadline.

## The cron entry (`crontab -l` on oci-free)
```
15 23 * * * cd /data/kaggle_playground_s6e10 && git pull -q --ff-only; .venv/bin/python scripts/submit_scheduler.py run >> logs/scheduler.log 2>&1
```
The VM's TZ is UTC. Remove the entry after 2026-10-31.

## Common mistakes
- **Running `enqueue` on the laptop.** It writes the *laptop's* queue file, which nobody reads.
- **"N slot(s) unfilled -- bank more runs" in the log.** Fewer eligible runs existed than free slots. Bank seed twins or probes earlier in the day.
