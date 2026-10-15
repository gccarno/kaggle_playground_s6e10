#!/usr/bin/env bash
# usage (on oci-free, usually from a one-shot cron): scripts/vm_launch.sh BATCH_SCRIPT TAG [extra args]
#   e.g. scripts/vm_launch.sh scripts/batch21_tfm.sh Z4 --launch-only
# Brings the VM checkout to origin/main without losing VM-only runs.csv rows (the submission-scheduler
# skill's cleanup recipe), pushes src/ as the code dataset at that HEAD, then launches the kernel.
# Collect later with the same batch script and --adopt, on this VM (the push dir lives here).
set -euo pipefail
cd /data/kaggle_playground_s6e10
export PATH="$PWD/.venv/bin:$PATH"
echo "=== $(date -u +%FT%TZ) vm_launch $*"
cp experiments/runs.csv /tmp/vm_runs.bak
if [ -n "$(git status --porcelain experiments/runs.csv)" ]; then
  git stash push -q experiments/runs.csv
  git pull -q --ff-only
  python scripts/sync_runs.py merge /tmp/vm_runs.bak
  git stash drop -q
else
  git pull -q --ff-only
fi
python scripts/push_code.py
bash "$@"
# Collection cron pattern (hourly; a no-op once the tag is logged, and safe while the kernel runs:
# --adopt logs nothing unless the kernel is COMPLETE):
#   0 12-22 17 10 * cd /data/kaggle_playground_s6e10 && ! grep -q ',Z4,' experiments/runs.csv && PATH=/data/kaggle_playground_s6e10/.venv/bin:$PATH bash scripts/batch21_tfm.sh Z4 --adopt >> logs/z4.log 2>&1
