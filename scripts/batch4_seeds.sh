#!/usr/bin/env bash
# Batch 4: T8 seed twins T16-T18, pre-registered in README Phase 2 (Oct 4). BANKED, not
# submitted here: the nightly slot scheduler (submit_scheduler.py) sends them at 23:15 UTC.
# Run on oci-free in tmux, one fit at a time (2 cores).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BASE='{"run_tag":"T8","profile_keys":["flight_distance"],"teacher":true}'
NOTE="SEED TWIN of champion T8 5ee7f1d3 (0.960590): fills unused slots (README s7) and adds seed-floor points at the champion. PREDICTION |d| < 0.00006 (T10 gave 0.000055). No gate: not a candidate. Ran on oci-free (aarch64)."
run() {  # tag seed
  echo "=================== $1"
  $PY scripts/run_local.py \
      --cfg "{\"run_tag\":\"$1\",\"profile_keys\":[\"flight_distance\"],\"teacher\":true,\"model_seed\":$2}" \
      --diff-vs "$BASE" --description "$1: T8 seed twin (model_seed $2)" --notes "$NOTE" 2>&1 \
    | grep -E "strict-twin|OOF AUC =|Appended|Error|error|Traceback|NOT A STRICT"
}
run T16 7
run T17 2024
run T18 31337
echo "=================== batch done"
$PY scripts/submit_scheduler.py list
