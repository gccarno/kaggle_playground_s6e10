#!/usr/bin/env bash
# Batch 5: plan Phase 2 probe (c), the original dataset as extra TE support. Pre-registered
# in README Phase 2 (Oct 4). Banked; the nightly slot scheduler submits T19 (T1r is NOSUBMIT).
# Run on oci-free in tmux, one fit at a time.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T1='{"run_tag":"T1","te_cols":["flight_distance"]}'
echo "=================== T1r (refactor parity: must be bit-identical to T1 0a155047)"
$PY scripts/run_local.py --cfg '{"run_tag":"T1r","te_cols":["flight_distance"]}' --diff-vs "$T1" \
    --description "T1r: T1 replayed after the te_origin_support refactor (parity)" \
    --notes "PARITY: archived T1 cfg on code with te_origin_support (default off). Max |dp| vs T1 must be 0. NOSUBMIT (public would repeat T1)." 2>&1 \
  | grep -E "strict-twin|OOF AUC =|Appended|Error|error|Traceback|NOT A STRICT"
echo "=================== T19"
$PY scripts/run_local.py --cfg '{"run_tag":"T19","te_cols":["flight_distance"],"te_origin_support":true}' --diff-vs "$T1" \
    --description "T19: T1 + original rows pooled into the FD target encoding" \
    --notes "HYPOTHESIS the original's per-route label rate adds support to FD TE (plan Phase 2 probe c). MECHANISM 130k extra (FD,label) pairs shrink per-route noise, but the original's route rates differ from the competition's (3.8% flips, generator drift) so they add bias. PREDICTION null: |d| < 0.00014 (per-key origin label was +0.000014 in Phase 1; T7 showed FD TE is already saturated next to profiles). GATE +0.00014 OOF vs T1 0a155047 (0.960139). Ran on oci-free (aarch64)." 2>&1 \
  | grep -E "strict-twin|OOF AUC =|Appended|Error|error|Traceback|NOT A STRICT"
echo "=================== batch done"
