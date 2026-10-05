#!/usr/bin/env bash
# Phase 3 batch 7: teacher variants on champion T8, plus xgb/cat seed twins for Phase 4.
# Pre-registered in README Phase 3 (Oct 5). Banked for the nightly slot scheduler
# (T8r2 is NOSUBMIT). Run on oci-free in tmux, one fit at a time.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
keep='strict-twin|OOF AUC =|teacher:|Appended|Error|error|Traceback|NOT A STRICT'
echo "=================== smoke (5%, 2 folds): both new teacher axes must run"
KCFG='{"sample_frac":0.05,"n_folds":2,"profile_keys":["flight_distance"],"teacher":true,"teacher_profile":true,"teacher_learner":"cat"}' \
  KOUT=/tmp/smoke7 $PY src/pipeline.py 2>&1 | grep -E "$keep" || { echo SMOKE FAILED; exit 1; }
T8='{"run_tag":"T8","profile_keys":["flight_distance"],"teacher":true}'
GATE="GATE +0.00014 OOF vs T8 5ee7f1d3 (0.960590). Ran on oci-free (aarch64)."
echo "=================== T8r2 (parity: must be bit-identical to T8 5ee7f1d3)"
$PY scripts/run_local.py --cfg '{"run_tag":"T8r2","profile_keys":["flight_distance"],"teacher":true}' \
    --description "T8r2: T8 replayed after the teacher_profile/teacher_learner axes (parity)" \
    --notes "PARITY: archived T8 cfg on code with teacher_profile/teacher_learner (default off). Max |dp| vs T8 must be 0. NOSUBMIT (public would repeat T8)." 2>&1 | grep -E "$keep"
run() {  # tag cfg-extra description notes [diff-vs]
  echo "=================== $1"
  $PY scripts/run_local.py \
      --cfg "{\"run_tag\":\"$1\",\"profile_keys\":[\"flight_distance\"],\"teacher\":true,$2}" \
      --diff-vs "${5:-$T8}" --description "$3" --notes "$4" 2>&1 | grep -E "$keep"
}
run T23 '"teacher_profile":true' "T23: T8 with a route-aware teacher (teacher also sees the FD profile)" \
  "HYPOTHESIS the teacher's value is an interaction with route (T3 +0.00010 raw vs T8 +0.00021 next to the profile); a teacher that sees the route profile learns the original's function per route. MECHANISM the original's decision boundary differs by route character, which a raw-column teacher must infer from FD alone. PREDICTION borderline, +0.0001 to +0.0003. $GATE"
run T24 '"teacher_learner":"cat"' "T24: T8 with a CatBoost teacher" \
  "HYPOTHESIS a different learner family for the teacher changes little. MECHANISM the original's function is learned almost perfectly (original CV AUC 0.995, flat from 250 to 5000 rounds), so a second learner reproduces the same function. PREDICTION below the gate. $GATE"
run T25 '"teacher_params":{"n_estimators":500}' "T25: T8 with a 500-round teacher (best original-CV setting)" \
  "HYPOTHESIS the round count is irrelevant. MECHANISM original CV AUC 0.99519 at 500 vs 0.99505 at 1500 (diag_teacher_rounds.py), a 0.00014 gap on a 0.995 model. PREDICTION below the gate; closes the rounds sub-axis. $GATE"
X14='{"run_tag":"T14","profile_keys":["flight_distance"],"teacher":true,"learner":"xgb"}'
C15='{"run_tag":"T15","profile_keys":["flight_distance"],"teacher":true,"learner":"cat"}'
run T26 '"learner":"xgb","model_seed":7' "T26: T14 seed twin (xgb leg, model_seed 7)" \
  "SEED TWIN of the xgb leg T14 4cde76c7 (0.960678): xgb seed floor at the champion representation, and a second xgb leg for Phase 4 pools. PREDICTION |d| < 0.00006. No gate: not a candidate. Ran on oci-free (aarch64)." "$X14"
run T27 '"learner":"cat","model_seed":7' "T27: T15 seed twin (cat leg, model_seed 7)" \
  "SEED TWIN of the cat leg T15 636833f2 (0.960683): cat seed floor at the champion representation, and a second cat leg for Phase 4 pools. PREDICTION |d| < 0.00006. No gate: not a candidate. Ran on oci-free (aarch64)." "$C15"
echo "=================== batch done"
$PY scripts/submit_scheduler.py list
