#!/usr/bin/env bash
# Phase 4 batch 12: FD as a native categorical across families, and CatBoost CTRs as their own
# route view. Pre-registered in README Phase 4 (Oct 8). Banked for the nightly scheduler.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
keep='strict-twin|OOF AUC =|Appended|Error|error|Traceback|NOT A STRICT'
FD='"flight_distance"'
BASE="\"profile_keys\":[$FD],\"teacher\":true"
T8="{\"run_tag\":\"T8\",$BASE}"
T14="{\"run_tag\":\"T14\",\"learner\":\"xgb\",$BASE}"
U4="{\"run_tag\":\"U4\",\"learner\":\"cat\",$BASE,\"num_as_cat\":[$FD]}"
GATE="GATE 1 vs strict twin +0.00014; GATE 2 on B3 pool {T8,T14,U4,T1,T24}: ADD > 2 sd and SWAP > -1 sd. Ran on oci-free (aarch64)."
run() {  # tag, cfg body, diff-vs, description, notes
  echo "=================== $1"
  $PY scripts/run_local.py --cfg "{\"run_tag\":\"$1\",$2}" --diff-vs "$3" \
      --description "$4" --notes "$5 $GATE" 2>&1 | grep -E "$keep"
}
run U7 "\"learner\":\"lgb\",$BASE,\"num_as_cat\":[$FD]" "$T8" "U7: T8 + FD as an lgb categorical" \
  "HYPOTHESIS FD identity as a gradient-grouped categorical. PREDICTION Gate 1 null to negative (T29: high-cardinality lgb categoricals overfit); fails Gate 2."
run U6 "\"learner\":\"xgb\",$BASE,\"num_as_cat\":[$FD]" "$T14" "U6: T14 + FD as an xgb categorical" \
  "HYPOTHESIS xgb partition splits on FD identity. PREDICTION Gate 1 null (no CTRs; identity already in profile); fails Gate 2."
run U9 "\"learner\":\"cat\",\"teacher\":true,\"num_as_cat\":[$FD]" "$U4" "U9: U4 without the profile (CatBoost CTRs as the only route view)" \
  "HYPOTHESIS CatBoost ordered-TS + combination CTRs are a third route view. PREDICTION solo ~0.9605 (< U4); Gate 2 borderline, passes only if CTR route errors differ from profile route errors."
run U4s "\"learner\":\"cat\",$BASE,\"num_as_cat\":[$FD],\"model_seed\":7" "$U4" "U4s: U4 seed twin (model_seed 7)" \
  "HYPOTHESIS cat seed floor at the best single. PREDICTION |delta| < 0.00005 vs U4; fails Gate 2."
run U4s2 "\"learner\":\"cat\",$BASE,\"num_as_cat\":[$FD],\"model_seed\":1337" "$U4" "U4s2: U4 seed twin (model_seed 1337)" \
  "HYPOTHESIS cat seed floor at the best single. PREDICTION |delta| < 0.00005 vs U4; fails Gate 2."
echo "=================== batch done"
