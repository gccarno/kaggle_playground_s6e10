#!/usr/bin/env bash
# Phase 4 batch 13 (oci-free half): the CatBoost CTR route view (U9) probed like the TE view was.
# Pre-registered in README Phase 4 (Oct 8). Banked for the nightly scheduler.
# U15/U16/U17 (best-leg capacity/lr/seed-bag) run as Kaggle kernels.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
keep='strict-twin|OOF AUC =|Appended|Error|error|Traceback|NOT A STRICT'
FD='"flight_distance"'
BASE="\"profile_keys\":[$FD],\"teacher\":true"
U4="{\"run_tag\":\"U4\",\"learner\":\"cat\",$BASE,\"num_as_cat\":[$FD]}"
U9="{\"run_tag\":\"U9\",\"learner\":\"cat\",\"teacher\":true,\"num_as_cat\":[$FD]}"
GATE="GATE 1 vs strict twin +0.00014; GATE 2 on B4 pool {T14,U4,U9,T1,T24}: ADD > 2 sd and SWAP > -1 sd. Ran on oci-free (aarch64)."
run() {  # tag, cfg body, diff-vs, description, notes
  echo "=================== $1"
  $PY scripts/run_local.py --cfg "{\"run_tag\":\"$1\",$2}" --diff-vs "$3" \
      --description "$4" --notes "$5 $GATE" 2>&1 | grep -E "$keep"
}
run U10 "\"learner\":\"cat\",\"num_as_cat\":[$FD]" "$U9" "U10: U9 without the teacher (pure CatBoost CTR route view)" \
  "HYPOTHESIS the teacher is the component U9 shares with the pool (U1 lesson). PREDICTION solo ~0.0002 below U9; Gate 2 ADD >= U9's if the teacher was the shared part."
run U13 "\"learner\":\"cat\",\"teacher\":true,\"num_as_cat\":[$FD],\"teacher_learner\":\"cat\"" "$U9" "U13: U9 with a CatBoost teacher" \
  "HYPOTHESIS a different teacher function on the CTR view (T24 analogue). PREDICTION solo +-0.00005 vs U9; fails Gate 2 next to U9."
run U14 "\"learner\":\"cat\",\"teacher\":true,\"num_as_cat\":[$FD],\"model_seed\":7" "$U9" "U14: U9 seed twin (model_seed 7)" \
  "HYPOTHESIS seed floor at the CTR leg. PREDICTION |delta| < 0.00005 vs U9."
run U11 "\"learner\":\"cat\",$BASE,\"num_as_cat\":[$FD,\"age\"]" "$U4" "U11: U4 + age as a CatBoost categorical" \
  "HYPOTHESIS age CTRs inside CatBoost (T29 failed only in lgb). PREDICTION Gate 1 null (age residual chi2 0.52)."
echo "=================== batch done"
