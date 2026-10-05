#!/usr/bin/env bash
# Phase 4 batch 10: the TE route representation (second view) built out across families.
# Pre-registered in README Phase 4 (Oct 5). Gate 2 on the B2 pool. Run on oci-free in tmux.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
keep='strict-twin|OOF AUC =|Appended|Error|error|Traceback|NOT A STRICT'
FD='"flight_distance"'
T1="{\"run_tag\":\"T1\",\"te_cols\":[$FD]}"
U1="{\"run_tag\":\"U1\",\"te_cols\":[$FD],\"teacher\":true}"
T15="{\"run_tag\":\"T15\",\"learner\":\"cat\",\"profile_keys\":[$FD],\"teacher\":true}"
GATE="GATE 2 on B2 pool {T8,T14,T15,T1,T24}: ADD > 2 sd and SWAP > -1 sd. Ran on oci-free (aarch64)."
run() {  # tag, cfg body, diff-vs, description, notes
  echo "=================== $1"
  $PY scripts/run_local.py --cfg "{\"run_tag\":\"$1\",$2}" --diff-vs "$3" \
      --description "$4" --notes "$5 $GATE" 2>&1 | grep -E "$keep"
}
run U1 "\"te_cols\":[$FD],\"teacher\":true" "$T1" "U1: T1 + teacher (TE route view, lgb leg)" \
  "HYPOTHESIS the teacher interacts with route knowledge on the TE view as it did on the profile view. PREDICTION solo 0.9603-0.9604; in the B2 blend SWAP for T1 > 0."
run U5 "\"te_cols\":[$FD],\"teacher\":true,\"te_pseudo\":true" "$U1" "U5: U1 + te_pseudo (test soft labels in the route TE support)" \
  "HYPOTHESIS pooling test-row soft labels into per-route TE support cuts route-rate variance. PREDICTION solo within +-0.00005 of U1; fails Gate 2 next to U1."
run U2 "\"te_cols\":[$FD],\"teacher\":true,\"learner\":\"xgb\"" "$U1" "U2: U1 with xgb (TE route view, xgb leg)" \
  "HYPOTHESIS a family leg on the second view adds as T14 did on the first. PREDICTION solo >= U1; passes Gate 2."
run U3 "\"te_cols\":[$FD],\"teacher\":true,\"learner\":\"cat\"" "$U1" "U3: U1 with cat (TE route view, cat leg)" \
  "HYPOTHESIS cat was the strongest contributor on view 1; same on view 2. PREDICTION solo >= U1+0.00005; passes Gate 2."
run U4 "\"learner\":\"cat\",\"profile_keys\":[$FD],\"teacher\":true,\"num_as_cat\":[$FD]" "$T15" "U4: T15 + FD as a CatBoost categorical (ordered-TS route view)" \
  "HYPOTHESIS CatBoost ordered target statistics + FD x category CTRs are a third route view. PREDICTION borderline: solo T15 +-0.00005, ADD +0.00002..+0.00005."
echo "=================== batch done"
