#!/usr/bin/env bash
# Phase 4 batch 11: family legs on T1's representation (TE route view, NO teacher).
# Pre-registered in README Phase 4 (Oct 5). Gate 2 on the B2 pool. Run on oci-free after batch 10.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
keep='strict-twin|OOF AUC =|Appended|Error|error|Traceback|NOT A STRICT'
FD='"flight_distance"'
T1="{\"run_tag\":\"T1\",\"te_cols\":[$FD]}"
GATE="GATE 2 on B2 pool {T8,T14,T15,T1,T24}: ADD > 2 sd and SWAP > -1 sd. Ran on oci-free (aarch64)."
run() {
  echo "=================== $1"
  $PY scripts/run_local.py --cfg "{\"run_tag\":\"$1\",$2}" --diff-vs "$T1" \
      --description "$3" --notes "$4 $GATE" 2>&1 | grep -E "$keep"
}
run V2 "\"te_cols\":[$FD],\"learner\":\"xgb\"" "V2: T1 with xgb (TE route view, no teacher)" \
  "HYPOTHESIS view-2 value lives in what it does not share with view 1 (U1 lesson); an xgb leg on the teacher-free TE view adds. PREDICTION borderline ADD +0.00001..+0.00003."
run V3 "\"te_cols\":[$FD],\"learner\":\"cat\"" "V3: T1 with cat (TE route view, no teacher)" \
  "HYPOTHESIS cat is the strongest family and V3 shares no teacher with view 1. PREDICTION passes Gate 2 (ADD > 2 sd)."
echo "=================== batch done"
