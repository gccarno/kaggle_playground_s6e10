#!/usr/bin/env bash
# Phase 1 calibration batch -- pre-registered in README Phase 1 (commit f454464).
# Every run is a strict twin of A0 and is submitted. Run as ONE harness background task.
set -u
cd "$(dirname "$0")/.."
NOTE="CALIBRATION point (README Phase 1): buys OOF<->LB spread for oof_lb_fit.py; not a shipping candidate, no gate."
run() {  # tag cfg description
  echo "=================== $1"
  python scripts/run_local.py --cfg "$2" --diff-vs '{"run_tag":"A0"}' \
      --description "$3" --notes "$NOTE ${4:-}" --submit 2>&1 \
    | grep -E "strict-twin|^fold|OOF AUC =|Public LB|Appended|Error|error|Traceback|NOT A STRICT"
}
RATINGS='["wifi","time_convenient","online_booking","gate_location","food_drink","online_boarding","seat_comfort","entertainment","onboard_service","leg_room","baggage","checkin","cleanliness"]'
run A0s  '{"run_tag":"A0s","model_seed":1337}'                    "A0s: A0 at model_seed=1337 (seed-noise floor)" "PURPOSE: seed-noise floor = |OOF(A0s)-OOF(A0)|."
run A0b3 '{"run_tag":"A0b3","seed_bag":3}'                        "A0b3: A0 with 3-seed bag per fold"
run L7   '{"run_tag":"L7","lgb_params":{"num_leaves":7}}'         "L7: A0 with num_leaves=7"
run L255 '{"run_tag":"L255","lgb_params":{"num_leaves":255}}'     "L255: A0 with num_leaves=255"
run D1   '{"run_tag":"D1","drop_cols":["online_boarding","wifi"]}' "D1: A0 minus online_boarding, wifi (degraded)"
run D2   "{\"run_tag\":\"D2\",\"drop_cols\":$RATINGS}"            "D2: A0 minus all 13 ratings (degraded)"
run G1   '{"run_tag":"G1","learner":"glm"}'                       "G1: logistic regression on raw features (non-GBDT family check)"
run X1   '{"run_tag":"X1","learner":"xgb"}'                       "X1: XGBoost on raw features (family point)"
run K1   '{"run_tag":"K1","learner":"cat"}'                       "K1: CatBoost on raw features (family point)"
echo "=================== batch done"
python scripts/submit_run.py --today
