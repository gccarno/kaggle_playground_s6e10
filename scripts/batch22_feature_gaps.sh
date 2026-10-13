#!/usr/bin/env bash
# usage: scripts/batch22_feature_gaps.sh TAG [--launch-only|--adopt] -- Phase 4 batch 22, feature-engineering gaps (README, Oct 13).
# Every probe is a strict twin of T24 (lgb, FD profile + CatBoost teacher, the lgb leg in B8), so a pass is
# also a direct SWAP candidate for T24 in B8. Kaggle CPU kernels.
cd "$(dirname "$0")/.."
B='"profile_keys":["flight_distance"],"teacher":true,"teacher_learner":"cat"'
base="{\"run_tag\":\"T24\",$B}"
G="GATE 1 vs T24 +0.00014; GATE 2 on the B8 pool: SWAP for T24 > -1 sd, ADD > 2 sd. Batch 22 closes axes left closed only by argument (Checkpoint 3)."
case $1 in
Q1) k='"fe_quant_cols":["flight_distance","age","dep_delay","arr_delay"]'
    d="Q1: T24 + quantized FD/age/delays (divisors 5/25/100)"
    n="HYPOTHESIS coarse value groups add a view trees lack. MECHANISM quantization. PREDICTION null, |d| < 0.00014: trees threshold raw values and the profile already addresses each route.";;
Q2) k='"te_window_cols":["flight_distance"]'
    d="Q2: T24 + window TE of FD (neighbour-route rates, radii 1-50)"
    n="HYPOTHESIS neighbour routes smooth rare/unseen routes. MECHANISM centred-window TE. PREDICTION null to +0.0001: FD residual chi2/df ~1 under T8-class models.";;
Q3) k='"te_bin_cols":["age","dep_delay","arr_delay"]'
    d="Q3: T24 + binned TE of age and delays"
    n="HYPOTHESIS binned label rates of the magnitudes help. MECHANISM binned TE. PREDICTION null: age TE null (T28), delays chi2/df <= 1.01.";;
Q4) k='"fe_delay_diff":true'
    d="Q4: T24 + arr_delay - dep_delay"
    n="HYPOTHESIS time made up in the air matters beyond either delay. MECHANISM a difference is awkward for axis-aligned trees. PREDICTION |d| < 0.0001: delays are weak overall.";;
Q5) k='"one_hot_cols":["gender","customer_type","travel_type","travel_class","wifi","time_convenient","online_booking","gate_location","food_drink","online_boarding","seat_comfort","entertainment","onboard_service","leg_room","baggage","checkin","cleanliness"]'
    d="Q5: T24 with the 4 categoricals and 13 ratings one-hot encoded"
    n="HYPOTHESIS one-hot level indicators beat native/ordinal splits. MECHANISM one-hot. PREDICTION null: low-cardinality columns gain nothing from one-hot in GBDTs.";;
*) echo "unknown tag"; exit 2;;
esac
cfg="{\"run_tag\":\"$1\",$B,$k}"
python scripts/collect_run.py --cfg "$cfg" --diff-vs "$base" --description "$d" --notes "$n $G" "${@:2}" 2>&1 | grep -vE "status: RUNNING|status: QUEUED"
