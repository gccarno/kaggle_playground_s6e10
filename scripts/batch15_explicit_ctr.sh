#!/usr/bin/env bash
# usage: scripts/batch15_explicit_ctr.sh TAG [--adopt] -- launches one Phase 3b batch-15 kernel via collect_run (pre-registered in README, Oct 10)
cd "$(dirname "$0")/.."
FD='"flight_distance"'
R='"wifi","time_convenient","online_booking","gate_location","food_drink","online_boarding","seat_comfort","entertainment","onboard_service","leg_room","baggage","checkin","cleanliness"'
P5='["age","customer_type"],["age","travel_type"],["age","travel_class"],["wifi","entertainment"],["online_boarding","wifi"]'
P7="$P5,[$FD,\"travel_class\"],[$FD,\"travel_type\"]"
PK='"te_pair_bins":4096,"te_pair_smooth":50'
T1B="\"te_cols\":[$FD],$PK"
G="GATE 1 vs strict twin +0.00014; GATE 2 on B6 pool {T14,U11,T1,T24,U13,U22,U18,U24}: ADD > 2 sd and SWAP > -1 sd. Kaggle kernel."
case $1 in
W1) cfg="{\"run_tag\":\"W1\",$T1B,\"te_pair_cols\":[$P7]}"; dv="{\"run_tag\":\"T1\",$T1B}"
    d="W1: T1 + explicit pair TEs (age x cats, rating pairs, FD x class/type), lgb"; n="HYPOTHESIS lgb gains the combination stats CatBoost builds internally (residual chi2 1.4-2.4 on these pairs under T1). PREDICTION Gate 1 +0.0001 to +0.0003 vs T1; SWAP for T1 passes on B6.";;
W2) cfg="{\"run_tag\":\"W2\",\"learner\":\"xgb\",$T1B,\"te_pair_cols\":[$P7]}"; dv="{\"run_tag\":\"V2\",\"learner\":\"xgb\",$T1B}"
    d="W2: V2 + explicit pair TEs, xgb TE view"; n="HYPOTHESIS the pair-TE gain is family-general. PREDICTION Gate 1 as W1; Gate 2 borderline.";;
W3) cfg="{\"run_tag\":\"W3\",\"learner\":\"xgb\",\"profile_keys\":[$FD],\"teacher\":true,$PK,\"te_pair_cols\":[$P5]}"; dv="{\"run_tag\":\"T14\",\"learner\":\"xgb\",\"profile_keys\":[$FD],\"teacher\":true,$PK}"
    d="W3: T14 + age/rating pair TEs (xgb, profile view)"; n="HYPOTHESIS view-1 xgb gains the age/rating interactions (T14 age x customer_type chi2 2.5). PREDICTION Gate 1 +0.00005 to +0.00015; SWAP for T14 passes.";;
W4) cfg="{\"run_tag\":\"W4\",\"learner\":\"cat\",\"num_as_cat\":[$FD,\"age\",$R]}"; dv="{\"run_tag\":\"U10\",\"learner\":\"cat\",\"num_as_cat\":[$FD]}"
    d="W4: U10 + age and 13 ratings as CatBoost categoricals (pure all-CTR view, no profile/teacher)"; n="HYPOTHESIS the least pool-like view: all CTRs, no shared teacher. PREDICTION Gate 1 +0.0003 to +0.0005 vs U10; ADD passes only if no-teacher errors differ.";;
W5) cfg="{\"run_tag\":\"W5\",\"learner\":\"cat\",\"teacher\":true,\"num_as_cat\":[$FD,\"age\",$R]}"; dv="{\"run_tag\":\"U18\",\"learner\":\"cat\",\"teacher\":true,\"num_as_cat\":[$FD,\"age\"]}"
    d="W5: U18 + the 13 ratings as CatBoost categoricals (CTR-only view)"; n="HYPOTHESIS U22's rating gain carries to the CTR-only view. PREDICTION Gate 1 +0.00005 to +0.00015 vs U18; SWAP for U18 passes.";;
W6) cfg="{\"run_tag\":\"W6\",\"learner\":\"glm\",$T1B,\"te_pair_cols\":[$P7]}"; dv="{\"run_tag\":\"W1\",$T1B,\"te_pair_cols\":[$P7]}"
    d="W6: W1 representation with glm"; n="HYPOTHESIS a linear family on explicit encodings. PREDICTION solo ~0.955; fails Gate 2 unless linear errors are uncorrelated.";;
*) echo "unknown tag"; exit 2;;
esac
python scripts/collect_run.py --cfg "$cfg" --diff-vs "$dv" --description "$d" --notes "$n $G" "${@:2}" 2>&1 | grep -vE "status: RUNNING|status: QUEUED"
