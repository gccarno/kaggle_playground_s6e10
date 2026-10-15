#!/usr/bin/env bash
# usage: scripts/batch24_tree_views.sh TAG [--launch-only|--adopt] -- Phase 4b batch 24, tree-structure views (README, Oct 15).
# Each probe changes only HOW the trees are grown (one *_params key), never the inputs: R1-R3 are strict
# twins of T24 (lgb leg of B11), R4-R5 of U11 (CatBoost leg of B11). Kaggle CPU kernels; no GPU quota.
cd "$(dirname "$0")/.."
T24='"profile_keys":["flight_distance"],"teacher":true,"teacher_learner":"cat"'
U11='"learner":"cat","num_as_cat":["flight_distance","age"],"profile_keys":["flight_distance"],"teacher":true'
G="GATE 1 vs twin +0.00014 (reported, not the decision); GATE 2 on the B11 pool: ADD > 2 sd, SWAP > -1 sd. Batch 24: tree-structure views on CPU while the GPU quota is spent."
case $1 in
R1) B=$T24; tw=T24; k='"lgb_params":{"boosting_type":"rf","num_leaves":1023,"colsample_bytree":0.5,"subsample":0.632,"n_estimators":500}'
    d="R1: T24 as a random forest (lgb rf, 1023 leaves, 500 trees)"
    n="HYPOTHESIS an averaged forest of deep independent trees errs differently from boosting. MECHANISM bagging, not boosting. PREDICTION solo -0.001 to -0.003 vs T24; ADD 0 to +0.00002.";;
R2) B=$T24; tw=T24; k='"lgb_params":{"extra_trees":true}'
    d="R2: T24 with extremely randomized split thresholds (lgb extra_trees)"
    n="HYPOTHESIS random thresholds smooth the step functions on FD/age/delays. MECHANISM extra_trees. PREDICTION solo within +-0.0001 of T24; ADD small, maybe a SWAP for T24.";;
R3) B=$T24; tw=T24; k='"lgb_params":{"linear_tree":true}'
    d="R3: T24 with linear models in the leaves (lgb linear_tree)"
    n="HYPOTHESIS magnitudes (delays, age, profile means) act linearly within a leaf. MECHANISM piecewise-linear trees. PREDICTION solo +-0.0002 vs T24; ADD uncertain.";;
R4) B=$U11; tw=U11; k='"cat_params":{"grow_policy":"Depthwise"}'
    d="R4: U11 with depthwise (non-symmetric) CatBoost trees"
    n="HYPOTHESIS every CatBoost leg uses oblivious trees; asymmetric trees keep the ordered CTRs but split differently. MECHANISM grow_policy Depthwise. PREDICTION solo +-0.0001 vs U11; ADD > 2 sd is the test.";;
R5) B=$U11; tw=U11; k='"cat_params":{"grow_policy":"Lossguide","max_leaves":64,"depth":10}'
    d="R5: U11 with leaf-wise CatBoost trees (Lossguide, 64 leaves)"
    n="HYPOTHESIS leaf-wise growth chases the hard routes deeper than symmetric trees. MECHANISM grow_policy Lossguide. PREDICTION solo +-0.0002 vs U11; ADD > 2 sd is the test.";;
*) echo "unknown tag"; exit 2;;
esac
cfg="{\"run_tag\":\"$1\",$B,$k}"
case $tw in T24) base="{\"run_tag\":\"T24\",$T24}";; U11) base="{\"run_tag\":\"U11\",$U11}";; esac
python scripts/collect_run.py --cfg "$cfg" --diff-vs "$base" --description "$d" --notes "$n $G" "${@:2}" 2>&1 | grep -vE "status: RUNNING|status: QUEUED"
