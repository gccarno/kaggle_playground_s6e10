#!/usr/bin/env bash
# usage: scripts/batch17_mlp.sh TAG [--launch-only|--adopt] -- Phase 3b batch 17, the MLP axis (README, Oct 10).
# Strict twins of W10; T4 GPU kernels.
cd "$(dirname "$0")/.."
FD='"flight_distance"'
R='"wifi","time_convenient","online_booking","gate_location","food_drink","online_boarding","seat_comfort","entertainment","onboard_service","leg_room","baggage","checkin","cleanliness"'
W10B="\"learner\":\"mlp\",\"profile_keys\":[$FD]"
W10="{\"run_tag\":\"W10\",$W10B,\"num_as_cat\":[$FD,\"age\"]}"
G="GATE 1 vs W10 +0.00014; GATE 2 on B6/B7 pool: ADD > 2 sd and SWAP > -1 sd. Kaggle T4 kernel."
case $1 in
W11) cfg="{\"run_tag\":\"W11\",$W10B,\"num_as_cat\":[$FD,\"age\"],\"seed_bag\":3}"
     d="W11: W10 with seed_bag 3"; n="HYPOTHESIS NN fit variance is large; bag inside folds. PREDICTION Gate 1 +0.0002 to +0.0005; ADD above W10's.";;
W12) cfg="{\"run_tag\":\"W12\",$W10B,\"num_as_cat\":[$FD,\"age\",$R]}"
     d="W12: W10 + rating embeddings"; n="HYPOTHESIS rating embeddings (U22 analogue). PREDICTION Gate 1 +-0.0002.";;
W13) cfg="{\"run_tag\":\"W13\",$W10B,\"num_as_cat\":[$FD,\"age\"],\"mlp_params\":{\"dropout\":0.25}}"
     d="W13: W10 with dropout 0.25"; n="HYPOTHESIS val AUC peaks early and decays: more regularisation. PREDICTION Gate 1 +0.0001 to +0.0003.";;
W14) cfg="{\"run_tag\":\"W14\",$W10B,\"num_as_cat\":[$FD,\"age\"],\"teacher\":true}"
     d="W14: W10 + teacher"; n="HYPOTHESIS U1 rule on a new family. PREDICTION solo +0.0003 or more; ADD below W10's.";;
W15) cfg="{\"run_tag\":\"W15\",$W10B,\"num_as_cat\":[$FD,\"age\"],\"te_cols\":[$FD]}"
     d="W15: W10 + FD target encoding"; n="HYPOTHESIS TE route view inside the MLP. PREDICTION Gate 1 +-0.0001.";;
*) echo "unknown tag"; exit 2;;
esac
acc=""; [ "$2" = "--launch-only" ] && acc="--accelerator NvidiaTeslaT4"
python scripts/collect_run.py --cfg "$cfg" --diff-vs "$W10" --description "$d" --notes "$n $G" $acc "${@:2}" 2>&1 | grep -vE "status: RUNNING|status: QUEUED"
