#!/usr/bin/env bash
# usage: scripts/batch18_mlp_bags.sh TAG [--launch-only|--adopt] -- Phase 3c batch 18, MLP read through bags (README, Oct 10).
# X1 is a strict twin of W11 (bag size only); X3-X6 are strict twins of X2 (W10 at seed_bag 5). T4 GPU kernels.
cd "$(dirname "$0")/.."
FD='"flight_distance"'
R='"wifi","time_convenient","online_booking","gate_location","food_drink","online_boarding","seat_comfort","entertainment","onboard_service","leg_room","baggage","checkin","cleanliness"'
M='"learner":"mlp"'
P="\"profile_keys\":[$FD]"
NC="\"num_as_cat\":[$FD,\"age\"]"
W11="{\"run_tag\":\"W11\",$M,$P,$NC,\"seed_bag\":3}"
X2="{\"run_tag\":\"X2\",$M,$P,$NC,\"seed_bag\":5}"
G="GATE 1 vs twin +0.00014 (bagged, so readable); GATE 2 on the B7 pool: ADD > 2 sd and SWAP > -1 sd (SWAP vs the MLP leg it would replace). Kaggle T4 kernel."
base=$X2
case $1 in
X1) cfg="{\"run_tag\":\"X1\",$M,$P,$NC,\"seed_bag\":10}"; base=$W11
    d="X1: W11 with seed_bag 10"; n="HYPOTHESIS the 1->3 bag gain (+0.0012) is fit variance; it shrinks like 1/n. PREDICTION Gate 1 +0.0001 to +0.0004 vs W11; SWAP for W11 passes.";;
X2) cfg=$X2; base=$W11
    d="X2: W10 with seed_bag 5 (base for X3-X6)"; n="HYPOTHESIS bag curve midpoint and the base for bagged variants. PREDICTION between W11 and X1.";;
X3) cfg="{\"run_tag\":\"X3\",$M,$P,\"num_as_cat\":[$FD,\"age\",$R],\"seed_bag\":5}"
    d="X3: X2 + rating embeddings"; n="HYPOTHESIS W12 re-read through a bag: rating embeddings, the U22 analogue. PREDICTION Gate 1 +-0.0002.";;
X4) cfg="{\"run_tag\":\"X4\",$M,$NC,\"seed_bag\":5}"
    d="X4: X2 without the FD profile (bagged W9)"; n="HYPOTHESIS the raw-column MLP is the second MLP view in B7 (W9); bagged it should SWAP for W9. PREDICTION solo -0.0002 vs X2; SWAP for W9 > 0.";;
X5) cfg="{\"run_tag\":\"X5\",$M,$P,$NC,\"seed_bag\":5,\"teacher\":true}"
    d="X5: X2 + teacher"; n="HYPOTHESIS W14 re-read through a bag; U1 rule: shared teacher raises solo, cuts ADD. PREDICTION solo +0.0001 to +0.0003; ADD below X2's.";;
X6) cfg="{\"run_tag\":\"X6\",$M,$P,$NC,\"seed_bag\":5,\"mlp_params\":{\"epochs\":10}}"
    d="X6: X2 with a 10-epoch OneCycle"; n="HYPOTHESIS best epoch is 8-9 of 16, so val AUC decays while lr anneals; a shorter cycle finishes annealing near the peak. PREDICTION Gate 1 +0.0001 to +0.0003; half the runtime.";;
*) echo "unknown tag"; exit 2;;
esac
acc=""; [ "$2" = "--launch-only" ] && acc="--accelerator NvidiaTeslaT4"
python scripts/collect_run.py --cfg "$cfg" --diff-vs "$base" --description "$d" --notes "$n $G" $acc "${@:2}" 2>&1 | grep -vE "status: RUNNING|status: QUEUED"
