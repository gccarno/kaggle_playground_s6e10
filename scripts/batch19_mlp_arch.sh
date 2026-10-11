#!/usr/bin/env bash
# usage: scripts/batch19_mlp_arch.sh TAG [--launch-only|--adopt] -- Phase 3c batch 19, MLP architecture (README, Oct 11).
# Y1-Y3, Y5 are strict twins of X3; Y4 of X4. All 5-seed bags. T4 GPU kernels.
cd "$(dirname "$0")/.."
FD='"flight_distance"'
R='"wifi","time_convenient","online_booking","gate_location","food_drink","online_boarding","seat_comfort","entertainment","onboard_service","leg_room","baggage","checkin","cleanliness"'
M='"learner":"mlp"'
P="\"profile_keys\":[$FD]"
NCR="\"num_as_cat\":[$FD,\"age\",$R]"
X3="{\"run_tag\":\"X3\",$M,$P,$NCR,\"seed_bag\":5}"
X4="{\"run_tag\":\"X4\",$M,\"num_as_cat\":[$FD,\"age\"],\"seed_bag\":5}"
G="GATE 1 vs twin +0.00014; GATE 2 on the B8 pool: ADD > 2 sd and SWAP > -1 sd. Kaggle T4 kernel."
base=$X3
mp() { echo "{\"run_tag\":\"$1\",$M,$P,$NCR,\"seed_bag\":5,\"mlp_params\":{$2}}"; }
case $1 in
Y1) cfg=$(mp Y1 '"epochs":10')
    d="Y1: X3 with a 10-epoch OneCycle"; n="HYPOTHESIS X6's shorter cycle (+0.000095) carries onto the best-ADD MLP. PREDICTION Gate 1 +0.00005 to +0.00015.";;
Y2) cfg=$(mp Y2 '"emb_dim":32')
    d="Y2: X3 with emb_dim 32"; n="HYPOTHESIS a 16-dim route embedding is too small for thousands of FD levels. PREDICTION Gate 1 +-0.0001; ADD above X3's if the embedding is the bottleneck.";;
Y3) cfg=$(mp Y3 '"plr_sigma":1.0')
    d="Y3: X3 with plr_sigma 1.0"; n="HYPOTHESIS higher-frequency periodic features resolve age and delays finer. PREDICTION Gate 1 +-0.0002.";;
Y4) cfg="{\"run_tag\":\"Y4\",$M,$NCR,\"seed_bag\":5}"; base=$X4
    d="Y4: X4 + rating embeddings (no-profile, all-embedding MLP)"; n="HYPOTHESIS the MLP analogue of W4's all-categorical view. PREDICTION solo +-0.0002 vs X4; ADD candidate next to X3.";;
Y5) cfg=$(mp Y5 '"hidden":512')
    d="Y5: X3 with hidden 512"; n="HYPOTHESIS capacity probe. PREDICTION null (S6E9: capacity is null); closes the axis.";;
*) echo "unknown tag"; exit 2;;
esac
acc=""; [ "$2" = "--launch-only" ] && acc="--accelerator NvidiaTeslaT4"
python scripts/collect_run.py --cfg "$cfg" --diff-vs "$base" --description "$d" --notes "$n $G" $acc "${@:2}" 2>&1 | grep -vE "status: RUNNING|status: QUEUED"
