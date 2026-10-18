#!/usr/bin/env bash
# usage: scripts/batch27_tabicl.sh TAG [--launch-only|--adopt] -- Phase 5 batch 27, our own TabICL v2 leg (README, Oct 19).
# T4 GPU kernels with internet (pip tabicl==2.1.1 behind the version guard). K1 twins Z1 (only `learner` changes).
cd "$(dirname "$0")/.."
G="GATE 1 vs twin Z1 is reported; GATE 2 on the B12w pool at TFM weight 2: ADD > 2 sd and SWAP > -1 sd. Kaggle T4 kernel."
base=""
case $1 in
K0s) cfg='{"run_tag":"K0s","learner":"tabicl","sample_frac":0.05,"n_folds":2}'
     d="K0s: TabICL v2 smoke (5% rows, 2 folds)"; n="NOSUBMIT smoke: install guard, HF checkpoint, T4 code path and per-fold timing.";;
K1)  cfg='{"run_tag":"K1","learner":"tabicl"}'; base='{"run_tag":"Z1","learner":"tabpfn"}'
     d="K1: TabICL v2, raw columns, 250k stratified context"; n="HYPOTHESIS a second in-context family (column-then-row attention, a different prior) whose errors are partly independent of TabPFN's. PREDICTION solo 0.9603-0.9610; Gate 2 ADD on B12w > +0.00002.";;
*) echo "unknown tag"; exit 2;;
esac
acc=""; [ "$2" = "--launch-only" ] && acc="--accelerator NvidiaTeslaT4"
dv=(); [ -n "$base" ] && dv=(--diff-vs "$base")
python scripts/collect_run.py --cfg "$cfg" "${dv[@]}" --description "$d" --notes "$n $G" $acc "${@:2}" 2>&1 | grep -vE "status: RUNNING|status: QUEUED"
