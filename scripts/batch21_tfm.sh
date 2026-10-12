#!/usr/bin/env bash
# usage: scripts/batch21_tfm.sh TAG [--launch-only|--adopt] -- Phase 4b batch 21, TabPFN-3.5 legs (README, Oct 12).
# T4 GPU kernels with internet (pip tabpfn==9.0.0 behind a version guard). Z2 twins Z1; Z3 twins Z2.
cd "$(dirname "$0")/.."
FD='"flight_distance"'
T='"learner":"tabpfn"'
G="GATE 1 vs twin +0.00014; GATE 2 on the B8 pool: ADD > 2 sd and SWAP > -1 sd. Kaggle T4 kernel, about 2.5 GPU-hours."
base=""
case $1 in
Z0s) cfg="{\"run_tag\":\"Z0s\",$T,\"sample_frac\":0.05,\"n_folds\":2}"
     d="Z0s: TabPFN-3.5 smoke (5% rows, 2 folds)"; n="NOSUBMIT smoke: install guard, checkpoint download, lean patch, code path.";;
Z1)  cfg="{\"run_tag\":\"Z1\",$T}"
     d="Z1: TabPFN-3.5 full context, raw columns"; n="HYPOTHESIS the TFM view (in-context learning over the full fold) is a new family view. PREDICTION solo 0.9600-0.9608; Gate 2 ADD > +0.00005 on B8.";;
Z2)  cfg="{\"run_tag\":\"Z2\",$T,\"num_as_cat\":[$FD,\"age\"]}"; base="{\"run_tag\":\"Z1\",$T}"
     d="Z2: Z1 + FD and age as categoricals"; n="HYPOTHESIS route identity as a categorical (goodpjw2008 idea). PREDICTION Gate 1 +0.0002 to +0.0006.";;
Z3)  cfg="{\"run_tag\":\"Z3\",$T,\"num_as_cat\":[$FD,\"age\"],\"teacher\":true}"; base="{\"run_tag\":\"Z2\",$T,\"num_as_cat\":[$FD,\"age\"]}"
     d="Z3: Z2 + teacher"; n="HYPOTHESIS the original's function as an input (samanyu1808 orig_pred idea). PREDICTION solo +0.0001 or more; U1 says ADD falls.";;
*) echo "unknown tag"; exit 2;;
esac
acc=""; [ "$2" = "--launch-only" ] && acc="--accelerator NvidiaTeslaT4"
dv=(); [ -n "$base" ] && dv=(--diff-vs "$base")
python scripts/collect_run.py --cfg "$cfg" "${dv[@]}" --description "$d" --notes "$n $G" $acc "${@:2}" 2>&1 | grep -vE "status: RUNNING|status: QUEUED"
