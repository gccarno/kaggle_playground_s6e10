#!/usr/bin/env bash
# Phase 3 batch 8: the age residual on champion T8 (residual_lookup.py). Pre-registered in
# README Phase 3 (Oct 5). Banked for the nightly slot scheduler. Run on oci-free in tmux.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
keep='strict-twin|OOF AUC =|Appended|Error|error|Traceback|NOT A STRICT'
T8='{"run_tag":"T8","profile_keys":["flight_distance"],"teacher":true}'
GATE="GATE +0.00014 OOF vs T8 5ee7f1d3 (0.960590). Ran on oci-free (aarch64)."
run() {
  echo "=================== $1"
  $PY scripts/run_local.py \
      --cfg "{\"run_tag\":\"$1\",\"profile_keys\":[\"flight_distance\"],\"teacher\":true,$2}" \
      --diff-vs "$T8" --description "$3" --notes "$4 $GATE" 2>&1 | grep -E "$keep"
  $PY scripts/residual_lookup.py "experiments/preds/$(tail -1 experiments/runs.csv | cut -d, -f1)" | grep -E "^  (age|flight_distance) "
}
run T28 '"te_cols":["age"]' "T28: T8 + per-value target encoding of age" \
  "HYPOTHESIS age carries a value-specific (sawtooth) label effect the champion misses: residual chi2/df 2.77 over 69 ages (p ~ 1e-13), 2.66 after removing a smooth trend. MECHANISM it surfaces once routes are resolved (A0 1.77, T2 2.55, T8 2.77); at A0 age TE removed it (1.12) but OOF was null (T6). SIZE excess chi2 ~120 vs FD's ~4300 at A0 (+0.0016) => expected ~+0.00005. PREDICTION below the gate; age chi2/df falls to ~1."
run T29 '"num_as_cat":["age"]' "T29: T8 + age as a categorical (identity access)" \
  "HYPOTHESIS same age signal reached through identity (LightGBM categorical splits group ages by gradient) instead of a label rate. MECHANISM the T4~T5 identity route, applied to age. PREDICTION below the gate, at most T28."
echo "=================== batch done"
$PY scripts/submit_scheduler.py list | tail -12
