#!/usr/bin/env bash
# Phase 3 batch 6: profile variants on champion T8. Pre-registered in README Phase 3
# (Oct 4). Banked for the nightly slot scheduler (T8r is NOSUBMIT). Run on oci-free in tmux.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T8='{"run_tag":"T8","profile_keys":["flight_distance"],"teacher":true}'
GATE="GATE +0.00014 OOF vs T8 5ee7f1d3 (0.960590). Ran on oci-free (aarch64)."
keep='strict-twin|OOF AUC =|Appended|Error|error|Traceback|NOT A STRICT'
echo "=================== T8r (parity: must be bit-identical to T8 5ee7f1d3)"
$PY scripts/run_local.py --cfg '{"run_tag":"T8r","profile_keys":["flight_distance"],"teacher":true}' \
    --description "T8r: T8 replayed after the Phase 3 profile axes (parity)" \
    --notes "PARITY: archived T8 cfg on code with profile_cat_shares/profile_stats/profile_pair_keys (default off). Max |dp| vs T8 must be 0. NOSUBMIT (public would repeat T8)." 2>&1 | grep -E "$keep"
run() {  # tag cfg-extra description notes
  echo "=================== $1"
  $PY scripts/run_local.py \
      --cfg "{\"run_tag\":\"$1\",\"profile_keys\":[\"flight_distance\"],\"teacher\":true,$2}" \
      --diff-vs "$T8" --description "$3" --notes "$4 $GATE" 2>&1 | grep -E "$keep"
}
run T20 '"profile_cat_shares":true' "T20: T8 + per-route categorical shares (class/type/customer/gender)" \
  "HYPOTHESIS the route's passenger MIX by category is route knowledge the means-only profile lacks. MECHANISM across routes the Business-class share varies 57x binomial (travel type 20x, loyal 17x); a row's own class is visible but not its route's class mix. PREDICTION clears, +0.0002 to +0.0006."
run T21 '"profile_stats":["std"]' "T21: T8 + per-route std of the other numerics" \
  "HYPOTHESIS route heterogeneity (spread of ages, delays, ratings) adds little over its mean. MECHANISM a second moment of the passenger mix; weaker than the mean. PREDICTION below the gate."
run T22 '"profile_pair_keys":[["flight_distance","travel_class"]]' "T22: T8 + profile keyed on route x class" \
  "HYPOTHESIS route x class is a fare product whose typical ratings differ from the route's overall mean. MECHANISM a Business passenger's satisfaction depends on that route's Business product, which the route-wide mean blends with Economy. PREDICTION borderline, +0.0001 to +0.0003."
echo "=================== batch done"
