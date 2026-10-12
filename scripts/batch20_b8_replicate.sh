#!/usr/bin/env bash
# usage: scripts/batch20_b8_replicate.sh LEG [--launch-only|--adopt] -- Phase 4 batch 20 (README, Oct 12).
# Replicates one B8 leg at model_seed 43 as <LEG>R, a strict twin of the archived leg cfg.
# MLP legs get a T4. The B8R blend of all 11 replicates measures how much of B8's public score is luck.
cd "$(dirname "$0")/.."
leg=$1; shift
# tr: Windows Python ends lines with \r, which would make gpu "1\r".
read -r cfg base gpu < <(python - "$leg" <<'PY' | tr -d '\r'
import json, sys, pandas as pd
sys.path[:0] = ["src"]; from pipeline import DEFAULTS
leg = sys.argv[1]
r = pd.read_csv("experiments/runs.csv").set_index("run_tag")
c = json.load(open(r.loc[leg, "preds_dir"] + "/cfg.json"))
base = {k: v for k, v in c.items() if k in DEFAULTS and DEFAULTS[k] != v and k != "run_tag"}
base = {"run_tag": leg, **base}
rep = {**base, "run_tag": leg + "R", "model_seed": 43}
print(json.dumps(rep, separators=(",", ":")), json.dumps(base, separators=(",", ":")),
      int(base.get("learner") == "mlp"))
PY
)
n="HYPOTHESIS the B8 recipe is seed-stable: ${leg}R (model_seed 43) is a replicate of $leg. PREDICTION solo within the leg's seed floor (|d| < 0.0001 cat, < 0.0002 bagged mlp); B8R OOF within +-0.00003 of B8; B8R vs B8 public gap measures B8's public luck (paired sd 0.000152). Phase 4 batch 20."
acc=""; [ "$gpu" = "1" ] && [ "$1" = "--launch-only" ] && acc="--accelerator NvidiaTeslaT4"
python scripts/collect_run.py --cfg "$cfg" --diff-vs "$base" --description "${leg}R: $leg replicate at model_seed 43 (B8R leg)" --notes "$n" $acc "$@" 2>&1 | grep -vE "status: RUNNING|status: QUEUED"
