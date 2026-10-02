#!/usr/bin/env python3
"""Generate a Kaggle kernel (notebook + kernel-metadata.json) that runs src/pipeline.py
from the private code dataset under one config override.

Output goes to .kaggle_output/_push_<slug>/ (gitignored) -- never the repo root, which S6E9
filled with ~100 generated probe notebooks and metadata files. The config that matters is
archived with the run in experiments/preds/<run_id>/cfg.json by collect_run.py.

Normally called by collect_run.py; standalone use is for inspecting a kernel:
    python scripts/make_probe_kernel.py A0 '{"run_tag":"A0"}' --sha <git sha> --run-id abcd1234
"""
import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from config import CODE_DATASET, COMPETITION, KAGGLE_USER, ORIGIN_DATASET  # noqa: E402

BODY = '''import glob, json, os, subprocess, sys
EXPECTED_SHA = {sha!r}
RUN_ID = {run_id!r}
cfg = json.loads({cfg_json!r})

hits = glob.glob("/kaggle/input/**/CODE_SHA", recursive=True)
assert len(hits) == 1, f"expected exactly one CODE_SHA under /kaggle/input, found {{hits}}"
code_dir = os.path.dirname(hits[0])
sha = open(hits[0]).read().strip()
print("code dataset:", code_dir, "sha", sha)
if sha != EXPECTED_SHA:
    raise RuntimeError(f"STALE CODE: dataset has {{sha}}, kernel was built for {{EXPECTED_SHA}}. "
                       "Re-run scripts/push_code.py and wait for the version to process.")

import importlib
for m in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost"):
    try:
        print(f"  {{m}} {{importlib.import_module(m).__version__}}")
    except Exception as e:
        print(f"  {{m}} unavailable: {{e}}")

env = dict(os.environ, KCFG=json.dumps(cfg), KRUN_ID=RUN_ID, PYTHONUNBUFFERED="1")
r = subprocess.run([sys.executable, "-u", os.path.join(code_dir, "pipeline.py")], env=env)
if r.returncode != 0:
    raise RuntimeError("pipeline.py failed")
'''


def slug_for(tag):
    return "s6e10-" + tag.lower().replace("_", "-")


def make(tag, cfg, sha, run_id, gpu=False):
    slug = slug_for(tag)
    out = REPO_ROOT / ".kaggle_output" / f"_push_{slug}"
    out.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": BODY.format(sha=sha, run_id=run_id,
                                         cfg_json=json.dumps(cfg)).splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (out / "kernel.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"{KAGGLE_USER}/{slug}", "title": slug, "code_file": "kernel.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": gpu, "enable_tpu": False,
        # No internet: everything the kernel needs is attached, and an offline kernel
        # cannot pip-install something that silently changes library versions.
        "enable_internet": False,
        "dataset_sources": [CODE_DATASET, ORIGIN_DATASET],
        "competition_sources": [COMPETITION], "kernel_sources": [], "model_sources": [],
        "keywords": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out, meta["id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag")
    ap.add_argument("cfg", help="inline JSON override")
    ap.add_argument("--sha", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    out, ref = make(args.tag, json.loads(args.cfg), args.sha, args.run_id)
    print(f"{ref} -> {out}")


if __name__ == "__main__":
    main()
