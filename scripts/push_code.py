#!/usr/bin/env python3
"""Publish src/ to Kaggle as a new version of the PRIVATE dataset config.CODE_DATASET.

Why a dataset and not `git clone` (S6E9's approach): the repo stays private during the
contest (rules: no code sharing outside the team except publicly on the forums), and every
kernel is tied to an exact code version. In S6E9 a kernel silently ran whatever GitHub HEAD
was -- "push before you push a kernel" was a rule people had to remember. Here the rule is
enforced:

  * refuses to run with a dirty git tree, so the uploaded code IS a commit;
  * writes that commit's sha into CODE_SHA next to the sources;
  * every probe kernel asserts CODE_SHA == the sha it was generated for, so a stale or
    still-processing dataset version fails loudly instead of running old code.

    python scripts/push_code.py            # upload a new version (creates it the first time)
    python scripts/push_code.py --check    # report whether Kaggle already has HEAD
"""
import argparse, json, shutil, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from config import CODE_DATASET  # noqa: E402
import kaggle_io  # noqa: E402

STAGE = REPO_ROOT / ".kaggle_output" / "_code"
PUSHED_RECORD = REPO_ROOT / ".kaggle_output" / "code_sha_pushed.txt"


def head_sha():
    return kaggle_io.run(["git", "rev-parse", "HEAD"]).stdout.strip()


def dirty_files():
    """Tracked or untracked changes under src/ only -- docs and logs can be dirty, the
    code that runs on Kaggle cannot."""
    out = kaggle_io.run(["git", "status", "--porcelain", "--", "src"]).stdout
    return [l for l in out.splitlines() if l.strip()]


def pushed_sha():
    return PUSHED_RECORD.read_text().strip() if PUSHED_RECORD.exists() else ""


def dataset_exists():
    out = kaggle_io.kaggle("datasets", "status", CODE_DATASET)
    return out.returncode == 0 and "ready" in (out.stdout + out.stderr).lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--wait-sec", type=int, default=60,
                    help="pause after upload so the new version finishes processing")
    args = ap.parse_args()

    sha = head_sha()
    if args.check:
        print(f"HEAD {sha}\nlast pushed {pushed_sha() or '(never)'}\n"
              f"up to date: {sha == pushed_sha()}")
        return
    dirty = dirty_files()
    if dirty:
        raise SystemExit("src/ has uncommitted changes -- commit first, so the uploaded code "
                         "is a real commit:\n  " + "\n  ".join(dirty))

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for p in (REPO_ROOT / "src").glob("*.py"):
        shutil.copy2(p, STAGE / p.name)
    (STAGE / "CODE_SHA").write_text(sha + "\n")
    user, slug = CODE_DATASET.split("/")
    (STAGE / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": CODE_DATASET, "licenses": [{"name": "other"}]}, indent=1))

    if dataset_exists():
        out = kaggle_io.kaggle("datasets", "version", "-p", str(STAGE), "-m", sha, "-q")
    else:
        # `datasets create` is PRIVATE unless --public is passed.
        out = kaggle_io.kaggle("datasets", "create", "-p", str(STAGE), "-q")
    blob = out.stdout + out.stderr
    print(blob.strip())
    if out.returncode != 0 or "error" in blob.lower():
        raise SystemExit("code upload failed")
    PUSHED_RECORD.write_text(sha + "\n")
    print(f"uploaded {CODE_DATASET} @ {sha}; waiting {args.wait_sec}s for processing")
    time.sleep(args.wait_sec)


if __name__ == "__main__":
    main()
