"""Competition-wide constants -- the one place a new contest's names get changed.

Every script and src/ module imports from here rather than hard-coding the slug, target or
user: in S6E9 "Will_Buy_EV" and "playground-series-s6e9" were spelled out in ~30 files.
"""
from pathlib import Path

COMPETITION = "playground-series-s6e10"
TARGET = "satisfaction"
ID = "id"
KAGGLE_USER = "gcarno"

# The code reaches Kaggle kernels as a PRIVATE dataset (scripts/push_code.py), not a git
# clone of a public repo: it keeps the code private during the contest and pins every
# kernel to an exact, recorded code version.
CODE_DATASET = f"{KAGGLE_USER}/s6e10-code"
# The 129,880-row source dataset the competition's generator was fit to (public, CC0).
ORIGIN_DATASET = "arseniyshutko/binary-aviation-satisfaction-129k"

# Frozen CV -- README section 2. Never change these after Phase 1.
SEED = 42
CV_SEED = 42
N_FOLDS = 5

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_CSV = REPO_ROOT / "experiments" / "runs.csv"
PREDS_DIR = REPO_ROOT / "experiments" / "preds"


def target_to_int(s):
    """train.csv stores the target as bool (pandas reads True/False); the original
    dataset stores it as the strings 'satisfied' / 'neutral or dissatisfied'."""
    if s.dtype == bool:
        return s.astype(int).to_numpy()
    return s.astype(str).str.strip().str.lower().isin(
        ["true", "1", "satisfied"]).astype(int).to_numpy()

# Kaggle's final-submission deadline (README, data facts table). The slot scheduler stops here.
DEADLINE_UTC = "2026-10-31T23:59:00+00:00"
