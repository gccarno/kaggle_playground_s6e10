"""Everything specific to the S6E10 DATA lives here, and only here.

Column names are normalised to snake_case on load: the raw headers contain spaces and a
slash ("Departure/Arrival time convenient"), which LightGBM rewrites with a warning and which
are awkward to spell inside JSON configs. Configs and features always use the clean names.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from config import ID, REPO_ROOT, TARGET, target_to_int

ON_KAGGLE = Path("/kaggle/input").exists()

RENAME = {
    "Age": "age",
    "Flight Distance": "flight_distance",
    "Inflight wifi service": "wifi",
    "Departure/Arrival time convenient": "time_convenient",
    "Ease of Online booking": "online_booking",
    "Gate location": "gate_location",
    "Food and drink": "food_drink",
    "Online boarding": "online_boarding",
    "Seat comfort": "seat_comfort",
    "Inflight entertainment": "entertainment",
    "On-board service": "onboard_service",
    "Leg room service": "leg_room",
    "Baggage handling": "baggage",
    "Checkin service": "checkin",
    "Cleanliness": "cleanliness",
    "Departure Delay in Minutes": "dep_delay",
    "Arrival Delay in Minutes": "arr_delay",
    "Gender": "gender",
    "Customer Type": "customer_type",
    "Type of Travel": "travel_type",
    "Class": "travel_class",
}

# The 13 survey ratings, integer 0-5 (0 = not answered in the source survey).
RATINGS = ["wifi", "time_convenient", "online_booking", "gate_location", "food_drink",
           "online_boarding", "seat_comfort", "entertainment", "onboard_service",
           "leg_room", "baggage", "checkin", "cleanliness"]
RAW_NUM = ["age", "flight_distance", "dep_delay", "arr_delay"] + RATINGS
RAW_CAT = ["gender", "customer_type", "travel_type", "travel_class"]


def _find(name, kaggle_glob):
    """Locally, files live under data/. On Kaggle, mount paths have changed between
    contests (S6E9's P0 smoke test hit a FileNotFoundError on the guessed path), so search
    /kaggle/input instead of hard-coding it."""
    if not ON_KAGGLE:
        return REPO_ROOT / "data" / name
    hits = sorted(Path("/kaggle/input").glob(kaggle_glob))
    if not hits:
        raise FileNotFoundError(f"{kaggle_glob} not found under /kaggle/input")
    return hits[0]


def load_competition():
    """Return (train, test, sample, y) with clean column names."""
    train = pd.read_csv(_find("train.csv", "**/train.csv")).rename(columns=RENAME)
    test = pd.read_csv(_find("test.csv", "**/test.csv")).rename(columns=RENAME)
    sample = pd.read_csv(_find("sample_submission.csv", "**/sample_submission.csv"))
    y = target_to_int(train[TARGET])
    return train, test, sample, y


def load_origin():
    """The 129,880-row source dataset (no id column), clean names, plus its labels."""
    og = pd.read_csv(_find("original/data.csv", "**/*aviation-satisfaction*/**/data.csv")
                     ).rename(columns=RENAME)
    return og, target_to_int(og[TARGET])


def dataset_features(tr, te, cfg, feats_num, feats_cat):
    """Dataset-specific UNSUPERVISED features (no target read; safe over train + test).
    Every block is a separate default-off config key so it can be strict-twin tested."""
    if cfg.get("fe_rating_stats"):
        for df in (tr, te):
            r = df[RATINGS].to_numpy(np.float64)
            df["rating_sum"] = r.sum(1)
            df["rating_n_zero"] = (r == 0).sum(1)
            df["rating_min"] = r.min(1)
        feats_num += ["rating_sum", "rating_n_zero", "rating_min"]
    if cfg.get("fe_delay_diff"):
        for df in (tr, te):
            df["delay_diff"] = df["arr_delay"] - df["dep_delay"]
        feats_num += ["delay_diff"]
    return tr, te, feats_num, feats_cat


__all__ = ["ID", "TARGET", "RAW_NUM", "RAW_CAT", "RATINGS", "ON_KAGGLE",
           "load_competition", "load_origin", "dataset_features"]
