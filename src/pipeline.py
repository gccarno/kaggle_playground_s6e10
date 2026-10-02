#!/usr/bin/env python3
"""S6E10 model pipeline -- the single source of truth for every run.

Every probe is a CONFIG OVERRIDE, never an edit to this file's body. The override arrives as
JSON in the KCFG env var and the output directory in KOUT; both have defaults, so
`python src/pipeline.py` runs the default recipe standalone.

Emits into KOUT:
    oof_proba_<learner>.csv    id, fold, proba   (aligned to the frozen split)
    test_proba_<learner>.csv   id, proba
    submission.csv             id, satisfaction
and prints one final `RUN_METRICS_JSON:{...}` line -- the parsing contract with
scripts/run_local.py and scripts/collect_run.py.

Leakage rule (README section 3): every SUPERVISED transform is fit inside the fold loop on
training-fold rows only. base_features() is UNSUPERVISED (no target read) and may see
train + test together.

Module layout:  config.py (names) | dataset.py (the only S6E10-specific code) |
                encoders.py (target encoders) | learners.py (fit_predict) | this file.
"""
import gc, json, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CV_SEED, ID, N_FOLDS, REPO_ROOT, SEED, TARGET  # noqa: E402
from dataset import (ON_KAGGLE, RAW_CAT, RAW_NUM, dataset_features,  # noqa: E402
                     load_competition, load_origin)
from encoders import (_pseudo_label_test, apply_key_target_encoding,  # noqa: E402
                      apply_shape_target_encoding, apply_target_encoding,
                      apply_window_target_encoding)
from learners import fit_predict  # noqa: E402

warnings.filterwarnings("ignore")

OUT_DIR = Path(os.environ.get("KOUT", "/kaggle/working" if ON_KAGGLE else
                              REPO_ROOT / ".kaggle_output" / "scratch"))

# ---------------------------------------------------------------------- config
DEFAULTS = {
    "run_tag": "baseline",
    "learner": "lgb",                # lgb | xgb | cat | glm
    "model_seed": SEED,
    "seed_bag": 1,                   # average this many model seeds inside each fold
    "n_folds": N_FOLDS,
    "cv_seed": CV_SEED,
    # SMOKE TESTS ONLY: train on a stratified fraction of train.csv. Any value < 1 makes the
    # OOF incomparable with every full run -- never use it for a probe.
    "sample_frac": 1.0,

    # -- feature set
    "drop_cols": [],                 # raw columns removed from the feature set
    "num_as_cat": [],                # numeric columns ALSO given to the learner as categoricals

    # -- UNSUPERVISED feature engineering (computed once over train + test)
    "freq_cols": [],                 # log1p(train+test value count)
    "fe_digit_cols": [],             # floor(v / 10^p) % 10 per power
    "fe_digit_powers": [0, 1, 2, 3],
    "fe_quant_cols": [],             # quantisation ladder floor(v / d)
    "fe_quant_divisors": [5, 25, 100],
    "fe_quant_divisors_by_col": {},
    "fe_rating_stats": False,        # dataset.py: sum / n_zero / min of the 13 ratings
    "fe_delay_diff": False,          # dataset.py: arr_delay - dep_delay

    # UNSUPERVISED group profile: for each key column, the mean of every other raw numeric
    # column over all train+test rows sharing that key value, plus log1p(group size).
    # MECHANISM (README Phase 2): if flight_distance is a route ID, a route's typical
    # passenger (age, class mix, ratings) is a property of the ROUTE the tree can read off
    # directly, instead of reconstructing it from 3,474 numeric split points.
    "profile_keys": [],

    # -- original dataset: append its rows to every fold's TRAINING rows only
    "use_origin_extra": False,
    # TEACHER: a model fit on the ORIGINAL dataset only (its own clean labels), whose
    # logit becomes one feature on every train/test row. Not a leak w.r.t. the competition
    # folds: it never sees a competition label. MECHANISM: competition label = the source's
    # decision function + ~3.8% flips, so a model of the clean function is a denoised
    # summary of all 21 columns at once (public: +0.0003 raw, +0.0006-0.0008 with routes).
    "teacher": False,
    "teacher_params": {"n_estimators": 1500, "learning_rate": 0.03, "num_leaves": 63,
                       "min_child_samples": 20, "colsample_bytree": 0.8,
                       "subsample": 0.8, "subsample_freq": 1},

    # -- SUPERVISED encoders (fit per fold; see encoders.py for each one's mechanism)
    "te_cols": [],
    "te_smooth": 20.0,
    "te_multi_smooth": [],
    "te_inner_folds": 5,
    "te_inner_repeats": 1,
    "te_backoff": "prior",           # "prior" | "neighborhood"
    "te_backoff_bins": 200,
    "te_backoff_smooth": 50.0,
    "te_count_feature": False,
    "te_pseudo": False,
    "te_pseudo_rounds": 400,
    "te_pseudo_lr": 0.05,
    "te_pseudo_leaves": 31,
    "te_bin_cols": [],
    "te_bins": 100,
    "te_bin_smooth": 500.0,
    "te_pair_cols": [],
    "te_pair_bins": 20,
    "te_pair_smooth": 1500.0,
    "te_shape_cols": [],
    "te_shape_bins": [1024, 4096],
    "te_shape_smooth": 10.0,
    "te_window_cols": [],
    "te_window_radii": [1, 2, 5, 10, 25, 50],
    "te_window_smooth": 10.0,
    "te_window_count": False,
    "te_window_sides": False,

    # -- learner constraints (lgb only)
    "monotone": {},                  # {column: +1 | -1}
    "additive_only": False,

    # -- per-learner params
    "early_stopping_rounds": 100,
    "lgb_params": {
        "n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 63,
        "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
        "min_child_samples": 20, "reg_lambda": 0.0,
    },
    "xgb_params": {
        "n_estimators": 3000, "learning_rate": 0.05, "max_depth": 0, "max_leaves": 63,
        "grow_policy": "lossguide", "colsample_bytree": 0.8, "subsample": 0.8,
        "reg_lambda": 1.0,
    },
    "cat_params": {"iterations": 3000, "learning_rate": 0.08, "depth": 6, "l2_leaf_reg": 3.0},
    "glm_params": {"C": 1.0},
}
DICT_KEYS = ["lgb_params", "xgb_params", "cat_params", "glm_params", "teacher_params"]


def load_cfg(override=None):
    """DEFAULTS + override. Param dicts merge one level deep, so an override of
    {"lgb_params": {"num_leaves": 15}} keeps every other lgb param."""
    over = json.loads(os.environ.get("KCFG", "{}")) if override is None else override
    unknown = set(over) - set(DEFAULTS)
    if unknown:
        # A misspelt key is a silent no-op that still logs a row -- S6E9 lost a probe to
        # exactly that (te_smooth ignored under te_multi_smooth). Refuse instead.
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **over}
    for k in DICT_KEYS:
        cfg[k] = {**DEFAULTS[k], **over.get(k, {})}
    return cfg


# ------------------------------------------------------------------ feature build
def base_features(train, test, cfg):
    """UNSUPERVISED transforms only -- safe to compute once over train + test."""
    feats_num = [c for c in RAW_NUM if c not in cfg["drop_cols"]]
    feats_cat = [c for c in RAW_CAT if c not in cfg["drop_cols"]]
    tr, te = train.copy(), test.copy()

    tr, te, feats_num, feats_cat = dataset_features(tr, te, cfg, feats_num, feats_cat)

    for c in cfg["freq_cols"]:
        counts = pd.concat([tr[c], te[c]]).value_counts()
        for df in (tr, te):
            df[f"freq_{c}"] = np.log1p(df[c].map(counts).astype(float).values)
        feats_num.append(f"freq_{c}")

    for c in cfg["fe_digit_cols"]:
        for p in cfg["fe_digit_powers"]:
            for df in (tr, te):
                df[f"digit_{c}_{p}"] = np.floor_divide(df[c].to_numpy(np.float64), 10.0 ** p) % 10
            feats_num.append(f"digit_{c}_{p}")

    by_col = cfg["fe_quant_divisors_by_col"] or {}
    for c in cfg["fe_quant_cols"]:
        for d in by_col.get(c, cfg["fe_quant_divisors"]):
            for df in (tr, te):
                df[f"q_{c}_{d}"] = np.floor_divide(df[c].to_numpy(np.float64), float(d))
            feats_num.append(f"q_{c}_{d}")

    for key in cfg["profile_keys"]:
        both = pd.concat([tr, te], ignore_index=True)
        others = [c for c in RAW_NUM if c != key]
        prof = both.groupby(key)[others].mean().add_prefix(f"prof_{key}__")
        prof[f"prof_{key}__logn"] = np.log1p(both.groupby(key).size())
        for df in (tr, te):
            block = prof.reindex(df[key].to_numpy())
            for c in prof.columns:
                df[c] = block[c].to_numpy()
        feats_num += list(prof.columns)

    if cfg["teacher"]:
        add_teacher(tr, te, cfg)
        feats_num.append("teacher_logit")

    for c in cfg["num_as_cat"]:
        # Value IDENTITY handed to the learner as a categorical (LightGBM groups levels by
        # gradient statistics inside the fold -- not a leak).
        for df in (tr, te):
            df[f"cat_{c}"] = df[c]
        feats_cat.append(f"cat_{c}")

    # Category vocabulary over train + test: unsupervised, and an unseen test level can
    # never raise at inference.
    for c in feats_cat:
        vocab = pd.api.types.CategoricalDtype(sorted(set(tr[c].dropna()) | set(te[c].dropna())))
        tr[c] = tr[c].astype(vocab)
        te[c] = te[c].astype(vocab)
    return tr, te, feats_num, feats_cat


def add_teacher(tr, te, cfg):
    """Fit LightGBM on the original dataset (raw columns, its own labels; fixed rounds, no
    competition rows anywhere) and write its logit onto every train/test row."""
    import lightgbm as lgb
    og, oy = load_origin()
    cols = RAW_NUM + RAW_CAT
    A = og[cols].copy()
    T, E = tr[cols].copy(), te[cols].copy()
    for c in RAW_CAT:
        vocab = pd.api.types.CategoricalDtype(sorted(set(A[c]) | set(T[c]) | set(E[c])))
        A[c], T[c], E[c] = A[c].astype(vocab), T[c].astype(vocab), E[c].astype(vocab)
    m = lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **cfg["teacher_params"])
    m.fit(A, oy)
    for df, X in ((tr, T), (te, E)):
        p = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        df["teacher_logit"] = np.log(p / (1 - p))
    msg = f"teacher: fit on {len(A)} original rows"
    if TARGET in tr:      # diagnostic only -- nothing downstream uses this number
        msg += (f"; teacher alone vs competition labels AUC = "
                f"{roc_auc_score(tr[TARGET].astype(int), tr['teacher_logit']):.6f}")
    print(msg)


def apply_supervised(cfg, Xtr, ytr, Xva, Xte, seed):
    """Every SUPERVISED transform, fit on the training fold only. The load-bearing call."""
    if cfg["te_cols"]:
        apply_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_cols"], cfg, seed)
        if cfg["te_pseudo"]:
            pseudo = _pseudo_label_test(cfg, Xtr, ytr, Xte)
            apply_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_cols"], cfg, seed, pseudo=pseudo)
            print(f"  te_pseudo pass1: mean soft label {pseudo.mean():.6f}", flush=True)
    if cfg["te_bin_cols"]:
        apply_key_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_bin_cols"], cfg["te_bin_smooth"],
                                  cfg["te_bins"], cfg, seed, "teb_")
    if cfg["te_pair_cols"]:
        apply_key_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_pair_cols"],
                                  cfg["te_pair_smooth"], cfg["te_pair_bins"], cfg, seed, "tep_")
    if cfg["te_shape_cols"]:
        apply_shape_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_shape_cols"], cfg, seed)
    if cfg["te_window_cols"]:
        apply_window_target_encoding(Xtr, ytr, Xva, Xte, cfg["te_window_cols"], cfg, seed)


def load_frame(cfg):
    """Load data, apply the (smoke-only) row sample and the unsupervised features.
    Returns (train, test, sample, y, feats, feats_cat, origin_X, origin_y)."""
    train, test, sample, y = load_competition()
    if cfg["sample_frac"] < 1.0:
        keep = (pd.Series(y).groupby(y, group_keys=False)
                .apply(lambda s: s.sample(frac=cfg["sample_frac"], random_state=SEED)).index)
        keep = np.sort(keep.to_numpy())
        train, y = train.iloc[keep].reset_index(drop=True), y[keep]
        print(f"SMOKE: sample_frac={cfg['sample_frac']} -> {len(train)} train rows")

    origin = load_origin() if cfg["use_origin_extra"] else None
    if origin is not None:
        # Run the origin rows through base_features WITH the test frame so they get the same
        # unsupervised columns and category vocabulary; they never enter a validation fold.
        og, og_y = origin
        og = og.drop(columns=[TARGET])
        og[ID] = -1 - np.arange(len(og))
        both = pd.concat([train.drop(columns=[TARGET]), og], ignore_index=True)
        both_f, test_f, feats_num, feats_cat = base_features(both, test, cfg)
        train_f, og_f = both_f.iloc[:len(train)], both_f.iloc[len(train):]
        feats = feats_num + feats_cat
        return (train_f, test_f, sample, y, feats, feats_cat,
                og_f[feats].reset_index(drop=True), og_y)

    train_f, test_f, feats_num, feats_cat = base_features(train, test, cfg)
    return train_f, test_f, sample, y, feats_num + feats_cat, feats_cat, None, None


def build_fold(cfg, frame, f, i, j):
    """Fold f's (Xtr, ytr, Xva, yva, Xte) with every supervised transform applied. Exposed
    so HPO or diagnostics reuse THIS fold builder instead of re-implementing it (S6E9's
    optuna_probe.py drifted from the pipeline exactly that way)."""
    train, test, _, y, feats, _, og_X, og_y = frame
    Xtr, Xva, Xte = train[feats].iloc[i].copy(), train[feats].iloc[j].copy(), test[feats].copy()
    ytr, yva = y[i], y[j]
    if og_X is not None:
        Xtr = pd.concat([Xtr.reset_index(drop=True), og_X], ignore_index=True)
        ytr = np.concatenate([ytr, og_y])
    apply_supervised(cfg, Xtr, ytr, Xva, Xte, cfg["cv_seed"] + f)
    return Xtr, ytr, Xva, yva, Xte


def code_sha():
    """The git sha this code was pushed from (scripts/push_code.py writes CODE_SHA next to
    the sources); locally, the live HEAD. Logged with every run so a kernel result can be
    tied to the exact code that produced it."""
    stamp = Path(__file__).resolve().parent / "CODE_SHA"
    if stamp.exists():
        return stamp.read_text().strip()
    import subprocess
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=REPO_ROOT)
    return out.stdout.strip()


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    cfg = load_cfg()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"cfg = {json.dumps(cfg)}\nout = {OUT_DIR}\ncode_sha = {code_sha()}\n")

    frame = load_frame(cfg)
    train, test, sample, y, feats, feats_cat, og_X, _ = frame
    print(f"{len(feats)} base features: {feats}")
    if og_X is not None:
        print(f"origin extra rows: {len(og_X)} (training folds only)")

    skf = StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=cfg["cv_seed"])
    oof = np.zeros(len(train))
    fold_id = np.full(len(train), -1, dtype=np.int8)
    test_proba = np.zeros(len(test))
    fold_aucs, best_iters = [], []

    for f, (i, j) in enumerate(skf.split(train[feats], y)):
        Xtr, ytr, Xva, yva, Xte = build_fold(cfg, frame, f, i, j)
        n_model_features = Xtr.shape[1]
        # Seed-bagging INSIDE a fold: averages the fitting randomness without touching the
        # split -- the one seed manipulation that paid in S6E7.
        bag = max(1, int(cfg["seed_bag"]))
        pv, pt, bi = np.zeros(len(Xva)), np.zeros(len(Xte)), 0
        for b in range(bag):
            c = {**cfg, "model_seed": cfg["model_seed"] + 100 * b}
            a, t_, i_ = fit_predict(c, Xtr, ytr, Xva, yva, Xte, feats_cat)
            pv += a / bag; pt += t_ / bag; bi = max(bi, i_)
        oof[j], fold_id[j] = pv, f
        test_proba += pt / cfg["n_folds"]
        fold_aucs.append(roc_auc_score(yva, pv))
        best_iters.append(bi)
        print(f"fold {f}  auc={fold_aucs[-1]:.6f}  best_iter={bi}", flush=True)
        # Release this fold's frames before the next fold builds its own (S6E9 OOM fix).
        del Xtr, Xva, Xte, pv, pt
        gc.collect()

    final_auc = roc_auc_score(y, oof)
    lr = cfg["learner"]
    print(f"\nOOF AUC = {final_auc:.6f}   folds {np.mean(fold_aucs):.6f} "
          f"+/- {np.std(fold_aucs):.6f}")

    pd.DataFrame({ID: train[ID], "fold": fold_id, "proba": oof}).to_csv(
        OUT_DIR / f"oof_proba_{lr}.csv", index=False)
    pd.DataFrame({ID: test[ID], "proba": test_proba}).to_csv(
        OUT_DIR / f"test_proba_{lr}.csv", index=False)
    sub = pd.DataFrame({ID: test[ID], TARGET: test_proba})
    # Validate against sample_submission before trusting the file.
    assert list(sub.columns) == list(sample.columns), (sub.columns, sample.columns)
    assert len(sub) == len(sample) and (sub[ID].values == sample[ID].values).all()
    assert sub[TARGET].between(0, 1).all() and sub[TARGET].notna().all()
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    metrics = {
        "run_tag": cfg["run_tag"], "learner": lr, f"{lr}_oof_auc": round(final_auc, 6),
        "final_oof_auc": round(final_auc, 6),
        "fold_auc_mean": round(float(np.mean(fold_aucs)), 6),
        "fold_auc_std": round(float(np.std(fold_aucs)), 6),
        "fold_aucs": [round(a, 6) for a in fold_aucs], "best_iters": best_iters,
        "n_features": int(n_model_features), "n_folds": cfg["n_folds"],
        "cv_seed": cfg["cv_seed"], "model_seed": cfg["model_seed"],
        "te_cols": ",".join(map(str, cfg["te_cols"])),
        "engineered": [c for c in feats if c not in RAW_NUM + RAW_CAT],
        "sample_frac": cfg["sample_frac"], "code_sha": code_sha(),
        # Echo of the caller's nonce: collect_run.py refuses a kernel output whose run_id is
        # not the one it just pushed (S6E9 archived a stale previous output twice).
        "run_id": os.environ.get("KRUN_ID", ""),
        "notebook_runtime_sec": round(time.time() - t0, 1),
    }
    print("RUN_METRICS_JSON:" + json.dumps(metrics))


if __name__ == "__main__":
    main()
