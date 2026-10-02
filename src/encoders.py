"""Supervised target-encoding machinery, ported VERBATIM from S6E9 src/pipeline.py
(lines 429-923). Generic for any binary target: nothing here names a column.

Leakage rule: every function here is SUPERVISED, so callers pass training-fold rows only.
Training rows get their own encodings from an inner K-fold, so a row never contributes to
its own encoded value. The one non-verbatim change is NaN handling in _key_codes (marked).
Docstrings still cite S6E9 numbers (income, 0.0748, ...) -- they are
the measured rationale for each design and are kept as history, not as S6E10 facts.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _te_stats(values, y):
    """Per-value (sum, count). Supervised -- callers must pass a train-fold slice."""
    return pd.DataFrame({"v": values, "y": y}).groupby("v")["y"].agg(["sum", "count"])


def _fit_encoder(values, y, bins, cfg, prior, smooth=None, extra=None):
    """Build a value -> encoded-rate mapping, plus the backoff needed for unseen values.

    Two backoff modes, and the difference is the whole point of the `te_backoff` knob:

    "prior"        a rare value falls back to the fold's global positive rate (0.1746).
    "neighborhood" a rare value falls back to the rate of its QUANTILE BIN first, and only
                   the bin falls back to the global prior. For a lookup-table column this
                   is strictly better: an income value seen 3 times should be shrunk toward
                   the rate of nearby incomes, not toward the population average, because
                   the column carries a real monotone trend UNDERNEATH the lookup
                   (Spearman(value, rate) = 0.68) that the global prior throws away.
    """
    if extra is not None:
        # extra = (values, SOFT targets in [0,1], bins-or-None) -- unlabelled rows given a
        # model's predicted probability and folded into the per-value statistics. _te_stats
        # is a groupby sum/count, so a float target is already well defined: a soft label
        # of 0.3 contributes 0.3 of a buyer and 1 of a row, exactly as a fractional
        # observation should. Callers must guarantee these rows carry no information about
        # the labels being predicted (see cfg["te_pseudo"]).
        ev, ey, eb = extra
        values = np.concatenate([np.asarray(values), np.asarray(ev)])
        y = np.concatenate([np.asarray(y, dtype=np.float64),
                            np.asarray(ey, dtype=np.float64)])
        if bins is not None:
            # No bins for the extra pool means the neighborhood backoff cannot be built
            # consistently, so drop to the prior rather than silently misalign lengths.
            bins = (pd.Series(np.concatenate([np.asarray(bins.values), np.asarray(eb)]))
                    if eb is not None else None)
    st = _te_stats(values, y)
    if cfg["te_backoff"] == "neighborhood" and bins is not None:
        bst = _te_stats(bins, y)
        bin_rate = ((bst["sum"] + prior * cfg["te_backoff_smooth"]) /
                    (bst["count"] + cfg["te_backoff_smooth"]))
        # each value's backoff target is its own bin's smoothed rate
        vbin = pd.Series(bins.values, index=values).groupby(level=0).first()
        target = vbin.reindex(st.index).map(bin_rate).fillna(prior)
    else:
        bin_rate, target = None, pd.Series(prior, index=st.index)
    if smooth == "auto":
        # Empirical-Bayes smoothing in the spirit of sklearn's TargetEncoder(smooth=
        # "auto") -- see te_multi_smooth in DEFAULTS: not a bit-exact port, but the same
        # idea of letting the data set the smoothing rather than a fixed constant. A
        # single scalar tau = target variance / mean within-value variance, applied to
        # every value in this column (sklearn's own "auto" is also one global tau, not
        # per-category).
        y_var = float(np.var(y)) if len(y) > 1 else 0.0
        cnt = st["count"].clip(lower=1)
        rate = st["sum"] / cnt
        within = rate * (1 - rate)
        mean_within = float((within * st["count"]).sum() / max(st["count"].sum(), 1))
        sm = y_var / mean_within if mean_within > 1e-9 else cfg["te_smooth"]
    else:
        sm = cfg["te_smooth"] if smooth is None else float(smooth)
    enc = (st["sum"] + target * sm) / (st["count"] + sm)
    return enc, st["count"], bin_rate


def _pseudo_label_test(cfg, Xtr, ytr, Xte):
    """Pass-1 soft labels for the TEST rows, for cfg["te_pseudo"].

    Deliberately NOT routed through fit_predict(): that fitter always passes
    eval_set=[(Xva, yva)] with early stopping, so the number of rounds -- and therefore
    the predictions -- would be chosen using the validation fold's labels. Those
    predictions then go on to build the encoder that the same validation fold is scored
    with, which is a leak with no assert anywhere to catch it. Fixed rounds, no eval_set,
    train-fold labels only.

    Lean by design: this is a support-weighting device for the encoders, not a model whose
    own accuracy is shipped, and Z5c measured the gain with a 400-round lr=0.05 pass 1.
    """
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=int(cfg["te_pseudo_rounds"]),
                           learning_rate=float(cfg["te_pseudo_lr"]),
                           num_leaves=int(cfg["te_pseudo_leaves"]),
                           random_state=cfg["model_seed"], n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def apply_target_encoding(Xtr, ytr, Xva, Xte, cols, cfg, rng_seed, pseudo=None):
    """Fit per-value target encodings on the TRAINING FOLD ONLY.

    Training rows get their encoding from an inner K-fold so a row never contributes to
    its own encoded value -- without this, a 13,214-value column like Annual_Income_USD
    memorizes the fold outright. Val and test rows use the full training-fold statistics.

    Why this transform matters here (README.md section 6): after fitting the best possible
    MONOTONE function of income, per-value residual rates still have SD 0.0748 against a
    binomial expectation of 0.0284. The column is a value->target lookup table, and a
    lookup is invisible to any model that reads the value as a magnitude.

    cfg["te_multi_smooth"], when non-empty, emits one te_{c}_s{smooth} column PER
    smoothing value instead of a single te_{c} column at cfg["te_smooth"] -- the public
    frontier recipe's "smooth=10 and smooth='auto' side by side" idea (README Phase 14).
    Empty (the default) reproduces the original single-column behaviour exactly, so every
    archived run stays bit-identical.
    """
    prior = float(ytr.mean())
    new_cols = []
    # `pseudo`, when given, is an array of soft labels aligned to Xte. The extra pool IS
    # Xte: its own raw column values plus these soft labels. Note it is added to BOTH the
    # inner-fold fits (which produce the training rows' encodings) and the full-fold fit
    # (which produces the val/test rows'), because a test row is never scored and so can
    # never contribute to its own encoded value the way a train row can.
    if pseudo is not None and len(pseudo) != len(Xte):
        raise ValueError(f"pseudo has {len(pseudo)} rows, Xte has {len(Xte)}")
    smooths = list(cfg["te_multi_smooth"]) if cfg["te_multi_smooth"] else [None]
    for c in cols:
        # A local de-categorized VIEW for the encoding arithmetic only -- Xtr[c]/Xva[c]/
        # Xte[c] themselves are left untouched (still native categorical, still handed to
        # the learner as such). pandas' Categorical.map() maps the .categories and
        # reconstructs a Categorical result, which then breaks plain float arithmetic
        # (`Categorical / int`) a few lines down -- str/object side-steps that entirely.
        cat_src = isinstance(Xtr[c].dtype, pd.CategoricalDtype)
        ctr = Xtr[c].astype(str) if cat_src else Xtr[c]
        cva = Xva[c].astype(str) if cat_src else Xva[c]
        cte = Xte[c].astype(str) if cat_src else Xte[c]

        # Quantile bin edges are fit on the TRAINING FOLD ONLY, per the leakage rule.
        # A neighborhood needs an ORDERING to be a neighbor in -- np.quantile has none
        # for a categorical column, so a non-numeric column always falls back to the
        # prior regardless of the global te_backoff setting, rather than erroring.
        btr = bva = bte = None
        if cfg["te_backoff"] == "neighborhood" and not cat_src:
            edges = np.unique(np.quantile(ctr.values,
                                          np.linspace(0, 1, cfg["te_backoff_bins"] + 1)))
            btr = pd.Series(np.searchsorted(edges, ctr.values), index=Xtr.index)
            bva = pd.Series(np.searchsorted(edges, cva.values), index=Xva.index)
            bte = pd.Series(np.searchsorted(edges, cte.values), index=Xte.index)

        reps = max(1, int(cfg["te_inner_repeats"]))
        last_n, last_cnt_tr = None, None
        for sm in smooths:
            name = f"te_{c}" if sm is None else f"te_{c}_s{sm}"
            new_cols.append(name)

            # The inner-fold TE a training row receives is itself noisy: it is built
            # from 4/5 of the fold, and WHICH 4/5 is an arbitrary draw. Averaging over
            # several independent inner splits cancels that draw without touching the
            # leakage property -- every split still excludes the row's own label from
            # its encoding. Pure variance reduction, which is what an additive
            # generator rewards.
            enc_tr = np.zeros(len(Xtr), dtype=np.float64)
            cnt_tr = np.zeros(len(Xtr), dtype=np.float64)
            for rep in range(reps):
                inner = StratifiedKFold(cfg["te_inner_folds"], shuffle=True,
                                        random_state=rng_seed + 1000 * rep)
                for i, j in inner.split(Xtr, ytr):
                    e, n, br = _fit_encoder(ctr.iloc[i].values, ytr[i],
                                            None if btr is None else btr.iloc[i], cfg,
                                            prior, sm,
                                            extra=None if pseudo is None else
                                            (cte.values, pseudo, None if bte is None
                                             else bte.values))
                    fb = (pd.Series(prior, index=Xtr.index[j]) if br is None
                          else btr.iloc[j].map(br).fillna(prior))
                    enc_tr[j] += ctr.iloc[j].map(e).fillna(fb).astype(np.float64).values / reps
                    if rep == 0:
                        cnt_tr[j] = ctr.iloc[j].map(n).fillna(0.0).astype(np.float64).values
            Xtr[name] = enc_tr
            last_cnt_tr = cnt_tr

            e, n, br = _fit_encoder(ctr.values, ytr, btr, cfg, prior, sm,
                                    extra=None if pseudo is None else
                                    (cte.values, pseudo, None if bte is None
                                     else bte.values))
            last_n = n
            for X, c_, b in ((Xva, cva, bva), (Xte, cte, bte)):
                fb = prior if br is None else b.map(br).fillna(prior)
                X[name] = c_.map(e).fillna(pd.Series(fb, index=X.index) if br is not None
                                           else prior).astype(np.float64).values

        if cfg["te_count_feature"]:
            # Counts don't depend on smoothing, so any smoothing pass's (n, cnt_tr)
            # gives the same values -- last_n/last_cnt_tr are just whichever ran last.
            cname = f"cnt_{c}"
            new_cols.append(cname)
            Xtr[cname] = np.log1p(last_cnt_tr)
            Xva[cname] = np.log1p(cva.map(last_n).fillna(0.0).astype(np.float64).values)
            Xte[cname] = np.log1p(cte.map(last_n).fillna(0.0).astype(np.float64).values)
    return new_cols


def _key_codes(Xtr, Xva, Xte, cols, nbins):
    """Integer key codes for one or more columns, aligned across the three splits.

    A high-cardinality NUMERIC component is replaced by its quantile-BIN index, with the
    edges fit on the TRAINING FOLD ONLY per the leakage rule. A low-cardinality or
    categorical component is label-encoded over train union test, which is an UNSUPERVISED
    mapping and therefore not a leak (README section 3) -- it exists only so that a level
    absent from one split cannot shift another split's codes.

    Multiple components are combined into one composite code by mixed-radix packing, which
    is what makes a PAIR a single lookup key rather than two independent columns.
    """
    codes = {"tr": None, "va": None, "te": None}
    src = {"tr": Xtr, "va": Xva, "te": Xte}
    for c in cols:
        col = {k: X[c] for k, X in src.items()}
        if pd.api.types.is_numeric_dtype(col["tr"]) and col["tr"].nunique() > nbins:
            edges = np.unique(np.quantile(col["tr"].values, np.linspace(0, 1, nbins + 1)))
            part = {k: np.searchsorted(edges, v.values) for k, v in col.items()}
            n_levels = len(edges) + 1
        else:
            # S6E10 deviation from the verbatim port: map(str) rather than astype(str), so
            # a NaN becomes the level "nan". Under pandas 3, astype(str) keeps NaN as a
            # float and sorted() then fails comparing it with strings (arr_delay has NaNs;
            # S6E9 had none).
            cats = pd.Index(sorted(set().union(*(set(v.map(str)) for v in col.values()))))
            part = {k: cats.get_indexer(v.map(str)) + 1 for k, v in col.items()}
            n_levels = len(cats) + 1
        for k in codes:
            codes[k] = part[k] if codes[k] is None else codes[k] * n_levels + part[k]
    return codes


def apply_key_target_encoding(Xtr, ytr, Xva, Xte, keys, smooth, nbins, cfg, rng_seed,
                              prefix):
    """SUPERVISED TE keyed on a quantile BIN or on a PAIR of columns, fit on the fold only.

    Why this is a different regime from `apply_target_encoding` rather than a contradiction
    of C3 (raw-value keys at smoothing 100 lost 0.000667): the key here is COARSE. A bin
    holds thousands of rows instead of ~50, so heavy smoothing shrinks a well-estimated
    rate only slightly, and what the feature carries is the denoised local trend rather
    than the per-value lookup. Training rows get their encoding from an inner K-fold, as
    everywhere else, so no row contributes to its own key's rate.

    Backoff is the global prior, not a neighborhood: a bin key HAS no neighborhood beyond
    itself, and a pair key's neighborhood is the very interaction the feature is testing.
    """
    prior = float(ytr.mean())
    kcfg = {**cfg, "te_backoff": "prior"}
    new_cols = []
    for key in keys:
        cols = [key] if isinstance(key, str) else list(key)
        name = prefix + "__".join(cols)
        new_cols.append(name)
        codes = _key_codes(Xtr, Xva, Xte, cols, nbins)
        ktr = pd.Series(codes["tr"], index=Xtr.index)

        enc_tr = np.zeros(len(Xtr), dtype=np.float64)
        inner = StratifiedKFold(cfg["te_inner_folds"], shuffle=True, random_state=rng_seed)
        for i, j in inner.split(Xtr, ytr):
            e, _, _ = _fit_encoder(ktr.iloc[i].values, ytr[i], None, kcfg, prior, smooth)
            enc_tr[j] = ktr.iloc[j].map(e).fillna(prior).values
        Xtr[name] = enc_tr

        e, _, _ = _fit_encoder(ktr.values, ytr, None, kcfg, prior, smooth)
        Xva[name] = pd.Series(codes["va"], index=Xva.index).map(e).fillna(prior).values
        Xte[name] = pd.Series(codes["te"], index=Xte.index).map(e).fillna(prior).values
    return new_cols


def _shape_bin_stats(codes, y, n_bins, prior, smooth):
    """Per-bin (sum, count) plus derived CENTRAL / NEIGHBOUR-POOLED rate channels.

    Ported from the public frontier recipe's `bin_statistics`
    (jazivxt/single-model-zoom-zoom, forked as najiama/pure-lgbm-model -- README Phase
    14): central = additive-smoothed rate of the bin itself; symmetric = a 3-bin
    gaussian-weighted neighbourhood rate (kernel sigma=0.8 bins); left/right = the
    ADJACENT bin's smoothed rate; slope = right - left; curvature = how far the bin's
    own rate sits from the average of its neighbours. All smoothed toward `prior` by
    `smooth`, same additive-smoothing arithmetic as _fit_encoder.

    Returns an (n_bins, 7) array: [central, symmetric, left, right, slope, curvature,
    log1p(count)].
    """
    sums = np.bincount(codes, weights=y, minlength=n_bins).astype(np.float64)
    counts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    central = (sums + smooth * prior) / (counts + smooth)
    left_sums, left_counts = np.r_[0.0, sums[:-1]], np.r_[0.0, counts[:-1]]
    right_sums, right_counts = np.r_[sums[1:], 0.0], np.r_[counts[1:], 0.0]
    left = (left_sums + smooth * prior) / (left_counts + smooth)
    right = (right_sums + smooth * prior) / (right_counts + smooth)
    kernel = np.exp(-0.5 * (np.arange(-1, 2) / 0.8) ** 2)
    neighbor_sums = np.convolve(sums, kernel, mode="same")
    neighbor_counts = np.convolve(counts, kernel, mode="same")
    symmetric = ((neighbor_sums + smooth * kernel.sum() * prior) /
                (neighbor_counts + smooth * kernel.sum()))
    slope = right - left
    curvature = central - 0.5 * (left + right)
    return np.column_stack([central, symmetric, left, right, slope, curvature,
                            np.log1p(counts)])


def apply_shape_target_encoding(Xtr, ytr, Xva, Xte, cols, cfg, rng_seed):
    """SUPERVISED multi-scale neighbour-pooled shape encoder for a NUMERIC column, fit
    on the TRAINING FOLD ONLY, one EQUAL-WIDTH bin grid per width in
    cfg["te_shape_bins"] (distinct from te_bin_cols, which uses QUANTILE bins).

    MECHANISM (README Phase 14, DEFAULTS): apply_target_encoding keys on the exact
    value at ~50 rows/value for Annual_Income_USD, so the per-value rate estimate's own
    SE (~0.053) is close to the real per-value SD (0.0748, README section 6) --
    signal-to-noise near 1.4:1. Pooling adjacent bins here cuts that SE substantially
    while the slope/curvature channels hand the tree the response's local DERIVATIVE.
    This is a materially finer regime than te_bin_cols/H1 (100 QUANTILE bins, smoothing
    500, ~132 distinct income values per bin): at te_shape_bins=[8192, 16384] over
    income's ~$158k range, each bin is roughly $10-19 wide.

    Leakage protocol mirrors apply_target_encoding: bin edges are fit on Xtr only;
    training rows get their channels from an inner K-fold so no row informs its own
    bin's statistic (the label of a $50,003 row can otherwise leak into that bin's rate
    via a wide, coarse bin, exactly like the per-value case); val/test use the full
    training-fold statistics.
    """
    prior = float(ytr.mean())
    smooth = cfg["te_shape_smooth"]
    channel_names = ["pos", "central", "symmetric", "left", "right", "slope",
                     "curvature", "logcount"]
    new_cols = []
    for c in cols:
        v_tr = Xtr[c].to_numpy(np.float64)
        v_va = Xva[c].to_numpy(np.float64)
        v_te = Xte[c].to_numpy(np.float64)
        lo, hi = v_tr.min(), v_tr.max()
        for width in cfg["te_shape_bins"]:
            edges = np.linspace(lo, hi, width + 1)

            def code(v, edges=edges, width=width):
                return np.clip(np.searchsorted(edges[1:-1], v), 0, width - 1)

            codes_tr, codes_va, codes_te = code(v_tr), code(v_va), code(v_te)
            prefix = f"teshape_{c}_{width}_"

            feat_tr = np.zeros((len(v_tr), len(channel_names)), dtype=np.float64)
            inner = StratifiedKFold(cfg["te_inner_folds"], shuffle=True,
                                    random_state=rng_seed)
            for i, j in inner.split(codes_tr, ytr):
                stats = _shape_bin_stats(codes_tr[i], ytr[i], width,
                                         float(ytr[i].mean()), smooth)
                feat_tr[j, 0] = codes_tr[j] / max(width - 1, 1)
                feat_tr[j, 1:] = stats[codes_tr[j]]
            for k, cn in enumerate(channel_names):
                Xtr[prefix + cn] = feat_tr[:, k]

            stats = _shape_bin_stats(codes_tr, ytr, width, prior, smooth)
            for X, codes in ((Xva, codes_va), (Xte, codes_te)):
                X[prefix + "pos"] = codes / max(width - 1, 1)
                block = stats[codes]
                for k, cn in enumerate(channel_names[1:]):
                    X[prefix + cn] = block[:, k]

            new_cols += [prefix + cn for cn in channel_names]
    return new_cols


def _window_rates(v_fit, y_fit, v_query, radii, prior, smooth):
    """Smoothed positive rate over every fit row whose value lies in [q-r, q+r].

    Supervised -- callers must pass a TRAINING-FOLD slice as (v_fit, y_fit).

    Sort the fit values once, take a prefix sum of the labels, and each radius is then
    two np.searchsorted calls per query row: O(n log n) once plus O(n log n) per radius,
    which keeps a 6-radius encoder inside the same wall-clock budget as the bin encoder.
    Smoothing arithmetic is _fit_encoder's: (sum + smooth*prior) / (count + smooth).

    Returns (rates, counts), each (len(v_query), len(radii)).
    """
    order = np.argsort(v_fit, kind="mergesort")
    vs, ys = v_fit[order], y_fit[order].astype(np.float64)
    cs_sum = np.r_[0.0, np.cumsum(ys)]
    cs_cnt = np.arange(len(vs) + 1, dtype=np.float64)
    rates = np.empty((len(v_query), len(radii)), dtype=np.float64)
    counts = np.empty_like(rates)
    for k, r in enumerate(radii):
        lo = np.searchsorted(vs, v_query - r, side="left")
        hi = np.searchsorted(vs, v_query + r, side="right")
        s, n = cs_sum[hi] - cs_sum[lo], cs_cnt[hi] - cs_cnt[lo]
        rates[:, k] = (s + smooth * prior) / (n + smooth)
        counts[:, k] = n
    return rates, counts


def _window_side_rates(v_fit, y_fit, v_query, radii, prior, smooth):
    """One-sided companion to _window_rates: the smoothed positive rate over fit rows in
    [v-r, v) and in (v, v+r], per radius.

    Same prefix-sum machinery, two extra searchsorted calls per radius to find where the
    query value itself starts and ends in the sorted fit values, so the centre is excluded
    from BOTH sides. That exclusion is why these channels are strictly more leak-resistant
    than the centred rate: a one-sided window never contains the query row's own value at
    all, let alone its own label.

    Returns (left, right), each (len(v_query), len(radii)).
    """
    order = np.argsort(v_fit, kind="mergesort")
    vs, ys = v_fit[order], y_fit[order].astype(np.float64)
    cs_sum = np.r_[0.0, np.cumsum(ys)]
    cs_cnt = np.arange(len(vs) + 1, dtype=np.float64)
    # where the query value itself sits in the sorted fit values -- radius-independent
    at_lo = np.searchsorted(vs, v_query, side="left")
    at_hi = np.searchsorted(vs, v_query, side="right")
    left = np.empty((len(v_query), len(radii)), dtype=np.float64)
    right = np.empty_like(left)
    for k, r in enumerate(radii):
        lo = np.searchsorted(vs, v_query - r, side="left")
        hi = np.searchsorted(vs, v_query + r, side="right")
        sl, nl = cs_sum[at_lo] - cs_sum[lo], cs_cnt[at_lo] - cs_cnt[lo]
        sr, nr = cs_sum[hi] - cs_sum[at_hi], cs_cnt[hi] - cs_cnt[at_hi]
        left[:, k] = (sl + smooth * prior) / (nl + smooth)
        right[:, k] = (sr + smooth * prior) / (nr + smooth)
    return left, right


def apply_window_target_encoding(Xtr, ytr, Xva, Xte, cols, cfg, rng_seed):
    """SUPERVISED CENTRED-WINDOW target rates for a NUMERIC column, one column per
    radius in cfg["te_window_radii"], fit on the TRAINING FOLD ONLY.

    MECHANISM, and why this is not te_shape_cols again (README Phase 14/15). The shape
    encoder pools neighbours through a FIXED equal-width bin grid: which rows a value
    pools with depends on where the value falls inside its bin, so a value sitting at a
    bin edge pools asymmetrically, and the pooling radius is whatever the grid width
    happens to be. A centred window pools SYMMETRICALLY around the value itself, at
    several radii at once, so the same value always sits at the centre of its own
    neighbourhood. Phase 15's X1 found the shape encoder flat from 1024 to 16384 bins --
    the grid resolution did not matter -- which is consistent with the grid's real
    limitation being its ARBITRARY ORIGIN rather than its width, and that is exactly what
    a centred window removes. Read as an idea (not as code or artifact) from the public
    frontier's "centred-window target rates" description; reimplemented here under this
    repo's own leakage protocol.

    Leakage protocol mirrors apply_target_encoding / apply_shape_target_encoding: the
    windows are built from Xtr only; training rows get theirs from an inner K-fold, so a
    row's own label never enters its own window (a window ALWAYS contains the query value
    itself, so without this the encoder memorizes outright -- the same failure per-value
    TE has); val and test rows use the full training-fold statistics.
    """
    prior = float(ytr.mean())
    radii = list(cfg["te_window_radii"])
    smooth = float(cfg["te_window_smooth"])
    sides = bool(cfg["te_window_sides"])
    new_cols = []
    for c in cols:
        v_tr = Xtr[c].to_numpy(np.float64)
        v_va = Xva[c].to_numpy(np.float64)
        v_te = Xte[c].to_numpy(np.float64)
        names = [f"tewin_{c}_r{r}" for r in radii]

        rate_tr = np.zeros((len(v_tr), len(radii)), dtype=np.float64)
        cnt_tr = np.zeros_like(rate_tr)
        lf_tr = np.zeros_like(rate_tr)
        rt_tr = np.zeros_like(rate_tr)
        inner = StratifiedKFold(cfg["te_inner_folds"], shuffle=True,
                                random_state=rng_seed)
        for i, j in inner.split(v_tr.reshape(-1, 1), ytr):
            rate_tr[j], cnt_tr[j] = _window_rates(v_tr[i], ytr[i], v_tr[j], radii,
                                                  float(ytr[i].mean()), smooth)
            if sides:
                lf_tr[j], rt_tr[j] = _window_side_rates(v_tr[i], ytr[i], v_tr[j], radii,
                                                        float(ytr[i].mean()), smooth)
        for k, nm in enumerate(names):
            Xtr[nm] = rate_tr[:, k]

        out, side = {}, {}
        for key, X, v in (("va", Xva, v_va), ("te", Xte, v_te)):
            out[key] = _window_rates(v_tr, ytr, v, radii, prior, smooth)
            for k, nm in enumerate(names):
                X[nm] = out[key][0][:, k]
            if sides:
                side[key] = _window_side_rates(v_tr, ytr, v, radii, prior, smooth)
        new_cols += names

        if sides:
            # The derivative channels the shape encoder had and the centred window lost.
            # slope = right - left is the response's local first difference at v; curvature
            # = centred - mean(left, right) is its second. A tree cannot construct either
            # from the symmetric rates alone, however many radii it is given.
            for k, r in enumerate(radii):
                for suf, tr_v, va_v, te_v in (
                        ("L", lf_tr[:, k], side["va"][0][:, k], side["te"][0][:, k]),
                        ("R", rt_tr[:, k], side["va"][1][:, k], side["te"][1][:, k]),
                        ("slope", rt_tr[:, k] - lf_tr[:, k],
                         side["va"][1][:, k] - side["va"][0][:, k],
                         side["te"][1][:, k] - side["te"][0][:, k]),
                        ("curv", rate_tr[:, k] - 0.5 * (lf_tr[:, k] + rt_tr[:, k]),
                         out["va"][0][:, k] - 0.5 * (side["va"][0][:, k] + side["va"][1][:, k]),
                         out["te"][0][:, k] - 0.5 * (side["te"][0][:, k] + side["te"][1][:, k]))):
                    nm = f"tewin_{c}_r{r}_{suf}"
                    Xtr[nm], Xva[nm], Xte[nm] = tr_v, va_v, te_v
                    new_cols.append(nm)

        if cfg["te_window_count"]:
            # Window occupancy: how many training-fold rows the rate was built from, so
            # the tree can discount a radius that is empty for this value. Counts do not
            # depend on the labels, but they are still fold-fit, so they ride along here
            # rather than in base_features.
            cnames = [f"tewin_{c}_r{r}_n" for r in radii]
            for k, nm in enumerate(cnames):
                Xtr[nm] = np.log1p(cnt_tr[:, k])
                Xva[nm] = np.log1p(out["va"][1][:, k])
                Xte[nm] = np.log1p(out["te"][1][:, k])
            new_cols += cnames
    return new_cols
