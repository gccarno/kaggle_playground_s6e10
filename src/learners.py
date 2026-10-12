"""Learners. Each returns (val_proba, test_proba, best_iteration); AUC is the eval metric.

Ported from S6E9 `fit_predict`, with one deliberate change: each learner reads its OWN
param dict (cfg["lgb_params"], cfg["xgb_params"], cfg["cat_params"], cfg["glm_params"]).
In S6E9 xgb and cat read LightGBM-named keys (num_leaves, colsample_bytree...) and mapped
them, so tuning one learner silently retuned the others.

The neural/EBM/TabICL learners were not ported: nine architectures were tested in S6E9 and
none reached the strong-and-decorrelated quadrant (FINDINGS section 4). Port one only when
a probe with a written mechanism needs it.
"""
import numpy as np
import pandas as pd


def fit_predict(cfg, Xtr, ytr, Xva, yva, Xte, feats_cat):
    name = cfg["learner"]
    esr = cfg["early_stopping_rounds"]

    if name == "lgb":
        import lightgbm as lgb
        p = cfg["lgb_params"]
        mono = None
        if cfg["monotone"]:
            mono = [int(cfg["monotone"].get(c, 0)) for c in Xtr.columns]
        # One group per column = every tree splits on one feature: a boosted GAM.
        inter = [[i] for i in range(Xtr.shape[1])] if cfg["additive_only"] else None
        m = lgb.LGBMClassifier(random_state=cfg["model_seed"], n_jobs=-1, verbose=-1,
                               monotone_constraints=mono, interaction_constraints=inter, **p)
        # eval_set=, not eval_X=/eval_y=: the latter only exists in LightGBM 4.7+ and broke
        # on Kaggle's pinned image in S6E9 (X3).
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        return (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1],
                int(m.best_iteration_ or p["n_estimators"]))

    if name == "xgb":
        import xgboost as xgb
        p = cfg["xgb_params"]
        m = xgb.XGBClassifier(random_state=cfg["model_seed"], n_jobs=-1, tree_method="hist",
                              enable_categorical=True, eval_metric="auc",
                              early_stopping_rounds=esr, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1],
                int(m.best_iteration or p["n_estimators"]))

    if name == "cat":
        from catboost import CatBoostClassifier, Pool
        p = cfg["cat_params"]
        cat_idx = [Xtr.columns.get_loc(c) for c in feats_cat if c in Xtr.columns]
        A, B, C = Xtr.copy(), Xva.copy(), Xte.copy()
        for c in feats_cat:
            if c in A.columns:
                # A NaN level must become an explicit string: categorical .astype(str) can keep
                # it as float NaN, which CatBoost rejects (U21, arr_delay). Columns without NaN
                # convert exactly as before, so earlier runs are bit-identical.
                A[c], B[c], C[c] = (X[c].astype(object).where(X[c].notna(), "nan").astype(str)
                                    for X in (A, B, C))
        m = CatBoostClassifier(random_seed=cfg["model_seed"], eval_metric="AUC", verbose=0,
                               early_stopping_rounds=esr, **p)
        m.fit(Pool(A, ytr, cat_features=cat_idx), eval_set=Pool(B, yva, cat_features=cat_idx))
        return (m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1],
                int(m.get_best_iteration() or p["iterations"]))

    if name == "glm":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        cats = [c for c in feats_cat if c in Xtr.columns]

        def design(X, cols=None):
            D = pd.get_dummies(X, columns=cats, drop_first=True).astype(float)
            D = D.fillna(0.0)
            return D if cols is None else D.reindex(columns=cols, fill_value=0)
        A = design(Xtr)
        B, C = design(Xva, A.columns), design(Xte, A.columns)
        sc = StandardScaler().fit(A)          # fit on the training fold only
        m = LogisticRegression(max_iter=3000, **cfg["glm_params"]).fit(sc.transform(A), ytr)
        return (m.predict_proba(sc.transform(B))[:, 1],
                m.predict_proba(sc.transform(C))[:, 1], 0)

    if name == "mlp":
        return _fit_predict_mlp(cfg, Xtr, ytr, Xva, yva, Xte, feats_cat)

    if name == "tabpfn":
        return _fit_predict_tabpfn(cfg, Xtr, ytr, Xva, yva, Xte, feats_cat)

    raise ValueError(f"unknown learner {name!r}")


def _fit_predict_tabpfn(cfg, Xtr, ytr, Xva, yva, Xte, feats_cat):
    """TabPFN-3.5 in-context learner: the whole training fold (~560k rows) is the context,
    with no subsampling and no gradient fitting. Phase 4b (README): the tabular foundation
    model view that every top public stack carries. The recipe (full context, KV cache,
    int8 cache, autocast, lean cache build) follows hemingweb's public notebook.

    Needs internet on the kernel: tabpfn==9.0.0 is pip-installed by the kernel cell (with a
    version guard), and the checkpoint is downloaded from Hugging Face once per process.
    TabPFN-3.5 carries Prior Labs' non-commercial licence, which permits Kaggle use."""
    import os, time
    import torch
    from huggingface_hub import hf_hub_download
    from tabpfn import TabPFNClassifier
    import tfm_lean
    p = cfg["tabpfn_params"]
    tfm_lean.apply(p["chunk"])
    ckpt = hf_hub_download("Prior-Labs/tabpfn_3_5", p["ckpt"], local_dir="/tmp/tabpfn_ckpt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cats = [c for c in feats_cat if c in Xtr.columns]

    def arr(X):
        # Category codes (NaN stays NaN); numerics as float32.
        D = X.copy()
        for c in cats:
            codes = D[c].cat.codes.astype("float32")
            D[c] = codes.where(codes >= 0)
        return D.to_numpy(np.float32)
    cat_idx = [Xtr.columns.get_loc(c) for c in cats]

    def oom(e):
        return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()

    t0 = time.time()
    clf = None
    for extra in ({"memory_saving_mode": "auto", "keep_cache_on_device": True},
                  {"memory_saving_mode": True, "keep_cache_on_device": True},
                  {"memory_saving_mode": True, "keep_cache_on_device": False}):
        try:
            if dev == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            clf = TabPFNClassifier(model_path=ckpt, n_estimators=1, random_state=cfg["model_seed"],
                                   device=dev, ignore_pretraining_limits=True,
                                   fit_mode="fit_with_cache", kv_cache_precision="int8",
                                   inference_precision="autocast",
                                   categorical_features_indices=cat_idx, **extra)
            clf.fit(arr(Xtr), np.asarray(ytr))
            break
        except Exception as e:
            if not oom(e) or extra["keep_cache_on_device"] is False:
                raise
            print(f"    tabpfn OOM in fit with {extra}; next fallback", flush=True)
            clf = None
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
    print(f"    tabpfn fit {(time.time() - t0) / 60:.1f} min, context {len(Xtr):,}, "
          f"peak {peak:.1f} GB", flush=True)

    def predict(X, name):
        A = arr(X)
        out = np.empty(len(A))
        bs, i, t = p["batch"], 0, time.time()
        while i < len(A):
            j = min(i + bs, len(A))
            try:
                out[i:j] = clf.predict_proba(A[i:j])[:, 1]
            except Exception as e:
                if oom(e) and bs > 1024:
                    bs //= 2
                    torch.cuda.empty_cache()
                    continue
                raise
            i = j
        print(f"    tabpfn {name}: {len(A):,} rows in {(time.time() - t) / 60:.1f} min", flush=True)
        return out
    return predict(Xva, "val"), predict(Xte, "test"), 0


def _fit_predict_mlp(cfg, Xtr, ytr, Xva, yva, Xte, feats_cat):
    """A RealMLP-lite: PLR (periodic) embeddings for every numeric, learned embeddings for
    every categorical (incl. num_as_cat identities such as cat_flight_distance), 3 SiLU
    layers. Phase 3b (README): the one family that builds its OWN representation of the
    route, from an embedding, instead of reading one we engineered.

    Numerics: train-fold median fill, then a rank-gauss QuantileTransformer fit on the
    training fold only. Early stopping on the validation fold's AUC picks the epoch, the
    same use of the validation fold as GBDT early stopping. torch is in the Kaggle image;
    it is imported here so the other learners never need it."""
    import torch
    from torch import nn
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import QuantileTransformer
    p = cfg["mlp_params"]
    seed = cfg["model_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cats = [c for c in feats_cat if c in Xtr.columns]
    nums = [c for c in Xtr.columns if c not in cats]
    med = Xtr[nums].median()
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             subsample=200_000, random_state=seed)
    qt.fit(Xtr[nums].fillna(med).to_numpy(np.float64))

    def tensors(X):
        xn = qt.transform(X[nums].fillna(med).to_numpy(np.float64)).astype(np.float32)
        # Category codes over the train+test vocabulary (base_features); NaN -> 0.
        xc = np.stack([X[c].cat.codes.to_numpy(np.int64) + 1 for c in cats], 1) if cats \
            else np.zeros((len(X), 0), np.int64)
        return torch.from_numpy(xn), torch.from_numpy(xc)
    card = [len(Xtr[c].cat.categories) + 1 for c in cats]

    class PLR(nn.Module):
        def __init__(self, n, k, d):
            super().__init__()
            self.c = nn.Parameter(torch.randn(n, k) * p["plr_sigma"])
            self.w = nn.Parameter(torch.randn(n, 2 * k, d) / np.sqrt(2 * k))
            self.b = nn.Parameter(torch.zeros(n, d))

        def forward(self, x):                      # (B, n) -> (B, n * d)
            v = 2 * np.pi * self.c[None] * x[..., None]
            v = torch.cat([torch.sin(v), torch.cos(v)], -1)
            return torch.relu(torch.einsum("bnk,nkd->bnd", v, self.w) + self.b).flatten(1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.plr = PLR(len(nums), p["plr_freq"], p["plr_dim"])
            self.emb = nn.ModuleList(nn.Embedding(k, min(p["emb_dim"], k)) for k in card)
            width = len(nums) * p["plr_dim"] + sum(min(p["emb_dim"], k) for k in card)
            layers = []
            for _ in range(p["layers"]):
                layers += [nn.Linear(width, p["hidden"]), nn.SiLU(), nn.Dropout(p["dropout"])]
                width = p["hidden"]
            self.mlp = nn.Sequential(*layers, nn.Linear(width, 1))

        def forward(self, xn, xc):
            parts = [self.plr(xn)] + [e(xc[:, i]) for i, e in enumerate(self.emb)]
            return self.mlp(torch.cat(parts, 1)).squeeze(1)

    tr_n, tr_c = tensors(Xtr)
    sets = [tensors(Xva), tensors(Xte)]
    yt = torch.from_numpy(np.asarray(ytr, np.float32))
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    steps = p["epochs"] * int(np.ceil(len(yt) / p["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)

    def predict(xn, xc):
        net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(xn), 16384):
                out.append(torch.sigmoid(net(xn[s:s + 16384].to(dev), xc[s:s + 16384].to(dev))).cpu())
        net.train()
        return torch.cat(out).numpy().astype(np.float64)

    best, best_ep, best_va, best_te, bad = -1.0, 0, None, None, 0
    for ep in range(p["epochs"]):
        perm = torch.randperm(len(yt), generator=gen)
        for s in range(0, len(perm), p["batch_size"]):
            idx = perm[s:s + p["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(net(tr_n[idx].to(dev), tr_c[idx].to(dev)), yt[idx].to(dev))
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(*sets[0])
        auc = roc_auc_score(yva, pv)
        print(f"    mlp epoch {ep + 1}: val auc {auc:.6f}", flush=True)
        if auc > best:
            best, best_ep, best_va, best_te, bad = auc, ep + 1, pv, predict(*sets[1]), 0
        else:
            bad += 1
            if bad >= p["patience"]:
                break
    return best_va, best_te, best_ep
