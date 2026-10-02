# kaggle_playground_s6e10

Kaggle Playground Series S6E10, **Predicting Airline Satisfaction**:
https://www.kaggle.com/competitions/playground-series-s6e10

This README is **the contract**: data facts, the frozen split, the instrument numbers, the
standing decisions, and the phase log. Where it disagrees with FINDINGS.md, this file wins.
The method comes from `KAGGLE_PLAYBOOK.md`, carried over from S6E9. Read §10 to §12 first.

---

## 1. Competition and data facts

| | |
|---|---|
| Target | `satisfaction` (bool in train.csv); submit a **probability** |
| Metric | ROC AUC |
| Rows | train 699,635 · test 299,844 · original 129,880 |
| Positive rate | train 0.443573 · original 0.434463 |
| Features | 4 numerics (`age`, `flight_distance`, `dep_delay`, `arr_delay`), 13 survey ratings (0 to 5), 4 binary or 3-level categoricals |
| Missing | `arr_delay`: 292 NaN in train (393 in the original) |
| Source | Synthetic, generated from `arseniyshutko/binary-aviation-satisfaction-129k` (public, CC0, allowed as external data) |
| Limits | **10 submissions/day** (reset 00:00 UTC, no rollover) · 2 final selections · teams of up to 3 |
| Deadline | **2026-10-31 23:59 UTC** |
| Public/private split | Not stated. Assume 20/80, the Playground default. |

Column names are normalised to snake_case on load (`src/dataset.py: RENAME`).

**Phase 1 facts** (from `scripts/eda_phase1.py`):
- **Duplicates:** none in train or test, and no test feature row appears in train.
- **Drift:** none. Adversarial train-vs-test AUC is **0.5010**, so imputation or drift handling has nothing to fix.
- **Cardinality:**
  - `flight_distance`: 3,474 values in train, 3,821 in the original, 61 test rows with values unseen in train;
  - `age`: 75 values;
  - `dep_delay` / `arr_delay`: about 167 values each (the original has about 470, so the generator coarsened the delays);
  - ratings: 6 levels each (0–5; `baggage` has no 0).
- **Rating zeros** ("not answered") are rare, between 0.1% and 3.7%.
- **The original dataset is not copied:**
  - 17 of 699,635 train rows (0.0024%) match an original row exactly;
  - matching on the key (`flight_distance`, `age`, `customer_type`, `travel_type`, `travel_class`) hits 54.7% of train rows, but the original label adds only **+0.000014 AUC** on top of A0 for those rows, which is noise.

## 2. Frozen CV (never change)

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over train.csv in file order (`src/config.py`). Every OOF file is aligned to this split. `compare_oof.py` refuses to compare files whose fold assignments differ.

## 3. Leakage rules

- Anything that reads the target is fitted **inside the fold, on training-fold rows only**, at every place it is used. Target-encoded training rows get their values from an inner K-fold.
- Unsupervised transforms (frequency counts, category vocabularies, quantisation) may see train + test together.
- Rows from the original dataset only ever go into **training** folds. They are never validated on and never predicted.
- `te_pseudo`'s first pass uses a fixed number of rounds with no `eval_set`, because early stopping on the validation fold would leak.

## 4. The instrument (the three numbers)

| Number | Value | Source |
|---|---|---|
| Seed-noise floor (\|OOF(A0) − OOF(A0s)\|) | **0.000039** (S6E9: 0.000038) | Phase 1 |
| Paired dAUC SD, seed-twin pair (A0 vs A0s) | public **0.000152** · private **0.000083** · full OOF 0.000047 | `split_resolution.py`, 200 boots |
| Paired dAUC SD, cross-family pair (A0 vs X1) | public 0.000159 · private 0.000087 · full OOF 0.000051 | same |
| OOF→public fit over 8 GBDT pairs | slope **0.971**, residual σ **0.000130**, Spearman 0.79 | `oof_lb_fit.py` |
| **SHIPPING GATE** (σ ÷ slope) | **+0.00013 OOF, PROVISIONAL** (8 pairs, below the 10-pair minimum). Re-fit once at ≥12 GBDT pairs in Phase 2, then freeze. | |

How to read these:
- **Any public LB delta below about 0.00015 is unreadable.** A0→A0s, a seed change alone, moved public by +0.00010.
- **Inside the near-champion band (OOF 0.9583–0.9590) the public LB ranks models at Spearman 0.49.** The band is only 0.0006 wide, comparable to the noise.
- **The private split resolves about 1.8× better than public.** Shakeup room between near-equal models is about ±0.0001.

Once the gate is set, it stays fixed. **Do not re-derive it later from small leaderboard deltas.**

## 5. Standing decisions (2026-10-01)

1. **Code reaches Kaggle as a private dataset** (`gcarno/s6e10-code`, via `scripts/push_code.py`). The GitHub repo stays private until Nov 1. *Why:* the rules forbid sharing code privately outside the team, and pinning code by sha removes S6E9's "kernel ran stale GitHub HEAD" trap.
2. **Artifact policy: ours-only until the Oct 18 review.** Public notebooks can be read for *ideas* from day 1. Public OOF/test pairs can enter blends only after the review, and only fold-aligned and with a manifest. Plan: Final A is ours-only, Final B is public-mixed. Price in leaderboard places: TBD in Phase 1.
3. **Phase checkpoints.** Each phase ends with a write-up and needs the user's approval before the next phase starts.
4. **What AUC is invariant to:** any monotone transform of the final score. That rules out calibration, temperature, clipping and probability post-processing as knobs.

## 6. Data model notes

**Lookup diagnostic, run against the model's residuals** (Phase 1). The question is whether A0's OOF residuals still vary by value more than binomial noise allows. This is the S6E9 income test, but conditioned on the model. A raw per-value rate test (which flags all four numerics at 2 to 18 times binomial) is confounded, because the other features already explain part of each value's rate.

| Column | Values (n≥30) | Residual χ²/df | Read |
|---|---|---|---|
| `flight_distance` | 2,243 | **2.20** | **The model misses value-specific information. This is the prime lookup suspect.** |
| `dep_delay` | 57 | 1.01 | Magnitude, handled |
| `arr_delay` | 54 | 0.93 | Magnitude, handled |
| `age` | 69 | 0.52 | Magnitude, handled |
| ratings | 6 | below 0.3 | Handled |

## 7. Submission schedule

| Days | Use of the 10 slots |
|---|---|
| Oct 2–3 | Calibration spread: about 8 deliberately different models plus the A0 baseline |
| Oct 4–17 | Every probe is submitted. Fill unused slots with champion seed twins. |
| Oct 18–30 | Ours-only and public-mixed candidates, near-twin rotation |
| Oct 30 | **Finals chosen by hand.** Commit a predicted private score and rank band. |
| Oct 31 | Spare slots only |

---

## 8. Phase log

### Phase 0: repository and harness (2026-10-01)

- Made the repo private. Downloaded the competition data and the original dataset into `data/`, which is gitignored.
- Ported the S6E9 harness, refactored:
  - `src/` split into config, dataset, encoders, learners and pipeline;
  - `scripts/kaggle_io.py` puts the run_id prefix on every submission message;
  - kernels run the code from a pinned private dataset, with a run_id nonce echo as a stale-output guard;
  - `load_cfg` rejects unknown keys.
- **Encoder deviation:** `_key_codes` now uses `map(str)`. S6E10 has NaNs, and under pandas 3 `astype(str)` leaves NaN as a float, so `sorted()` failed.
- **Harness verified:**
  - `P0_smoke_local` (10% sample, 2 folds) scored OOF 0.957039.
  - `P0_smoke_kernel`, the same config run on a Kaggle CPU kernel from the code dataset, also scored 0.957039, with identical fold AUCs and best iterations. Max |Δp| was 4.4e-16 on both OOF and test, which is float rounding.
  - The kernel took about 1 minute from push to COMPLETE.
- **The Kaggle image's library versions differ from local:** numpy 2.0.2, pandas 2.3.3, sklearn 1.6.1, lightgbm 4.6.0, xgboost 3.2.0, catboost 1.2.10. Locally we have pandas 3.0.5, sklearn 1.9.0 and lightgbm 4.7.0. Identical results on this config is evidence, not a guarantee. Re-check local-vs-kernel parity whenever a run is moved between the two.
- **Deferred to Phase 4,** when blending starts: porting `stack_logit`, `leg_probe`, `leg_swap`, `subset_ceiling`, `public_blend` and `backfill_private`.

### Phase 1: EDA, seed floor and calibration spread (2026-10-02)

**Baseline.** A0 (`ee4536ff`): raw 21 features, 5-fold LightGBM. OOF 0.958787, public **0.95829**. Folds 0.958808 ± 0.000668. `best_iters` are 482/589/553/234/272, so folds 3 and 4 stopped training at about half the rounds of the others. Keep an eye on that.

**Pre-registered calibration batch (9 slots, committed before any of it runs).** The purpose is *not* to improve the score. It is to buy OOF↔LB pairs spread across a range, so that `oof_lb_fit.py` can measure the slope and residual σ (playbook §4). Deliberately worse models are the "cheapest paired points" (S6E9 Phase 8). Nothing in this batch is a shipping candidate, so none of it has a gate.

| Tag | Change vs A0 | Role | Expected OOF |
|---|---|---|---|
| A0s | `model_seed` 1337 | **seed-noise floor** = \|OOF(A0s) − OOF(A0)\|; also the near-twin pair for `split_resolution.py` | about 0.9588 |
| A0b3 | `seed_bag` 3 | near-twin point; variance reduction inside each fold | ≥ A0 by noise |
| L7 | `num_leaves` 7 | capacity ladder, low end | slightly below A0 |
| L255 | `num_leaves` 255 | capacity ladder, high end | about A0 |
| X1 | `learner` xgb | family point (S6E9 family offset for xgb: −0.00004) | about A0 |
| K1 | `learner` cat | family point (S6E9 family offset for cat: −0.0001) | about A0 |
| D1 | drop `online_boarding`, `wifi` | deliberately degraded GBDT, mid-range point | below A0 |
| D2 | drop all 13 ratings | deliberately degraded GBDT, low point | far below |
| G1 | `learner` glm | non-GBDT. **Excluded from the GBDT fit** (in S6E9 one neural point inflated σ 3×) and kept as a family check. | about 0.92 |

The run order is cheapest first, and every run is submitted. Some predictions are recorded so the post-mortem can check the instrument:
- the seed floor is below 0.0001;
- Spearman(OOF, LB) over the GBDT points is at least 0.9.

**Leaderboard reading, 2026-10-02 02:16 UTC.** From the Explore-agent survey of 22 public notebooks; the notes live in the session scratchpad.
- **The board:**
  - 277 teams; #1 has 0.96137, #10 0.96129, #50 0.96109, #100 0.96028, #200 0.95816;
  - A0 (0.95829) sits around rank 180, the plain raw-feature level, which public XGBoost on raw columns also reaches (0.958763 CV);
  - the **0.96124 band is mostly shared blend files** (0.995 × someone's csv + ε), plus stacks of public OOF libraries such as "Flight Log V5".
- **Split:** almost every serious notebook uses **SKF(5, shuffle, 42), which is our frozen split**, so public OOF libraries are fold-aligned with ours. Exceptions: evgendvorkin uses seed 123, and ou20040313 uses 3 folds.
- **Independent confirmation of our residual test.** Public authors found `flight_distance` is a lookup key, and "route" features carry the largest public gain, **+0.0016** for XGBoost:
  - per-distance profiles of the other columns over train+test;
  - counts;
  - in-fold TE of FD, FD×Age, FD×Class and similar;
  - `max_bin=4096` alone also helps (fold 0: +0.0007).
- **Second lever, a "teacher".** A model trained *only on the original 129,880 rows* is used as a feature: +0.0003 on raw features, +0.0006–0.0008 next to the route features. Its mechanism is that competition label = the original's decision function + about 3.8% flips (dariushafshar). This is a *different* channel from the per-key label lookup we measured as null: the teacher generalises the original's function to every row.
- **Third lever, RealMLP as the decorrelated stack member** (about +0.0003 in stacks). This contradicts S6E9's "neural is null", so it has to be re-tested here rather than assumed either way.
- **Public nulls**, which are cheap to believe but still unverified by us:
  - appending the original rows (−0.00013 to −0.00045);
  - rating aggregates;
  - label cleaning (−0.002);
  - 10 folds;
  - TabM, FT-Transformer and GPT-2.
- **Noise:**
  - thisray's resampling study: the public LB orders a 0.0001 gap correctly only about 85% of the time;
  - our A0→A0s seed change alone moved public by +0.0001;
  - dariushafshar estimates a label-noise ceiling of about 0.9613.

**Artifact policy price** (§5.2, written down before it gets tempting):
- The best published ours-style single models are about 0.9608 to 0.9611 public, and the best honest stacks of *our own* members are about 0.9612 to 0.9613. The shared-file frontier is 0.96124 to 0.96137.
- Today, 0.9611 against 0.96124 is about **30 places** (48 teams at or above 0.96110).
- At S6E9's final size (about 3.5k teams, roughly 12× today's count), the same gap scales to roughly **300 to 400 places**. The gap is inside the shakeup noise: the private paired SD is still to be measured.
- **Policy unchanged:** ours-only until the Oct 18 review. *Ideas* from the public notebooks are allowed now.

**Calibration batch results.** All are strict twins of A0, and all were submitted except K1.

| Tag | run_id | OOF | Public | Note |
|---|---|---|---|---|
| A0 | ee4536ff | 0.958787 | 0.95829 | baseline |
| A0s | cbbda46e | 0.958826 | 0.95839 | seed floor 0.000039 OOF, but **+0.00010 public** |
| A0b3 | ea8495f9 | 0.958971 | 0.95833 | 3-seed bag: **+0.000184 OOF (4.7× the seed floor)**, public +0.00004 |
| L7 | 1936c4df | 0.958347 | 0.95781 | every fold hit about 3,000 rounds, so it is capacity-starved |
| L255 | 7ecc574a | 0.958840 | 0.95834 | ≈ A0; it stops at about 150–230 rounds |
| D1 | e8fbc4b0 | 0.953116 | 0.95301 | without `online_boarding` and `wifi`: −0.0057 |
| D2 | e48f031c | 0.874472 | 0.87640 | without all ratings: −0.084. The ratings are the signal. |
| G1 | 8d189ab3 | 0.921556 | 0.92227 | GLM; non-GBDT, excluded from the fit |
| X1 | 061c20f8 | 0.958838 | 0.95818 | xgb ≈ lgb on OOF; public −0.00011 vs A0 (inside noise) |
| K1 | — | — | — | **Not run.** Claude Code stopped the batch on low system memory (0.4 GB free of 15.7) during CatBoost training. Orphan processes were stopped before they could submit. To be re-run. |

Checks against the pre-registered predictions:
- Seed floor < 0.0001: **held** (0.000039).
- Spearman ≥ 0.9 over the GBDT points: **missed** (0.79). The near-champion cluster is ordered by noise, which is a finding, not a failure.
