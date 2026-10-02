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
| Seed-noise floor (\|OOF(A0) − OOF(A0s)\|) | **UNSET** | Phase 1 |
| Paired dAUC SD at public / private split size | **UNSET** | `split_resolution.py`, Phase 2 |
| OOF→public slope, residual σ | **UNSET** | `oof_lb_fit.py` over ≥10 pairs, Phase 2 |
| **SHIPPING GATE** (σ ÷ slope, in OOF units) | **UNSET**; use a placeholder of +0.0001 until then | |

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
