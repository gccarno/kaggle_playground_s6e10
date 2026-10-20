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
| OOF→public fit over 13 GBDT pairs (G1 excluded) | slope **0.971**, residual σ **0.000135**, Spearman 0.89 | `oof_lb_fit.py` re-fit, Phase 2 (Oct 3) |
| **SHIPPING GATE** (σ ÷ slope) | **+0.00014 OOF, FROZEN 2026-10-03.** The 8-pair provisional value was 0.00013, and the re-fit moved it by less than its own precision. | |

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

**Automation (from Oct 4).** `scripts/submit_scheduler.py` runs from cron on `oci-free` at 23:15 UTC. It spends whatever slots are still free that day: the explicit queue first, then banked full-data runs that haven't been submitted, newest first.
- Auto-fill stops after Oct 29. From then on, only queued runs are sent.
- The scheduler never writes runs.csv. Scores come back through `submit_run.py --fill-missing`.
- See the `submission-scheduler` skill.

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

### Phase 2: the lookup hunt (2026-10-02)

**Compute policy (user, 2026-10-02):** full runs go to Kaggle CPU kernels, up to 5 at a time. The laptop is for analysis only, after Phase 1's low-memory stop.

**New config axes, default off, so every archived config is unchanged:**
- `profile_keys`: per-key means of the other raw numerics over train+test, plus log group size. Unsupervised.
- `teacher`: LightGBM fit on the original rows only (fixed 1,500 rounds at lr 0.03); its logit becomes a feature. Smoke check: the teacher alone scores 0.9558 AUC on competition labels.

**Pre-registered probes.** Every probe is a strict twin of A0 (OOF 0.958787). The gate is the provisional **+0.00013 OOF**. All are run on Kaggle in parallel.

| Tag | Change | Hypothesis / mechanism | Prediction |
|---|---|---|---|
| T1 | `te_cols: [flight_distance]` | FD is a route ID, not a magnitude (residual χ²/df 2.20). Per-value TE hands the tree the lookup directly. | **clears the gate**, +0.0005 to +0.0015 |
| T2 | `profile_keys: [flight_distance]` | A route's typical passenger mix is a route property. Unsupervised, so it carries no label noise. | clears the gate |
| T3 | `teacher: true` | Competition label = the source's function + about 3.8% flips; the teacher is a denoised summary of that function. | clears the gate (public reports +0.0003) |
| T4 | `lgb_params.max_bin: 4096` | With 256 histogram bins, adjacent distances share a bin, so a lookup is invisible. Finer bins partially restore it. | small gain, below T1 |
| T5 | `freq_cols: [flight_distance]` | **Control for T1:** is the gain group *size* (unsupervised) or the label rate? | below the gate |
| T6 | `te_cols: [age]` | **Negative control:** the residual test says age is already handled (χ²/df 0.52), so TE should add nothing. | \|Δ\| < gate |
| K1 | `learner: cat` | Phase 1 calibration point, re-run on Kaggle | about A0 |

**Falsification:**
- If T1 fails the gate, the residual test is wrong as a lookup detector.
- If T6 clears the gate, the residual test misses information it should have caught.
- Either outcome gets logged as a finding.

**Slots:** only 1 is left today, and it goes to **T1**. The rest are submitted after 00:00 UTC on Oct 3.

**Compute change (user, Oct 3).** T1–T5 ran on the OCI VM `oci-free` instead of Kaggle kernels.
- **The VM:** aarch64, 2 OCPU / 12 GB, managed in `~/Documents/oci-infra`. `uv sync` gives the same package versions as the laptop, and the data md5s match.
- **Parity:** A0arm (`6391ae86`, A0 replayed on ARM, not submitted) gives OOF 0.958787. Max |Δp| against A0 is **4.4e-16** on both OOF and test, the same rounding-level parity as the Phase 0 kernel. So deltas are read against A0 directly, with no platform offset.
- **Speed:** about 3–4 min per probe, 23 min for all six runs.

**Results, batch of Oct 3.** T1 went out with the batch rather than in the Oct 2 slot. All five were submitted, each a strict twin of A0 (OOF 0.958787, public 0.95829), gated at **+0.00014**.

| Tag | run_id | OOF | ΔOOF | Gate | Public | ΔPublic |
|---|---|---|---|---|---|---|
| T1 FD target encoding | 0a155047 | 0.960139 | +0.00135 | **pass** | 0.95971 | +0.00142 |
| T2 FD profiles | 46e7df22 | **0.960379** | **+0.00159** | **pass** | **0.96004** | **+0.00175** |
| T3 teacher | f7b789ea | 0.958887 | +0.00010 | fail | 0.95820 | −0.00009 |
| T4 max_bin 4096 | 17bd6d92 | 0.959498 | +0.00071 | **pass** | 0.95904 | +0.00075 |
| T5 FD frequency (control) | 5cdc1a57 | 0.959480 | +0.00069 | **pass** | 0.95906 | +0.00077 |

**Checks against the pre-registration:**
- **T1 held.** It cleared the gate inside the predicted +0.0005 to +0.0015 band, so the residual χ²/df test works as a lookup detector. The first falsification arm is not triggered.
- **T2 held, and it is the best probe.** The *unsupervised* route profile beats the supervised TE. The route is described better by who flies it than by its noisy label rate, and the profile carries no in-fold estimation noise.
- **T3 missed.** It came in below the gate on both OOF and public. Public notebooks report +0.0003 on raw features; we measured a third of that. They also report +0.0006 to +0.0008 *next to route features*, so the teacher is not closed. Re-test it on top of the new champion.
- **T4 held** (a small gain, below T1).
- **T5 is falsified, and it's the informative result.** Frequency alone is worth +0.00069, which is half of T1. FD counts over train+test are close to unique per value, so frequency works as a *second, re-ordered copy of the route ID*. Trees can then isolate single routes that sit between other routes on the distance axis. T4 (finer bins, +0.00071) is the same mechanism reached a different way: T4 ≈ T5.
  - **Reading:** about +0.0007 is *identity access* (any way of letting a tree address a single route). The extra +0.0007 to +0.0009 in T1 and T2 is *what is known about the route* (its label rate, or its passenger mix).

**Still open in this phase:** T6 (the age-TE negative control) and K1 (CatBoost).

**Pre-registered, batch 2 (Oct 3, 5 slots, on `oci-free`).**
- **T6 and K1** run as already registered: strict twins of A0.
- **K1 now runs on the 12 GB VM.** The Phase 1 failure was *system* memory on a laptop that was busy with other work, not CatBoost's own footprint (about 21 columns × 700k rows). The script runs it last, and a crash stops the run before it can submit.
- **T7–T9 stack Phase 2's levers on the new champion T2** (`46e7df22`, OOF 0.960379). They are strict twins of `{"run_tag":"T2","profile_keys":["flight_distance"]}`, gated at **+0.00014 vs T2**.

| Tag | Change vs T2 | Hypothesis / mechanism | Prediction |
|---|---|---|---|
| T7 | `te_cols: [flight_distance]` | The label rate (T1) and the passenger mix (T2) are different knowledge about a route, so they should partly add. | **clears**, +0.0002 to +0.0006 |
| T8 | `teacher: true` | Public notebooks report +0.0006–0.0008 for the teacher *next to route features*, against our +0.0001 on raw features (T3). That would mean the teacher interacts with route identity. | borderline. Clears only if the interaction is real. |
| T9 | `lgb_params.max_bin: 4096` | T2's 16 profile columns are close to unique per route, so T2 already gives *identity access* (the T4 ≈ T5 mechanism). Finer bins on FD should add nothing. | **below the gate** (\|Δ\| < 0.00014) |

**Falsification:**
- If T9 clears the gate, T2 does not fully provide identity access, and the "two components" reading of T5 is incomplete.
- If T7 is null, the profile already encodes the route's label rate.

**Results, batch 2.** All five were submitted. Δ is measured against the stated baseline, and the gate is +0.00014.

| Tag | run_id | Baseline | OOF | ΔOOF | Public | ΔPublic | Read |
|---|---|---|---|---|---|---|---|
| T6 age TE | b9272bbf | A0 | 0.958762 | −0.000025 | 0.95833 | +0.00004 | **held**: null, as predicted |
| T7 T2 + FD TE | 6844d407 | T2 | 0.960386 | +0.000007 | 0.96018 | +0.00014 | **prediction failed**: null on OOF |
| T8 T2 + teacher | 5ee7f1d3 | T2 | **0.960590** | **+0.000211** | **0.96022** | +0.00018 | **clears. New champion.** |
| T9 T2 + max_bin 4096 | 4a621eb8 | T2 | 0.960353 | −0.000026 | 0.96000 | −0.00004 | **held**: null, as predicted |
| K1 CatBoost | 4ecb3d67 | A0 | 0.958674 | −0.000113 | 0.95827 | −0.00002 | about A0. The family offset matches S6E9's −0.0001. |

**What the results mean:**
- **T6 closes the falsification check from the other side.** The residual χ²/df test flagged FD and not age, and target encoding helped FD and not age. The test is now validated both ways, as a lookup detector.
- **T7: the passenger-mix profile already contains the route's label rate.** Adding the supervised target encoding on top contributed nothing to OOF. Its +0.00014 on public is below the readable threshold (0.00015).
- **T8: the teacher is an interaction, not a main effect.** On raw features it added +0.00010, below the gate (T3). Next to the route profile it adds +0.00021. That's the same direction public notebooks report, at about a third of their size.
- **T9: T2 already gives identity access.** Finer bins on FD add nothing once the profile is present. This confirms T5's two-component reading.
- **K1 ran fine on the 12 GB VM** (52 minutes on 2 cores). The Phase 1 failure was laptop-wide memory pressure, not CatBoost itself.

**Pre-registered, batch 3: banked, not submitted (Oct 3, `oci-free`).** Today's 10 slots are spent. These runs are archived now and submitted later with `submit_run.py <tag>` as slots allow. All are strict twins of T8, `{"profile_keys":["flight_distance"],"teacher":true}`, OOF 0.960590, with the gate at +0.00014.

| Tag | Change vs T8 | Purpose / mechanism | Prediction |
|---|---|---|---|
| T10 | `model_seed: 1337` | Seed twin of the champion: the seed floor at the champion, and a near-twin for slot rotation | \|Δ\| < 0.00006 |
| T11 | `seed_bag: 3` | Variance reduction inside each fold (A0b3 gained +0.000184 at A0) | +0.0001 to +0.0002, near the gate |
| T12 | `use_origin_extra: true` | Original rows added to training. The teacher already carries the original's function, and public notebooks report −0.0001 to −0.0004. | below the gate or negative |
| T13 | `fe_rating_stats: true` | Sum, zero-count and minimum of the 13 ratings. Public notebooks report null. | below the gate |
| T14 | `learner: xgb` | Diversity leg for Phase 4. It is judged later by ADD/SWAP, **not by its own score**. | about T8 − 0.00004 |
| T15 | `learner: cat` | Diversity leg (about 1 h on the VM) | about T8 − 0.0001 |

**Results, batch 3** (banked; nothing submitted yet). Δ is against T8 (0.960590), and the gate is +0.00014.

| Tag | run_id | OOF | ΔOOF | Read |
|---|---|---|---|---|
| T10 seed twin | 2212dd7c | 0.960535 | −0.000055 | **held** (\|Δ\| < 0.00006). The seed floor at the champion is 0.000055, against 0.000039 at A0. |
| T11 seed_bag 3 | f19b5e12 | **0.960734** | **+0.000144** | **held. It clears the gate by a hair.** The bag gain repeats at A0's size. |
| T12 + original rows | 949548bc | 0.959646 | **−0.000944** | **held**: clearly negative. The public null holds here, and it's 2–9× worse than public reported. |
| T13 rating aggregates | e295bfa7 | 0.960587 | −0.000003 | **held**: null |
| T14 xgb leg | 4cde76c7 | 0.960678 | +0.000088 | **prediction wrong in sign**: xgb is *above* lgb on this representation |
| T15 cat leg | 636833f2 | 0.960683 | +0.000093 | **prediction wrong in sign**: cat is above lgb too |

**Why the T12 penalty makes sense:** the teacher already delivers the original's decision function in a denoised form. The original rows themselves add a second distribution with different label noise, which the trees then have to fit.

**Early blend read.** This is a quick fixed rank-mean over OOF, *not* the Phase 4 Gate 2. Pool size is held constant, so a cross-family pair is compared with a seed pair.

| Pool | OOF | Compare with |
|---|---|---|
| T8 + T10 (2 lgb seeds) | 0.960712 | — |
| T8 + T14 (lgb + xgb) | 0.960789 | +0.000077 vs the seed pair |
| T8 + T15 (lgb + cat) | 0.960833 | +0.000121 vs the seed pair |
| T11 (3 lgb seeds, bagged inside folds) | 0.960734 | — |
| T8 + T14 + T15 (3 families) | 0.960889 | **+0.000155 vs T11** |

- **Correlations:** the rank correlation between lgb and xgb is 0.985, and between lgb and cat it is 0.982.
- **The family mix beats extra seeds at the same pool size.** That runs against S6E9's "cross-family pools are null", so it's a Phase 4 question to test properly with ADD/SWAP, not a rule yet.
- **These blends are not submittable yet.** A submission needs a `runs.csv` row, and the blend scripts are deferred to Phase 4.

**Submission queue for Oct 4** (the banked singles, every one a paired point): T11, T15, T14, T10, T12, T13. That leaves 4 slots free.

**Pre-registered, batch 4: seed twins to fill the Oct 4 slots (`oci-free`, banked for the nightly scheduler).**
- **What:** T16, T17 and T18 are T8 (`{"profile_keys":["flight_distance"],"teacher":true}`) with `model_seed` set to 7, 2024 and 31337. Each is a strict twin of T8 (OOF 0.960590).
- **Why:** three slots would otherwise go unused tonight (§7). Each twin is also a seed-floor point at the champion on both OOF and public.
- **Prediction:** |ΔOOF| < 0.00006 each, given T10's 0.000055. Public spread across T8/T10/T16–T18 within about ±0.0001.
- **No gate.** These are noise measurements, not candidates.
- **A0arm** is now marked `NOSUBMIT`: it's a parity replay, so its public score would just repeat A0's.

**Results, batch 4** (banked; the scheduler submits them tonight). Δ is against T8 (0.960590).

| Tag | run_id | Seed | OOF | ΔOOF | Read |
|---|---|---|---|---|---|
| T16 | 3c4ec810 | 7 | 0.960616 | +0.000026 | held |
| T17 | 8dc6ed00 | 2024 | 0.960587 | −0.000003 | held |
| T18 | fc3188ab | 31337 | 0.960568 | −0.000022 | held |

- **The seed floor at the champion:** across T8, T10 and T16–T18, OOF has a standard deviation of about 0.00003. The range is 0.000081, from T10 to T16. The gate of +0.00014 is about 4.7 sd, so it still holds.
- **T11 public came back at 0.96025**, which is +0.00003 over T8 against +0.000144 on OOF. That's one point and within public noise, so it's not read as a contradiction yet. The twins give the public seed spread to judge it against tomorrow.

**Pre-registered, batch 5: the original dataset as TE support (plan probe c, `oci-free`, banked).**
- **New axis:** `te_origin_support`, default off. The original's (FD, label) pairs are pooled into every per-value TE statistic. Its rows are never fit on and never encoded. The original's labels carry no information about competition labels, so pooling them in every inner fold is leak-free.
- **T1r** replays T1's archived config on the new code. Its OOF must be **bit-identical** to T1 (`0a155047`), per the refactor rule. It's marked NOSUBMIT.
- **T19** is T1 + `te_origin_support: true`, a strict twin of T1 (OOF 0.960139), gated at +0.00014. It's tested on T1 because that's where FD TE is the active channel. On the T8 champion, FD TE is already null (T7).
  - **Mechanism:** 130k extra pairs shrink per-route noise. But the original's route rates differ from the competition's (3.8% flips, generator drift), so they also add bias.
  - **Prediction: null** (|Δ| < 0.00014). The original's per-key label added +0.000014 in Phase 1.
  - **Falsification:** if T19 clears, the original's per-route rate holds information that per-key matching missed. It would then be re-tested next to the teacher on T8.
- **Plan status after this:** probes (a) T12, (b) Phase 1 matching, and (c) T19 close the original-dataset arm, and the lookup hunt is closed: only FD is a key (§6). So batch 5 ends Phase 2, and Checkpoint 2 follows.

**Results, batch 5** (T19 banked for tonight's scheduler; T1r is NOSUBMIT).

| Tag | run_id | Baseline | OOF | ΔOOF | Read |
|---|---|---|---|---|---|
| T1r parity | 02728c4e | T1 | 0.960139 | 0 | **bit-identical**: max \|Δp\| = 0.0 on all OOF and test rows. The refactor is clean. |
| T19 + origin TE support | 1f8aa334 | T1 | 0.960161 | +0.000022 | **held**: null, as predicted |

- **The original-dataset arm is closed.** The original's labels add nothing as rows (T12, −0.00094), nothing through key matching (Phase 1, +0.000014), and nothing as TE support (T19, +0.00002). Its only useful channel is the **teacher**, which carries its *decision function* rather than its labels (T8, +0.00021).
- **A process note:** a parity replay can't use `--diff-vs`, because the strict-twin check refuses a run that changes zero keys. Run parity replays without it.

### Checkpoint 2 (2026-10-04): the instrument, and which columns are lookups

**Instrument (§4, frozen Oct 3).**
- **Seed floor:** 0.000039 at A0. At the champion it's about 0.00003 (one sd over T8, T10 and T16–T18).
- **OOF→public:** slope 0.971, residual σ 0.000135.
- **Gate:** **+0.00014 OOF.**
- **Readability:** public deltas below about 0.00015 can't be read. The private split resolves about 1.8× better than public.

**Lookups.** `flight_distance` is the only lookup key. It's a **route ID** (residual χ²/df 2.20). Age, both delays and the 13 ratings read as handled magnitudes (χ²/df ≤ 1.01), and T6 confirmed age TE is null. The route gain splits into two parts:
1. **Identity access, about +0.0007.** This is anything that lets a tree address a single route: frequency (T5) or 4096 bins (T4).
2. **Route knowledge, another +0.0007 to +0.0009.** This is either the route's label rate (T1, TE) or its passenger mix (T2, an *unsupervised* profile). The profile already contains the label rate, so T7 adds nothing on top of it.

**The champion is T8:** the FD profile plus the teacher. OOF 0.960590 (+0.00180 over A0), public 0.96022 (+0.00193).
- The teacher is an *interaction*. On raw features it adds +0.00010; next to the route profile it adds +0.00021.

**Closed with nulls:** FD TE on top of the profile, finer bins on top of the profile, rating aggregates (T13), original rows (T12), original key matching, original TE support (T19), age TE.

**Carried to Phase 3/4** (not acted on yet):
- **Seed-bagging** clears the gate at the champion (T11, +0.000144), but its public score is only +0.00003, which is one point and unreadable.
- **xgb and cat legs** on the T8 representation are *above* lgb (+0.00009), and a 3-family rank-mean gives +0.000155 over T11 at the same pool size.
- **Next:** profile variants (other keys, profile statistics) and FD×X pair keys. The cross-family blend has to pass Gate 2 (ADD/SWAP) in Phase 4.

**Slots:** tonight's run (23:15 UTC) sends T18, T17, T16, T19, T15, T14, T13, T12 and T10. That's 9 of the 9 free slots.

### Phase 3: representation build-out (from 2026-10-04)

**Data check before choosing probes** (laptop, light):
- **No FD jitter.** Every train FD value exists in the original dataset.
- **Route sizes:** the median route has 72 rows. Only 1.1% of rows sit in routes with fewer than 30 rows, so a window encoder that borrows from neighbouring distances has almost nothing to work with. It's **deprioritised**.
- **The adjacent-distance label rates correlate at 0.70.** That's a smooth trend that trees already capture.
- **The profile's gap:** across routes, the share of each categorical level varies far more than binomial: Business class **57×**, business travel 20×, loyal customer 17×. The T2 profile averages only the numerics, so the route's *class mix* is missing, even though the code comment claimed it.

**New axes, default off:** `profile_cat_shares`, `profile_stats`, `profile_pair_keys`.

**Pre-registered, batch 6** (banked for the nightly scheduler). Every run is a strict twin of T8 (0.960590), gated at **+0.00014**. T8r replays T8 on the new code and must be bit-identical; it's NOSUBMIT.

| Tag | Change vs T8 | Mechanism | Prediction |
|---|---|---|---|
| T20 | `profile_cat_shares: true` | The route's passenger mix *by category* (57× / 20× / 17× over binomial). A row's own class is visible, but its route's class mix isn't. | **clears**, +0.0002 to +0.0006 |
| T21 | `profile_stats: ["std"]` | Route heterogeneity, a second moment, is weaker than the mean | below the gate |
| T22 | `profile_pair_keys: [[FD, travel_class]]` | Route × class is a fare product. A Business passenger's satisfaction depends on that route's Business product, which the route-wide mean blends with Economy. | borderline, +0.0001 to +0.0003 |

**Falsification:** if T20 is null, the categorical mix is already recoverable from the numeric means (ratings correlate with class), and "who flies it" is saturated.

**Stop rule (plan):** three consecutive gate failures close an axis.

**Results, batch 6** (banked). Δ is against T8 (0.960590); the gate is +0.00014.

| Tag | run_id | OOF | ΔOOF | Read |
|---|---|---|---|---|
| T8r parity | a15c7e3c | 0.960590 | 0 | **bit-identical** (max \|Δp\| = 0.0 on OOF and test). NOSUBMIT. |
| T20 categorical shares | d981cc08 | 0.960539 | −0.000051 | **prediction failed**: null |
| T21 per-route std | f27d4805 | 0.960589 | −0.000001 | held: null |
| T22 route × class profile | 7a4d0a7c | 0.960500 | −0.000090 | prediction failed: null to slightly negative |

**Why T20 failed: a measurement mistake worth keeping.**
- The 57× "over binomial" figure measured the *total* across-route spread of the Business share. Most of that spread is a **smooth function of distance**, because long routes carry more Business passengers.
- On routes with at least 50 rows, distance alone predicts the route's Business share with **R² 0.92** (shuffled 5-fold, HistGBR). The 16 numeric means give 0.92, and both together give **0.97**. For travel type the figures are 0.78, 0.88 and 0.91; for loyal customers, 0.78, 0.80 and 0.88.
- The tree already sees raw distance and the means, so the shares told it nothing new.
- **Rule:** a candidate feature has to vary more than noise *after conditioning on what the model already has*. That's the residual χ² logic from Phase 1, applied to features. The unconditioned binomial ratio is the wrong test.

**Stop rule fired: the profile-variants axis is CLOSED.** T20, T21 and T22 are three consecutive gate failures.

**What's left of the Phase 3 list.** It is mostly the TE family on FD (multi-smoothing, quantile ladders, digits, window). Phase 2 already showed that family is saturated next to the profile: TE added nothing (T7) and finer bins added nothing (T9). These are **not run blind**, and each would need a conditioned mechanism first. The one lever that showed an *interaction* with the route features is the teacher (T8), which is still untuned. That's next.

**Public scores from the Oct 4 scheduler run** (9 submitted at 23:15 UTC, no slot wasted). Read with `submit_run.py --fill-missing` on Oct 5.

| Group | OOF | Public |
|---|---|---|
| T8 + seed twins T16/T17/T18 | 0.960568–0.960616 | 0.96022–0.96033 (range **0.00011**, sd about 0.00005) |
| T11 seed_bag 3 | 0.960734 | 0.96025: **inside the seed band**, so its +0.000144 OOF is unconfirmed on public |
| T14 xgb leg / T15 cat leg | 0.960678 / 0.960683 | **0.96039 / 0.96040**, +0.00012 over the T8-twin mean, the same direction as OOF (+0.00009) |
| T19 / T20 / T21 / T22 | as above | 0.95977 / 0.96028 / 0.96026 / 0.96021, all consistent with their OOF nulls |

- **Public seed spread at the champion is 0.00011.** That's the yardstick for every public delta from here on.
- **The best single models are the xgb and cat legs on T8's representation.** They're still below the gate on OOF.

**Teacher diagnostic** (`scripts/diag_teacher_rounds.py`, run on `oci-free`). Selection used only the original dataset's own CV; the competition AUC is shown for reference.

| Rounds | 250 | 500 | 1000 | 1500 (current) | 3000 | 5000 |
|---|---|---|---|---|---|---|
| Original CV AUC | 0.99500 | **0.99519** | 0.99513 | 0.99505 | 0.99486 | 0.99470 |
| vs competition labels | 0.95471 | 0.95434 | 0.95426 | 0.95426 | 0.95420 | 0.95433 |

The curve is flat. The original's function is learned almost perfectly at any round count, so tuning rounds can't be the lever.

**Pre-registered, batch 7** (`oci-free`, banked for the nightly scheduler).
- **New axes, default off:** `teacher_profile` (the teacher also sees the FD profile, mapped onto original rows by route), `teacher_learner` (`lgb`/`cat`), `teacher_cat_params`.
- **T8r2** replays T8 on the new code and must be bit-identical; it's NOSUBMIT.
- **T23–T25** are strict twins of T8, gated at **+0.00014**.
- **T26 and T27** are seed twins of the T14 and T15 legs. They aren't gated: they set the xgb/cat seed floor and give Phase 4 a second leg per family.

| Tag | Change | Mechanism | Prediction |
|---|---|---|---|
| T23 | `teacher_profile: true` | The teacher's value is an interaction with route (+0.00010 raw vs +0.00021 next to the profile). A route-aware teacher learns the original's function per route. | borderline, +0.0001 to +0.0003 |
| T24 | `teacher_learner: cat` | The original function is learned almost perfectly (CV AUC 0.995), so a second learner family reproduces it | below the gate |
| T25 | `teacher_params.n_estimators: 500` | The best original-CV setting, but the curve is flat | below the gate (closes the rounds sub-axis) |
| T26 | T14 + `model_seed: 7` | xgb seed floor | \|Δ\| < 0.00006 vs T14 |
| T27 | T15 + `model_seed: 7` | cat seed floor | \|Δ\| < 0.00006 vs T15 |

**Falsification:** if T23 is null, the teacher-route interaction is already fully formed downstream (the competition model combines the teacher logit with the route profile itself), and the teacher axis is done.

**Results, batch 7** (banked). Δ is against T8 (0.960590) unless noted; the gate is +0.00014.

| Tag | run_id | OOF | ΔOOF | Read |
|---|---|---|---|---|
| T8r2 parity | 8af8446e | 0.960590 | 0 | bit-identical OOF (NOSUBMIT) |
| T23 route-aware teacher | 1af30fff | 0.960558 | −0.000032 | **prediction failed**: null. The teacher × route interaction already forms downstream. |
| T24 CatBoost teacher | 6363a28e | 0.960650 | +0.000060 | held: below the gate |
| T25 500-round teacher | 81ae7cd7 | 0.960580 | −0.000010 | held: null |
| T26 xgb seed twin | 9ed7ebee | 0.960672 | −0.000006 vs T14 | held: xgb seed floor about 0.00001 |
| T27 cat seed twin | f9677b17 | 0.960715 | +0.000032 vs T15 | held: cat seed floor about 0.00003 |

**Stop rule fired: the teacher axis is CLOSED** (T23, T24 and T25 failed the gate in a row).

**The conditioned residual test** (`scripts/residual_lookup.py`, new). It computes χ²/df of per-value OOF residuals against calibrated noise, for each column and for FD pairs.

| Key | A0 | T6 (A0 + age TE) | T3 (teacher) | T2 (profile) | T8 | T14 xgb | T15 cat |
|---|---|---|---|---|---|---|---|
| `flight_distance` | **2.91** | 2.95 | 3.01 | 0.84 | 0.97 | 0.88 | 1.01 |
| FD × class/type/customer | 2.4–2.6 | 2.5–2.6 | 2.5–2.6 | 0.84–0.91 | 0.98–1.02 | 0.90–0.95 | 1.02–1.07 |
| `age` | 1.77 | **1.12** | 2.50 | 2.55 | **2.77** | 2.53 | 2.39 |

- **The route lookup is fully used.** FD and every FD pair sit at about 1.0 under the champion. That's evidence that the FD representation is complete, not just that the later probes missed. It's also why the TE family on FD (multi-smoothing, ladders, digits, window) is not run.
- **Age is the only signal left.** χ² ≈ 191 on 69 df (p ≈ 1e-13). Removing a smooth trend only takes it to 2.66, so it's value-specific: a sawtooth, sharpest above age 60 (61 +, 62 −, 63 +, 64 +, 65 −, 66 +, 67 −, 68 +, 69 −).
- **Where it comes from:** it **surfaces as the model gets stronger elsewhere** (A0 1.77 → T2 2.55 → T8 2.77). The teacher doesn't create it.

**Pre-registered, batch 8** (`oci-free`, banked). Strict twins of T8, gate +0.00014.

| Tag | Change | Prediction |
|---|---|---|
| T28 | `te_cols: [age]` | **below the gate** (excess χ² is about 120, against FD's 4,300 at A0, which bought +0.0016, so expect about +0.00005). Age χ²/df should fall to about 1. |
| T29 | `num_as_cat: [age]` | below the gate, at most T28 |

**Falsification:** if T28 clears, the gain from removing residual χ² is not proportional to the excess. The sizing rule would then be wrong, and age becomes a Phase 3 axis worth building out.

**Results, batch 8** (banked). Δ is against T8 (0.960590).

| Tag | run_id | OOF | ΔOOF | Age χ²/df | Read |
|---|---|---|---|---|---|
| T28 age TE | f29ba752 | 0.960557 | −0.000033 | 2.77 → 1.99 | held: below the gate |
| T29 age as categorical | 757b4901 | 0.960244 | **−0.000346** | 2.77 → **0.05** | held, and it **erased** the age residual while *losing* AUC. FD residuals rose to 1.24. |

**T29's lesson:** the residual χ² measures *per-group calibration in the mean*. A model can drive it to zero, here with 69 categorical age levels, and still rank worse, because those splits cost it elsewhere. A high residual χ² is a candidate signal, not a promised AUC gain. The sizing rule (excess χ² → expected gain) held: about +0.00005 was predicted, and ≤ 0 is what came back.

### Checkpoint 3 (2026-10-05): the champion recipe and the closed axes

**Champion recipe: the T8 representation.**
- **Config:** `{"profile_keys":["flight_distance"],"teacher":true}`. That's the FD route profile (means of the 16 other numerics plus log size, over train+test) and a LightGBM teacher fit on the original dataset.
- **Scores:** OOF 0.960590 with lgb; public 0.96022, with seed twins spanning 0.96022–0.96033.
- **Best singles on this representation:**
  - cat legs T15/T27: 0.960683 / 0.960715 OOF, public 0.96040;
  - xgb legs T14/T26: 0.960678 / 0.960672 OOF, public 0.96039.

**Evidence that representation is done, not just stalled:** under the champion, `residual_lookup.py` puts FD and every FD pair at χ²/df ≈ 1.0 (A0: 2.9). The only residual left is age, a calibration-type signal that two probes showed is not worth AUC.

**Closed axes:**

| Axis | Probes | Why it's closed |
|---|---|---|
| Profile variants (categorical shares, std, route × class) | T20–T22 | Stop rule. The categorical mix is a smooth function of distance plus the means (R² 0.97). |
| Teacher (route-aware, CatBoost teacher, rounds) | T23–T25 + diagnostic | Stop rule. The original's function is learned at CV AUC 0.995 at any setting. |
| FD TE family (TE, multi-smoothing, ladders, digits, window) | T7, T9 + residual test | FD residual χ²/df ≈ 1 under the champion |
| Original dataset as labels (rows, key matching, TE support) | T12, Phase 1, T19 | Only the teacher channel carries it |
| Age (TE, categorical) | T6, T28, T29 | A calibration residual; no AUC |
| Rating aggregates | T13 | null |

**Carried to Phase 4:**
- **Family legs:** cat and xgb are above lgb on this representation. Their seed floors are about 0.00003 (cat) and about 0.00001 (xgb).
- **Early read:** a 3-family rank-mean gave +0.000155 over T11 at the same pool size. It has to pass Gate 2 (ADD/SWAP) properly.
- **seed_bag (T11)** is +0.000144 on OOF but unconfirmed on public.
- **RealMLP**, which public notebooks report at +0.0003 in stacks, is an untested neural leg.

**Slots:** tonight sends T29, T28, T27, T26, T25, T24, T23, T13, T12 and T10, filling 10 of 10.

**Tooling (Oct 4).** Three new skills:
- `kaggle-cli` (this repo);
- `oci-vm` (user-level);
- `submission-scheduler` (this repo, together with `scripts/submit_scheduler.py` and the cron entry on `oci-free`).

The submit guards in `submit_run.py` were moved into a shared `eligible()` function, and `tests/test_submit_scheduler.py` covers the pick order. This is plumbing only: no probe or OOF changed.

`scripts/sync_runs.py` (`pull` / `push <tag>`) merges runs.csv between the laptop and the VM by run_id, and copies the preds dirs along with their rows. Never scp one runs.csv over the other.

### Phase 4: diversity and blending (from 2026-10-05)

**Tooling.** `scripts/blend.py` builds fixed, unfitted `rank_mean` blends from archived OOF/test pairs.
- Each leg's OOF is turned into ranks over the whole OOF, and the legs get equal weight.
- AUC only sees order, so ranking removes each family's calibration scale without fitting anything.
- `gate2` reports ADD and SWAP for a candidate leg, each with a paired-bootstrap SD over 200 resamples. Its O(n) weighted AUC matches `roc_auc_score` exactly, ties and weights included.
- `write` archives a blend as an ordinary run (`learner=blend`, `manifest.json` listing the legs), so `submit_run.py` and the nightly scheduler treat it like any other run.

**Gate 2, defined before any blend was scored.**
- ADD(c | P) = AUC(P + c) − AUC(P).
- SWAP(c | P) = AUC(P − w + c) − AUC(P), where w is P's weakest leave-one-out contributor. The pool size stays constant, so a gain can't come just from one more averaged member.
- **A leg passes when ADD > 2 × its paired SD and SWAP > 0.**
- **The shipping gate is unchanged:** a blend replaces the champion only at **≥ +0.00014 OOF over T8 (0.960590)**. It's also reported against the best single leg (T27, 0.960715).

**Pre-registered, batch 9.** Laptop only, with no training, over legs that are already archived.

| ID | Test | Prediction |
|---|---|---|
| B9a | `score` on {T8, T14, T15}, the 3-family rank_mean | OOF 0.96085–0.96092. That clears the shipping gate over T8 (+0.00025 or more) and beats T27 by +0.00014 or more. |
| B9b | Leave-one-out within {T8, T14, T15} | All three contribute more than 0. lgb (T8) is the weakest. |
| B9c | `gate2` with seed twins T16, T26, T27 as candidates | ADD is small (+0.00001 to +0.00005). SWAP is at most 0 except for a cat twin replacing lgb. **No twin passes** (seed noise isn't diversity). |
| B9d | `gate2` with the representation near-twins T7, T9, T11, T13, T20–T25 and T28 as candidates | **None passes.** T24 (CatBoost teacher, a different teacher function) is the only one with a mechanism, and is borderline at best. |
| B9e | `gate2` with weaker or other-representation legs T1, T2, A0, X1, K1 and G1 as candidates | None passes, with ADD ≤ 0. Each is below the pool floor (playbook §6). |
| B9f | `stack` on {T8, T14, T15}: a nested-CV logit stack against rank_mean | The stack washes, \|Δ\| < 0.00003. That's the 5th S6E9-style wash; the legs are near-twins, so there's no correlated error to subtract. |
| B9g | `disagree --ref T15 --all` | Spearman(solo, disagreement) ≤ −0.7. No leg is both strong (≥ 0.9605) and decorrelated (more than 2× the seed-twin disagreement). |

**Falsification:**
- If B9a lands under +0.00014 over T8, cross-family diversity on a single representation is too small to ship, and Phase 4 has to find a **second representation** rather than more learners.
- If B9d finds a passing near-twin, solo-null representation probes carry blend value, and the closed axes have to be re-read as legs.

**Results, batch 9.** All are OOF; the paired SDs come from 200 bootstrap resamples.

| ID | Result | Read |
|---|---|---|
| B9a | {T8, T14, T15} = **0.960889**: +0.000299 over T8, +0.000174 over T27 | **held**, archived as **B1** (`69b9c704`) |
| B9b | Leave-one-out: T8 +0.000028, T14 +0.000056, T15 +0.000101 | held: lgb is the weakest |
| B9c | Seed twins: T16 / T26 fail. T27 has ADD +0.000029 (sd 0.000008), but its SWAP is +0.000002 (sd 0.000017). | **failed by the letter** (T27 "passes"). In substance it's a seed-bag of the dominant cat leg, and its SWAP is 0.1 SD. |
| B9d | Near-twins on the 3-pool: **T7 +0.000115, T9 +0.000095, T24 +0.000081, T23 +0.000046, T21 +0.000033 pass.** T11, T13, T20, T22, T25 and T28 fail. | **prediction failed** |
| B9e | **T1 +0.000141 (7 SD, the largest of all)** and T2 +0.000102 pass. A0, X1, K1, G1 and T12 fail. | **failed for T1/T2**, held for the raw-feature legs |
| B9f | Nested logit 0.960885 against rank_mean 0.960889 (Δ −0.000005, sd 0.000007). Weights 0.26 / 0.29 / 0.45. | held: the stack washes |
| B9g | Spearman(solo, disagreement vs T15) = **−0.954** (n = 43) | held |

**Why B9d and B9e failed, and the mechanism that replaces them.**
- Every leg that passed represents the **route** differently from T8:
  - T1: FD as a *label rate*, with no profile and no teacher;
  - T2: the profile with no teacher;
  - T7: profile plus TE;
  - T9: profile with 4096 bins;
  - T24: a *different teacher function*.
- Legs that pass on the 3-pool differ from T8 only in what was added *beside* the route (T13 ratings, T20 shares, T28 age, T11 seeds), and they fail.
- Different route representations make **different route errors**, and averaging cancels them.
- **T1 is weaker solo (0.96014) but the best leg**, because its route errors are the least correlated with T8's (disagreement 0.036 against about 0.018).
- This is playbook §7's "the representation is the unit of ensemble value". The **pool-floor rule** (§6: a leg below the pool's weakest solo contributes nothing) is **wrong here**. It holds for a weaker leg on the *same* representation (A0, X1, K1), not for a weaker leg on a different one.

**Sequential Gate 2 (mechanical membership, not greedy):**
- **{T8, T14, T15} + T1:** 0.961030.
- **+ T24:** ADD +0.000043 (sd 0.000010), SWAP +0.000052, giving **0.961073**.
- T7 also passed by the letter, but its SWAP was +0.000001.
- **On the 5-pool nothing passes**: every candidate has ADD ≤ +0.000018.
- **B2** (`ead789bd`) = rank_mean{T8, T14, T15, T1, T24} = **0.961073, which is +0.000483 over T8 and clears the shipping gate.**

**Gate 2 weakness, noted and not changed mid-batch:**
- "SWAP > 0" lets through a SWAP of 0.1 SD (T27, T7).
- From batch 10 on, **SWAP must also exceed −1 SD and ADD must exceed 2 SD.** Both have to hold, so a pure noise SWAP can't carry a leg that has no real ADD.
- In practice ADD > 2 SD did the work in every batch 9 decision.

**Pre-registered, batch 10: the TE route representation, built out across families** (`oci-free`, banked). Batch 9 found that the TE route representation is the most valuable second view. But it exists only as T1: lgb, no teacher. Every leg below is gated by Gate 2 on the B2 pool {T8, T14, T15, T1, T24}. Gate 1 (solo against its strict twin) is reported for information.

| Tag | Config (strict twin of) | Mechanism | Prediction |
|---|---|---|---|
| U1 | `te_cols:[FD], teacher:true` (T1 + teacher) | The teacher interacts with route knowledge (T3 +0.00010 raw, T8 +0.00021 next to the profile) | solo ≈ 0.9603–0.9604. On the B2 pool: **SWAP for T1 > 0**, a stronger leg on the same view. |
| U2 | U1 + `learner: xgb` | A family leg on the second view, as T14 was on the first | solo ≥ U1. **Passes** Gate 2. |
| U3 | U1 + `learner: cat` | Same, with cat. cat was the strongest contributor on view 1. | solo ≥ U1 + 0.00005. **Passes.** |
| U4 | T15 + `num_as_cat:[FD]` | A third route view: CatBoost's ordered target statistics and FD×category CTRs, inside the booster | borderline. Solo T15 ± 0.00005, ADD about +0.00002 to +0.00005. |
| U5 | U1 + `te_pseudo: true` | Test-row soft labels added to the per-route TE support. A variance cut for the route rate. | solo within ±0.00005 of U1. **Fails** Gate 2 next to U1 (a near-twin). |

**Falsification:**
- If U2 and U3 fail with U1 in the pool, the second view's value is saturated by one leg, and the next lever is a **third view** (U4), not more families.
- If U1 doesn't beat T1 in the blend, the teacher's value is already carried by the view-1 legs.

**Slots tonight:** B2 and B1 go first in the queue. They're the first public read on blending. Auto-fill takes the 8 remaining slots.

**Interim results, batch 10** (U1 and U5; Gate 2 on the B2 pool).

| Tag | run_id | Solo | Gate 1 | ADD | SWAP | T1→leg in B2 | Read |
|---|---|---|---|---|---|---|---|
| U1 | c736b2cf | 0.960247 | +0.000108 vs T1 | −0.000023 | −0.000027 | **−0.000079** (sd 0.000013) | **prediction failed** |
| U5 | 65ffd758 | 0.960210 | −0.000037 vs U1 | −0.000027 | −0.000029 | −0.000085 | held (fails) |

**Why U1 failed: the teacher is shared.**
- Every view-1 leg (T8, T14, T15, T24) carries a teacher logit.
- Adding it to the TE view raised the leg's solo score (+0.000108), but it also moved the leg toward view 1. Disagreement with T15 fell from 0.036 (T1) to 0.029 (U1).
- In a blend, the shared component's errors no longer cancel.
- **Rule:** a leg's blend value comes from what it *doesn't* share with the pool. Improving a diversity leg with the pool's own best feature can destroy its value.
- Also: inside B2, T8's leave-one-out is now −0.000009 (B2 − T8 = 0.961082, +0.000009, sd 0.000008, 1 SD). It's inside the noise, so **membership is not re-fitted.**

**This changes what U2 and U3 test.** They're left running as pre-registered. The mechanism above now predicts that both **fail**, because they carry the teacher too.

**Pre-registered, batch 11** (`oci-free`, queued straight after batch 10). These are the family legs on the second view *as it actually earned its place*: T1's representation, with no teacher. Gate 2 is on the B2 pool, with ADD > 2 SD and SWAP > −1 SD.

| Tag | Config (strict twin of T1) | Prediction |
|---|---|---|
| V2 | T1 + `learner: xgb` | borderline, ADD about +0.00001 to +0.00003 (xgb added least on view 1) |
| V3 | T1 + `learner: cat` | **passes**, ADD > 2 SD (cat is the strongest family and shares no teacher) |

**Falsification:** if V3 fails too, a single leg (T1) saturates view 2, and B2 sits at the elbow for the representations we have. The next lever would then be a genuinely new view, not more families.

**Results, batches 10–11** (complete). Gate 2 is on the B2 pool, or on B2 with U4 in place of T15 where noted.

| Tag | run_id | Solo | Public | Gate 2 | Read |
|---|---|---|---|---|---|
| U2 xgb on U1 | 94b087c0 | 0.960333 | 0.95960 | ADD −0.000009: fail | held (shared teacher) |
| U3 cat on U1 | 1c1c3d66 | 0.960354 | 0.95974 | ADD −0.000003: fail | **prediction failed** (prediction was pass). Explained by the U1 mechanism. |
| **U4** T15 + FD categorical | 17c85549 | **0.960835** (Gate 1 **+0.000152** vs T15) | **0.96056** | ADD +0.000030 (4 SD), SWAP +0.000043 (4 SD): **PASS** | prediction failed **upward**. **Best single.** |
| V2 xgb on T1 | 2ef8460b | 0.960135 | 0.95962 | on the U4 pool: ADD −0.000004: fail | held |
| V3 cat on T1 | 9f93364b | 0.959894 | 0.95959 | ADD +0.000016 (1.3 SD): fail | **prediction failed.** Falsification branch: **T1 alone saturates view 2.** |

**Public read on blending** (Oct 5 slots): B1 0.96044 and **B2 0.96053**, against T8 0.96022 (+0.00031, same direction as OOF +0.00048). U4 0.96056.

**U4's mechanism.** CatBoost turns a categorical into ordered target statistics *and* builds combination CTRs (FD × class, FD × travel type) on the fly. That's the route × class lookup T22 tried to build as an lgb profile, which failed. The difference is that CatBoost estimates it with ordered (leak-free) statistics, inside the booster.

**B3** = rank_mean{T8, T14, **U4**, T1, T24} = **0.961106**. It's B2 with T15 swapped for U4 (+0.000033, sd 0.000008). This is the **champion blend**. No remaining candidate passes on it (T15 and T27 fail as well).

**Pre-registered, batch 12: FD as a native categorical, across families and as its own view** (`oci-free`, banked). Gate 2 is on the B3 pool (ADD > 2 SD, SWAP > −1 SD), and Gate 1 is against the strict twin named.

| Tag | Config (strict twin of) | Mechanism | Prediction |
|---|---|---|---|
| U6 | T14 + `num_as_cat:[FD]` | xgb's partition-based categorical splits as the route identity | Gate 1 null (xgb has no CTRs; FD identity is already in the profile). Fails Gate 2. |
| U7 | T8 + `num_as_cat:[FD]` | lgb gradient-grouped categorical | Gate 1 null to negative (T29 lesson: high-cardinality categoricals overfit in lgb). Fails. |
| U9 | U4 − `profile_keys` | CatBoost's CTRs as the *only* route knowledge: a third view, without the profile | solo below U4 (about 0.9605). Gate 2 borderline: **passes only if CTR route errors differ from profile route errors.** |
| U4s / U4s2 | U4 + `model_seed` 7 / 1337 | cat seed floor at the new best leg | \|Δ\| < 0.00005 vs U4. As blend legs, they fail Gate 2 (B9c). |

**Falsification:** if U6 or U7 clears Gate 1, FD-as-categorical is a family-general representation gain, not a CatBoost-CTR effect. Phase 3's "representation done" call would then be wrong for categorical encodings.

**Oct 8 slots.**
- **B3 submitted: public 0.96056.** B2 was 0.96053, so +0.00003 is unreadable, consistent with the +0.000033 OOF.
- **The Oct 7 slots were lost (0 of 10).** No runs were banked while Phase 4 was paused, and the scheduler correctly had nothing to send. **Lesson: bank filler before any pause.**
- Banked near-twin blends of B3 (OOF): **B3s** 0.961101 (seed-twin legs), **B3c** 0.961102 (+T15), **B3m** 0.961121 (−T8), **B3x** 0.961083 (+3 seed twins). All are within ±0.00002 of B3, so their public spread measures blend noise.
- Batch 12's 5 runs, plus these 4, fill the 9 slots that are left.

**Batch 12, first result:** U7 (`f1290490`, T8 + FD as an lgb categorical) = **0.959408, −0.00118 against T8.** That matches the prediction: lgb overfits a 3.5k-level categorical, the same failure as T29.

**Results, batch 12** (complete). Gate 2 is on the B3 pool {T8, T14, U4, T1, T24} = 0.961106.

| Tag | run_id | Solo | Gate 1 | ADD (sd) | SWAP (sd) | Read |
|---|---|---|---|---|---|---|
| U7 lgb FD-cat | f1290490 | 0.959408 | −0.00118 vs T8 | −0.000018 (0.000013) | −0.000009 | held (fails) |
| U6 xgb FD-cat | f86c8c1f | 0.958229 | −0.00245 vs T14 | −0.000036 (0.000016) | −0.000043 | held (fails; worse than predicted) |
| **U9** CTR-only route | cbf31660 | **0.960831** | −0.000004 vs U4 | **+0.000052 (0.000008)** | **+0.000067** | **PASS, 6.5 SD.** Prediction (borderline, solo ≈ 0.9605) failed **upward**. |
| U4s seed 7 | 2c7c01dc | 0.960849 | +0.000014 | +0.000024 (0.000007) | +0.000040 | passes by the letter |
| U4s2 seed 1337 | 9399f1b0 | 0.960801 | −0.000034 | +0.000019 (0.000007) | +0.000036 | passes by the letter |

**Read.**
- U9 matches U4's solo score with no profile at all. CatBoost's CTRs fully replace the hand-built route profile. Its disagreement with U4 is 0.0147, against 0.0071 for the seed twin U4s. **The CTR route view and the profile route view make different errors.** That's a third route view.
- The seed twins of U4 pass (B9c said they wouldn't). U4 is now the dominant leg, so seed-bagging it is a variance cut on the leg that carries the most weight. They are recorded as a seed bag, not as diversity.
- T8's leave-one-out in B3 is now −0.000016: it's the weakest leg, and it's the SWAP target. B3 + U9 (6 legs) = 0.961158.

**Pre-registered, batch 13: the CTR route view** (blends on the laptop; U10, U13, U14, U11 on `oci-free`; U15, U16, U17 on Kaggle kernels). Gate 2 is on the **B4 pool {T14, U4, U9, T1, T24}** (ADD > 2 SD, SWAP > −1 SD). Gate 1 is against the strict twin named.

| Tag | Config (strict twin of) | Mechanism | Prediction |
|---|---|---|---|
| **B4** | blend {T14, U4, U9, T1, T24} | B3 with U9 in place of T8 (the SWAP that passed) | OOF ≈ 0.96117. Public within the noise of B3. |
| B4x | B4 + U4s + U4s2, if both pass on the B4 pool | Seed bag of the dominant cat leg | ADD ≈ +0.00002 |
| U10 | U9 − `teacher` | The pure CTR view. The U1 lesson says the shared teacher costs diversity. | solo about 0.0002 below U9. Gate 2 ADD ≥ U9's on B3 if the teacher was the shared part. |
| U13 | U9 + `teacher_learner: cat` | A different teacher function on the CTR view (T24 analogue) | solo ±0.00005. Fails Gate 2 next to U9. |
| U14 | U9 + `model_seed: 7` | Seed floor at the new leg; seed-bag material | \|Δ\| < 0.00005 vs U9 |
| U11 | U4 + `num_as_cat: [FD, age]` | Age CTRs inside CatBoost (T29 failed only in lgb) | Gate 1 null (age residual χ² 0.52) |
| U15 | U4 + `cat_params.depth: 8` | More capacity for CTR combinations at the best leg | Gate 1 ±0.0001 |
| U16 | U4 + `seed_bag: 3` | In-leg seed bag at the best single (T11 analogue) | +0.00003 to +0.00005 over U4 |
| U17 | U4 + `cat_params` lr 0.04, 6000 iterations | A lower learning rate at the best leg (best_iter 520–1030, far from the cap) | Gate 1 +0.00002 to +0.00008 |
| B5 | Sequential Gate 2 over the batch-13 legs on the B4 pool | Mechanical membership | ≥ B4 |

**Falsification:**
- If U10 passes Gate 2 with a larger ADD than U9, the "teacher is shared" rule generalises across views, and diversity legs drop the teacher.
- If B4 doesn't beat B3 + U9 (0.961158), T8 still carries something, and B5 keeps it.

**Slots:** B4 goes first in tonight's queue and displaces U4s2 to Oct 9. On Oct 9: U4s2, plus B4x, B5 and the seven probes.

**Batch 13, first results** (laptop, no training).
- **B4** (`fb906814`) = **0.961173**. That's +0.000067 over B3, +0.000583 over T8, and above B3 + U9 (0.961158). The prediction (≈ 0.96117) held. **B4 is the champion blend**, and it's queued first for tonight.
- On the B4 pool, U4s (ADD +0.000005), U4s2 (+0.000001) and T8 (−0.000015) all **fail**. U9 already gives the pool a second cat leg, so the seed-bag value the twins had on B3 is used up. **B4x is not built**, and its slot goes to the next leg from batch 13 that passes. The falsification for B4 didn't trigger: T8 stays out.

**Results, batch 13** (complete; Gate 2 on the B4 pool {T14, U4, U9, T1, T24} = 0.961173). U16 and U17 finished on Kaggle but their pollers died. They were logged with the new `collect_run.py --adopt`, which runs the same run_id/code_sha echo checks but skips the push.

| Tag | run_id | Solo | Gate 1 | Public | ADD (sd) | SWAP (sd) | Read |
|---|---|---|---|---|---|---|---|
| U10 pure CTR (no teacher) | b7f0c8a9 | 0.960278 | −0.000553 vs U9 | 0.96014 | +0.000030 (0.000012) | −0.000003 | fails on B4. Prediction (solo about 0.0002 below U9) failed: it's 0.00055 below. |
| **U11** U4 + age as cat | 00a6fe3c | **0.961009** | **+0.000174 vs U4** | **0.96066** | **+0.000040 (0.000007)** | +0.000039 | **PASS.** Prediction (Gate 1 null) **failed upward**. It's the best single, and it clears the shipping gate on its own. |
| U13 U9 + cat teacher | 6ea54be6 | 0.960881 | +0.000050 vs U9 | 0.96027 | +0.000039 (0.000007) | +0.000017 | **PASS.** Prediction (fails next to U9) failed. |
| U14 U9 seed 7 | fc4a9df0 | 0.960867 | +0.000036 | 0.96036 | +0.000017 | −0.000001 | held (seed floor; fails) |
| U15 U4 depth 8 | cd1698fa | 0.960762 | −0.000073 | 0.96060 | −0.000001 | −0.000010 | held (±0.0001); fails |
| U16 U4 seed_bag 3 | dfa83498 | 0.960951 | +0.000116 | pending | +0.000008 | +0.000003 | the bag gain is above the predicted range, but it fails Gate 2: B4 already averages two cat legs |
| U17 U4 lr 0.04 | 2222cda3 | 0.960889 | +0.000054 | pending | +0.000002 | −0.000006 | held (+0.00002 to +0.00008); fails |

**Public, Oct 8:** B4 **0.96064** (B3 0.96056: +0.00008, same sign as OOF +0.000067). U11 **0.96066**, the best public score (U4 0.96056: +0.00010, same sign as OOF).

**Why U11 works (mechanism, provisional).** Age's residual χ²/df was 0.52, so the model wasn't missing any *per-value* age information. The gain has to come from what CatBoost does with a categorical beyond its own target statistic: **combination CTRs** (age × class, age × travel type, age × FD). That's the same mechanism U4 found for FD. It also explains why T29 (age as an lgb categorical) failed: lgb gets the identity but not the combinations. The residual-χ² diagnostic tests only marginal lookups, so it **can't see interaction lookups**. A null χ² doesn't close a column for CatBoost.

**B5, sequential Gate 2 on the B4 pool.** U11 passes, ADD +0.000040 (5.7 SD). With U11 in, U4's leave-one-out is −0.000005, so U11 **replaces** U4 (it's U4 plus age). On {T14, U11, U9, T1, T24}, U13 passes (ADD +0.000031, 4.4 SD, SWAP +0.000017). After that, nothing passes: U4, U10, U14, U16 and U17 all fail. **B5** (`d2de2320`) = rank_mean{T14, U11, U9, T1, T24, U13} = **0.961249**, +0.000076 over B4. That's under the shipping gate against B4, so B5 is a candidate, not yet a confirmed champion.

**Pre-registered, batch 14: other numerics as CatBoost categoricals** (Kaggle kernels, banked). Gate 2 is on the **B5 pool** (ADD > 2 SD, SWAP > −1 SD). Gate 1 is against the strict twin named.

| Tag | Config (strict twin of) | Mechanism | Prediction |
|---|---|---|---|
| U18 | U9 + `num_as_cat: [FD, age]` | Age CTRs on the CTR-only view: does U11's gain carry to the other cat leg? | Gate 1 +0.0001 to +0.0002 vs U9. On B5, a SWAP for U9 > 0. |
| U19 | U11 + `seed_bag: 3` | The two batch-13 gains combined (U16's bag gain was +0.000116 at U4) | +0.00005 to +0.00012 over U11. Fails Gate 2 next to U11 (U16 did next to U4). |
| U20 | U11 + `dep_delay` in num_as_cat (167 levels) | Delay × class CTRs | Gate 1 ±0.00005. Delay is monotone in satisfaction, so there's little interaction left to find. |
| ~~U21~~ → **U24** | U11 + `teacher_learner: cat` | U21 dropped before launch: a NaN `arr_delay` level crashes CatBoost (the cat branch passes NaN categories through, and fixing it is a learner edit that needs a bit-identity re-check). U24 applies U13's passing change (a cat teacher) to the best single. | Gate 1 ±0.00005 vs U11 (T24 was −0.00002 vs T8). On B5, it's a SWAP for U11 that passes only if the teacher function, not the age CTRs, made U13's errors different. |
| U22 | U11 + the 13 RATINGS in num_as_cat | Rating × class / rating × travel-type CTRs. Ratings are the strongest features. | Gate 1 +0.00005 to +0.00015. **The largest open question.** |
| U23 | U11 + `model_seed: 7` | Seed floor at the new best single | \|Δ\| < 0.00005 vs U11 |
| B6 | Sequential Gate 2 of U18–U23 on the B5 pool | Mechanical membership | ≥ B5 |

**Falsification:**
- If U20, U21 and U22 are all null, the CTR gain is specific to columns with **many levels and non-monotone effects** (FD, age), not to "numerics as categoricals".
- If U18 fails Gate 1, age's CTRs need the profile beside them, so the gain isn't independent of view 1.

**Oct 9 slots (queue order):** B5, U16, U17, U18, U19, U20, U22, U23, U24, B6 (11 candidates for 10 slots; the last probe to bank goes to Oct 10). Unqueued auto-fill (B3 near-twins, U6, U7) is the fallback for any kernel that fails.

**Results, batch 14** (complete; Kaggle kernels). The local pollers for U18–U22 and U24 were killed by laptop memory pressure while the kernels kept running. All six were logged with `collect_run.py --adopt`, and all echo run_id and code_sha `3bd80b4`. Gate 2 on the B5 pool {T14, U11, U9, T1, T24, U13} = 0.961249.

| Tag | run_id | Solo | Gate 1 | ADD on B5 (sd) | SWAP (sd) | Read |
|---|---|---|---|---|---|---|
| U18 U9 + age cat | beb81d09 | 0.960998 | **+0.000167 vs U9** | +0.000019 (0.000005) | +0.000031 | **held** (+0.0001 to +0.0002). PASS, SWAP for U9. |
| U19 U11 seed_bag 3 | 0b12edd1 | **0.961090** | +0.000081 | +0.000020 (0.000005) | +0.000021 | held (+0.00005 to +0.00012). Best single. Fails sequentially (next to U11 + U22). |
| U20 U11 + dep_delay cat | 35a4ace3 | 0.960962 | −0.000047 | +0.000014 (0.000006) | +0.000013 | held (±0.00005). Fails sequentially. |
| U22 U11 + 13 ratings cat | f964e00c | 0.961084 | +0.000075 | **+0.000040 (0.000006)** | +0.000043 | held (+0.00005 to +0.00015). The strongest new leg. |
| U23 U11 seed 7 | a8316f82 | 0.960907 | **−0.000102** | −0.000004 | −0.000000 | **prediction failed** (\|Δ\| < 0.00005). See below. |
| U24 U11 + cat teacher | fa7bdc80 | 0.960992 | −0.000017 | +0.000028 (0.000006) | +0.000023 | held (±0.00005). PASS: the teacher function is a diversity axis at the best leg too. |

**The U23 caveat.** The seed floor at the CatBoost legs is about 0.0001, not A0's 0.000039, so single-seed Gate 1 numbers for CatBoost are about ±0.0001. Compared seed-matched, age-as-cat is U23 − U4s = **+0.000058**. U11's +0.000174 was partly a lucky seed. The direction is confirmed (U18 repeats it at U9, and U11's public score is +0.00010 over U4), but the size is about +0.0001, not +0.00017.

**B6, sequential Gate 2 on the B5 pool:**
1. **+U22** (ADD +0.000040, 6.7 SD).
2. U9's leave-one-out drops to −0.000003, so **U9 comes out**. U18, its age-CTR successor on the same view, comes in next (ADD +0.000022, 3.7 SD).
3. **+U24** (ADD +0.000014, 2.8 SD).
4. Closure: U9, U19, U20 and U23 all fail. T14's leave-one-out is −0.000002, which is noise, so it's kept (the batch-10 rule).

**B6** (`c4086055`) = rank_mean{T14, U11, T1, T24, U13, U22, U18, U24} = **0.961328**: +0.000079 over B5 and **+0.000155 over B4**, which clears the shipping gate against B4.

**Oct 9 slots (final):** B5, B6, U22, U19, U18, U24, U16, U17, U20, U23. The dry run confirms 10/10.

**Oct 10 filler, banked:** the B6 near-twins **B6m** (−T14) 0.961330, **B6b** (U19 in place of U11) 0.961324 and **B6s** (+U23) 0.961318. Their public spread around B6 measures blend noise. Together with the 7 older unsubmitted runs, the auto-fill holds 10.

### Phase 3b: a second representation round, crossed with model families (from 2026-10-10)

**Why reopen Phase 3.** We're about 8 days ahead of plan, and the user chose to spend that time on representations. Phase 3 called "representation done" on **lgb solo scores**. Phase 4 then found two things that call was blind to:
- **Value comes from different views of the route.** T1 (TE), T8 (profile) and U9 (CatBoost CTRs) each make different errors.
- **CatBoost's combination CTRs found interaction lookups** (age×class, rating×class) that a marginal residual test can't see (FINDINGS 10).

So the round crosses **new representations × model families**, and judges every leg by Gate 1 (solo against its strict twin, +0.00014) *and* Gate 2 on the **B6 pool** {T14, U11, T1, T24, U13, U22, U18, U24} = 0.961328 (ADD > 2 SD, SWAP > −1 SD). The Phase 3 stop rule applies per axis.

**Step 0: residual diagnostic on pair keys.** `residual_lookup.py --keys` now takes arbitrary keys. Values are χ²/df; ≈ 1 means nothing is left.

| Key | T1 (lgb, TE view) | T14 (xgb, profile view) | U22 (cat, all CTRs) |
|---|---|---|---|
| age × customer_type | 1.92 | **2.50** | 1.59 |
| age × travel_type | 1.56 | 2.10 | 1.47 |
| age × travel_class | 1.60 | 1.94 | 1.40 |
| age (marginal) | 1.79 | **2.53** | 1.44 |
| wifi × entertainment | **2.35** | 1.60 | 1.11 |
| online_boarding × wifi | 2.11 | 1.80 | 1.00 |
| FD × travel_class / travel_type | 1.41 / 1.44 | 0.95 / 0.92 | 0.97 / 0.98 |
| full 13-rating tuple | 1.04 | 1.03 | 1.04 |

**What the diagnostic says.**
- **The rating pairs are left over in lgb and xgb, and gone in CatBoost.** U22's rating CTRs drove wifi×entertainment from 2.35 to 1.11. That's U22's gain made visible, and it predicts lgb/xgb gain from explicit rating-pair encodings.
- **Age pairs are the largest residual everywhere, even in U22 (1.4–1.6).** This is not the "calibration-type" age residual Phase 3 closed: T29 erased the *marginal* age χ² in lgb and lost AUC. The pairs are interactions.
- **FD pairs are left over only in the TE view (T1)**, because the profile view already carries them.
- **The rating tuple is not a lookup key.** 337,881 distinct tuples in 699,635 rows; only 18.7% of rows sit in a tuple seen at least 30 times; χ²/df = 1.04 under all three models. By the rule set in the plan (build only if > 1.3), **the rating-tuple feature (W8) is dropped before being built.**

**Pre-registered, batch 15: explicit pair encodings for lgb/xgb/glm, and a pure all-categorical CatBoost view** (Kaggle kernels, config only). The pair encodings use the existing `te_pair_cols` (`encoders.apply_key_target_encoding`, in-fold with an inner K-fold), with `te_pair_bins: 4096` so FD and age stay exact keys and `te_pair_smooth: 50`. The baselines carry the same two keys, which are no-ops without pairs, so only `te_pair_cols` differs.
- **PAIRS7** = age×{customer_type, travel_type, travel_class}, wifi×entertainment, online_boarding×wifi, FD×{travel_class, travel_type}.
- **PAIRS5** = PAIRS7 without the FD pairs. The profile view has no FD residual.

| Tag | Config (strict twin of) | Mechanism | Prediction |
|---|---|---|---|
| W1 | T1 + `te_pair_cols: PAIRS7` (lgb) | Hands lgb the combination statistics CatBoost builds internally | Gate 1 +0.0001 to +0.0003 vs T1. On B6, a SWAP for T1 passes. |
| W2 | V2 + PAIRS7 (xgb, TE view) | The same, for xgb: is the gain family-general? | Gate 1 as W1. Gate 2 borderline (V2 failed next to T1). |
| W3 | T14 + `te_pair_cols: PAIRS5` (xgb, profile + teacher) | Can view 1's xgb get the age/rating interactions? T14 has the largest age residual (2.5). | Gate 1 +0.00005 to +0.00015. A SWAP for T14 passes. |
| W4 | U10 + `num_as_cat: [FD, age, 13 ratings]` (cat, no profile, no teacher) | The view least like the pool: all-CTR, sharing no teacher (U1 lesson) | Gate 1 +0.0003 to +0.0005 vs U10 (U10 is weak). ADD passes only if no-teacher errors differ. |
| W5 | U18 + the 13 ratings in `num_as_cat` | U22's rating gain carried over to the CTR-only view | Gate 1 +0.00005 to +0.00015 vs U18. A SWAP for U18 passes. |
| W6 | W1 + `learner: glm` | A linear family on explicit encodings | solo about 0.955. Gate 2 fails (G1 precedent) unless linear errors are uncorrelated. |

**Falsification:**
- If W1–W3 are all null on Gate 1, the combinations are only useful when estimated *inside* the booster (CatBoost's ordered statistics), and explicit pair TE is closed.
- If W4 solo is below U9, the teacher plus profile aren't replaceable by CTRs alone.

**Batch 16 (code, pre-registered with its commit):**
- the CatBoost NaN-level fix, then W7 = U11 + arr_delay as a categorical;
- an MLP learner (W9, W10).

**Pre-registered, batch 16: two code additions, each defaulting to off.**
- **The CatBoost NaN level** (`learners.py`, cat branch). A categorical NaN now becomes the string level `"nan"`. Before, `.astype(str)` on a pandas categorical left it as float NaN, and CatBoost rejected it (U21). Columns without NaN convert to exactly the same strings.
  - **Bit-identity check:** U10r, a replay of U10's cfg (NOSUBMIT), must reproduce U10's OOF exactly.
- **An MLP learner** (`learner: "mlp"`, `mlp_params`; torch from the Kaggle image, T4 GPU).
  - **Architecture:** PLR periodic embeddings per numeric, learned embeddings per categorical (including `num_as_cat` identities), 3 × 256 SiLU layers, AdamW with OneCycle, and the best epoch chosen by validation AUC.
  - **Why:** it's the only family that builds its own route representation from an embedding. Public notebooks report RealMLP at +0.0003 in stacks; S6E9 found neural legs null.
  - **Smoke test first:** W9s (`sample_frac 0.05`, 2 folds), never submitted.

| Tag | Config (strict twin of) | Mechanism | Prediction |
|---|---|---|---|
| W7 | U11 + `arr_delay` in num_as_cat (the old U21) | Delay × class CTRs on the other delay | Gate 1 ±0.00005 (U20, dep_delay, was −0.000047). Fails Gate 2. |
| W9 | `learner: mlp`, `num_as_cat: [FD, age]`, raw columns (twin: A0 + num_as_cat as lgb is not meaningful; reported against A0 and T1) | An embedding route view | solo 0.9590–0.9600. **Gate 2 passes only if solo ≥ 0.9600** (T1 precedent: different but strong enough). |
| W10 | W9 + `profile_keys: [FD]` | The MLP with the route profile | solo +0.0002 to +0.0005 over W9. Gate 2 as W9. |

**Batch 15/16, first results** (Gate 2 on the B6 pool = 0.961328).

| Tag | run_id | Solo | Gate 1 | ADD (sd) | SWAP (sd) | Read |
|---|---|---|---|---|---|---|
| W1 lgb + PAIRS7 | fd2029e9 | 0.960050 | **−0.000089** vs T1 | +0.000002 (0.000007) | −0.000002 | **prediction failed**: fails |
| W2 xgb + PAIRS7 | a19789bf | 0.960197 | +0.000062 vs V2 | +0.000012 (0.000007) | +0.000010 | fails |
| W3 xgb view 1 + PAIRS5 | c7dd9841 | 0.960692 | +0.000014 vs T14 | −0.000003 (0.000005) | +0.000006 | **prediction failed**: fails |
| W6 glm + PAIRS7 | 1c181629 | 0.948634 | (vs W1 −0.0114) | −0.000241 | −0.000291 | held (fails) |
| **W9** MLP, raw columns | 34748cd9 | 0.959085 | (A0 +0.000298) | **+0.000056 (0.000010)** | +0.000059 | **PASS, 5.6 SD.** Prediction (pass only if solo ≥ 0.9600) failed upward. |
| **W10** MLP + FD profile | 4e0159fc | 0.959295 | +0.000210 vs W9 | **+0.000057 (0.000010)** | +0.000061 | **PASS** |

**Explicit pair TE is closed** (stop rule: W1, W2, W3). The falsification arm triggered: the age, rating and route interactions pay **only when estimated inside the booster**, as CatBoost's ordered CTRs, even though the residual test shows them as left over in lgb/xgb. A plausible mechanism is that an inner-K-fold pair TE on 100–3,000 groups adds estimation noise of the same order as the interaction it carries. CatBoost's ordered statistics, recomputed per tree permutation, average that noise away.

**The MLP is a new view.** It's 0.002 below the pool's legs and still has the largest ADD of any candidate on B6. It builds its own route representation, an FD embedding, and makes different errors. Its fold AUCs average 0.95958 against a pooled OOF of 0.95909, so per-fold calibration varies; rank-mean over the pooled OOF slightly understates it. The best epoch is 8–9 of 16 under OneCycle, so validation AUC falls in the last third.

**Pre-registered, batch 17: the MLP axis** (T4 kernels, about 4 minutes each). These are strict twins of W10, with Gate 2 on the B6 pool and the B7 pool once that's built.

| Tag | Change vs W10 | Mechanism | Prediction |
|---|---|---|---|
| W11 | `seed_bag: 3` | NN fitting variance is large; a bag averages it inside each fold | Gate 1 +0.0002 to +0.0005; ADD above W10's |
| W12 | 13 ratings added to `num_as_cat` | Rating embeddings, the U22 analogue | Gate 1 ±0.0002 |
| W13 | `mlp_params.dropout: 0.25` | Validation AUC peaks early and decays: more regularisation | Gate 1 +0.0001 to +0.0003 |
| W14 | `teacher: true` | Tests the U1 rule on a new family: the shared teacher should raise solo and cut ADD | solo +0.0003 or more; ADD below W10's |
| W15 | `te_cols: [FD]` | The TE route view inside the MLP | Gate 1 ±0.0001 |

**Falsification:** if W14's ADD is at least W10's, the "shared teacher costs diversity" rule doesn't hold for a different family.

**Results, batch 17 and the round's close.** Gate 2 on the B6 pool, single-candidate.

| Tag | run_id | Solo | Gate 1 vs W10 | ADD on B6 (sd) | Read |
|---|---|---|---|---|---|
| W11 seed_bag 3 | f8520c7f | **0.960534** | **+0.001239** | **+0.000069 (0.000008)** | prediction (+0.0002 to +0.0005) failed **upward**. The single-seed MLP is very noisy. |
| W12 rating embeddings | dcd70b08 | 0.958551 | −0.000744 | +0.000044 | fails Gate 1 (single seed; unreadable at this noise level) |
| W13 dropout 0.25 | 08c3738f | 0.959338 | +0.000043 | +0.000049 | null |
| W14 + teacher | 7a185d50 | 0.959247 | −0.000048 | +0.000034 | **prediction failed**: solo didn't rise. ADD is below W10's, as the U1 rule predicts. |
| W15 + FD TE | a7efacb6 | 0.958979 | −0.000316 | +0.000045 | null to negative |
| W4 all-CTR cat, no teacher | 41d4d547 | 0.960703 | +0.000425 vs U10 | +0.000048 (0.000007) | **held** (+0.0003 to +0.0005) |
| W5 CTR view + ratings | f5df566f | **0.961140** | +0.000142 vs U18 | +0.000029 (0.000005) | **held.** Best single. |

**The bit-identity check for the NaN fix.** U18r (`17c0bc5a`, a replay of the Kaggle-run U18 at `18e4f7ad`) reproduces U18's OOF and test **exactly** (max \|Δ\| = 0.0). The first replay, U10r, was an invalid check: U10 had run on oci-free (aarch64, a different CatBoost build), and the cross-platform replay differs (OOF 0.960303 vs 0.960278). That's a reminder that **runs are bit-reproducible only on the same platform**. W7 (U11 + arr_delay) is running.

**B7, sequential Gate 2 on the B6 pool:**
1. **+W11** (ADD +0.000069).
2. **+W9** (+0.000031).
3. The other MLP variants now fail (ADD ≤ +0.000013). **Two MLP legs saturate the view.**
4. **+W5** (+0.000020).
5. **+W4** (+0.000025, 5 SD).
6. **+W14** (+0.000014, 2.0 SD).
7. With W4 in, T1's ADD back is −0.000004, so **T1 comes out**. T14's ADD back is −0.000015, so **T14 comes out**.
8. Closure: nothing passes. U13's leave-one-out of −0.000004 is noise, so it's kept.

**B7** (`be455188`) = rank_mean{U11, T24, U13, U22, U18, U24, W11, W9, W5, W4, W14} = **0.961497**: **+0.000169 over B6**, which clears the shipping gate.

**Oct 10 slots:** B7, W5, W11, W4, W9, W14, W10, W2, W3, W1 (dry run 10/10). **Oct 11 is banked:** W12, W13, W15, W6, plus W7 and the B6 near-twins.

### Checkpoint 3b (2026-10-10)
- **Done:**
  - 15 probes across 3 new axes, plus the CatBoost NaN fix (bit-identical) and an MLP learner;
  - `collect_run --launch-only`: the laptop never idles a poller, so memory pressure can't lose a run.
- **Learned:**
  - A new family (the MLP) is a new view. A new encoding for an old family (explicit pair TE) isn't.
  - Interactions pay only inside CatBoost's ordered CTRs.
  - The champion moved +0.000169 (B7 0.961497).
- **Closed:** explicit pair TE (W1–W3), glm (W6), and the rating-tuple key (dropped by the diagnostic before it was built).
- **Open:**
  - **The MLP axis.** Bag size is the obvious lever: the 3-bag gave +0.0012 solo. Then MLP representation variants, each read only through bags.
  - W7.

**W7** (`12bedad2`, U11 + arr_delay as a CatBoost categorical, now possible after the NaN fix) = 0.960942: Gate 1 **−0.000067** vs U11, and Gate 2 on the B7 pool ADD −0.000009. **Held** (±0.00005, fails). Both delays are now null as CatBoost categoricals (U20, W7). That confirms the batch-14 reading: CTRs pay on **many-level, non-monotone** columns (FD, age, ratings), not on monotone magnitudes. Banked for Oct 11.

### Phase 3c: the MLP axis, read through bags (2026-10-10)

The user approved continuing the MLP axis at Checkpoint 3b. **Why bags first:** W11 (3 seeds) beat W10 (1 seed) by +0.0012. That's 9× the Gate 1 threshold, so a single-seed MLP's Gate 1 is mostly seed noise, and W12–W15 are unread rather than closed. This round reads every MLP variant from a 5-seed bag. The bag's `model_seed + 100·b` seeds differ by bag size only, so X1/X2 vs W11 is a strict twin. Config-only, no code change.

**Pre-registered, batch 18** (T4 kernels, at most 2 at once; a 5-bag takes about 15 minutes). Gate 2 is on the **B7 pool** (0.961497). W11 and W9 are already in it, so an MLP candidate passes as an ADD or as a SWAP for the MLP leg it resembles.

| Tag | Config (strict twin of) | Mechanism | Prediction |
|---|---|---|---|
| X1 | W11 with `seed_bag: 10` (W11) | The 1→3 gain is fitting variance, which shrinks like 1/n | Gate 1 +0.0001 to +0.0004; SWAP for W11 passes |
| X2 | W10 with `seed_bag: 5` (W11) | The bag curve's midpoint, and the base for X3–X6 | between W11 and X1 |
| X3 | X2 + 13 ratings in `num_as_cat` (X2) | W12 re-read: rating embeddings, the U22 analogue | Gate 1 ±0.0002 |
| X4 | X2 without `profile_keys` (X2) | W9 bagged: the raw-column MLP view | solo −0.0002 vs X2; SWAP for W9 > 0 |
| X5 | X2 + `teacher` (X2) | W14 re-read; the U1 rule | solo +0.0001 to +0.0003; ADD below X2's |
| X6 | X2 with `mlp_params.epochs: 10` (X2) | The best epoch is 8–9 of 16, so validation AUC falls while the LR anneals; a shorter cycle finishes annealing near the peak | Gate 1 +0.0001 to +0.0003, at about 60% of the runtime |

**Falsification:**
- If X1 ≤ W11 + 0.0001, the bag curve is flat after 3 seeds, and bag size stops being a lever.
- If X3, X5 and X6 all fail both gates, the MLP representation axis is closed under the stop rule, and the MLP's blend value is just its family (W11 + W9).

**A tag collision.** The tag "X1" had already been used on Oct 2 (`061c20f8`, xgb on raw features). The bag-10 run is logged as **X1m** (`cbd6a24f`); the rename was made in runs.csv and in its cfg.json. Check the tag namespace in runs.csv before pre-registering a new letter.

**Results, batch 18.** Gate 2 is single-candidate on the B7 pool (0.961497).

| Tag | run_id | Solo | Gate 1 (twin) | ADD (sd) | SWAP | Read |
|---|---|---|---|---|---|---|
| X1m bag 10 | cbd6a24f | **0.960892** | +0.000358 (W11) | +0.000012 (0.000004) | +0.000014 | **held** (+0.0001 to +0.0004). As a SWAP for W11 the blend is unchanged (0.961496). |
| X2 bag 5 | 3bcd5d1c | 0.960702 | +0.000168 (W11) | +0.000009 (0.000005) | +0.000010 | held (between W11 and X1m). Fails Gate 2. |
| X3 + rating embeddings | bd496856 | 0.960684 | −0.000018 | **+0.000016 (0.000005)** | +0.000019 | held (±0.0002): null solo, **the best ADD of the round** |
| X4 without profile | 8832a8d6 | 0.960610 | −0.000092 | +0.000011 (0.000005) | +0.000012 | held. As a SWAP for W9: +0.000013. |
| X5 + teacher | b9497ceb | 0.960584 | −0.000118 | −0.000002 | −0.000003 | **prediction failed again**: the teacher *lowers* the MLP's solo score (W14 too). The U1 half held. Closed. |
| X6 10-epoch cycle | 1be77029 | 0.960797 | +0.000095 | +0.000013 (0.000005) | +0.000014 | short of Gate 1 (+0.00014), below the prediction. The best epoch is now the last one (9–10 of 10), and runtime fell only 6%. |

**The bag curve: solo 0.959295 → 0.960534 → 0.960702 → 0.960892 at 1/3/5/10 seeds. The blend hardly moves.** The blend averages over legs, so it already averages the MLP's seed noise. A bagged leg is worth about +0.00001 to it.

**B8, sequential Gate 2 on B7:**
1. +X3 (ADD +0.000016) → 0.961514. After that, every other MLP candidate's ADD is +0.000004 (fail). Three MLP legs saturate.
2. Swap the single-seed legs for their bags: X4 for W9 (+0.000010) and X1m for W11 (0). That gives 0.961524. X1m is kept as the more stable estimate of the same view.
3. U13's ADD back is −0.000005, so **U13 comes out** (the T1 rule) → 0.961529.
4. Closure: no remaining leg has a negative LOO.

**B8** (`5f5eb27a`) = rank_mean{U11, T24, U22, U18, U24, X1m, X4, W5, W4, W14, X3} = **0.961529**, +0.000032 over B7. That's below the shipping gate, but it's queued first for Oct 11 as a read on public.

**Oct 11 slots:** B8, X6, X3, X1m, X4, X2, W7, W12, W13, W15. W6 and X5 are banked for Oct 12.

**Pre-registered, batch 19: MLP architecture** (Oct 11, filling the Oct 12 slots; plan "10 more submissions"). These are 5-seed bags on T4 kernels. Y1–Y3 and Y5 are strict twins of X3; Y4 is a strict twin of X4. Gate 2 is on the **B8 pool** (0.961529).

| Tag | Change | Mechanism | Prediction |
|---|---|---|---|
| Y1 | `mlp_params.epochs: 10` | X6's shorter cycle (+0.000095), carried onto the best-ADD MLP | Gate 1 +0.00005 to +0.00015 |
| Y2 | `mlp_params.emb_dim: 32` | A 16-dim route embedding may be too small for thousands of FD levels | Gate 1 ±0.0001; ADD > X3's if the embedding is the bottleneck |
| Y3 | `mlp_params.plr_sigma: 1.0` | Higher-frequency periodic features resolve age and delays more finely | Gate 1 ±0.0002 |
| Y4 | X4 + 13 ratings in `num_as_cat` | The all-embedding MLP, the analogue of W4 | solo ±0.0002 vs X4; an ADD candidate next to X3 |
| Y5 | `mlp_params.hidden: 512` | A capacity probe | null (S6E9: capacity is null) |

**Stop rule:** if Y1–Y5 all fail Gate 2 on B8, the MLP axis is closed.

**Also for Oct 12:**
- **B9:** the sequential Gate 2 build of the Y-runs on B8.
- **B8st:** a nested-CV logit stack on B8's legs (`blend.py stack --write`). B9f washed on near-twin legs (Δ −0.000005). B8's legs differ in family and calibration. Prediction: Δ vs rank_mean ±0.00005; it's submitted only if the nested OOF is at least rank_mean's.
- **Why these slots:** the public paired SD is 0.000152, so single-leg probes are record entries. Blends are where public can tell us something (B7: +0.00024 public).

**Results, batch 19** (Gate 2 single-candidate on the B8 pool, 0.961529).

| Tag | run_id | Solo | Gate 1 | ADD (sd) | Read |
|---|---|---|---|---|---|
| Y1 10-epoch cycle | cdb0b9e7 | 0.960695 | +0.000011 vs X3 | −0.000004 (0.000005) | **prediction failed** (+0.00005 to +0.00015). As a SWAP for X3: 0.961525. |
| Y2 emb_dim 32 | 22e5a5c4 | 0.960541 | −0.000143 vs X3 | −0.000004 | the route embedding isn't the bottleneck |
| Y3 plr_sigma 1.0 | 614a47ec | 0.960442 | −0.000242 vs X3 | −0.000014 | held (±0.0002) at its edge: negative |
| Y4 all-embedding, no profile | a73476b5 | 0.960626 | +0.000016 vs X4 | +0.000004 | null. As a SWAP for X4: 0.961529, unchanged. |
| Y5 hidden 512 | 292f2bdb | 0.960622 | −0.000062 vs X3 | −0.000010 (0.000004) | held: capacity is null |

**B8st** (`8ad571dd`, NOSUBMIT): a nested-CV logit stack on B8's legs scored 0.961519 against rank_mean 0.961529 (Δ −0.000010, SD 0.000012). That's the sixth stack wash, and the first on a diverse pool. The fitted weights favour W5 (0.22), X3 (0.18) and W4 (0.14), and give X1m and U11 about 0. That's the same ordering as the leave-one-out contributions, so the equal-weight rank_mean already captures it. `blend.py stack --write` now archives stacks.

**The MLP axis is closed** (stop rule: Y1–Y5 all fail Gate 2). No B9 was built; **B8 stays the champion candidate.** The MLP's value lives in three views: the profile MLP (X1m), the raw-column MLP (X4), and the rating-embedding MLP (X3). Schedule, embedding width, PLR frequency and capacity don't add a view.

**Oct 12 slots:** W6, X5, Y4, Y1, Y2, Y3, Y5, plus three auto-fill near-twins. Oct 11 (tonight) is B8 first.

### Phase 4, resumed (2026-10-12)

The user approved returning to Phase 4 at Checkpoint 3c. There are two jobs before the Oct 18 public-policy review: the midpoint instrument checks, and the review's evidence.

**Oct 11 public scores:** **B8 0.96105** (a new best, +0.00007 over B7; OOF +0.00003). The bagged MLPs are now our best *single* models on public: X3 0.96073, X6 0.96072, X1m 0.96069, X2 0.96067, X4 0.96065, against U11's 0.96066.

**Instrument refit** (`oof_lb_fit.py`, 88 runs):
- Spearman 0.914; slope 0.973;
- **σ 0.000373**, up from 0.000259;
- shipping gate 0.000383 OOF.

σ grew because of the MLPs. The public−OOF offset by family (OOF > 0.957):

| Family | n | Mean offset | SD |
|---|---|---|---|
| blend | 8 | −0.000529 | 0.000047 |
| cat | 25 | −0.000370 | 0.000122 |
| lgb | 33 | −0.000436 | 0.000133 |
| xgb | 7 | −0.000489 | 0.000173 |
| **mlp** | 12 | **+0.000440** | 0.000545 |

Single-seed MLPs run +0.0006 to +0.0015; bagged MLPs about 0.

**Mechanism, in two parts:**
- **Cross-fold calibration** (measured). The fold-mean AUC minus the pooled OOF AUC is about 0.00001 for every GBDT, 0.00025–0.0006 for single-seed MLPs, and 0.00003–0.00012 for bagged MLPs. Pooled OOF penalises fold models whose probabilities sit on different scales.
- **Test-side fold averaging** (inferred). A test prediction averages 5 fold models, an implicit 5× bag; an OOF row comes from one model. For a high-variance learner, test is stronger than OOF.

**Consequence:** OOF-based Gate 2 undervalues MLP legs.
- Ranking within folds before blending fixes the calibration half honestly: B8's OOF goes 0.961529 → **0.961540** (+0.000011, 1.8 SD).
- It doesn't change membership, since no candidate passes under it, and it doesn't change B8's test file.
- The test-side half can't be measured on OOF. It's recorded as a Final B question, not a rule.

**Pre-registered, batch 20: a B8 recipe replicate** (`scripts/batch20_b8_replicate.sh`):
- Every B8 leg is re-run at `model_seed: 43` as `<leg>R`, a strict twin of its archived cfg, and blended as **B8R**.
- **Prediction:** each leg is within its seed floor; B8R's OOF is within ±0.00003 of B8.
- **Use:** the B8R−B8 public gap is the public noise of *this recipe*. Final selection needs that number. These are the Oct 13–14 slots.

**Leaderboard reading, 2026-10-12 01:00 UTC** (for the Oct 18 review).
- **The board:**
  - 1,554 teams;
  - #1 0.96243, #10 0.96187, #25–#100 **0.96184** (a flat band), #200 0.96177, #500 0.96093;
  - **B8 (0.96105) ranks about 466.**
- **What the band is:** meta-stacks (L2 logistic, rank or hill-climb) over **124–340 public OOF/test libraries**. Examples: kratosyan's "Stacking every public OOF library", goodpjw2008's "228-member meta stack", cdeotte's "Stacking TFMs with public OOF". karttikjangid05 finds that **181 of 340 public OOFs are near-copies** (logit correlation > 0.999), and that 22 distinct OOFs nearly match the full stack (0.96181). Several 0.96184 files are reposts of one credited "Top18" file that has no OOF behind it.
- **Fold alignment:** every library audited uses **SKF(5, shuffle, 42), our split**. They're blendable with our legs on OOF, so a Final B blend could be honestly validated.
- **The missing view is tabular foundation models.**
  - TabPFN-3.5 and TabICL members appear in every top stack. goodpjw2008's TabPFN + route-category notebook scores **0.96160 public**, against our best single (X3, 0.96073) and B8 (0.96105).
  - cdeotte publishes OOF/test for 9 TFMs (`cdeotte/s6e10-tfm-oof-and-test-predictions`).
  - A public honest-OOF stack reports **CV 0.96208** (najiama, LB 0.96177), against B8's 0.96154 (fold-ranked). So **the gap is real on CV, not just public overfitting.**
  - TabPFN-3.5 carries Prior Labs' non-commercial licence; its authors say it permits Kaggle use.
- **Price of the ours-only policy:** 0.96105 against 0.96184 is **about 0.0008, or about 440 places** today. On Oct 2 it was about 30 places.
  - The gap is now 5× the public paired SD and about 2× the OOF gap of any leg we've found.
  - It isn't shakeup noise. It's a missing model family, plus 100+ diverse members.

### Phase 4b: our own tabular-foundation-model leg (2026-10-12)

On Oct 12 the user chose to train our own TFM leg now, keeping Final A ours-only, with the Oct 18 review unchanged. The *recipe* comes from public notebooks (ideas are allowed), with credit:
- **hemingweb, "S6E10 EXP01 TabPFN35 Full Context":** the full-context, KV-cache, int8, lean-cache-build recipe.
- **goodpjw2008:** route categories.
- **samanyu1808:** the `orig_pred` input.

No public OOF or test artifact is used.

**Code (each piece defaults to off):**
- `src/tfm_lean.py`: the memory-lean KV-cache build, ported with attribution.
- `learners._fit_predict_tabpfn`:
  - TabPFN-3.5 (`tabpfn==9.0.0`, checkpoint `Prior-Labs/tabpfn_3_5`), with the full fold context (about 560k rows) and `n_estimators=1`;
  - `fit_with_cache`, an int8 KV cache, and autocast;
  - an OOM fallback chain, and batched prediction with a halving back-off.
- `tabpfn_params` in DEFAULTS.
- `make_probe_kernel`: only a learner that needs a pip package gets internet and an install cell. That cell **aborts if numpy, pandas, scikit-learn or torch change**, the S6E9 shap lesson.
- **Bit-identity:** an archived kernel (U18r) regenerates **byte-identical**, both ipynb and metadata. The change is purely additive, so no training replay is needed.
- **Licence:** TabPFN-3.5 has Prior Labs' non-commercial licence; its authors state it permits Kaggle competitions.
- **Cost:** about 2.5 T4-hours per single-seed run.

**Pre-registered, batch 21** (`scripts/batch21_tfm.sh`; Gate 2 on the B8 pool):

| Tag | Config (twin) | Mechanism | Prediction |
|---|---|---|---|
| Z0s | `learner: tabpfn`, 5% of rows, 2 folds (NOSUBMIT) | smoke test of install, weights, patch and code path | runs; AUC about 0.95 |
| Z1 | `learner: tabpfn`, raw columns | in-context learning over the full fold: a new family view | solo 0.9600–0.9608; **ADD > +0.00005** |
| Z2 | Z1 + `num_as_cat: [FD, age]` (Z1) | route identity as a categorical | Gate 1 +0.0002 to +0.0006 |
| Z3 | Z2 + `teacher` (Z2) | the original's function as an input | solo +0.0001 or more; U1 says ADD falls |

**Stop rule:** if Z1–Z3 all fail Gate 2, the TFM axis is closed. The public TFMs' value would then be in stacking 100+ members, not in the view itself, which is evidence for the Oct 18 review. Passing legs go into **B10**; the tag "B9" is skipped to avoid confusion with B9f.

**B8 replicate (batch 20), first legs.** Each is at `model_seed` 43, compared with its original:

| Replicate | Δ OOF |
|---|---|
| U11R | −0.000062 |
| U22R | +0.000044 |
| U18R | −0.000032 |
| U24R | −0.000051 |
| W5R | +0.000054 |
| T24R | +0.000007 |
| X3R | −0.000090 |
| X4R | +0.000096 |
| X1mR | +0.000047 |
| W4R | +0.000008 |
| W14R | +0.000206 |

**Batch 20 result (Oct 12): B8R (`67dc16eb`) OOF 0.961520, against B8's 0.961529 (Δ −0.000009). That's inside the pre-registered ±0.00003, so the B8 recipe is seed-stable on OOF.**
- Ten of the 11 legs are within their seed floors.
- The exception is W14R at +0.000206. W14 is a 3-seed MLP bag with a teacher, and its floor sits at the top of the bagged-MLP range. It's also the weakest leg.
- The leg deltas average out in the blend: their spread is ±0.00009, but the blend moves 0.000009. This is the blend averaging seed noise again (finding 15).
- B8R submits in tonight's slot (Oct 12, queue position 8), and the other 11 replicate legs fill Oct 13. The B8R−B8 public gap goes to the Oct 18 review as the public noise of this recipe.

**Z1, first attempt (Oct 12): out of GPU memory at fold 1, no run logged.**
- Fold 0 finished: **AUC 0.960700**, near the CatBoost legs' fold 0 (W4R 0.960907). The fit took 57 min (peak 13.2 GB of a T4's 14.6), validation 15 min, test 32 min.
- Fold 1 ran out of memory in every fallback, including `keep_cache_on_device: False`, so fold 0's model and cache had not been freed. The allocator also showed 2.7 GB reserved but unused.
- **Fix** (`learners._fit_predict_tabpfn`): `del clf`, `gc.collect()` and `torch.cuda.empty_cache()` around each fit, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The config is unchanged, so Z1 is relaunched under its own tag. Nothing else moves.
- **Cost revised:** about 104 min per fold, so **about 9 GPU-hours per 5-fold run**, not 2.5 (the public notebooks' 28 min/fold doesn't hold on our kernel). That fits the 12 h session limit, but the weekly GPU quota (about 30 h) now covers about 3 runs. Z1 and Z2 go now. Z3 and any bag wait for the Gate 2 reading.

**Batch 20 public (Oct 12 slots): B8R 0.96101, against B8's 0.96105 (Δ −0.00004; OOF Δ −0.000009).**
- The gap is a quarter of the public paired SD (0.000152), so B8's public score is the recipe's, not a lucky seed. For final selection, a B8-recipe blend is worth 0.9610 ± noise on public.
- The replicate legs land where their originals did: U22R 0.96067 against 0.96064, and W5R 0.96059 against 0.96059.
- **Other Oct 12 public scores:** X5 0.96042, Y1 0.96075, Y2 0.96068, Y3 0.96063, Y4 0.96070, Y5 0.96065, W6 0.94847.
- **Y1 (10-epoch OneCycle) is the best public MLP so far**, but it failed Gate 2 on OOF (ADD −0.000004). A public-only effect with no OOF movement is provisional (§5), so the MLP axis stays closed.

**Batch 22, feature-engineering gaps (pre-registered Oct 13).**
- **Why now:** the user asked whether every feature technique had been tried. An audit of every archived `cfg.json` against `DEFAULTS` showed which encoders had never run, though the code exists (ported from S6E9):
  - quantization (`fe_quant_cols`);
  - binned, window and shape TE;
  - TE multi-smoothing and backoff;
  - digit features;
  - the arrival − departure delay difference;
  - one-hot, which had no axis at all.
- **Checkpoint 3 closed the FD-TE family by argument** (FD residual χ²/df ≈ 1 under the champion), not by runs. This batch tests that argument on the cheapest base.
- **The twin is T24:** lgb with the FD profile plus a CatBoost teacher, OOF 0.960650, and B8's lgb leg. A passing probe is therefore also a direct SWAP for T24 in B8.
- **New axis `one_hot_cols`** (default `[]`): it replaces each listed column with 0/1 indicators over the train+test levels. It's additive, and a 5% T24 replay is byte-identical before and after.

| Tag | One key changed | Mechanism | Prediction |
|---|---|---|---|
| Q1 | `fe_quant_cols: [FD, age, dep_delay, arr_delay]` | quantization (divisors 5/25/100) | null: trees threshold raw values, and the profile already addresses each route |
| Q2 | `te_window_cols: [FD]` | neighbour-route label rates, radii 1–50 | null to +0.0001 (FD χ²/df ≈ 1) |
| Q3 | `te_bin_cols: [age, dep_delay, arr_delay]` | binned TE of the magnitudes | null (T28; χ²/df ≤ 1.01) |
| Q4 | `fe_delay_diff: true` | arr − dep delay, time made up in the air | \|Δ\| < 0.0001 |
| Q5 | `one_hot_cols`: 4 categoricals + 13 ratings | one-hot instead of native or ordinal | null |

- **Gates:**
  - Gate 1: +0.00014 vs T24.
  - Gate 2 on the B8 pool: SWAP for T24 > −1 SD, ADD > 2 SD.
- **Stop rule:** if all five are null, these axes are closed *empirically*, and FINDINGS records that the residual test's "nothing left" call held for every encoder family. Any pass goes into B10 with the Z results.

**Batch 22 result (Oct 13): all five null, as pre-registered. The feature-engineering axes are closed empirically.**

| Tag | OOF | Δ vs T24 | ADD on B8 (SD) | SWAP for U11 |
|---|---|---|---|---|
| Q1 quantization | 0.960631 | −0.000019 | −0.000010 (0.000004) | −0.000010 |
| Q2 window TE | 0.960693 | +0.000043 | −0.000002 (0.000004) | −0.000002 |
| Q3 binned TE | 0.960627 | −0.000023 | −0.000006 (0.000004) | −0.000006 |
| Q4 delay diff | 0.960643 | −0.000007 | −0.000007 (0.000005) | −0.000008 |
| Q5 one-hot | 0.960546 | **−0.000104** | −0.000012 (0.000004) | −0.000013 |

- The residual test's "nothing left in single columns" call held for every encoder family it was used to skip. Only Q2, the one aimed at the route, moved in the positive direction, and it is a third of Gate 1.
- **Q5 hurts.** One-hot costs lgb 0.0001, so native categorical splits and ordinal ratings are the better representation (finding 20).

**Batch 21 result (Oct 13): both TabPFN legs pass Gate 2 by a wide margin. B10 is the new champion.**

| Tag | Solo OOF | Gate 1 | ADD on B8 (SD) | SWAP for U11 | Runtime |
|---|---|---|---|---|---|
| Z1 raw | 0.960859 | (no twin) | **+0.000067** (0.000006) | +0.000070 | 8.3 h T4 |
| Z2 + FD/age categorical | 0.961070 | +0.000211 vs Z1 | **+0.000076** (0.000006) | +0.000080 | 8.4 h T4 |

- **Predictions held:**
  - Z1's solo score is a touch above the predicted 0.9600–0.9608;
  - ADD > +0.00005 for both;
  - Z2's Gate 1 is at the bottom of the predicted +0.0002–0.0006.
- **Z2 is within 0.00007 of the best single model** (W5, CatBoost, 0.961140), and it is the largest ADD since the first MLP (W11 +0.000069).
- **B10** (`98754488`) is the rank mean of B8's 11 legs + Z1 + Z2: **OOF 0.961652, +0.000123 over B8.** The two TabPFN legs stack: B8 + Z2 alone is 0.961605.
- B10, Z2 and Z1 go first in the Oct 14 slots.

**Batch 23, feature probes on the TabPFN view (pre-registered Oct 13).**
- **Why:** batch 22 says the GBDT feature space is exhausted, and finding 17 says new views come from new inputs. TabPFN is the one view whose inputs we haven't varied beyond route identity.
- **Budget:** each run is about 8.4 T4-hours. Z1 + Z2 used about 16.7 of the 30-hour weekly quota, so one run fits before the Saturday reset.

| Tag | Config (twin) | Mechanism | Prediction |
|---|---|---|---|
| Z3 | Z2 + `teacher` (Z2), already pre-registered in batch 21 | the original's function as an input column | solo +0.0001 or more; ADD at or below Z2's (U1: the teacher costs diversity) |
| Z4 | Z2 + `profile_keys: [FD]` (Z2), after the quota resets on Oct 17 | the route's passenger-mix profile, the biggest GBDT gain (T2 +0.0016) | solo +0.0001 to +0.0003: TabPFN sees the route as a category but not as a profile; ADD uncertain |

- **Gates:** Gate 1 +0.00014 vs Z2; Gate 2 ADD on the B10 pool > 2 SD, or SWAP for Z2 > −1 SD.

**Batch 23 result, Z3 (Oct 15): Gate 1 fails, Gate 2 passes. B11 is the new champion.**

| Tag | Solo OOF | Gate 1 vs Z2 | ADD on B10 (SD) | SWAP for T24 | Public |
|---|---|---|---|---|---|
| Z3 Z2 + teacher (`538c0c67`) | 0.960987 | **−0.000083** | **+0.000023** (0.000004) | +0.000030 | 0.96056 |

- **The prediction was half right.**
  - The ADD fell, as U1 predicted: +0.000023, against Z2's +0.000076.
  - But the solo score *also* fell, where we had predicted +0.0001. For TabPFN, the teacher column is a strong, smooth input. It pulls in-context learning toward the original's function and away from the route-level detail that Z2's categoricals gave it.
- **The leg still ships**, because a solo loss with a blend gain means the teacher-conditioned TabPFN errs in different places from both Z2 and the GBDT teacher legs. Only Gate 2 is a shipping decision (playbook §3).
- **B11** (`0571d27a`) is B10's 13 legs + Z3, a rank mean of 14: **OOF 0.961675, +0.000023 over B10.** That's below Gate 1's single-model bar, but it's a pre-registered Gate 2 pass at about 6 SD.
- **Oct 14 public scores:** B10 **0.96116**, Z2 0.96068, Z1 0.96049, and Q1–Q5 0.96014–0.96030.
  - B10 − B8 public is **+0.00011**, against the OOF gain of +0.000123. So the TabPFN gain transfers, at above the public paired SD (0.000152) × 0.7.
  - B11 public is 0.96118 (+0.00002 vs B10), which is unreadable as expected.
- **OOF→LB fit (120 runs):** slope 0.973, residual σ 0.000349 LB.
- **Next:** Z4 (Z2 + route profile) launches when the GPU quota resets on Oct 17. Checkpoint 4 is Oct 18.

**Batch 24, tree-structure views on CPU (pre-registered Oct 15).**
- **Why now:** the GPU quota is spent until Sat Oct 17, and the user asked to keep probing in the meantime.
- **The open question:** findings 12 and 19 say blend value comes from a different inductive bias. Every GBDT leg in B11 grows trees one way: CatBoost's oblivious (symmetric) trees, or lgb's boosted leaf-wise trees.
- **The test:** these probes change only *how the trees are grown*, never the inputs. Each is one `*_params` key, so no code changes.
- **Smoke test:** every config ran at 5% of rows and 2 folds on the laptop first.

| Tag | Twin | One key changed | Mechanism | Prediction |
|---|---|---|---|---|
| R1 | T24 | `lgb_params`: `boosting_type rf`, 1023 leaves, 500 trees, bagging 0.632, colsample 0.5 | averaged deep independent trees (bagging, not boosting) | solo −0.001 to −0.003; ADD 0 to +0.00002 |
| R2 | T24 | `lgb_params`: `extra_trees true` | random split thresholds smooth the step functions | solo ±0.0001; ADD small |
| R3 | T24 | `lgb_params`: `linear_tree true` | linear models in the leaves for the magnitudes | solo ±0.0002; ADD uncertain |
| R4 | U11 | `cat_params`: `grow_policy Depthwise` | asymmetric trees that keep the ordered CTRs | solo ±0.0001; **ADD > 2 SD is the test** |
| R5 | U11 | `cat_params`: `grow_policy Lossguide`, 64 leaves, depth ≤ 10 | leaf-wise growth chases the hard routes deeper | solo ±0.0002; **ADD > 2 SD is the test** |

- **Gates:**
  - Gate 1 vs the twin is reported but isn't the decision.
  - Gate 2 on the B11 pool: ADD > 2 SD and SWAP > −1 SD.
  - Passing legs form B12.
- **Stop rule:** if all five fail Gate 2, the tree-structure axis is closed. The GBDT views would then be saturated in the inputs (batch 22) *and* in the tree shape. Remaining blend headroom would come only from new families, such as TFM inputs.
- **Slots:** every probe is submitted as it lands (Oct 15). Unused slots auto-fill.

**Z4 is scheduled for the quota reset.** A one-shot cron on oci-free runs `scripts/vm_launch.sh scripts/batch21_tfm.sh Z4 --launch-only` at **Sat Oct 17 03:00 UTC**.
- **Why 03:00:** it leaves a margin past the weekly reset, because a T4 run started on the remaining ~5 h would die mid-fold.
- **What `vm_launch.sh` does:** it pulls main without losing VM-only runs.csv rows, then pushes `src/` as the code dataset at that HEAD.
- **Collection:** a second one-shot entry at Sat 14:00 UTC adopts the output on the VM (`batch21_tfm.sh Z4 --adopt`). That night's scheduler auto-fill then submits Z4.
- **On the laptop:** `sync_runs.py pull` brings the row over, and Gate 2 on the then-champion pool follows.

**Batch 24 result (Oct 15): all five fail Gate 2. The tree-structure axis is closed.**

| Tag | Solo OOF | Δ vs twin | ADD on B11 (SD) | SWAP | Public |
|---|---|---|---|---|---|
| R1 random forest | 0.958202 | −0.002448 (T24) | −0.000040 (0.000006) | −0.000041 | 0.95767 |
| R2 extra-trees | 0.960491 | −0.000159 (T24) | −0.000006 (0.000004) | −0.000002 | 0.95984 |
| R3 linear leaves | 0.960465 | −0.000185 (T24) | −0.000013 (0.000004) | −0.000010 | 0.96012 |
| R4 CatBoost Depthwise | 0.960729 | −0.000280 (U11) | −0.000017 (0.000003) | −0.000011 | 0.96046 |
| R5 CatBoost Lossguide | 0.960736 | −0.000273 (U11) | −0.000016 (0.000004) | −0.000010 | 0.96059 |

- **Every variant is weaker solo, and every ADD is negative.** So a changed tree shape on the same inputs gives a slightly worse copy of the same view, not a new view.
- **The default growers are the right ones for this data.** CatBoost's oblivious trees beat the asymmetric ones by 0.00027. That fits the regularising role symmetric trees play with ordered CTRs on a high-cardinality route key. This mechanism is plausible but not tested.
- **R1 was the only prediction that held in size** (−0.0024 in a predicted −0.001 to −0.003). R4/R5 were predicted within ±0.0002 and fell slightly outside it.
- **Stop rule applied:** GBDT views are saturated in both inputs (batch 22) and tree shape (batch 24), as finding 22 records. B11 stays champion.

**Batch 25, TabPFN weight in the blend (pre-registered Oct 15, before scoring).**
- **Why:** B11's leave-one-out puts Z1 (+0.000047) and Z2 (+0.000056) far above every other leg, and seven legs are slightly negative. The equal-weight rank mean may dilute the one family that is genuinely different.
- **Probe B12w:** B11 with Z1, Z2 and Z3 each listed twice, so the TFM share of the blend goes from 3/14 to 6/17. This is one structural choice, not a fitted weight (finding 18).
- **Prediction:** +0.00001 to +0.00004 over B11.
- **Gate:** the paired bootstrap Δ vs B11 must exceed 2 SD. Weight 3 is computed as information only and can't ship, so the choice of k isn't tuned on OOF.

**Batch 25 result (Oct 15): B12w passes. TabPFN at weight 2 gives OOF 0.961718, +0.000043 over B11 (paired SD 0.000008, 5.3 SD), at the top of the predicted range. Public 0.96122 vs B11 0.96118 (+0.00004), the same direction and unreadable in size.**
- Weight 3 (information only) adds +0.000049 at 3.6 SD, so the curve is flat past 2 and k isn't tuned.
- **B12w (`1337a5f8`) is the champion.** From now on, new TFM legs are gated on the B12w pool at weight 2 (finding 23).

**Batch 26, finalist blends to fill slots (pre-registered Oct 16, before writing).**
- **Why:** after tonight only one banked run is left for auto-fill, so the remaining slots need candidates. Public deltas this small are unreadable, so these are finalist candidates and calibration points, not decisions.
- **Already seen (scored Oct 16 while planning, so not blind):** TabPFN weight 3 is +0.000007 over B12w. Dropping W14, X1m, T24 or X4 alone is +0.000003 to +0.000006.
- **B13k3:** B12w at TabPFN weight 3.
- **B13p:** greedy backward elimination from B12w over the candidates W14, X1m, T24, X4, in that order. A drop is kept only while OOF does not fall.
- **B13pk3:** the B13p pool at TabPFN weight 3.
- **Gate:** none of these can replace B12w as champion. Any gain is inside 2 SD, and the pruning order was chosen after looking at leave-one-out scores. They enter the final-selection pool only.

**Batch 26 result (Oct 16): three finalist blends written and queued for tonight's slots. B12w stays champion.**
- **B13p** (`cc839b3d`): greedy pruning dropped W14, X1m and T24 and kept X4. OOF 0.961731, +0.000013 over B12w. The order was picked after seeing leave-one-out scores, so this gain is optimistic.
- **B13k3** (`0cfc1174`): OOF 0.961725, +0.000007.
- **B13pk3** (`eda537d3`): OOF 0.961724. On the pruned pool, weight 3 is worse than weight 2.
- All three sit ahead of auto-fill in the oci-free queue. The other 7 slots tonight go to leftover runs (W14, W11, W5, W10, W9, W2, W3), which add low-OOF points to the OOF-to-public fit.

**Z4 result (Oct 17 kernel, collected Oct 19): it fails Gate 1 but passes Gate 2 marginally. B14 is a Final A candidate, not a new champion.**
- **Solo:** Z4 (`6ea49b38`) OOF 0.960847, −0.000223 vs Z2, public 0.96064. The prediction (+0.0001 to +0.0003) failed in sign. The route profile is the largest GBDT gain (T2 +0.0016), but TabPFN with 17 extra profile columns is weaker. A plausible, untested mechanism: TabPFN already reads the route from the category, so the profile columns add width without information.
- **Gate 2 on B12w at TFM weight 2:** ADD +0.0000118 (SD 0.0000058, **2.05 SD**). SWAP Z4 for Z3 is −0.0000000, and SWAP Z4 for Z1 is −0.000017. At weight 1 the ADD is +0.0000096 (3.2 SD).
- **B14** (`754f9b07`) = B12w + Z4, TFM weight 2: OOF 0.961730. That's a pass at the edge of the gate, the same size as B13p's post-hoc pruning gain. It's banked as a Final A candidate, and B12w keeps the champion title because the margin is the gate itself.

### Checkpoint 4 (2026-10-18): the public-artifact review
- **The user's decision:** "borrow more from the leaderboard". Public OOF/test pairs may now enter **Final B** (fold-aligned, with a manifest). **Final A stays ours-only** (§5.2).
- **The GPU for the rest of the week (~22 T4-hours) goes to our own TabICL leg** (user's choice over a TabPFN seed bag). Public TabICL and TabPFN members enter Final B either way, so this buys Final A a second TFM family.
- **Board, Oct 18 23:40 UTC:** #1 0.96309, #6–9 about 0.96200, us 0.96122.

### Phase 5: finals (from 2026-10-19)

**Batch 27, our TabICL leg (pre-registered Oct 19, before launch).**
- **Code (default off):** `learners._fit_predict_tabicl` (TabICL v2, `tabicl==2.1.1`, checkpoint `tabicl-classifier-v2-20260212.ckpt`, BSD-3) and `tabicl_params` in DEFAULTS: a 250k stratified context per fold, 4 ensemble members, predict batch 8192 with a halving back-off. The kernel installs the package behind the pinned-stack guard. The idea comes from koumeimaki's c250k TabICL member and cdeotte's tabicl2. No public artifact is used.
- **Bit-identity:** the U18r kernel regenerates byte-identical. A 2% CPU smoke (3k context, 1 member) gives OOF 0.9539, so the code path is sound.

| Tag | Config (twin) | Mechanism | Prediction |
|---|---|---|---|
| K0s | `learner: tabicl`, 5% rows, 2 folds (NOSUBMIT) | install, checkpoint, T4 path, timing | runs; AUC about 0.955 |
| K1 | `learner: tabicl` (Z1; only `learner` changes) | a second in-context family with column-then-row attention and a different prior | solo 0.9603–0.9610; **ADD on B12w > +0.00002** |

- **Gate:** Gate 2 on the B12w pool at TFM weight 2 (ADD > 2 SD, SWAP > −1 SD). A pass makes **B15** = B14 + K1 at TFM weight 2.
- **Pre-registered fallback:** if K0s or K1's first fold projects past the 12 h session, relaunch K1 with `tabicl_params.context` 150k. That's the same probe at a smaller context, not a new one.

**Batch 27 launch (Oct 19 00:00 UTC):** K0s (`446cbf82`) and K1 (`845df7a2`) run on Kaggle T4s, pushed from oci-free at code `321db97`. Hourly adopt crons on oci-free collect them on Oct 19–20.

**Batch 28, Final B: the public library (pre-registered Oct 19, before any F blend is scored).**
- **Import (`scripts/public_lib.py`):** 214 OOF/test pairs from 8 public datasets. 180 are kept, as `P_*` pseudo-runs (NOSUBMIT, Final B only, ledger axis "public"). Every member's source, audit numbers and cluster are in `experiments/public_manifest.csv`. The datasets:
  - cdeotte's TFM library and his 124-member library;
  - goodpjw2008's extra members;
  - koumeimaki, kagankoral, dmitriykolmikov, sachith7, amanatar and kratosyan.
- **Excluded:**
  - 32 near-copies (rank correlation > 0.999 with a stronger kept member);
  - goodpjw2008's `stack245` (a meta-model);
  - `nj_01` in cdeotte's 124 library, which is najiama's blend re-hosted. It was imported by a script bug before the exclusion took, so its row is marked EXCLUDED and is never a leg.
  - No member exceeds the 0.9625 leak line. The best single is goodpjw2008's TabPFN (0.961491).
  - megayak's parquet library is skipped (no parquet engine, and we don't add packages for it). dariushafshar's golem library returns 404.
- **Fold alignment:** koumeimaki ships its fold file, and it matches our partition exactly. The others state SKF(5, shuffle, 42). An unaligned member would still be out-of-fold for every row, so the nested blend scores stay honest either way.
- **New tool:** `blend.py climb` does Caruana forward selection with replacement. It selects on 4 folds and scores the 5th, and that nested OOF is the decision number. Sanity check on our 15 legs: nested 0.961722 against B12w's 0.961718, with in-sample 0.961741. The 0.00002 optimism is what the nested score removes. `blend.py stack` gains `--C`. Blends with `P_` legs must be tagged `F*`.

| Tag | Pool | Method | Prediction (nested OOF) |
|---|---|---|---|
| F1 | 15 own legs (B12w + Z4) + 180 public | climb | 0.96200–0.96215 (najiama's honest stack CV is 0.96208) |
| F2 | same | nested L2 logit stack, C = 1 | within ±0.00005 of F1 |
| F3 | 180 public only | climb | F1 − F3 > 0, about +0.00002 to +0.00008 = what our legs are worth to Final B |
| F5 | B12w legs + public TFM members, TFMs at weight 2 | rank mean, unfitted | 0.9618–0.9620 |
| F6 | F1's pool | climb capped at 25 picks | within 0.00002 of F1 (selection regularisation) |
| F7 | F2's pool | L2 stack, C = 0.01 | within ±0.00003 of F2 |

- **Decision rule:** the Final B candidate is the best nested OOF among F1, F2 and F6. Public is read only for direction (§5).
- **Leak tripwire:** if an F blend's public score sits more than 0.0003 above the OOF→public line, a member is suspect. Then the members flagged by their fold-AUC spread get dropped, and the blend is refit.
- **Slots:** F1–F7 plus B14 take the Oct 19 slots. B14 is the Final A candidate.
- **Tag rename:** `K1` was already taken (Phase 2's CatBoost on raw features), so the TabICL runs are logged as **TI0s** (`446cbf82`) and **TI1** (`845df7a2`). The kernel slugs stay `s6e10-k0s`/`s6e10-k1`, and the adopt crons rename the rows by run_id. **TI0s passes:** the install guard held, folds 0.9576/0.9580 at 5% rows. Test prediction takes 4.2 min per 300k rows at a 17k context, so TI1 at 250k projects to about 90 min per fold, inside the 12 h session.
- **F5 result: it fails.** OOF 0.961342, which is −0.000376 against B12w. The prediction was 0.9618–0.9620. Equal weight on 23 public TFMs of mixed strength (mitra2 0.9571, causilo 0.9583, among others) dilutes the strong legs, so Final B needs selection (F1) or fitting (F2), not an unfitted mean.
- **F1 and F2 were stopped** before finishing, because the laptop ran out of memory. Each holds about 195 legs × 1M rows in memory. Neither wrote a run.

**Batch 28 rerun on oci-burst (Oct 20 01:20 UTC).** The user brought up oci-burst (16 vCPU, 62 GB).
- **Pool:** `pack_pool.py` packed the 195 legs into float32 logit columns. The public members were read from their raw downloads, and every one hashed to its audited `oof_sha`.
- **Blends:** `burst_blends.py` ran F1, F2, F3, F6 and F7 in parallel. The stacks took 2.5 min each, the climbs 23–27 min.

**F8 pre-registration (Oct 20, after F1–F7 were scored and before F8 runs).**
- **Why:** F3 says our 15 legs are worth nothing to the climb (F1 − F3 = −0.000005), yet the stack gives W4, Z2 and X3 weights of +0.05 to +0.07. F8 prices our legs in the method that wins.
- **F8:** F7's stack (C = 0.01) on the 180 public members only.
- **Prediction:** F7 − F8 between +0.00002 and +0.0001. If it's ≤ 0, our legs add nothing to Final B.

**Batch 28 result (Oct 20): a fitted stack over the public library is the Final B candidate, and our legs add nothing to it.**

| Tag | run_id | Nested OOF | vs B12w | Prediction | Verdict |
|---|---|---|---|---|---|
| F7 | `4398fb27` | **0.962152** | +0.000434 | within 0.00003 of F2 | pass, **Final B candidate** |
| F8 | `b8a4fdd9` | 0.962152 | +0.000434 | F7 − F8 = +0.00002…+0.0001 | fail: F7 − F8 = +0.0000005 |
| F2 | `9967474a` | 0.962135 | +0.000417 | within 0.00005 of F1 | missed, in its favour (+0.0002) |
| F3 | `35ae3ef3` | 0.961943 | +0.000225 | F1 − F3 > 0 | fail: −0.000005 |
| F1 | `fe775e2e` | 0.961938 | +0.000220 | 0.96200–0.96215 | fail |
| F6 | — | = F1 | — | within 0.00002 of F1 | NULL: the 25-pick cap never binds (the climb stops at 8–21 picks), so not archived |

- **Stack beats climb by +0.0002.** The climb picks 8–9 legs with integer weights. The stack fits all 195, including negative weights: cdeotte's `causilo` gets −0.23 against `tabfm` +0.31. That subtracts shared error, which a rank mean of positive picks can't do. Finding 18 (rank mean ≥ stack) was measured on 11 correlated legs of our own and doesn't carry over to 195 diverse ones.
- **C hardly matters:** C = 1 and C = 0.01 agree to within 0.00005 in every fold. With 700k rows per fit, 195 coefficients are well determined, so this isn't overfitting.
- **Our legs are worth about zero in Final B.** Both F3 (climb) and F8 (stack) match their twins that include our legs. The stack does give W4, Z2 and X3 weights of +0.05 to +0.07, but those weights only replace public members that carry the same information. Final A (ours-only) remains the hedge if the public library turns out to leak.
- **Leak tripwire for tonight:** with F7 at 0.96215 OOF, our OOF→public line predicts about 0.9616–0.9617 public. A public score above about 0.9620 would trigger the §8 batch 28 tripwire.
- **TI1 (batch 27) fails on budget.** The kernel hit the 12 h limit inside fold 0. Validation took 236 min for 140k rows at a 250k context, and the 300k test rows would need about 500 min. The 90 min/fold projection came from the 17k-context smoke: cost scales with context × rows, not rows alone. The pre-registered 150k fallback would still need about 37 T4-hours, above this week's quota, so it isn't launched. The axis stays closed unless the user reopens it at a much smaller context. TI0s is renamed from K0s on both machines.
- **Slots, Oct 20 23:15 UTC:** queued F7, F2, F1, F3, then auto-fill (B10, Z2, Z1, Q5, Q4, Q3). Oct 19's 10 went to B14 (public **0.96125**, our best so far) plus 9 replicates.
- **Process note:** F8's pre-registration was written to the README before F8 ran but committed after, in this same entry's commit.

### Leaderboard review (Oct 20 06:45 UTC)
- **Board:** 2,442 teams. #1 0.96341, #2 0.96326, #3–6 0.9622–0.9624. About 85 teams are at 0.9620 or above, 38 of them at exactly 0.96206, which is one public notebook. We're about **#680 at 0.96125** (B14).
- **OOF→public fit** (131 runs): slope 0.973, residual σ 0.00034. Our blends land about **0.0005 below their OOF** (B14 0.961730 → 0.96125), while honest public stacks report CV ≈ LB (najiama CV 0.96208, LB 0.96206).
- **Correction to batch 28's tripwire:** given those public stacks, an F7 score near 0.9621 is normal, not a sign of leakage. The tripwire moves to **public − OOF > +0.0003**, so F7 above 0.9625 is suspect.
- **What F7's score will tell us:** 0.9617 means the public split is harder for every model. 0.9621 means the 0.0005 gap was specific to our own models.

**Batch 29, a bigger library and non-linear meta-stackers (pre-registered Oct 20, before any of these runs).**
- **New import** (from the same audit, re-run on oci-burst over all 246 pairs):
  - megayak's parquet library and arhancanli12's library, converted to `.npy` in a throwaway uv environment with pyarrow, so the project's packages are unchanged. Both ship fold columns, and **both match our partition exactly**.
  - goodpjw2008's TabPFN-member CSVs, and najiama's one non-blend member (07 XGBoost).
  - **Net:** 10 new kept members, 186 in total. arhancanli12's 4 originals replace cdeotte's 4 re-hosts (those `P_cdeotte_ar_*` are now near-copies and leave the pool). The new information is megayak's 5 RealMLP-v4 seeds (up to 0.961206) and najiama's XGBoost. Every goodpjw2008 and other megayak column is a near-copy.
  - dariushafshar's golem library still fails to download. Submission-only datasets (no OOF) are excluded: they can't be validated or stacked honestly, and they're already tuned to the public LB.
- **New tool `scripts/burst_meta.py`:** a nested LightGBM meta-stacker (15 leaves, lr 0.03, L2 10, min leaf 200). For each outer fold it trains on three folds, early-stops on the fourth, and predicts the outer fold. Test = mean of the five models.

| Tag | Pool | Method | Mechanism | Prediction (nested OOF) |
|---|---|---|---|---|
| F9 | 15 own + 186 public | L2 logit stack, C = 0.01 (F7's method) | 6 new partly-independent members | F7 + 0.000000 to +0.00003 |
| F10 | same | LightGBM meta on the 201 logits | non-linear trust: e.g. believe the TFMs where the GBDTs disagree | F9 − 0.0001 to F9 + 0.0001 |
| F11 | same + 21 raw columns | LightGBM meta | trust that depends on the region of feature space | F10 + 0.00000 to +0.00010 |
| F12 | — | rank mean of F9 and the better of F10/F11 | stacks with different inductive biases make partly independent errors | ≥ max of the two, by up to +0.00005 |

- **Gate (Final B):** the best nested OOF becomes the Final B candidate if it beats F7 by more than the paired SD of their difference. Otherwise F7 stays.
- **Slots:** every F9–F12 run that beats F7 is queued for Oct 21 in OOF order.

**Batch 29 partial results and the F13 pre-registration (Oct 20, after F9–F11, before F12/F13).**
- **F9: 0.962155**, F7 + 0.000003. Inside the predicted range (+0 to +0.00003). The six new members are nearly redundant.
- **F10: 0.961924**, F9 − 0.00023. **Fails** (predicted within ±0.0001). LightGBM as the meta-model stops after 190–300 trees and loses to the linear stack. The member logits are already calibrated, so a weighted sum is close to the right shape, and trees cut each one into steps.
- **F11: 0.961828**, F10 − 0.0001. **Fails** (predicted ≥ F10). Adding the raw columns to a tree meta-model costs more.
- **F12 (unchanged, pre-registered):** rank mean of F9 and F10, the better of the two LightGBM metas.
- **F13, new: residual boosting.** The L2 stack (C = 0.01) is fit on the three training folds, and its logit is LightGBM's `init_score` (lr 0.01, 7 leaves, early stop 200, on logits + raw columns). Early stopping uses a fold outside the stack's fit.
  - **Mechanism:** keep the linear structure, and let the trees learn only the regions where it is systematically wrong.
  - **Prediction:** F9 + 0.00000 to +0.00006. If early stopping lands at about 0 trees, there's no residual structure (NULL).

**Batch 29 results: F12 and F13. F14 pre-registered (Oct 20, before F14 runs).**
- **F12: 0.962132.** **Fails** (predicted ≥ F9). The tree meta drags the rank mean down.
- **F13: 0.962103, NULL.** Residual boosting stopped at 1–458 trees, and per-fold changes stayed within ±0.00003 of the stack alone. Its base is a 3-fold stack, which is why it sits below F9. The linear stack has no regional errors left for trees to fix.
- **Conclusion:** the L2 stack is at this library's ceiling (about 0.96215). F9's +0.000003 over F7 is noise, so **F7 stays the Final B candidate**. No more GPU: Z3 already gave TabPFN the teacher, a TabPFN run costs about 8 T4-hours, and our TFM legs carry no weight in Final B.
- **F14 (leak hedge):** F7's method (L2 stack, C = 0.01) on our 15 legs plus the **25 public members whose folds we verified**:
  - the fold-file libraries (koumeimaki, megayak, arhancanli12);
  - cdeotte's re-hosts that are near-copies of those members.
  - **Mechanism:** a member whose OOF was contaminated in-fold would look strong in CV and fail on private. The stack would then overweight it, and nested CV cannot detect it. F14 drops the 161 unverifiable members.
  - **Prediction:** 0.9619–0.9621, between B14 and F9.
  - **Use:** if F7's public score comes in > 0.0003 above its OOF (the tripwire), F14 replaces it as Final B.

**Batch 29 result (Oct 20): F14 and leave-one-library-out.**
- **F14: 0.961889** (`a975808f`). The prediction (0.9619–0.9621) missed by 0.00001. With only fold-verified members it's still +0.00016 over B14, so that's what Final B is worth without trusting any unverifiable member.
- **Leave-one-library-out** (F9's stack, information only, no runs written):

| Library dropped | Members | Δ nested OOF |
|---|---|---|
| goodpjw2008 extra members | 15 | −0.000038 |
| cdeotte 124-library | 117 | −0.000036 |
| cdeotte TFM | 11 | −0.000026 |
| amanatar | 6 | −0.000024 |
| kagankoral | 17 | −0.000023 |
| sachith7 | 2 | −0.000014 |
| koumeimaki | 7 | −0.000008 |
| dmitriykolmikov, arhancanli12 | 1, 4 | −0.000005 |
| megayak, najiama, **our 15** | 5, 1, 15 | −0.000001 to 0 |

- **What it means:** Final B's +0.0004 comes from breadth. No library carries more than 0.00004, so a leak hidden in any one library couldn't produce the gain. Our own legs carry nothing.
- **Slots, Oct 20 (queued in priority order):** F7, F14, F9, F2, F1, F3, F10, F13, F12, F11. 79 eligible runs remain banked for auto-fill through Oct 29.
