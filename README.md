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
