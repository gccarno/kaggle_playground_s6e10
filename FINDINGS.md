# S6E10 Findings

This is the explanatory layer: *why* things worked or failed, written as results arrive. README.md §8 holds the dated phase log and overrides this file wherever the two disagree.

## Carried in from S6E9 (hypotheses for this contest, not facts about it)

1. **The lookup column.** Most of S6E9's gain (+0.0030 of about +0.0042) came from finding a column the model read as a *magnitude* when it really behaved as a *lookup key* into the source dataset. In S6E10, test `flight_distance`, `age` and the delays for this first.
2. **Representation, then stop.** Representation work added about +0.0012. Everything after that was about +0.00005 on private.
3. **Expected nulls:** HPO, capacity, fitted stacks, cross-family pools, and neural architectures.

## Phase findings

### Phase 2: the lookup hunt (Oct 2–4)

1. **The S6E9 lesson carried over, and it's a route, not an income.** `flight_distance` is a value→label lookup: the synthetic generator copied each route's character from the source data. A tree that reads distance as a magnitude can't see it.
2. **A lookup has two separable parts.** *Identity* is being able to address one route; frequency and fine bins provide it, for about +0.0007. *Knowledge* is what's true of that route, for about another +0.0008. The knowledge can come from labels (TE) or **without labels**, from the route's passenger mix. The unsupervised profile wins, because it carries no in-fold estimation noise.
3. **The residual χ²/df test is a working lookup detector.** It flagged FD (2.20) and not age (0.52), and target encoding then confirmed both calls (T1 helped, T6 was null). Run it first in future contests, before trying any feature.
4. **The original dataset helps through its function, not its labels.** As extra rows, as key-matched labels, or as TE support, the original's labels are null to negative. As a *teacher* (a model fit on the original, used as a feature), it helps, but only next to the route features. Its signal is the original's decision boundary, which a denoised model carries and individual noisy labels don't.

### Phase 3: representation build-out (Oct 4–5)

5. **Test a candidate feature *after conditioning on what the model already has*.** The Business-class share varied 57× more than binomial across routes, which looked like a strong signal. But distance and the means already predicted it (R² 0.97), and T20 was null. Raw variance is the wrong test; the conditioned residual is the right one.
6. **The conditioned residual test also says when to stop.** Under the champion, FD and every FD pair sit at χ²/df ≈ 1. The route lookup is fully used, so further FD encoders can't help, and the test shows that before any of them is run.
7. **A residual signal can be calibration, not ranking.** Age showed χ²/df 2.77 under the champion, value-specific and highly significant. Giving the model age as a categorical drove it to 0.05 and *cost* 0.00035 AUC. Expected gain scales with the *excess* χ² mass (about 120 for age against about 4,300 for FD at A0), and even that is an upper bound.
8. **Signals surface as the model improves.** Age's residual grew from 1.77 at A0 to 2.77 at the champion, because the large route errors no longer masked it. Re-run the residual test at every new champion, not just once.

### Phase 4: diversity and blending (Oct 5–)

9. **A leg's blend value is what it doesn't share with the pool.** Adding the pool's best feature (the teacher) to the TE route view raised its solo score and destroyed its blend value (U1). Ensemble value comes from different *representations* of the route (label rate, profile, CatBoost CTRs), not from more families on one representation.
10. **Residual χ² tests only marginal lookups; CatBoost's combination CTRs find interaction lookups.** Age had χ²/df 0.52 at A0, and as an lgb categorical it lost 0.00035 (T29, finding 7). As a CatBoost categorical, it gained +0.000174 at U4 (U11) and +0.000167 at U9 (U18). Ratings as CatBoost categoricals gained another +0.000075 (U22). The difference is that CatBoost builds age × class and rating × travel-type statistics with ordered, leak-free estimates. lgb only gets the identity, and overfits it. A column the χ² test clears is closed for lgb, not for CatBoost.
11. **The seed floor at the CatBoost legs is larger than at A0.** U11's seed twin U23 landed −0.000102, against a predicted |Δ| < 0.00005 and an A0 floor of 0.000039. Compared seed-matched (U23 vs U4s), the age gain is +0.000058. A single-seed CatBoost Gate 1 near +0.00014 can't be read without a twin. Blends and seed bags (U19) are the reliable way to read these legs.

### Phase 3b: representations × families (Oct 10)

12. **A new *family* can be a new view; a new *encoding* for an old family usually isn't.** The MLP (PLR numerics, FD/age embeddings) is 0.0015–0.002 below the GBDT legs solo, and it had the largest Gate 2 ADD of the round (W11, +0.000069, 8.6 SD). It builds its own route representation from an embedding, so its errors differ. Two MLP legs saturate it. After that, more MLP variants are seed noise, not new views. The single-seed MLP is very noisy: a 3-seed bag added +0.0012 solo (W11 vs W10). So read MLP Gate 1 results only from bags.
13. **Interactions pay only when estimated inside the booster.** The residual test showed age × customer-type and rating-pair interactions left over in lgb/xgb (χ²/df 1.6–2.5). Handing them over as explicit in-fold pair target encodings was null to negative (W1 −0.00009, W2 +0.00006, W3 +0.00001). CatBoost's ordered CTRs of the same pairs paid (U11, U22, W5). The working mechanism: an inner-K-fold TE on groups of 100–3,000 rows adds estimation noise about as large as the interaction. Ordered statistics, averaged over permutations, don't. A residual χ² above 1 is evidence of a signal, not proof that a given encoder can extract it (compare finding 7).
14. **The teacher-free view replaced the TE view.** In B7, W4 (all-CTR CatBoost, no profile, no teacher) made T1, the first teacher-free leg, redundant (T1's ADD back = −0.000004). T14 dropped out too (−0.000015). Membership kept moving toward "one leg per distinct way of seeing the route": profile+teacher CatBoost, CTR-only CatBoost, all-CTR without a teacher, and the MLP.
