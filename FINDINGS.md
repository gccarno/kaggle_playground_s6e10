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
