# S6E10 Findings

This is the explanatory layer: *why* things worked or failed, written as results arrive. README.md §8 holds the dated phase log and overrides this file wherever the two disagree.

## Carried in from S6E9 (hypotheses for this contest, not facts about it)

1. **The lookup column.** Most of S6E9's gain (+0.0030 of about +0.0042) came from finding a column the model read as a *magnitude* when it really behaved as a *lookup key* into the source dataset. In S6E10, test `flight_distance`, `age` and the delays for this first.
2. **Representation, then stop.** Representation work added about +0.0012. Everything after that was about +0.00005 on private.
3. **Expected nulls:** HPO, capacity, fitted stacks, cross-family pools, and neural architectures.

## Phase findings

*(none yet)*
