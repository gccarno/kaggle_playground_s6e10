# S6E10 status: start here

*Rewritten in place after every batch (last: 2026-10-13, batch 22 launched). For the visual version, see the dashboard (tabs **Experiments**, **Features**, **EDA**): run `python scripts/dashboard.py`, which writes `reports/dashboard.html` and is published as the "S6E10 Experiment Board" artifact. README.md is the contract and FINDINGS.md has the explanations. This page only orients.*

## Where we are
- **Champion: B8.** It's a rank-mean blend of 11 legs.
  - **Scores:** OOF **0.961529**, public **0.96105**.
  - **Legs:** 6 CatBoost, 1 LightGBM and 4 MLP views of the data.
  - **Seed check:** a full rebuild at a new seed (B8R) scored 0.961520 / 0.96101, so B8's score isn't luck.
- **Phase:** 4b, a tabular foundation model leg (TabPFN-3.5). It's the one model family we've never had.
- **Side batch 22:** an audit found feature techniques we had never run (quantization, binned/window TE, delay difference, one-hot). We're testing them instead of arguing them closed. The dashboard's **Features** tab lists every technique and the runs that used it.
- **Deadline:** 2026-10-31 23:59 UTC. A cron on oci-free spends 10 submission slots a day.

## The story so far (one line per phase)
1. **Phase 1, instrument:** measured the noise. Seed changes move OOF by 0.00004 and public by up to 0.00015. So a probe must beat its twin by **+0.00014 OOF** (Gate 1), and public deltas below about 0.00015 are unreadable.
2. **Phase 2, the lookup hunt:** `flight_distance` is a *route ID*, not a distance. Giving the model the route's identity and profile was worth **+0.0016**, the biggest gain of the contest (FINDINGS 1–4).
3. **Phase 3, representation:** a teacher model fit on the original data added +0.0002. Extra route statistics were predictable from what the model had, so they were null (FINDINGS 5–8).
4. **Phase 4, diversity and blending:** CatBoost's ordered CTRs see the route and age × class in a different way. Blends of *different views* beat any single model by about 0.0005 (FINDINGS 9–11, 14).
5. **Phase 3b/3c, MLP:** a neural net is weaker solo, but it was the biggest blend contributor. Architecture tweaks never helped; new *inputs* did (FINDINGS 12–17).
6. **Phase 4, B8 and the replicate:** stacking lost to a plain rank mean (FINDINGS 18). B8 was reproduced at a new seed.
7. **Phase 4b, now:** TabPFN-3.5, full-context in-context learning. The public leaderboard's frontier (0.9618) is built on models like this.

## How results are judged
- **Gate 1 (single model):** OOF minus its strict twin (exactly one config key changed) ≥ **+0.00014**.
- **Gate 2 (blend leg):** adding it to the champion pool lifts the blend by > 2 bootstrap SD, and swapping it in costs > −1 SD.
- **Public score** is read only for direction. Offsets measured on public get their sign trusted and their size halved.

## Axes
- **Closed:**
  - route lookup;
  - teacher;
  - original data as rows or labels;
  - extra route stats;
  - explicit interactions;
  - capacity and HPO;
  - seed bagging for blend value;
  - MLP architecture;
  - fitted stacks.
- **Open:** blending (champion B8).
- **Running:**
  - TFM (TabPFN-3.5);
  - feature-gap probes (batch 22: quantization, window TE, binned TE, delay difference, one-hot).

## Running now
- **Z1** (TabPFN-3.5 on raw columns) and **Z2** (+ FD/age as categoricals) are on Kaggle T4s.
  - Launched about 23:30 UTC Oct 12, expected about 09:00 UTC Oct 13.
  - About 9 GPU-hours each.
- **Q1–Q5** (batch 22) are on Kaggle CPU, launched about 10:25 UTC Oct 13, about 6 minutes each. Each is a strict twin of T24 (the lgb leg in B8) with one feature technique added.
- **The Oct 13 slot queue holds** the B8 replicate legs (X1mR, U11R, …) plus auto-fill.

## Next decisions
1. **When Q1–Q5 land:** Gate 1 vs T24, then Gate 2 (ADD to B8, SWAP for T24). If all five are null, the feature-engineering axes close with data.
2. **When Z1 and Z2 land:** run Gate 2 on the B8 pool. Passing legs go into **B10**, which is queued for the Oct 14 slots.
3. **Z3** (Z2 + teacher) runs after that. If Z1–Z3 all fail Gate 2, the TFM axis closes.
4. **Oct 18 review (user's call):** whether to keep the final submission ours-only, or allow public artifacts.
