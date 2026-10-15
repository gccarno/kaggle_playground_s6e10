# S6E10 status: start here

*Rewritten in place after every batch (last: 2026-10-15 01:00 UTC: Z3 collected, B11 champion). For the visual version, see the dashboard (tabs **Experiments**, **Features**, **EDA**): run `python scripts/dashboard.py`, which writes `reports/dashboard.html` and is published as the "S6E10 Experiment Board" artifact. README.md is the contract and FINDINGS.md has the explanations. This page only orients.*

## Where we are
- **Champion: B11.** It's B8's 11 legs plus three TabPFN-3.5 legs (Z1, Z2, Z3), rank-mean blended.
  - **Score:** OOF **0.961675**, public **0.96118**.
  - **Against B10:** B10 scored OOF 0.961652 and public 0.96116.
  - **Against B8:** B8 scored public 0.96105, so the TabPFN gain transferred to public (+0.00011 for B10).
  - **Why it moved:** TabPFN predicts by in-context learning, so its errors differ from every CatBoost, lgb and MLP leg.
- **Z3 (Z2 + teacher)** was weaker solo (−0.000083 vs Z2) but still added +0.000023 (6 SD) to the blend. Only the blend gate decides what ships (finding 21).
- **Feature engineering on GBDTs is closed with data.** Batch 22 tried every encoder we had skipped (quantization, window TE, binned TE, delay difference, one-hot). All five were null; one-hot actually hurts (−0.0001). See the dashboard's **Features** tab.
- **Phase:** 4b, batch 23 (feature probes on the TabPFN view).
- **Deadline:** 2026-10-31 23:59 UTC. A cron on oci-free spends 10 submission slots a day.

## The story so far (one line per phase)
1. **Phase 1, instrument:** measured the noise. Seed changes move OOF by 0.00004 and public by up to 0.00015. So a probe must beat its twin by **+0.00014 OOF** (Gate 1), and public deltas below about 0.00015 are unreadable.
2. **Phase 2, the lookup hunt:** `flight_distance` is a *route ID*, not a distance. Giving the model the route's identity and profile was worth **+0.0016**, the biggest gain of the contest (FINDINGS 1–4).
3. **Phase 3, representation:** a teacher model fit on the original data added +0.0002. Extra route statistics were predictable from what the model had, so they were null (FINDINGS 5–8).
4. **Phase 4, diversity and blending:** CatBoost's ordered CTRs see the route and age × class in a different way. Blends of *different views* beat any single model by about 0.0005 (FINDINGS 9–11, 14).
5. **Phase 3b/3c, MLP:** a neural net is weaker solo, but it was the biggest blend contributor. Architecture tweaks never helped; new *inputs* did (FINDINGS 12–17).
6. **Phase 4, B8 and the replicate:** stacking lost to a plain rank mean (FINDINGS 18). B8 was reproduced at a new seed.
7. **Phase 4b:** TabPFN-3.5, full-context in-context learning. It's a new view: B10 is +0.000123 over B8 (FINDINGS 19).
8. **Batch 22:** every untried GBDT feature encoder was null (FINDINGS 20).
9. **Batch 23, Z3:** the teacher lowers TabPFN's solo score but adds blend value, which makes B11 (FINDINGS 21).

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
  - fitted stacks;
  - GBDT feature encoders and one-hot (batch 22).
- **Open:**
  - blending (champion B11);
  - TFM inputs (batch 23).

## Running now
- **Nothing is training.** The weekly GPU quota is nearly spent until Sat Oct 17.
- **Oct 15 slots:** B11 and Z3 were submitted by hand. The oci-free cron auto-fills the other 8 at 23:15 UTC.

## Next decisions
1. **Z4** (Z2 + route profile) launches when the weekly GPU quota resets (Sat Oct 17). Each TabPFN run costs about 8.4 of the 30 weekly T4-hours.
2. **Oct 18 review (user's call):** whether to keep the final submission ours-only, or allow public artifacts.
