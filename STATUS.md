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
  - TFM inputs (batch 23);
  - tree structure (batch 24).

## Running now
- **Batch 24 (CPU kernels): R1–R5.** Same inputs as the T24 / U11 legs, but the trees grow differently: random forest, extra-trees, linear leaves, and asymmetric CatBoost. The question is whether a different tree shape is a new blend view.
- **Z4 is scheduled.** A one-shot cron on oci-free launches it Sat Oct 17 03:00 UTC, after the GPU quota reset, and adopts it hourly from 12:00 UTC.
- **Oct 15 slots:** B11 and Z3 are submitted. R1–R5 are submitted as they land, and the cron auto-fills the rest at 23:15 UTC.

## Next decisions
1. **When R1–R5 land:** Gate 2 on the B11 pool. Passing legs make B12. If all fail, the tree-structure axis is closed.
2. **Z4** (Z2 + route profile) is automatic on Oct 17. Each TabPFN run costs about 8.4 of the 30 weekly T4-hours.
3. **Oct 18 review (user's call):** whether to keep the final submission ours-only, or allow public artifacts.
