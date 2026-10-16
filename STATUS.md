# S6E10 status: start here

*Rewritten in place after every batch (last: 2026-10-16 02:00 UTC: batch 26 finalist blends queued). For the visual version, see the dashboard (tabs **Experiments**, **Features**, **EDA**): run `python scripts/dashboard.py`, which writes `reports/dashboard.html` and is published as the "S6E10 Experiment Board" artifact. README.md is the contract and FINDINGS.md has the explanations. This page only orients.*

## Where we are
- **Champion: B12w.** It's B8's 11 legs plus three TabPFN-3.5 legs (Z1, Z2, Z3), with the TabPFN legs at double weight in the rank mean.
  - **Score:** OOF **0.961718**, public **0.96122**.
  - **Against the earlier champions:**

    | Run | OOF | Public |
    |---|---|---|
    | B11 | 0.961675 | 0.96118 |
    | B10 | 0.961652 | 0.96116 |
    | B8 | 0.961529 | 0.96105 |
  - **Why it moved:** TabPFN reads the data by in-context learning, so its errors differ from every tree and MLP leg. Equal weights diluted that, and doubling its weight is +0.000043 (5.3 SD).
- **The GBDT views are saturated.** New encoders (batch 22) and new tree shapes (batch 24: random forest, extra-trees, linear leaves, asymmetric CatBoost) were all null or negative (findings 20, 22).
- **Phase:** 4b. **Deadline:** 2026-10-31 23:59 UTC. A cron on oci-free spends the leftover slots each night.

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
10. **Batch 24:** tree shape is not a view, and all five probes were null (FINDINGS 22).
11. **Batch 25:** TabPFN legs at double weight make B12w (FINDINGS 23).
12. **Batch 26:** three finalist blends (pruned pool, TabPFN weight 3) to fill slots. Each is within noise of B12w, so none can take champion.

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
  - GBDT feature encoders and one-hot (batch 22);
  - tree structure (batch 24).
- **Open:**
  - blending (champion B12w);
  - TFM inputs (batch 23).

## Running now
- **Nothing is training.** The GPU quota is spent, and batch 24 is closed.
- **Z4 is scheduled.** A one-shot cron on oci-free launches it Sat Oct 17 03:00 UTC and adopts it hourly from 12:00 UTC. That night's scheduler submits it. Check `logs/z4.log` on oci-free.
- **Oct 16 slots (0 of 10 used yet):** tonight's 23:15 UTC cron sends the queue first, then 7 leftover runs:
  - B13p, B13k3, B13pk3 (batch 26);
  - W14, W11, W5, W10, W9, W2, W3 (auto-fill).
- **After tonight only W1, B6 and U23 are left in auto-fill**, plus Z4 and its blends on Oct 17. From Oct 18, slots go unfilled unless new runs are banked. That's deliberate: public deltas this small are unreadable.

## Next decisions
1. **When Z4 lands (Oct 17):** Gate 2 on the B12w pool, keeping TFM legs at weight 2. Then queue B14 = B12w + Z4, plus its pruned and weight-3 versions.
2. **Oct 18 review (user's call):** whether to keep the final submission ours-only, or allow public artifacts.
