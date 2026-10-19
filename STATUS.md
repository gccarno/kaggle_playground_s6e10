# S6E10 status: start here

*Rewritten in place after every batch (last: 2026-10-19 01:30 UTC, batches 27–28 underway). For the visual version, see the dashboard (tabs **Experiments**, **Features**, **EDA**): run `python scripts/dashboard.py`, which writes `reports/dashboard.html` and is published as the "S6E10 Experiment Board" artifact. README.md is the contract and FINDINGS.md has the explanations. This page only orients.*

## Where we are
- **Phase 5 (finals).** At Checkpoint 4 (Oct 18) the user opened public artifacts for **Final B**. **Final A stays ours-only.**
- **Champion (Final A): B12w.** OOF **0.961718**, public **0.96122**.
  - **B14** = B12w + Z4: OOF 0.961730. Z4 fails Gate 1 (−0.00022 solo) but passes Gate 2 by a hair (2.05 SD), so it's banked as a Final A candidate.
  - **B13p, B13k3, B13pk3** all scored public 0.96123, so they're unreadable against B12w.
- **Board (Oct 18):** #1 0.96309, #6–9 about 0.96200, us 0.96122. The gap comes from stacks over public OOF libraries plus TFMs, which is what Final B targets.
- **Deadline:** 2026-10-31 23:59 UTC.

## Running now
- **TI1 (our TabICL v2 leg, Final A)** is on a Kaggle T4. It's projected at about 7.5 h, so it lands around Oct 19 08:00 UTC. A cron on oci-free adopts it hourly and renames the row from K1 to TI1 (the tag K1 was already taken). The smoke run TI0s passed.
- **Final B library:** 180 audited public members are imported as `P_*` pseudo-runs (`experiments/public_manifest.csv`). The audit removed 32 near-copies, one stack, and one re-hosted blend.
- **F5 failed** (OOF 0.961342). An unfitted mean over public TFMs dilutes the strong legs, so Final B needs selection or fitting.
- **F1 (nested Caruana) and F2 (nested L2 stack) were stopped by the session** when laptop RAM ran out: 195 legs × 1M rows, two jobs at once. They need a restart decision (see below).

## Tonight (Oct 19 slots, 23:15 UTC cron)
- **Queue is empty.** Auto-fill will send B14 first, then replicate legs.
- **Intended:** F1, F2, F3, F6, F7 and B14, once the F blends are computed.

## Next decisions
1. **Where to rerun F1/F2/F3/F6/F7:**
   - one at a time on the laptop, about 1 h each;
   - or on oci-burst after packing the 195 rank columns into one ~550 MB float32 matrix.
2. **When TI1 lands:** Gate 2 on B12w at TFM weight 2. A pass makes B15.
