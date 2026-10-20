# S6E10 status: start here

*Rewritten in place after every batch (last: 2026-10-20 08:30 UTC, batch 29 done). For the visual version, see the dashboard (tabs **Experiments**, **Features**, **EDA**): run `python scripts/dashboard.py`, which writes `reports/dashboard.html` and is published as the "S6E10 Experiment Board" artifact. README.md is the contract and FINDINGS.md has the explanations. This page only orients.*

## Where we are
- **Board (Oct 20 06:45 UTC):** 2,442 teams. #1 0.96341, #3–6 0.9622–0.9624, and about 85 teams at ≥ 0.9620 (38 tied at 0.96206, one public notebook). **Us: about #680 at 0.96125 (B14).**
- **Final A (ours-only): B14.** OOF 0.961730, public 0.96125.
- **Final B (public library): F7.** A nested L2 logit stack over 15 own + 180 public members: **OOF 0.962152, +0.00043.** In batch 29 it held up against everything we tried:

| Tried | Nested OOF | Read |
|---|---|---|
| F9: same stack, 186 public (6 new) | 0.962155 | the new members are redundant |
| F10/F11: LightGBM as meta-model | 0.96192 / 0.96183 | trees lose to the linear stack |
| F13: stack + residual LightGBM | NULL | no regional errors left |
| F14: only fold-verified members (leak hedge) | 0.961889 | +0.00016 over B14 without trusting any unverified member |
| Leave-one-library-out | −0.00004 at most | the lift is breadth; no library dominates, ours adds 0 |

- **GPU:** none planned. TI1 (TabICL) failed on budget, TabPFN runs cost about 8 h, and our TFM legs carry no weight in Final B.

## Priority queue for submissions
**Tonight, Oct 20 23:15 UTC (queued on oci-free, in this order):**
1. **F7:** the Final B candidate. Its public score answers the offset question (see below).
2. **F14:** the leak hedge.
3. **F9, F2:** stack variants, i.e. the noise floor between near-identical stacks.
4. **F1, F3:** climbs, the selection-vs-fitting comparison on public.
5. **F10, F13, F12, F11:** meta-stacker variants.

**Oct 21–28:** auto-fill takes the 79 banked runs (calibration points). Anything new jumps the queue:
- a re-stack if a new public OOF library appears (check daily; 2 min on burst);
- F14 to the front if the tripwire fires.

**Oct 29–31:** auto-fill stops (by design). Slots go only to hand-picked finalists.

## Decision after tonight's scores
- **Normal:** F7's public − OOF lies between −0.0008 and +0.0003. Our blends ran −0.0005 and honest public stacks about 0.
  - **Proposed final pair:** **F7 + B14**, or **F7 + F14** (F14 is a stronger hedge than B14 but uses verified public members). That's the user's call: §5 says Final A is ours-only.
- **Tripwire:** public − OOF > +0.0003. F14 becomes Final B, and we drop the most suspect library and refit.

## Infra
- **oci-burst is up** (16 vCPU / 62 GB) and holds the 201-leg pool (`~/s6e10/pool201`). It's needed only for re-stacks. **Proposal:** tear it down after Oct 21's read unless a new library appears, and by Oct 29 at the latest. A `terraform apply` needs the user's OK.
- **Cleanup:** remove the TabICL adopt crons on oci-free after today. Remove the scheduler cron after Oct 31.
