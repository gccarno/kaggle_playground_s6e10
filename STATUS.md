# S6E10 status: start here

*Rewritten in place after every batch. Last update: 2026-10-21 00:50 UTC, batch 29's public scores are in. For the visual version, see the dashboard (tabs **Experiments**, **Features**, **EDA**). It comes from `python scripts/dashboard.py`, which writes `reports/dashboard.html`, published as the "S6E10 Experiment Board" artifact. README.md is the contract and FINDINGS.md has the explanations. This page only orients.*

## Where we are
- **Board (Oct 20 06:45 UTC):** 2,442 teams. #1 is at 0.96341, #3–6 at 0.9622–0.9624, and about 85 teams are at ≥ 0.9620. We were about #680 at 0.96125 (B14). **F7 at 0.96178 is now our best public score** and should land around the 0.9618 band.
- **Final A (ours-only): B14.** OOF 0.961730, public 0.96125.
- **Final B (public library): F7.** OOF 0.962152, **public 0.96178** (public − OOF −0.00037: the normal band, so the tripwire didn't fire).

## What the Oct 20 slots told us
| Run | OOF | Public | Public − OOF |
|---|---|---|---|
| F2 / **F7** / F9 / F13 (L2 stacks) | 0.96210–0.96216 | 0.96178–0.96179 | −0.00032 to −0.00038 |
| F12 (stack + tree meta) | 0.962132 | 0.96171 | −0.00042 |
| F10 / F11 (LightGBM meta) | 0.96192 / 0.96183 | 0.96152 | about −0.0004 |
| F14 (fold-verified members only) | 0.961889 | 0.96149 | −0.00040 |
| F1 / F3 (climbs) | 0.96194 | 0.96142 / 0.96147 | −0.0005 |
| B14 / B12w (ours only) | 0.96173 / 0.96172 | 0.96125 / 0.96122 | −0.0005 |

- **No sign of a leak.** Stacks that use public members keep the same public − OOF offset as our own blends. A leaked member would push public above the line.
- **The OOF gains transfer.** F7 − B14 is +0.00042 on OOF and +0.00053 on public. The OOF ranking of the stacks holds on public to about 0.00001.

## Decision for the user: the final pair
- **Recommended: F7 + B14.** F7 is our best on both OOF and public. B14 keeps one entry fully ours-only (§5.2) in case a public library is contaminated in a way public can't show.
- **Alternative: F7 + F14.** F14 is +0.00024 on public over B14, but it uses 25 public members (fold-verified). It's a stronger second entry, though not an ours-only one.

## Submission queue
- **Oct 21–28:** auto-fill takes the 79 banked runs (calibration points). Anything new jumps the queue: re-stack if a new public OOF library appears (check daily; 2 min on burst).
- **Oct 29–31:** auto-fill stops (by design). Slots go to the two finalists only.

## Infra
- **oci-burst is up** (16 vCPU / 62 GB) and holds the 201-leg pool (`~/s6e10/pool201`). It's now also the build box for the kaggle-harness tests, so keep it until **Oct 29**. Tearing it down then needs the user's OK for the `terraform apply`.
- **Cleanup:**
  - remove the TabICL adopt crons on oci-free (now due);
  - remove the scheduler cron after Oct 31.
