# S6E10 status: start here

*Rewritten in place after every batch (last: 2026-10-20 02:30 UTC, batch 28 scored). For the visual version, see the dashboard (tabs **Experiments**, **Features**, **EDA**): run `python scripts/dashboard.py`, which writes `reports/dashboard.html` and is published as the "S6E10 Experiment Board" artifact. README.md is the contract and FINDINGS.md has the explanations. This page only orients.*

## Where we are
- **Phase 5 (finals).** Two finals: **Final A** is ours-only, **Final B** may use public OOF libraries (Checkpoint 4).
- **Final A candidate: B14** (B12w + Z4). OOF 0.961730, public **0.96125**, our best public so far. B12w (0.961718 / 0.96122) stays champion by the gate rule.
- **Final B candidate: F7.** A nested L2 logit stack (C = 0.01) over 15 own + 180 public legs: **OOF 0.962152, +0.00043 vs B12w.** That's about the gap between us and the 0.962 band on the board.
  - **Stack beats climb:** F1/F3 (Caruana) land at 0.96194. Negative weights let the stack subtract shared error.
  - **Our legs add nothing to Final B:** the public-only stack F8 also scores 0.962152.
- **TI1 (our TabICL) failed on budget.** It hit the 12 h limit inside fold 0: about 12.4 T4-hours per fold at a 250k context, so about 62 h for the full run. It isn't relaunched.
- **Deadline:** 2026-10-31 23:59 UTC.

## Tonight (Oct 20 slots, 23:15 UTC cron on oci-free)
- **Queued:** F7, F2, F1, F3, then auto-fill: B10, Z2, Z1, Q5, Q4, Q3.
- **Leak tripwire:** F7 should land around 0.9616–0.9617 public. **Above about 0.9620**, a public member is suspect, so drop flagged members and refit.

## Infra
- **oci-burst is up** (16 vCPU / 62 GB). It holds the packed 195-leg pool (`~/s6e10/pool195`), so new stacks take 2–3 min there. **Tear it down by Oct 29** (`terraform apply` without `burst_enabled`, which needs the user's OK).

## Next decisions
1. **After Oct 20's public scores:** confirm F7 as Final B. Possible follow-ups on burst: drop weak or suspicious members, or add a second-level feature (fold-AUC-weighted).
2. **GPU budget (TI1 dropped):** spend it on more TabPFN seeds for Final A, a TabICL run at a 50k context (about 2.5 h per fold), or nothing.
3. **Final selection (two picks):** B14 (Final A) and F7 (Final B) are the current pair.
