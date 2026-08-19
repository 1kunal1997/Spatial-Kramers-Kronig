# RESUME — read this first on waking

**Written 2026-08-18 ~21:20 PT, at the end of cycle 1, with the Captain's usage window closing.**
He expects to revive the voyage in ~3 hours. This file is the cold-start handoff.

## ⏱ FIRST ACTION ON WAKING, BEFORE ANYTHING ELSE

Run `research/hb.sh first-mate "session revived"`, then read `research/MEASURE.md` and compute the
previous expedition's duration from `research/HEARTBEAT.log` (last timestamp − EXPEDITION START).
**The Captain wants this in the morning report — a Pro-vs-Max decision rests on it.** Report duration
AND the configuration that produced it; the two are meaningless apart.

## How to resume

Say/do this: read `MISSION.md` → `PROTOCOL.md` → this file → `CLAIMS.md` → `BOARD.md`.
Everything durable is on disk and committed. No context from the previous session is required.

**Note:** `.claude/agents/` now holds all 9 crew definitions. They were written mid-session and so
were NOT registered in the session that created them. A fresh session **can address the crew by
name** (`theorist`, `numericist`, `red-team`, `replicator`, `auditor`, `referee`, `analyst`,
`librarian`, `navigator`) rather than dispatching general-purpose hands with charter pointers.

## ⚠ BUDGET — the binding constraint, read before dispatching anything

The Captain is on **Claude Pro**, not Max. Cycle 1 burned **~75% of a 5-hour window in ~25 minutes**
running 4 agents in parallel (2× opus at xhigh effort). **That configuration is not sustainable.**

**⚠️ CAPTAIN'S ORDER 2026-08-19 — DO NOT DOWNGRADE MODELS. Quality is the priority.** The earlier
"lean = sonnet by default" policy is **RESCINDED**. See `BUDGET.md`. The judgment seats
(`theorist`, `red-team`, `referee`, `navigator`) stay on **opus**, theorist and red-team at
**xhigh**. If speed becomes binding the Captain upgrades the plan — that is his lever, not the crew's.

**Control cost by SEQUENCING and SCOPE, never by tier:**
- **One agent at a time**, sequential — not parallel. This is the dominant lever.
- Sharp, bounded questions rather than three loose jobs per hand
- Adversaries fire only on **promotion attempts**, not every cycle
- Never re-derive context that is already in `LOG.md` / `RESUME.md`

Cycle 1 was also front-loaded with one-time costs (reading 3 waves of prior work, two long PDFs, 98
figures, building the ledger). Steady-state cycles should be cheaper than it regardless.

The 5-hour cron job is **summarize-and-stop**: it reports and explicitly does NOT dispatch crew.
Honour that. Wait for the Captain's order to sail.

## Where cycle 1 got to

**Returned and integrated into the ledger:**
- **librarian** → `lit/cycle1_domination_and_complexity.md`. Lead: Cartier–Fell–Meyer / Choquet
  dilation order (H11). Trap: Cartan domination + balayage are one-way false friends. Front 3
  (complexity) is genuinely open — and notably **C15 has no literature support either**.
- **numericist** → `results/cycle1_h6_crossover/`, `results/cycle1_h7_endpoints/`.
  C13 → `supported` (V5 is free). **α_c = 0.66623** (C18). **C19: the whole-domain min ε″ is a blind
  indicator** — the likely reason the Captain's pattern resisted quantification.

**ALL FOUR CREW RETURNED AND ARE FULLY INTEGRATED** into `LOG.md`, `CLAIMS.md`, `KILLED.md`,
`BOARD.md`. Cycle 1 is closed and committed (3362d89). Nothing is pending integration.

Headline results — see `LOG.md` for the full account:
- **theorist: H9 CONFIRMED.** The domination relation is the organizing law. Thm 4.2 and Thm 5.2 are
  ONE theorem. **Parsimony 14 → 3.** New two-sided theorem **Cor 4.B** (a + b ≤ L) supersedes Thm 4.2.
  Source: `theory/h9_potential_domination.md`, cold-start handoff in §12.
- **red-team: C15 NARROWED**, and it killed the "margin 0" premise (K16/C21), reopening cheap
  approximate certificates. Source: `results/cycle1_c15_redteam/redteam.md` (559 lines), §E1(a) names
  the next experiment.

## Open items for the Captain (non-blocking, in CAPTAIN_QUEUE.md)

0. 🚩 **CHECKPOINT: V2 is unreachable as written** — it asks for a superlinear time lower bound on a
   problem reducing to nonnegativity testing of a degree-N/2 trigonometric polynomial. `MISSION.md` is
   flagged, NOT rewritten; victory conditions are the Captain's call. A reachable restatement is ~90%
   established in `results/cycle1_c15_redteam/redteam.md`.
1. Durable auto-resurrection needs his hands: GitHub App access unverified, and key mission inputs
   (`verify_passivity.py`, the Aug PDFs, the 98-figure corpus) are **untracked** — `*.pdf`/`*.png` are
   gitignored, so a cloud agent would clone a repo missing the evidence.
2. Whether to commit `verify_passivity.py` and/or the PDFs and figures.

## Immediate next steps, in priority order — ALL are cheap, ONE agent at a time

1. **T3 — fix the 233/240 defect** (~20 min). (L1) disagrees with the corpus on 7 edge-slope-unmatched
   families; diagnosed as a ξ-space quadrature defect, unfixed and unconfirmed. It is the weakest
   point in the strongest result and exactly what a referee attacks. Then have `auditor` confirm the
   diagnosis independently rather than accepting the theorist's own account of it.
2. **E1(a) — move the cascade onto the decimated grid** (~20 lines, ~1 h). Red-team's split test is
   already correct on every physical profile; if the cascade survives at n = 512 the test lands in
   **10–30× territory and C15 dies outright.** Cheapest route to V1 ever on the board.
3. **`replicator` on Cor 4.B (C25)** — the mission's first two-sided non-trivial claim. Give it the
   statement and parameters ONLY; it must NOT see `theory/h9_potential_domination.md` or
   `h9_domination_check.py`. Blind re-derivation is the whole point.
4. **T1 — Open Problem 8.1 (skew tolerance)**, now a finite computation via (L2): 3 constraints ⟹
   ≤3-atom extreme points ⟹ degree-3 Q ⟹ Cardano. Theorist expects it to close completely.
5. Re-label the 98-figure corpus by **interior** minimum (C19) so margins become visible.
6. H6 leftover: vectorize `results/cycle1_h6_crossover/clausen_general_form.py` (slow O(N²) loop).

**Do NOT dispatch all of these at once.** That is what burned 75% of a window. One at a time.
