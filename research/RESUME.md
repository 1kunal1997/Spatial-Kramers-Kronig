# RESUME — read this first on waking

**Written 2026-08-18 ~21:20 PT, at the end of cycle 1, with the Captain's usage window closing.**
He expects to revive the voyage in ~3 hours. This file is the cold-start handoff.

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

**LEAN CONFIG IS THE DEFAULT NOW** unless the Captain says otherwise:
- **One agent at a time**, sequential — not parallel
- **sonnet by default**; opus only for genuinely hard theory, and at `high` not `xhigh`
- Adversaries fire only on **promotion attempts**, not every cycle
- Estimated 5–10× cheaper per cycle → roughly one solid cycle per 5-hour window

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

**On disk but NOT YET READ OR INTEGRATED — do this first:**
- `theory/h9_potential_domination.md` (22 KB) + `theory/h9_domination_check.py` — theorist's H9
  handoff. **Highest priority read.**
- `results/cycle1_c15_redteam/redteam.md` (23 KB) + `bench.md` (13 KB) + a full experiment suite
  (m1–m5, bench.py, cert.py, split_test.py). Red-team's attack on C15.

Both were written under a final checkpoint order, so they are structured as cold-start handoffs with
explicit "next step" sections. **Read them, integrate verdicts into `CLAIMS.md` and `LOG.md`, and
report to the Captain BEFORE dispatching anything new.** That costs almost nothing and is the highest
value first move.

## Open items for the Captain (non-blocking, in CAPTAIN_QUEUE.md)

1. Durable auto-resurrection needs his hands: GitHub App access unverified, and key mission inputs
   (`verify_passivity.py`, the Aug PDFs, the 98-figure corpus) are **untracked** — `*.pdf`/`*.png` are
   gitignored, so a cloud agent would clone a repo missing the evidence.
2. Whether to commit `verify_passivity.py` and/or the PDFs and figures.

## Immediate next steps, in order

1. Read the two unintegrated reports; update `CLAIMS.md` / `LOG.md`; report to Captain. **Cheap.**
2. Finish H6: vectorize `results/cycle1_h6_crossover/clausen_general_form.py` (currently a slow
   O(N²) loop) and test whether the general signed layer-cake/Clausen form predicts α_c = 0.66623.
   Note it is O(N) — **V3 tier, not the cheap certificate of V1**.
3. Re-label the 98-figure corpus by **interior** minimum rather than whole-domain min (C19), so
   margins become visible and near-threshold cases usable as a test set.
4. Whatever the theorist's and red-team's "next step" sections name.
