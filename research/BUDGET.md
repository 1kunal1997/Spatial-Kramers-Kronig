# Budget & Burn

## ⚠️ CAPTAIN'S STANDING ORDER (2026-08-19) — DO NOT DOWNGRADE MODELS

**Quality is the priority. The end goal is to find the answer.** No agent's model or effort may be
lowered to save tokens. If speed becomes the binding constraint, the Captain will upgrade the plan —
that is his lever, not the crew's. The First Mate's earlier "lean config = downgrade to sonnet"
default is **RESCINDED**.

This is the correct call and the evidence from cycle 1 supports it: the opus/xhigh theorist produced
the mission's largest result (H9, 14 statements → 3, Cor 4.B). Downgrading that seat would have been
a false economy — it would have cost the voyage its best day to save a fraction of one window.

## The real cost levers — use these instead of tier

Cycle 1 burned ~75% of a 5-hour window. The cause was **not** model quality. It was:

1. **Concurrency** — 4 agents dispatched simultaneously. This is the dominant lever. Sequencing costs
   wall-clock time, not quality.
2. **One-time front-loading** — reading 3 waves of prior work, two long PDFs, 98 figures, and building
   the ledger from scratch. Already paid; steady-state cycles are cheaper by construction.
3. **Task scope** — a hand given three loosely-bounded jobs explores more than one given a sharp
   question. Tighten the question, not the model.
4. **Re-reading** — the ledger exists so nobody re-derives context. `RESUME.md` and `LOG.md` are the
   cheapest tokens in the system.

## Model policy — match tool to job, never downgrade the judgment seats

| Seat | Model | Rationale |
|---|---|---|
| `theorist` | **opus / xhigh** | Judgment IS the product. Never downgrade. |
| `red-team` | **opus / xhigh** | Same. Its nested hand caught two of its own errors in cycle 1. |
| `referee` | **opus / high** | Final gate. |
| `navigator` | **opus / high** | Chooses what the voyage does next. |
| `numericist` | sonnet / medium | Mechanical execution. Delivered α_c = 0.66623 and the C19 blind-indicator finding without difficulty. Raise to opus if a task turns genuinely subtle. |
| `analyst`, `librarian`, `replicator`, `auditor` | sonnet | Search, plotting, mechanical verification. Raise per task if warranted. |

**This is tool-matching, not cost-cutting.** The distinction matters: sonnet did excellent work in the
seats above and there is no evidence a larger model would have changed those outputs. If any of these
seats produces work that looks limited by the model rather than the task, raise it and say so.

## Circuit breakers (unchanged)
- Per-agent `maxTurns` caps in each agent definition
- Adversaries fire on **promotion attempts**, not every cycle
- Captain checkpoint if burn crosses a threshold

## Ledger
| Cycle | Date | Agents run | Burn | Notes |
|-------|------|-----------|------|-------|
| 0 | 2026-08-18 | none | ~0 | fitting out |
| 1 | 2026-08-18 | 4 parallel (2 opus/xhigh, 2 sonnet) + 1 nested | ~75% of a 5-h window | Front-loaded. Delivered H9 confirmation, C15 narrowed, V5 settled, α_c, C19. High yield. |
