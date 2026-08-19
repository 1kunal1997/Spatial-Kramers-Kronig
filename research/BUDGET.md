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

## 🛑 DISPATCH BUDGET — the Captain's stop-at-75% control

The First Mate **cannot see usage telemetry**. So the budget is enforced by **counting dispatches**,
not by reading a meter.

**Measured costs (Pro plan, 5-hour window):**
| Hand | Approx. share of one Pro window |
|---|---|
| opus / xhigh (theorist, red-team) | **~15%** each |
| sonnet (numericist, auditor, replicator, librarian) | **~7%** each |
| First Mate integration per cycle | ~5% |

So **4 parallel hands (2 opus + 2 sonnet) ≈ 45–50%** of a Pro window, plus integration.
On Max (5×), divide all figures by ~5.

**How the Captain sets a cap:** state a budget at the start of a session —
*"you get 60% of this window"* or *"you get 6 hand-dispatches."* The First Mate keeps a running tally
below, **stops when the cap is reached regardless of progress**, and reports rather than continuing.

**Default cap if the Captain says nothing: 60% of a window.** Never spend the whole window — the
Captain needs headroom to work interactively.

**Tally (reset each session):** dispatches used: 0

## Circuit breakers (unchanged)
- Per-agent `maxTurns` caps in each agent definition
- Adversaries fire on **promotion attempts**, not every cycle
- Captain checkpoint if burn crosses a threshold

## Ledger
| Cycle | Date | Agents run | Burn | Notes |
|-------|------|-----------|------|-------|
| 0 | 2026-08-18 | none | ~0 | fitting out |
| 1 | 2026-08-18 | 4 parallel (2 opus/xhigh, 2 sonnet) + 1 nested | ~75% of a 5-h window | Front-loaded. Delivered H9 confirmation, C15 narrowed, V5 settled, α_c, C19. High yield. |
