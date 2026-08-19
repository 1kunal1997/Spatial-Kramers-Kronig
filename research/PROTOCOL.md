# Voyage Protocol

Read this first, every cycle, in any session. This file plus the ledger is the complete handoff —
a cold session with no memory can resume the voyage from these alone.

## Roles

- **Captain** — Kunal. Sets the mission, gives strategic direction at checkpoints. NOT a bottleneck:
  work never blocks on his reply.
- **First Mate** — the main Claude session. Runs cycles, dispatches crew, maintains the ledger.
  Owns nothing scientifically: the First Mate's opinion never promotes a claim.
- **Crew** — subagents in `.claude/agents/`, hired per task.

## The claim ladder

```
conjecture → supported → replicated → hardened → ESTABLISHED
```

| Rung | Requirement | Who signs off |
|---|---|---|
| `conjecture` | An idea. Default state. Free. | anyone |
| `supported` | Evidence exists (numerics or derivation) | line crew |
| `replicated` | Reproduced blind by an independent route | `replicator`: MATCH |
| `hardened` | Survived specific named attacks; code clean | `red-team`: SURVIVED **and** `auditor`: CLEAN |
| `ESTABLISHED` | Novel, significant, not overclaimed | `referee`: ESTABLISH |

**Hard rules:**
1. No claim advances a rung without its signing agent's written verdict in `results/<exp-id>/`.
2. The First Mate cannot promote a claim by judgment. Ever. Confidence is not a rung.
3. Any adversary may demote any claim at any time, including ESTABLISHED ones.
4. A demotion re-opens the hypothesis on `BOARD.md`.
5. Skipping a rung is forbidden. A claim cannot go `supported → hardened`.

## One cycle

1. Read `MISSION.md`, `BOARD.md`, `KILLED.md`, `BUDGET.md`
2. Select the top hypothesis by expected information gain. Check `KILLED.md` first — never re-run a
   corpse without a stated reason why the kill no longer applies.
3. Dispatch line crew **in parallel** on independent threads
4. Results land in `results/<exp-id>/` with a `manifest.json` (params, code path, timestamp, git SHA)
5. For any claim attempting promotion, dispatch the required adversary. Adversaries fire on
   **promotion attempts**, not every cycle.
6. Update `CLAIMS.md`, append `LOG.md`, move dead ends to `KILLED.md` **with the reason**
7. Evaluate checkpoint triggers; schedule the next wake

## Dispatch hygiene

- **`replicator` must never be given the originating code path.** Give it the claim and the physical
  parameters only. Violating this silently destroys the gate — it will reproduce the bug and certify it.
- `red-team` may hire its own hands to actually run counter-experiments (nesting is allowed 3 deep).
- Match tier to job: volume work on sonnet/haiku, opus only where judgment is the product.

## Checkpoint triggers — ping the Captain ONLY for

1. A claim reaches **ESTABLISHED** (victory candidate)
2. The board is exhausted of viable hypotheses (genuine need for direction)
3. A red-team kill invalidates a **foundational** assumption in `MISSION.md`
4. Budget threshold crossed

Everything else queues in `CAPTAIN_QUEUE.md`. **Work continues regardless of whether he answers.**
Silence from the Captain is never a reason to stop.

## Interruption discipline (Captain's order, 2026-08-19)

Work may be terminated without warning when the usage window closes. Therefore:

- **No caps on how long a hand may work.** If the theorist needs 20 minutes it gets 20 minutes. Turn
  limits have been removed from every charter. Never hand out artificially small tasks to bound loss —
  that trades research quality for a saving the checkpoint protocol already provides.
- **Every hand writes a RESUMABLE WORK STATE to disk at least every ~15 minutes.** Not a findings
  dump: goal / established / ruled out / in progress / next step / blockers, kept accurate in place.
  A fresh agent must be able to resume from that file alone after an interruption of hours.
- Maximum exposure is therefore ~15 minutes of one hand's work, regardless of task length.

## Heartbeat (Captain's order, 2026-08-19)

Every hand calls `research/hb.sh <agent> "<note>"` at every checkpoint; the First Mate calls it at
every cycle boundary. Only a live model can write it, so the last line of `HEARTBEAT.log` marks when
requests stopped succeeding. This measures expedition duration for the Captain's **Pro vs Max**
decision — see `MEASURE.md`. Never batch or pre-write heartbeat lines; that destroys their meaning.

## ⚠️ Common-mode error — a limit of independent replication (learned 2026-08-19)

The auditor independently reimplemented C22's benchmark and returned **CLEAN** (1.58× vs the reported
1.70×). Both were **wrong in the same way**: both wrote `np.linspace(-L, L, N)`, which duplicates the
endpoint, so `scipy.signal.hilbert` saw period 2L·N/(N−1). Independent reimplementation did not catch
it because both implementers reached for the same idiom.

**Independent ≠ immune.** When two implementations share a language, a library, or an obvious idiom,
they can share a bug. Guard against it explicitly: vary the *construction*, not merely the author —
grid conventions, endpoint handling, N's factorization (the bug hid because N was a power of two),
and where possible check against a closed form rather than a second numerical route.

## ⚠️ Dispatch hygiene: QUOTE claims, never paraphrase them (learned 2026-08-19)

The C30 replication was dispatched with a statement of the claim that the **First Mate paraphrased**
from a summary rather than quoting from the source document. The replicator then did careful,
pre-registered, rigorous work — possibly against a construction the proof never used. Hours of opus
and sonnet time now hang on which.

**When dispatching a replicator or an adversary against a claim, quote the claim verbatim from its
source document**, with the file and section. If the claim must be restated for a blind agent that
cannot read the source, restate it and then say explicitly: *"this is a restatement; if it is
ambiguous, flag it rather than guessing."* A blind agent cannot check your paraphrase — that is the
whole point of blinding it, and it makes the dispatcher's precision load-bearing.

## Honesty rules

- Negative results are results. Log them with equal care; they shrink the search space.
- A `MISMATCH` from the replicator is the most valuable output in the system. Never tune it away.
- Never report a claim at a higher rung than its verdicts support, including in Captain checkpoints.
