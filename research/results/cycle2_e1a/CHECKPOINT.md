# CHECKPOINT — Cycle 2 Red Team — E1(a) + V2 write-up
*Resumable work state. Reader = fresh agent, no memory. Updated in place.*

**Last updated:** 2026-08-19 (cycle start)

## 1. GOAL
Two tasks from the First Mate:
- **TASK 1 / E1(a):** move the cycle-1 split test's O(N) "cascade" onto the DECIMATED grid (n=512),
  so the whole test costs O(n log n) instead of O(N)+O(n log n). Cycle 1 measured the split test at
  0.88x wall clock vs the naive `dihiti` baseline; the profiler blamed N transcendental evaluations
  in the cascade. If decimating the cascade preserves accuracy, the test should land at 10-30x and
  claim V1 (>=3x), killing C15.
  DISCIPLINE ORDERS: benchmark vs the BEST baseline (`onehilb`, 1.70x faster than `dihiti`), report
  WALL CLOCK, test the adversarial cosine dressings (4/24 misses in cycle 1), and verify any bound
  rather than assuming it (K13 was an unsound-bound death).
- **TASK 2 / V2:** write `research/theory/v2_lower_bound.md` — (V2a) Omega(N) query lower bound with
  an explicit query model; (V2b) structural/CONDITIONAL barrier explaining why the log factor
  resists. Must never be written as an unconditional lower bound.

## 2. ESTABLISHED SO FAR
(nothing yet this cycle)

## 3. RULED OUT
(nothing yet this cycle; inherited kills K13-K17 in KILLED.md)

## 4. CURRENTLY IN PROGRESS
Setting up `results/cycle2_e1a/` harness, ported from `results/cycle1_c15_redteam/split_test.py`.

## 5. NEXT STEP
Write `e1a.py`: baselines (dihiti, onehilb), decimated-cascade split test, 24+ profile accuracy
table, timeit wall clock at N=2^12..2^16.

## 6. OPEN QUESTIONS / BLOCKERS
- **Foreseen tension (must be confronted, not hidden):** if the cascade moves to the decimated grid,
  the test reads only n of N samples => it is a SUBLINEAR-QUERY procedure. My own TASK 2 (V2a,
  Omega(N)) says no sublinear-query procedure can be correct in the worst case. So E1(a) can at best
  be a heuristic for a smoothness class, NOT a decision procedure. Need to state the input model
  under which any V1 claim is made.
