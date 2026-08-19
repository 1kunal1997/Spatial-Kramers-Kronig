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

## 2. ESTABLISHED SO FAR  (scripts: `e1a.py`, `diag_onehilb.py`; raw: `e1a_results.json`)

**F1 (verified-numerically). The cycle-1 split test NEVER USED its cascade.** In
`results/cycle1_c15_redteam/split_test.py`, `split_verdict()` returns the decimated minimum alone;
`cascade()` is computed and PRINTED but never gates the answer. So cycle 1's headline "20/24
sign-correct, and the 4 misses are only the adversarial cosines" is an accuracy figure for
DECIMATION ALONE (row S2 here), while the 0.88x wall clock was paid for a cascade that bought
nothing. Cycle 1's claim that "the cascade catches exactly the adversary that defeats decimation"
was INDICATED BY A TABLE, NEVER WIRED IN OR TESTED. My own cycle-1 report overstated this.

**F2 (verified-numerically). `onehilb` is NOT interchangeable with `dihiti` at fixed N — it is
~20x less accurate per grid point, and the deficit exceeds the margin.** `diag_onehilb.py`,
logistic family, interior |x|<=0.95L, value/D:
  w=0.05: N=2048 dihiti +1.0556e-3 vs onehilb +2.4383e-4 (bias -8.1e-4)
          N=2^18 dihiti +1.0043e-3 vs onehilb +9.9213e-4 (bias -1.2e-5)
  Bias decays ~N^-0.82 (ratio 0.566 per doubling), same shape for every w. Both converge to the
  same limit, so C22's algebraic identity is fine; but at the DECIMATED size n=512 the onehilb bias
  is ~2.4e-3.D, i.e. LARGER THAN THE +1.0e-3.D PASS MARGIN. Consequence: a decimated test must use
  `dihiti` (whose grid bias at n=2048 is only 5e-5.D), not the "faster" `onehilb`. **The 1.70x
  baseline win (C22) is a speed win only; it costs ~4-8x in grid resolution at fixed accuracy.**

**F3 (verified-numerically). THE CASCADE DOES NOT SURVIVE DECIMATION — E1(a)'s core premise fails.**
The screen quantity is |g(-L)-g(+L)|/D. Full grid N=2^16 vs decimated n=512:
  | family | |slope|/D at N | at n=512 |
  | logistic w=0.05 | 1.5e-11 | 3.5e-09 |
  | logistic w=0.3  | 8.3e-12 | 1.4e-03 |
  | bumps (all a)   | ~1.2e-11 | 2.3e-03 .. 5.7e-03 |
  | cos m=257 (adversary) | 3.98e-01 | 3.5e-09 |
  | cos m=512/1024/2048   | 1.6 / 6.3 / 25 | 2.4e-05 / 9.6e-05 / 3.8e-04 |
The ten-orders-of-magnitude separation cycle 1 relied on is a FULL-GRID property. On the decimated
grid it collapses TWO WAYS at once:
  (i) a truncation FLOOR ~ h_n^2*||g'''|| ~ 1e-3.D contaminates smooth profiles (false DEFERs), and
  (ii) high-m dressings ALIAS to nothing (cos m=257 reads 3.5e-9, i.e. cleaner than the logistic),
       so the screen is blind to precisely the adversary it exists to catch (false ANSWER).

## 6. OPEN QUESTIONS / BLOCKERS
- **Foreseen tension (must be confronted, not hidden):** if the cascade moves to the decimated grid,
  the test reads only n of N samples => it is a SUBLINEAR-QUERY procedure. My own TASK 2 (V2a,
  Omega(N)) says no sublinear-query procedure can be correct in the worst case. So E1(a) can at best
  be a heuristic for a smoothness class, NOT a decision procedure. Need to state the input model
  under which any V1 claim is made.
