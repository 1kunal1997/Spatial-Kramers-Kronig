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

---
## TASK 1 CLOSED (2026-08-19). Verdict written to `results/cycle2_e1a/redteam.md`.
E1(a) KILLED. V1 not achieved (best 2.86x, uncertified). C15 stronger. C22 narrowed. Cycle-1
attack 2.2 withdrawn (it was the endpoint-duplication bug F7).

## TASK 2 IN PROGRESS — V2 write-up, `research/theory/v2_lower_bound.md`
Backbone experiments in `results/cycle2_v2a/`:
- `v2a_hard_family.py` (Gaussian bumps, sigma = c*h): a_min/D = 8.95e-3 at c=2, **identical to 3
  digits across N = 2^11..2^15** (8.974e-3 -> 8.952e-3). a_min/m0 ~ 9.0. TV(p)/D = 1.79e-2,
  N-independent. ||p||_1/(D L) -> 0 like 1/N. 25/25 tested positions flip the label.
- `v2a_compact.py` (C^inf COMPACTLY supported bumps on exactly 2k grid points, k=4/8/16):
  41/41 positions flip at every N. max a/D = **1.680 / 1.680 / 1.680** at N = 4096/8192/16384
  (k=4) -- a UNIFORM, N-INDEPENDENT amplitude bound, which is exactly what the pigeonhole needs.
  Number of DISJOINT supports = floor(0.95 L/(k h)) = Theta(N) (1945 at N=2^14, k=4).
  min a/D ~ 1.4e-3 (cheap positions, near the seam where eps''_0 is small); the spread between
  min and max is explained by a_min(x0) ~ pi * eps''_0(x0) -- being confirmed.
NEXT: position-resolved a_min(x0) vs eps''_0(x0) at N=2^13, then write the doc.

## TASK 2 CLOSED (2026-08-19). `research/theory/v2_lower_bound.md` (400 lines, standalone).
- **V2a PROVED.** Lemma 1 (local flip law a_min(x0) = kappa*eps''_0(x0), kappa/pi = 0.418, INDEPENDENT
  of bump width) proved analytically from the master identity C1 -- the singular part of the DiHiTI
  kernel is the Hilbert kernel, which commutes with dilation, so the dip depth is width-independent.
  Verified to +-3% over three decades of eps''_0 and to 0.3% over N = 2^11..2^15.
  Theorem 2 (deterministic pigeonhole, Theta(N)) + Remark (holds against the FINITE M+1 input family)
  + Corollary 3 (Yao, randomized) + Corollary 4 (TIGHT: query complexity is Theta(N)).
- **V2b written as a CONDITIONAL/method-relative barrier**, with an explicit scope statement at the
  top forbidding the unconditional reading. Core: both known superlinear techniques (Ben-Or
  component counting, Bjorner-Lovasz-Yao face counting) are entropy counts of combinatorial
  multiplicity; a convex polyhedral cone with Theta(N) facets has 1 component and <=2^Theta(N)
  faces, so both are capped at O(N) BY ARITHMETIC. Element distinctness gets N log N from N!
  orderings; there is no analogue. Morgenstern's O(N log N) is for COMPUTING the DFT in
  bounded-coefficient linear circuits and does not transfer to deciding one bit.
  Three named escape routes R1/R2/R3, with R3 (bounded-coefficient linear DECISION trees) flagged as
  the one this crew could actually attempt. Explicit non-claims section.
- Section 3.5 states the honest boundary and, importantly, explains TASK 1's failure abstractly:
  a bandwidth promise defeats the lower bound, but VERIFYING a bandwidth promise is itself Omega(N),
  and bounding between-sample behaviour from samples is impossible without an a priori class --
  which is exactly why F5/F8's certificates were unsound.

## STATUS: BOTH TASKS COMPLETE. Nothing in progress.
## NEXT STEP for a successor: O2 (compiled/numba implementation of the split test -- the only
untested lever on V1) or O1/R3 (Morgenstern-type bound for bounded-coefficient linear decision
trees, the crew-sized version of V2b).
