# CHECKPOINT — adjudicator, C30 / V2a dispute
## GOAL
Adjudicate: cycle-2 red team claimed C30 (V2a, Theta(N) query complexity for DiHiTI passivity) PROVED
(research/theory/v2_lower_bound.md). Cycle-3 blind replicator returned MISMATCH
(results/cycle3_v2a_replication/replication.md), arguing the "local law" (bump of any width delta
depresses eps'' by -a/(pi C_phi), delta-independent) is provably false because H[even]=0 at the
bump centre. Determine: PROOF STANDS / NARROWED / BROKEN / PARAPHRASE ERROR.

## ESTABLISHED SO FAR
1. **[read, decisive]** The cycle-2 numerics DO NOT measure the depression at x0. `v2a_compact.py`
   and `v2a_hard_family.py` both bisect on `imin(dihiti(u0 + a*bump))` where
   `imin(v,x) = v[|x|<=0.95L].min()` -- the MINIMUM OVER THE WHOLE INTERIOR WINDOW.
   Lemma 1 in v2_lower_bound.md likewise states `min_x [A(u0+p)](x)` and its proof evaluates
   Delta(x0 + delta*sigma) = -a H[phi](sigma) and then takes `min_sigma`, explicitly noting
   min_sigma Delta = -a max_sigma(-H[phi](sigma)). So the proof's probe point tracks a FIXED
   RELATIVE position sigma* inside the bump -- i.e. the replicator's "offset" construction, NOT its
   "centered/measured-at-x0" construction.
   => The replicator's headline MISMATCH tests a construction the proof never used.
   => H[phi](0)=0 is TRUE and CONSISTENT with the proof (sigma*=0 is not the min; it's a zero).
2. **[read, real defect]** v2_lower_bound.md contains an internal arithmetic inconsistency:
   it states kappa = pi*C_phi ~ 1.31 AND "C_phi = 1/0.418 ~ 2.39". pi*2.39 = 7.5 != 1.31.
   From the table a_min/(pi eps0)=0.418 => kappa=1.31 => C_phi = 0.418. The "1/0.418" is wrong.
3. **[read, suspicious]** In v2a_compact.py, at FIXED N=16384, k=4->8->16 (delta x4) moves the
   uniform bound 1.680 -> 1.500 -> 1.447 (14%), i.e. a_min GROWS as delta shrinks; but at FIXED k,
   4x change in N (delta shrinks 4x physically) leaves it at 1.680 to 4 digits. That pattern says
   a_min depends on GRID POINTS PER BUMP, not on physical delta -> possible discretization lock.
   NEEDS TESTING (see NEXT STEP).

## RULED OUT
- (nothing yet)

## CURRENTLY IN PROGRESS
Writing `adj1_law.py`: with the cycle-2 dihiti, measure for a compact C^inf bump at x0:
 (a) Delta(x0)/a  vs delta      [expect ~0, replicator's fact]
 (b) min_x Delta /a vs delta    [Lemma 1's actual quantity; expect plateau]
 over decades of PHYSICAL delta at fixed large N with the bump always well resolved.
Also: recompute max(-H[phi]) directly for the cbump shape and compare to 1/(pi C_phi).

## NEXT STEP
Run adj1_law.py. Then adj2: a_min vs delta at fixed N with k from 4 to 512 to see whether the
14% "drift" is resolution (small k) or genuine O(delta) (large k).

## OPEN QUESTIONS
- Does the uniform bound r stay finite as delta->0 in the CONTINUUM (not just at fixed k)?
  The theorem only needs boundedness, not exact delta-independence.
- Is the L^inf+TV ball actually a smoothness constraint? TV(bump)=2a regardless of width, so TV
  adds nothing for single bumps -- possible overreach in the doc's rhetoric.
