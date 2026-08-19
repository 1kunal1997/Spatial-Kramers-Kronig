# THEORIST — CYCLE 2 RESUMABLE WORK STATE
Started 2026-08-19. Updated in place. Read this alone to resume.

## 1. GOAL
T3 (first): fix the 233/240 corpus-agreement defect in (L1) verification; re-run all 240;
report new count; VERIFY the mechanism (do not trust my own cycle-1 diagnosis, C28, which an
auditor is checking independently this cycle).
T1: close Wave-3 Open Problem 8.1 (skew tolerance) as a finite computation via (L2)
(3 linear constraints -> <=3-atom extreme points -> degree-3 Q -> Cardano).
T2 (if time): prove/refute the CFM/Choquet conjecture (harmonic test functions vacuous vs Cauchy
omega once M1=0).
Constraint V6: stay at 3 statements (L1,L2,L3). New things must be corollaries.

## 2. ESTABLISHED SO FAR (this cycle)
(nothing yet)

## 3. RULED OUT
(nothing yet)

## 4. CURRENTLY IN PROGRESS
Reproducing block C of research/theory/h9_fixes.py (233/240).

## 5. NEXT STEP
Run h9_fixes.py block C, record the 7 disagreeing profiles by name+params.

## 6. OPEN QUESTIONS / BLOCKERS
- My cycle-1 diagnosis said "conditionally convergent Cayley integral". On inspection the
  integral int ln|eta-xi(t)| g(t) dt in t-coords has only a LOG singularity at t=+-L and is
  ABSOLUTELY convergent whenever hypothesis (H) holds. So "conditional convergence" may be the
  WRONG diagnosis. Suspects instead: (a) the |x|<0.97L evaluation mask (eta capped at ~21) cuts
  the near-seam minimum; (b) the singular-cell fix `logln` misfires near the seam where the
  xi-spacing dxi blows up (bad-cell test d<0.5*dxi triggers for |xi| > 2N/pi and replaces
  ln|xi| by ~2 ln|xi|).
