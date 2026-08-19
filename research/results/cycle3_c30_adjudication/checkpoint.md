# CHECKPOINT — adjudicator, C30 / V2a dispute — **COMPLETE**
## GOAL
Adjudicate cycle-2 red team (C30 / V2a proved) vs cycle-3 blind replicator (MISMATCH).

## VERDICT (final, written to adjudication.md)
**PARAPHRASE ERROR** primary + **PROOF NARROWED** secondary.

## ESTABLISHED (all verified numerically, scripts in this dir)
1. Lemma 1 measures `min_x` of the perturbed eps'', NOT the value at x0. Cycle-2 code bisects on
   `imin(...)= v[|x|<=0.95L].min()`. The replicator tested value-at-x0. Wrong construction.
2. H[phi](x0)=0 exactly (both bump shapes) — TRUE, and consistent with the proof: sigma=0 is a ZERO
   of the depression profile; Lemma 1 uses its MINIMUM at sigma*=-0.7231.
3. Lemma 1 VERIFIED: min_x Delta/a = -0.79758 = -max(-H[phi]) exactly; flat to 0.1% over 256x in
   physical delta (adj1_law.py); Gaussian at fixed x0 flat to 0.2% over 50x (adj3 part ii).
4. x0-independence holds: 3.6% in min_x Delta/a, kappa spread 1.078x / 1.8% RMS (adj2).
5. Replicator's factor-2.6, 15% drift and 1.64x x0-spread are ONE artifact: probing at sigma~-1
   (bump edge) where -H = 0.388 instead of the max 0.661/0.798; max/edge ratio = 1.64-1.70 (adj4).
6. Cycle-2's r/D=1.680 explained exactly: 1.1411/0.67910 (k=4 discrete bump). k-drift is resolution,
   converging UP to 1.431, direction opposite to the document's stated reading.
7. C22 endpoint bug irrelevant here (6th sig fig).
8. M4 (hand B) non-monotonicity is a WINDOW-EDGE artifact (easiest x0*=+0.97L, outside the 0.95L
   scored window); the document's "under-resolved at 82 grid points" excuse is FALSE.
9. NARROWINGS: (N1) C_phi constants self-contradictory, pi spurious, correct kappa=1/h_phi=1.2538;
   (N2) TV bound vacuous for bumps, any N-independent Lipschitz bound collapses Theta(N) to Theta(1);
   (N3) intervals must be inset by 2kh from the scored window; (N4) x0-independence only O(delta).

## RULED OUT
- Grid convention / endpoint duplication as a cause. - N-dependence of any kind (0.003%).
- Any fatal flaw in Theorem 2 / Cor 3 / Cor 4 logic.

## NEXT STEP (for whoever follows)
Update CLAIMS.md: C30 -> `supported` reworded per adjudication.md §8/§9; C37 -> RESOLVED, not upheld.
Edit theory/v2_lower_bound.md per N1-N4.
