# CHECKPOINT — solo Fable session, 2026-08-19: the two-sided question

## GOAL
Answer the ONE QUESTION (FABLE_BRIEF): necessary-and-sufficient conditions on ε′ for the DiHiTI
partner ε″ ≥ 0 — or a rigorous proof that no criterion of the wanted kind exists.

## PLAN (formed after reading STATE/MISSION/BOARD/KILLED/CLAIMS/h9/redteam/v2_lower_bound)
Deliverable = one theory document `research/theory/ns_answer.md` containing:

1. **Theorem 1 (density of essential test points).** Any test set E ⊂ ℝ (Cayley η-points) that is
   sound and complete for passivity must be DENSE. Upgrade of C27 (no finite node set). Proof via
   boundary two-atom family (Cor 4.A) + threshold depth law (C40) + uniform tail bound.
2. **Theorem 2 (no finite linear battery).** For ANY finite set of linear functionals ℓ₁..ℓ_k of g
   (moments, samples of ε′, corner masses, M₁, M₂, σ, ...) and ANY decision function F:ℝ^k→{P,F},
   the rule F(ℓ(g)) misclassifies — witnesses are two positive balanced atomic mixtures with the
   SAME atoms, differing only in weights. Kills N1-as-criterion, subsumes the "clean condition" class.
   Proof: (k+2)-pair mixture cone; f(m)=sup_η Δ; boundary point with unique active η*; Danskin +
   independence lemma (log singularities) ⟹ some kernel direction crosses the boundary.
3. **Proposition 3 (simplex rigidity).** Passive cone = A⁻¹(positive cone), lattice/simplex cone;
   unique extremal representation = ε″ itself ⟹ every complete certificate re-represents ε″.
4. **Positive side.** The N&S that DOES exist: (L1) one law; exact finite collapse for n-layer
   staircases (L3 node test = the per-profile two-sided criterion for every fabricable profile);
   (L2) family-level two-sided laws. Frame per board item P4 ((L1) = frame, (L2)/(L3) = content).
5. **N1 closure.** Closed-form λ*(box, bulk) for corner-mass mixtures; show bulk-dependence ⟹ no
   universal corner-mass threshold; control = red-team's table (0.262 @ 1.30L bulk ±0.1, etc.).

## NUMERICAL CONVENTION (P0-aware, stated up front)
All constructions atomic with M₁ = 0 exact, atoms bounded away from seam, E = 0.
Primary evaluator: exact closed-form Δ(η) + node collapse (roots of Q) — grid-free.
Cross-check evaluator: production ht_derivative on smoothed staircase, min over interior nodes
(endpoints excluded, structural zeros), argmin reported; seam-blindness not in play since M₁ = 0
exactly and gain (when present) sits at finite interior η. Controls per claim.

## STATUS
- [x] Ledger read (STATE, MISSION, BOARD, KILLED, CLAIMS, h9, c25 redteam, v2_lower_bound, P0, C30)
- [x] common.py evaluators + controls (e0: Cor4.A threshold 1.000000000 x3; C40 slopes 4 digits;
      production match 3-4 digits on gain cases; red-team lambda* 0.262/0.278 reproduced)
- [x] Theorem 1 numerics (e1: violation confined to any target window, delta=0 control exact;
      positivity width ~ sqrt(delta); outer branches strictly negative)
- [x] Theorem 2 numerics (e2: k=8 battery incl. FULL known cascade defeated, match 6e-16,
      verdicts corroborated by production; k=16 defeated, match 8e-12, production corroborates
      at N=2^20-2^22 after identifying an O(h) common-mode production offset — the DIFFERENTIAL
      prod(gP)-prod(gG) = 3.382e-5 matches closed form 2x1.691e-5 at every N)
- [x] N1 closure numerics (e3: lambda* controls 4/4; lambda* range [0.030, 0.383] for the SAME
      corner geometry as bulk varies; killer pair with identical corner distribution PASSIVE/GAIN)
- [x] e4: Lemma E envelope construction — active point sweeps contiguous [1.45, 2.96]
      (152/281 grid theta, no gaps; sup = 0 AND argmax = theta at each; Delta'' in
      [-0.506, -0.069], all strictly negative => touching max nondegenerate)
- [x] ns_answer.md written with proofs + status labels (research/theory/ns_answer.md)
- [x] STATE.md, BOARD.md (N1 closed, N2 refuted, F1/F2 added), CLAIMS.md (C48-C54)
- [x] HANDOFF.md written (this directory) — successor pickup point

## SESSION COMPLETE 2026-08-19. Pick up from HANDOFF.md; top task = board F1
## (analytic closure of Lemma E), then red-team + blind replication of C49.

## NEW FACTS worth ledger entries (beyond plan)
- Production ht_derivative carries a PROFILE-DEPENDENT O(h) additive offset on staircase
  profiles (3.55e-5 at N=2^19 -> halves per N doubling; s-independent below s~6e-4).
  At the corpus's N=2048-2^14 this is ~1e-3-scale: near-threshold production verdicts are
  biased. Cancels in differences. Feeds P0.
- Outer-branch strict negativity of pair Delta PROVED (closes red-team residual risk #3).
- T2/CFM conjecture (theorist 12.3, board N2): REFUTED as stated. f(z)=cos(x)e^{-y} is
  harmonic, bounded on C+, omega-integrable; P=delta_0 passive yet int f dP = 1 > e^{-1}.
  The subharmonic cone is strictly bigger than the log-potential cone here.

## DEAD ENDS HIT THIS SESSION (log as they happen)
- T2-as-full-Choquet-order is FALSE: counterexample f(z) = cos(x)e^{−y} (harmonic, bounded on ℂ₊,
  ω-integrable): P = δ₀ passive but ∫cos dP = 1 > ∫cos dω = e^{−1}. The subharmonic test cone is
  strictly too big; the theorist's §12.3 hunch ("harmonic parts essentially affine") fails against
  oscillatory bounded harmonics. The correct classical home is likely the potential order of the
  SYMMETRIC CAUCHY PROCESS on ℝ (recurrent log potential kernel) and Rost's filling scheme /
  stopping-distribution order — recorded in the doc as a labelled identification, not leaned on.
