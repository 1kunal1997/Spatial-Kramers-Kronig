# CHECKPOINT — F1: analytic closure of Lemma E + thickening (Theorem 2, ns_answer.md §4)

**Agent:** theorist-F1. **Started** 2026-08-19. Git SHA at start: 9c1b0d2.
**Write target:** `research/theory/lemmaE_closure.md`; scripts here.

## GOAL
Remove the caveat "proved, with one numerically-certified constructive ingredient" from Theorem 2 of
`research/theory/ns_answer.md` §4: (1) prove analytically that the active-point set A* contains an
interval (Lemma E), (2) discharge the thickening step in full. Target label: **proved**.

## ESTABLISHED SO FAR (all analytic, checked by hand; numerics NOT yet run)

Notation (from `results/fable_ns_answer/common.py`): L=1, ξ(t)=tan(πt/2), balanced pair P(a,b) has
atoms ξ=β=tan(πb/2), ξ=−γ=−tan(πa/2), weights γ/(β+γ) at β and β/(β+γ) at −γ (mass 1, M₁=0).
Δ_g(η)=Σw_j ln|η−ξ_j| − (W/2)ln(1+η²); passive ⟺ Δ_g ≤ 0 on ℝ.

**(P1) [proved, re-derived by hand]** For a mass-1 balanced pair,
Δ′(η) = [η(1+βγ) + (γ−β)] / [(η−β)(η+γ)(1+η²)] = (1+βγ)(η−η*) / [(η−β)(η+γ)(1+η²)],
η* = (β−γ)/(1+βγ). So Δ has exactly ONE finite critical point, at η*; unimodal on (−γ,β).
(This is Lemma B of ns_answer.md; I verified the algebra independently.)

**(P2) [proved]** Convex-analytic restatement of the passive cone over a fixed pair set P₁..P_n:
M(m) := max_{η∈H} Δ_m(η) (H = convex hull of all atoms) is a finite, convex, positively homogeneous
function of m ≥ 0, and 𝒫∩ℝⁿ₊ = {m ≥ 0 : M(m) ≤ 0}. ∂M(m) = conv{v(η) : η ∈ A(m)} (Danskin),
v(η) := (Δ_{P_j}(η))_j. Step (D) of ns_answer.md is exactly: v(η) ⊥ ker M_battery for every active η.

**(P3) THE BASE POINT — the key new object. [proved, elementary algebra]**
Take **two SYMMETRIC balanced pairs** (a=b, so atoms ±β, weights ½,½) with **reciprocal** β's:
  G = pair with atoms ξ = ±1/2  (t = ±(2/π)arctan(1/2) ≈ ±0.29517), mass 1
  B = pair with atoms ξ = ±2    (t = ±(2/π)arctan(2)   ≈ ±0.70483), mass 1
p₀ := G + B, i.e. atoms ξ ∈ {−2,−1/2,1/2,2} each weight 1/2, W=2, M₁=0.
Then, with y = η²,
  Δ_{p₀}(η) = ½ ln[ |η²−4|·|η²−¼| / (1+η²)² ]  =: ½ ψ(y),  ψ(y)=ln[|y−4||y−¼|/(1+y)²].
**Passivity ⟺ |y−4||y−¼| ≤ (1+y)² for y ≥ 0**, proved by three elementary cases:
  y∈[0,¼]: (4−y)(¼−y) − (1+y)² = −25y/4 ≤ 0, equality iff y=0.
  y∈[¼,4]: (1+y)² − (4−y)(y−¼) = 2y² − 9y/4 + 2, discriminant 81/16 − 16 < 0 ⟹ > 0.
  y≥4:     (1+y)² − (y−4)(y−¼) = 25y/4 > 0.
So **Δ_{p₀} ≤ 0 on ℝ with equality EXACTLY at η=0** — an explicit boundary point of 𝒫.
Nondegeneracy is free and exact: Δ″_{p₀}(0) = ψ′(0) = −(Σ m_j/β_j² + W) = −(4 + ¼) − 2 = **−25/4 < 0**.
(General fact behind it: for symmetric-pair mixtures Δ(η)=½ψ(η²), Δ′(0)=0 always and
Δ″(0)=ψ′(0)=−Σm_j/β_j² − W < 0 ALWAYS. Nondegeneracy needs no measurement.)

**(P3a) [proved]** General reciprocal family: β_G=1/β_B=β<1, equal masses ⟹ passive ⟺ c:=β²+β^{−2} ≤ 6
⟺ β ≥ √2−1. At β=√2−1 a second touching point appears at y=1 (bifurcation). β=1/2 gives c=17/4,
comfortably inside.

**(P4) THE SWEEP + THICKENING IN ONE (Master Lemma). [proved modulo my writing it out]**
Fix n pairs: P₁=G, P₂=B, and P₃..P_n arbitrary balanced pairs, all 2n atoms pairwise distinct and
none equal to 0. Let u = Σ_{j≥3} e_j (direction adding the missing pairs), w = e₁ (the G coordinate).
Solve F(η,r;t) = (Δ_m(η), Δ′_m(η)) = 0 with m = p₀ + t u + r w.
Jacobian ∂(F₁,F₂)/∂(η,r) at (0,0,0) = [[Δ′_{p₀}(0), Δ_w(0)],[Δ″_{p₀}(0), Δ′_w(0)]]
                                     = [[0, −ln2],[−25/4, 0]], det = −(25/4)(ln2) ≠ 0.
IFT ⟹ analytic η(t), r(t), η(0)=0, r(0)=0, and
  ṙ(0) = −Δ_u(0)/Δ_w(0),  η̇(0) = −[Δ′_u(0) + Δ′_w(0)ṙ(0)]/Δ″_{p₀}(0) = (4/25)·Δ′_u(0)
because Δ′_w(0)=0 (w symmetric). For a pair with atoms (β,−γ): Δ′_P(0) = (β−γ)/(βγ).
So **choose P₃..P_n asymmetric with Σ_{j≥3}(β_j−γ_j)/(β_jγ_j) ≠ 0** ⟹ η̇(0) ≠ 0 explicitly.
Global passivity of m(t) for small t>0: (i) outside H, Δ_{m(t)}<0 by Lemma B (all weights >0);
(ii) on H∖(−ρ,ρ), Δ_{p₀} ≤ −ε (u.s.c. max on compact) and the perturbation is ≤ C t uniformly
(each Δ_{P_j} is bounded ABOVE on H); (iii) on [−ρ,ρ] chosen atom-free with Δ″_{p₀} ≤ −25/8,
Δ_{m(t)} is strictly concave, so its critical value 0 at η(t) is the max there.
⟹ m(t) ∈ ∂𝒫, all n coordinates > 0 (they are (1+r, 1, t,…,t)), unique active point η(t),
η(t) strictly monotone near 0 ⟹ **A* ⊇ an open interval**. Lemma E + thickening, done, analytic.

## RULED OUT (do not re-walk)
- **Original explicit triple** (bad₁,bad₂,good asymmetric pairs): globality of the max has no clean
  closed form; would need degree-7 Sturm certification. Abandoned for the symmetric construction,
  where y=η² turns Δ into a rational-argument log and the node polynomial N(y) is LINEAR.
- **"Active point must be infinite, get it from Bolzano–Weierstrass"**: works logically (real-analytic
  vanishing on a set with an accumulation point ⟹ Lemma A), but still needs a moving active point.
- **Union of A(p) over a connected family is connected**: FALSE in general (argmax can jump between
  two humps). Do not use.
- **Mixture of threshold pairs (a+b=1) as base point**: strictly INTERIOR to 𝒫 (their Δ's touch 0 at
  different η*), so it is not a boundary point. Dead.
- **Small-δ ballast with fixed bad pair**: the re-touch parameter ρ_c is O(1), not small, so the
  "positivity interval shrinks" trick does not apply at fixed pair set. Dead.
- Asymmetric-pair base points: Δ′(0)≠0 breaks the free nondegeneracy; symmetric is the right choice.

## IN PROGRESS
Nothing half-done in the algebra. Next is verification + writing.

## NEXT STEP
Write `results/f1_lemmaE_closure/e1_basepoint.py`: (a) verify p₀ is on ∂𝒫 with unique max at η=0
via the exact node test in `results/fable_ns_answer/common.py` AND a dense independent grid;
(b) verify Δ″(0) = −25/4 by finite differences; (c) CONTROL: β=0.3 (< √2−1) must come out NOT
passive (sup Δ > 0), and β=1/2 with the balance broken (β_B = 3 instead of 2) must give Δ(0) ≠ 0.
Then e2_sweep.py: numerically continue (η(t), r(t)) for an explicit n=3 (or 5) configuration,
confirm componentwise positivity, sup Δ = 0 attained at η(t), and match η̇(0) = (4/25)Δ′_u(0).

## BLOCKERS
None.
