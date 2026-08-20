# The minimal spanning set of one-sided conditions (Goal B / V8)

**Solo Fable session, 2026-08-20.** Scripts and raw output: `research/results/fable_goalB/`
(`exp1_parents.py` … `exp4_ladder_gapstats.py`, outputs `exp*_output.txt`, `manifest.json`).
All verdicts below use the audited exact node test (`results/fable_ns_answer/common.py`,
`exact_sup_gap`) — no FFT, no linspace grid, no production `ht_derivative` anywhere.

---

## 0. Summary — the answer to V8 in five statements

The one-sided corpus collapses into a **primal/dual pair** for the single convex cone 𝒫:

1. **Theorem S (the sufficient parent, primal side).** g ≥ 0, M₁ = 0, and the drop
   measure splits as ν = ν₀ + ν₁ with supp ν₀ ⊆ [−a, b], a + b ≤ L (balanced) and ν₁
   even unimodal ⟹ passive. C3, C25′, C4, C5 are one-line corollaries; C6/C7 are
   evaluations of its generator margin. Two closure lemmas (dilation, inversion — §2,
   both new, both one-line proofs) enlarge S for free.
2. **Theorem N (the necessary parent, dual side).** Passivity restricted to the closed
   outer half {|x| ≥ L/2} plus the centre. The entire C8 cascade, C9 and C13 are its
   Taylor coefficients at the seam (1–2 lines each); σ ≥ 0 is the ω-average of the full
   law (one line, §3.4). Properly one-sided: quantified below.
3. **Proposition NG (no finite zoo — NEW, proved).** On every Lemma-E2 atom set the
   passive cone is **non-polyhedral**, hence *no finite list of generator families can
   ever span 𝒫*. This is the sufficient-side mirror of Theorem 2/C49, and it is proved
   by the **same** Lemma E2. One lemma kills finite batteries (dual) and finite zoos
   (primal). The minimal spanning set is therefore bounded below on *both* sides: {S, N}
   is not completable, only extendable forever.
4. **The ladder (the gap has self-similar structure).** The first rung beyond S is exactly
   solvable: **Theorem S-τ** — inversion-symmetric drops with all radii in [L/4, 3L/4]
   are passive, sharp at L/4 (proved from Lemma E1′/C55; arctan(√2−1) = π/8 is why the
   constant is L/4). It reproduces C3's "middle half" motif one level up, on the
   *radius* axis. Beyond it sits a two-scale witness family with a closed-form threshold
   (§5.3), outside every known generator cone.
5. **The gap, quantified (B3).** Exact bands: a single symmetric pair needs radius ≤ L/2
   to be passive but only ≤ 2L/3 to pass N. Ensembles: 1.5% of random even profiles
   satisfying N hide gain; S certifies 39.8% of random passive atomic profiles.

Corpus count: **ten one-sided statements → two parents**, plus two no-go theorems that
prove two is where compression stops. Labels: everything in §2–§4 and §6 is **proved**
(§5.1 proved modulo C55's rung); §5.3's binding-point claim and all percentages are
**verified numerically**.

---

## 1. Setting

L = 1, x ∈ (−1, 1), Cayley coordinate ξ(t) = tan(πt/2). Drop density g = −dε′/dx ≥ 0
pushed to ξ gives the drop measure μ, total mass W. Gap function

    Δ_μ(η) = ∫ ln|η − ξ| dμ(ξ) − (W/2) ln(1 + η²),      π ε″(x) = −Δ_μ(ξ(x)),

**passive ⟺ Δ_μ ≤ 0 on ℝ.** Regions: **outer** O = {|η| ≥ 1} ⟺ |x| ≥ L/2 (where, by
C56/G4, Δ can be convex — the danger zone); **middle** |η| < 1 ⟺ |x| < L/2 (where Δ is
strictly concave off atoms, for every profile). The **radius** of an atom is |t|; the
**log-radius** is τ = 2 ln|ξ|. The full-window uniform drop (equilibrium measure λ, =
Cauchy in ξ) has Δ ≡ 0 — the unique neutral profile.

---

## 2. Two covariance lemmas (new, proved, one line each)

**Lemma D (dilation).** For c > 0 let μ_c be the push-forward of μ under ξ ↦ cξ. Then

    Δ_{μ_c}(η) = Δ_μ(η/c) + (W/2) ln[(c² + η²)/(1 + η²)].

*Proof.* ln|η − cξ| = ln c + ln|η/c − ξ|; add and subtract (W/2)ln(1 + η²/c²). ∎

The correction is ≤ 0 for every η iff c ≤ 1. Hence:

> **Corollary D1 (contraction closure).** The passive cone is closed under Cayley
> contractions ξ → cξ, c ≤ 1 (in window coordinates: t ↦ (2/π)arctan(c·tan(πt/2)),
> which pulls every drop toward the centre). Expansions (c > 1) are *not* safe (control:
> 380/800 random passive profiles lose passivity under a random expansion; contraction
> violations 0/800; identity verified to 1e−16 — `exp2_output.txt` block B).

One-line reproofs it buys: S(β) passive for β ≤ 1 (contract the boundary pair S(1));
scaled-Cauchy profiles (logistic-like ε′) passive for scale c ≤ 1 (contract λ).

**Lemma I (inversion).** For μ with 0 ∉ supp μ let μ* be the push-forward under
ξ ↦ 1/ξ (window coordinates: |t| ↦ 1 − |t|, reflection about the |t| = L/2 circle). Then

    Δ_{μ*}(η) = Δ_μ(1/η) − Δ_μ(0),        Δ_μ(0) = ∫ ln|ξ| dμ.

*Proof.* ln|η − 1/ξ| = ln|1/η − ξ| + ln|η| − ln|ξ|, and ½ln(1+1/η²) = ½ln(1+η²) − ln|η|. ∎
(Verified to 5.6e−17 on random asymmetric measures.)

> **Corollary I1.** If μ is passive with Δ_μ(0) = 0, then μ* is passive. (The
> inversion-symmetric sector of §5 lives exactly on this facet.)
> **Corollary I2 (the criterion factors through the outer half).** μ is passive ⟺
> Δ_μ ≤ 0 on O **and** Δ_{μ*} ≤ −Δ_μ(0) on O. The middle-half constraints of μ are the
> outer-half constraints of μ*, shifted by the centre value. The outer half is the
> universal danger zone in an exact sense, not only the concavity sense of C56.
> **Corollary I3 (centre condition).** Passivity requires Δ_μ(0) ≤ 0, i.e.
> ∫ g(t) ln tan(π|t|/2L) dt ≤ 0: the **geometric mean of the Cayley radii is ≤ 1**
> (equivalently the log-radius mean is ≤ 0 — the "τ-balance", the second balance law
> after M₁ = 0).

---

## 3. Theorem N — the necessary parent, and the corpus cascade as its jet

> **Theorem N.** If the DiHiTI partner is passive then
> (N1) Δ ≤ 0 on the closed outer region O = {|η| ≥ 1}  (ε″ ≥ 0 on |x| ≥ L/2), and
> (N2) Δ(0) ≤ 0  (ε″(0) ≥ 0, the centre/τ-balance condition).
> N is **properly** one-sided: §7 quantifies the profiles it passes that have gain.

N1/N2 are restrictions of the criterion — their content is which *evaluations* the corpus
conditions are. Every prior necessary condition is ⟨−Δ, ν⟩ ≥ 0 for one specific
nonnegative test object ν:

**3.1 The seam jet (⟸ N1, η → ±∞).** With M₁ = ∫ξ dμ, M₂ = ∫(1+ξ²) dμ = ∫g sec²(πt/2L)dt:

    Δ(η) = −M₁/η − M₂/(2η²) − M₃′/(3η³) + O(η⁻⁴)          (M₃′ = ∫ξ³dμ)

- **O(η⁰): C13.** Δ(±∞) = 0 — ε″(±L) = 0 is the zeroth coefficient, an identity.
- **PV-convergence of M₁: edge-slope condition** g(+L) = g(−L) (the ξ-density tails must
  cancel; H9 §6).
- **O(η⁻¹): M₁ = 0.** Δ ≤ 0 for both signs of large η kills the odd leading term. [1 line]
- **O(η⁻²): M₂ ≥ 0.** [1 line; automatic for g ≥ 0, binding for signed g.]
- **C9 depth law.** Maximise the two-term truncation: η* = −M₂/M₁, max Δ = M₁²/(2M₂),
  hence min ε″ ≈ −M₁²/(2πM₂). [2 lines]

**3.2 The centre (⟸ N2).** Corollary I3 above. New as a *named* condition; it is the
τ-analogue of M₁ = 0 and the first coefficient of the seam jet **of μ\*** (Lemma I turns
the η → 0 expansion of Δ_μ into the η → ∞ expansion of Δ_{μ*}).

**3.3 Sharpness of the L/2 boundary in N1.** C56 (universal strict concavity of Δ on
exactly the middle half) is the local reason O is the right region: outside O nothing
protects Δ; inside, every atom-free interval carries a concave cap. But the middle is
**not redundant**: see §7.

**3.4 The one member N does not imply: σ ≥ 0.** σ = ∫g ln(2cos(πt/2L))dt satisfies the
identity **σ = −∫Δ dω** (ω = Cauchy/harmonic measure): h = −Δ extended harmonically to
ℂ₊ has h(i) = ∫h dω by the Poisson formula, and h(i) = W ln2 − ∫ln sec(πt/2L)dμ = σ.
[2 lines] So σ ≥ 0 is the **ω-average of the full criterion** — it samples the middle,
and N1+N2 alone do not imply it. This is the precise sense in which the necessary side's
true parent is the single function Δ tested against nonnegative measures: the corpus is
five specific test measures (seam jet ×3, centre point, ω), and by **C49 no finite set of
such tests is ever complete**. N is the strongest *clean region* restriction, not a
completable list.

---

## 4. Theorem S — the sufficient parent

> **Theorem S.** Let g ≥ 0 with M₁ = 0 and suppose the drop measure splits as
> ν = ν₀ + ν₁ where (i) supp ν₀ ⊆ [−a, b] with a + b ≤ L and ν₀ balanced (M₁(ν₀) = 0),
> and (ii) ν₁ is even and unimodal. Then the partner is passive. Moreover the passive
> cone — hence S — is closed under the contractions of Corollary D1.

*Proof.* ν₀ is passive by C25′ (red-team SURVIVED); ν₁ by C5 (Khinchin mixture of
centered plateaus, each a contraction/restriction of λ); Δ is additive, and a sum of
nonpositive functions is nonpositive. ∎

**Corollary derivations (the V8 mechanical test, sufficient side):**
- **C3 (middle-half, Thm 4.2):** ν₁ = 0, a = b = L/2. [1 line]
- **C25′ (sharp box):** ν₁ = 0. [1 line]
- **C4 (Lemma 4.1):** the generator check for ν₀'s extreme pairs; one-line form (★) in
  H9 §3.1 — sin A ≤ cos B and sin B ≤ cos A when A + B ≤ π/2. [cited, 1 line]
- **C5 (unimodality, Thm 5.2):** ν₀ = 0. [1 line]
- **C6 (plateau Clausen form):** the *evaluation* of Δ on the generator 1_{[−s,s]} — the
  margin function M(α,θ). Not an independent condition; it is the kernel S is built from.
- **C7 (dip tolerance d\*):** the exact trace of 𝒫 on the 2-plane span{1_{[−a,a]},
  1_{[−b,b]}}: d\* = min_θ M(α_a,θ)/M(α_b,θ) — a ratio of two generator margins (H9 §5).
  [1 line given C6]
- **C10 / C11:** the single-atom limit of the pair law, and the ξ → −ξ symmetry of Δ.

**Why S is stated as a split and not something weaker.** The two obvious weakenings are
both *false*, killed this session before any proof attempt (`exp1`, `exp3`):
- *Radial-CDF dominance* ("every centred window wider than L/2 holds at least the
  equilibrium share of the drop": F(u) ≥ Wu/L on [L/2, L]) — 28/1315 random
  condition-satisfying profiles have gain; the exact boundary case ½δ₀ + λ|outer has
  gain +0.168 (converged in discretisation). It fails to secure even the centre
  condition I3.
- *"Outer + centre clean ⟹ passive"* (which would make S two-sided) — false even for
  even profiles: 805/30000 even counterexamples, middle bulge up to +0.70 with outer
  margin −0.013 and Δ(0) = −∞. Note the cheat that kills every point condition: a
  **vanishing centre atom** (mass 0.006) drives Δ(0) to −∞ without disturbing anything
  else — the concrete face of C48's density theorem.

---

## 5. The ladder — the gap between S and 𝒫 is structured and self-similar

**5.1 Theorem S-τ (the second rung; proved modulo C55's rung).**

> Let g be even and **inversion-symmetric** (radial measure invariant under
> |t| ↦ L − |t|), with every drop radius in **[L/4, 3L/4]**. Then the partner is
> passive, with ε″(0) = 0 exactly (the sector lies on the centre facet Δ(0) = 0).
> The constant L/4 is **sharp**: the pure inversion pair at radius u < L/4 has gain.

*Proof.* An inversion-symmetric radial measure disintegrates into equal-mass inversion
pairs {u, L−u}, u ∈ [L/4, L/2] (an atom at u = L/2 is self-paired = S(1), passive
boundary). Each pair is S(β) + S(1/β) with β = tan(πu/2) ∈ [tan(π/8), 1] =
[√2−1, 1], passive by **Lemma E1′** (`lemmaE_closure.md`, C55: passive ⟺
c = β² + β⁻² ≤ 6 ⟺ β ≥ √2−1). Mixtures of passives are passive. Sharpness is E1′'s
bifurcation at c = 6. ∎ (Control: 0/2000 random sector members violate; the pure pair
flips at u = 0.25 with sup = −6e−17 / +3.1e−2 at 0.24 — `exp4_output.txt`.)

Why the constant is L/4: arctan(√2−1) = π/8 exactly. **S-τ is C3's "middle half" motif
reproduced one level up, on the radius axis** — C3 confines *positions* to a half-window
under the balance M₁ = 0; S-τ confines *radii* to a half-window under the τ-balance
Δ(0) = 0. The same theorem shape regenerates in the log-radius variable.

**5.2 S-τ is disjoint from S (proved).** An inversion pair with u < L/2 has atoms at
radius L−u > L/2 with positive mass. In any S-split, ν₁ (even unimodal) has no atoms off
0, so all four atoms must lie in ν₀'s box: a, b ≥ L−u ⟹ a+b ≥ 2(L−u) > L. ∎ So the
entire sector {u ∈ [L/4, L/2)} is passive and invisible to S.

**5.3 The third rung: a two-scale witness with a closed-form threshold (verified
numerically).** The family m₁·S(0.6) + m₂·S(2.2) (Cayley radii 0.6 and 2.2; not a
passive pair set since 0.6·2.2 > 1, not reciprocal since 1/0.6 ≠ 2.2) is passive
exactly for m₂/m₁ ≤ m₂* with

    m₂* (bisected) = 0.64787983 ;   ln(1/0.6)/ln(2.2) = 0.64787997 ,

i.e. the boundary is the **τ-balance point Δ(0) = 0**, binding at the centre (7-digit
match; label: verified numerically, the "binding at η = 0" claim is not proved for the
family). Every passive member with m₂ > 0 is outside cone(S ∪ S-τ) on its own atom set
(the only passive-pair generator available is S(0.6) alone). The ladder does not stop
at the second rung.

---

## 6. Proposition NG — no finite zoo can ever span the passive cone (proved)

> **Proposition NG.** Fix any Lemma-E2 atom set A (n ≥ 3 balanced pairs satisfying
> (H1)–(H3) of `lemmaE_closure.md`; such sets exist for every n). The passive cone
> 𝒫_A ⊂ ℝⁿ₊ (weights on those pairs) is **not polyhedral**. Consequently, for any finite
> list 𝒢₁, …, 𝒢_k of generator families, each contributing finitely many members
> supported on any given finite atom set (every family in the corpus: balanced pairs,
> plateaus, reciprocal pairs, …), the cone they generate is a **strict** subcone of 𝒫.
> The sufficient side of the theory can be extended forever and completed never.

*Proof.* (1) *Non-polyhedrality.* Lemma E2 (C55, proved) provides a real-analytic curve
t ↦ m(t) ∈ ∂𝒫_A, in the open orthant, whose unique active point η(t) is strictly
monotone in t. Suppose 𝒫_A were polyhedral. Its boundary near the curve is covered by
finitely many closed faces; by Baire on the parameter interval, some face G (affine hull
q + T, dim T = d ≥ 1 since the curve is nonconstant) contains m(J′) for a subinterval
J′, with m(t) ∈ relint G after descending to the minimal such face. For q = m(t) ∈
relint G and any κ ∈ T, q ± sκ ∈ G ⊆ 𝒫_A for small s; evaluating Δ_{q±sκ} at the active
point η(t) (where Δ_q = 0) pinches ⟨v(η(t)), κ⟩ = 0 — the same two-sided perturbation as
Theorem 2's step (D). Hence {v(η(t)) : t ∈ J′} ⊆ T^⊥, a subspace of dimension ≤ n−1.
But η(J′) is a nondegenerate atom-free interval, and Corollary A″ (Lemma A′) says
span{v(η) : η ∈ η(J′)} = ℝⁿ. Contradiction. (2) *Strictness.* A positive mixture of
positive measures is supported on the union of their supports, so any member of
cone(𝒢) supported on A is a positive combination of the finitely many 𝒢-generators
supported inside A — a polyhedral subcone of ℝⁿ, which by (1) cannot exhaust 𝒫_A. ∎

**Remarks.** (i) This is the exact primal mirror of Theorem 2/C49 — both are the
statement "the boundary normals v(η) sweep a continuum spanning ℝⁿ", read once against
test batteries (dual) and once against generator families (primal). One lemma, two
no-gos. (ii) Scope: NG does not forbid a sufficient condition described by *inequalities*
(a family with infinitely many members per atom set) — but such a description is a
battery of functionals, and C48/C49 bound those from the other side. Together the two
no-gos close the loop: neither finitely many generators nor finitely many tests. (iii)
NG leans only on C55 (Lemma E2 + Corollary A″) — real-variable, no Thm 6.1, no numerics.

---

## 7. B3 — the gap, quantified

**Exact bands (closed form):**
- **Single symmetric pair at radius u** (β = tan(πu/2)): passive ⟺ u ≤ L/2;
  satisfies N ⟺ u ≤ 2L/3 (Δ(±1) = ½ln|1−β²| − ½ln2 ≤ 0 ⟺ β ≤ √3 ⟺ u = (2/π)·(π/3);
  outer sup binds at η = ±1 since the pair has no outer critical point). **N-gap band:
  u ∈ (L/2, 2L/3]** — the middle third of the outer half. Verified: Δ(1) = +3.3e−16 at
  β = √3 (`exp2` block D).
- **Inversion sector:** passive pairs at radii {u, L−u}, u ∈ [L/4, L/2) — entirely
  outside S (§5.2). **S-gap band on the radius axis: [L/4, L/2).**
- **Reciprocal-pair family** x·S(β) + (1−x)·S(1/β) (`gapmap_reciprocal.npz`, 120×101
  grid): area(passive) = 0.390, area(N-satisfying) = 0.390 (N-but-gain 0.001 — N is
  nearly exact here), S covers only the line {x = 1, β ≤ 1}. The x = ½ slice reproduces
  E1′: passive down to β = 0.4168 vs √2−1 = 0.4142 (grid resolution).

**Ensembles (random even atomic profiles, 12000 draws, exact node test):**
- of 5255 N-satisfying profiles, **79 (1.5%) have gain** — the honest false-pass rate of
  the necessary parent on this ensemble;
- of 5176 passive profiles, **2060 (39.8%) are certified by S** (via the pair-cone LP —
  for atomic profiles the ν₁ component is empty, so this is S's full atomic reach).
- Asymmetric general mixtures: 1583/30000 random balanced-pair mixtures satisfy N with
  gain up to +1.24 (`exp2` block C) — N's gap is not an even-symmetry artifact.

**What sits in the gap, structurally:** the ladder of §5 — sectors indexed by which
balance law (M₁ = 0, τ-balance, …) is saturated in aggregate. Each rung is exactly
solvable so far; NG guarantees the ladder never closes.

---

## 8. Parsimony check (the V8 mechanical test)

| old statement | now | derivation |
|---|---|---|
| C3 middle-half | corollary of **S** | ν₁ = 0, a = b = L/2 [1 line, §4] |
| C25′ sharp box | corollary of **S** | ν₁ = 0 [1 line] |
| C4 two-point lemma | generator check of **S** | (★) of H9 §3.1 [1 line] |
| C5 unimodality | corollary of **S** | ν₀ = 0 [1 line] |
| C6 plateau form | evaluation of **S**'s generator margin | §4 |
| C7 dip tolerance | ratio of two such margins | [1 line given C6] |
| C8a edge slopes | **N**, PV of M₁ at seam | §3.1 |
| C8b M₁ = 0 | **N**, O(1/η) | [1 line] |
| C8c M₂ ≥ 0 | **N**, O(1/η²) | [1 line] |
| C8d σ ≥ 0 | ω-average of the full law | σ = −∫Δdω [2 lines, §3.4] |
| C9 depth law | **N**, two-term max | [2 lines] |
| C13 endpoints | O(η⁰) identity | [1 line] |
| C10, C11 | pair-law limit; ξ→−ξ symmetry | [1 line each] |

Count of independent one-sided conditions: **10+ → 2** (S and N), plus two *two-sided*
no-go theorems (C49 and NG) proving the pair cannot be completed, plus the covariance
lemmas (D, I) which are identities, not conditions. New one-sided statements introduced:
S-τ and the §5.3 witness — both positioned inside the no-go's proof of structure (they
are the measured rungs of the ladder), not as screening conditions; a practitioner who
wants them as screens inherits them from E1′, which C55 already proved.

---

## 9. What a hostile referee can still demand

1. **S-τ's rung.** It inherits C55's rung (`supported`; red-team pending on the F1
   chain). If C55 falls, §5.1–5.2 fall with it; §2–§4, §6-part(2)… no — §6 also leans on
   C55 (Lemma E2 + A″). Only §2, §3, §4, §7's ensembles survive a C55 collapse.
2. **§5.3's "binding at the centre"** is numerics (7 digits, one family). The closed
   form m₂* = ln(1/β₁)/ln(β₂) is a conjecture for the general two-scale family.
3. **NG's scope** (remark ii of §6): conditions with infinitely many members per atom
   set are not covered — deliberately, since those are batteries and belong to C49's
   side. A referee may ask for the two no-gos to be merged into one statement; they
   should be, eventually ("no finitely-described one-sided theory is complete").
4. **The smooth-profile caveat** of Theorem 2 (`lemmaE_closure.md` §11.5) applies to NG
   identically: the witnesses are staircases; mollification is not carried out.
5. ~~**σ ≥ 0 vs N.**~~ **Closed.** 15/805 of the exp3 N-satisfying-with-gain profiles
   have σ < 0; explicit witness: radii (0.528, 0.924, 0.790, 0.716), weights (0.363,
   0.133, 0.124, 0.375), centre mass 0.006 — outer margin −0.013, Δ(0) = −∞, middle
   gain +0.70, σ = −0.185. So σ ≥ 0 is genuinely independent of N1+N2, as §3.4 claims.
6. Percentages in §7 are ensemble-dependent (stated with their samplers); they quantify,
   they do not characterize.

---

## 10. Numerical verification (controls, all firing)

| control | expectation | measured |
|---|---|---|
| pair threshold a+b = L | boundary at 0 | sup = −6.2e−17 at a=0.4, b=0.6; +7.3e−2 at b=0.65 |
| E1′ bifurcation | flip at β = √2−1 | −2.2e−16 at β = 0.414215; +0.226 at β = 0.35 |
| uniform staircase (λ) | sup = 0 | ±4e−16 at K = 50…400 |
| dilation identity | exact | diff ≤ 3.6e−16 |
| inversion identity | exact | diff ≤ 5.6e−17 |
| contraction closure | 0 violations, expansions must fail | 0/800; expansions 380/800 fail |
| S-τ sector | 0 violations, sharp at L/4 | 0/2000; u = 0.25 → −6e−17, u = 0.24 → +3.1e−2 |
| N threshold √3 | Δ(1) = 0 at β = √3 | +3.3e−16 |
| ladder threshold | ln(1/0.6)/ln2.2 | bisect 0.64787983 vs 0.64787997 |
| E1′ slice of gap map | β* = √2−1 | 0.4168 (grid step ~0.008) |

Killed candidates (do not re-walk): radial-CDF dominance (`exp1`), outer+centre
two-sidedness in any symmetry class (`exp2` C, `exp3`), and — inherited — the
balanced-pair cone (brief §3), CFM/convex order (C52).
