# Analytic closure of Lemma E and the thickening step: Theorem 2 is proved outright

**Theorist, board item F1, 2026-08-19.** Target: victory condition **V7**. This document removes the
label *"proved, with one numerically-certified constructive ingredient"* from Theorem 2 of
`research/theory/ns_answer.md` §4 and replaces it with **proved**.

Scripts and raw output: `research/results/f1_lemmaE_closure/` (`e2_sweep.py`, `e2_output.txt`,
`manifest.json`, `CHECKPOINT.md`). One script only: it carries all verifications and all six
controls. (A separate base-point script was not needed — the First Mate independently re-derived
and re-verified §4's base point before this document was written; those numbers are reproduced
inside `e2_sweep.py` anyway, from an independently written evaluator.)

---

## 0. Summary — what changed

| Item | Before (ns_answer.md §4) | After (this document) |
|---|---|---|
| Lemma E (active points sweep an interval) | grid-certified on 152 θ-values in [1.45, 2.96] for an explicit **asymmetric** triple; globality of the max checked numerically at each θ | **proved.** One explicit base mixture whose passivity is a *quadratic inequality with negative discriminant*, plus the implicit function theorem |
| Thickening to the open orthant | continuity argument *stated with* its hypotheses; nondegeneracy Δ″ ∈ [−0.506, −0.069] **measured** | **proved.** Nondegeneracy is exact and free: Δ″(0) = −25/4, an identity, not a measurement. Thickening is now the *same* IFT step as the sweep, not a separate perturbation |
| Theorem 2 | proved modulo the above | **proved** |
| Lemma A | proved via harmonic continuation in ℂ (Cauchy data on a segment) | **proved** by a three-line real-variable argument (Lemma A′ below); the complex-analytic route is no longer needed |

**Step (E) of `ns_answer.md` §4 is superseded, not supplemented.** The asymmetric triple
bad₁ = (−0.25, +0.90), bad₂ = (−0.40, +0.80), good = (−0.15, +0.45), the window [1.45, 2.96] and
the measured Δ″ range should be **deleted** and replaced by §5 of this document (splice text in §10).
A reader must not be left with two versions of the same lemma.

**The whole of Theorem 2 is now a real-variable argument.** No numerics are load-bearing anywhere in
the proof; the numerics in §8 are checks, not evidence.

---

## 1. Setting and notation

Identical to `h9_potential_domination.md` and `ns_answer.md`. Window L = 1, x ∈ (−1, 1); Cayley
coordinate ξ(t) = tan(πt/2), an increasing bijection (−1,1) → ℝ. The drop density is g = −dε′/dx ≥ 0;
for a staircase, g = Σ_j w_j δ_{t_j} with w_j > 0 and total mass W = Σ_j w_j. Write ξ_j = ξ(t_j).

    Δ_g(η) = Σ_j w_j ln|η − ξ_j| − (W/2) ln(1 + η²),        π ε″(x) = − Δ_g(ξ(x)),

so **passivity ⟺ Δ_g(η) ≤ 0 for every η ∈ ℝ**. Δ_g(η) → 0 as |η| → ∞, so the constraint is
"Δ ≤ 0", never "Δ ≤ −δ".

A **balanced pair** P(a,b), a, b ∈ (0,1), is the mass-1 measure with atoms at t = +b and t = −a,
i.e. ξ = β := tan(πb/2) > 0 and ξ = −γ := −tan(πa/2) < 0, carrying weights

    w₊ = γ/(β+γ)  at ξ = β,      w₋ = β/(β+γ)  at ξ = −γ,

chosen so that M₁ = Σ w_j ξ_j = 0. Every finite positive measure with M₁ = 0 is a positive
combination of balanced pairs on its atoms. Δ is additive: Δ_{Σ m_j P_j} = Σ m_j Δ_{P_j}.

For a fixed list of pairs P₁,…,P_n write, for m ∈ ℝⁿ,

    Δ_m := Σ_j m_j Δ_{P_j},     v(η) := (Δ_{P₁}(η), …, Δ_{P_n}(η)) ∈ ℝⁿ,
    𝒫 := {m ∈ ℝⁿ₊ : Δ_m ≤ 0 on ℝ}   (the passive cone in weight space).

So Δ_m(η) = ⟨v(η), m⟩ and 𝒫 is a closed convex cone cut out by the linear constraints
⟨v(η), ·⟩ ≤ 0, η ∈ ℝ. The **active set** of m ∈ 𝒫 is A(m) := {η ∈ ℝ : Δ_m(η) = 0}, and

    A* := ⋃ { A(p) : p ∈ ∂𝒫, p in the OPEN orthant (all n coordinates > 0) }.

**Lemma E is the statement that A* contains an open interval.** That is what §5 proves.

---

## 2. Two facts inherited (both already proved; re-derived here)

### Lemma B (single pair). PROVED — algebra re-derived independently for this document.

For a mass-1 balanced pair with atoms β, −γ,

    Δ′_P(η) = w₊/(η−β) + w₋/(η+γ) − η/(1+η²)
            = [ (1+βγ)η − (β−γ) ] / [ (η−β)(η+γ)(1+η²) ]
            = (1+βγ)·(η − η*) / [ (η−β)(η+γ)(1+η²) ],     η* = (β−γ)/(1+βγ) = tan(π(b−a)/2).

*Derivation.* w₊/(η−β) + w₋/(η+γ) = [γ(η+γ) + β(η−β)]/[(β+γ)(η−β)(η+γ)] = (η+γ−β)/[(η−β)(η+γ)].
Then (η+γ−β)(1+η²) − η(η−β)(η+γ) = η + γ − β + βγη = (1+βγ)η − (β−γ). ∎

**Consequences used below.** (i) Δ_P has *exactly one* finite critical point, at η*; it is a maximum
on (−γ, β). (ii) Δ_P < 0 strictly on (β, ∞) and on (−∞, −γ) — outside the hull of its own atoms Δ_P
is strictly monotone toward its limit 0. Hence for **any** m with all coordinates > 0, Δ_m < 0
strictly outside the hull H of all atoms. (iii) Each Δ_{P_j} is *bounded above* on any compact set
(it is a sum of two logarithms minus a nonnegative term).

### Lemma A′ (independence of pair gaps). PROVED — new, shorter, real-variable proof.

> Let P₁,…,P_n be balanced pairs whose 2n atoms are **pairwise distinct**. If Σ_j c_j Δ_{P_j}(η) = 0
> for every η in some nonempty open interval I ⊂ ℝ containing no atom, then c = 0.

*Proof.* Let S = {j : c_j ≠ 0} and suppose S ≠ ∅. F := Σ_{j∈S} c_j Δ_{P_j} is real-analytic on
ℝ ∖ A(S), where A(S) is the (nonempty, finite) set of atoms of the pairs in S. F ≡ 0 on I, hence
F ≡ 0 on the connected component J of ℝ ∖ A(S) containing I, by real-analyticity. J is a component
of the complement of a finite nonempty set, so it has a finite endpoint ξ ∈ A(S). Because all atoms
are pairwise distinct, ξ belongs to exactly one pair P_{j₀}, j₀ ∈ S, with some weight w > 0, and

    F(η) = c_{j₀} w ln|η − ξ| + O(1)     as η → ξ within J.

The O(1) term is bounded (all other logarithms and the −(W/2)ln(1+η²) term are continuous at ξ),
while ln|η−ξ| → −∞. F ≡ 0 forces c_{j₀} w = 0, i.e. c_{j₀} = 0 — contradicting j₀ ∈ S. ∎

**Corollary A″.** For any open interval I containing no atom, span{v(η) : η ∈ I} = ℝⁿ.

*Remark (parsimony).* This replaces the harmonic-continuation proof of Lemma A in `ns_answer.md` §1.
That proof is correct, but it invokes conjugation symmetry, vanishing Cauchy data on a segment, and
connectedness of ℂ minus a finite set. The argument above needs only that a real-analytic function
vanishing on an interval vanishes on its component, plus the log blow-up at an atom. Same statement,
one third the machinery, and it keeps the entire Theorem 2 inside real analysis.

---

## 3. The symmetric-pair calculus (why this construction is exact)

Let **S(β)** denote the mass-1 balanced pair with a = b, i.e. atoms at ξ = ±β with weights ½, ½
(β = tan(πa/2), so β < 1 ⟺ a + b < 1 ⟺ the pair is passive on its own). Then

    Δ_{S(β)}(η) = ½ ln|η² − β²| − ½ ln(1+η²).

For a mixture m = Σ_i m_i S(β_i) with W = Σ_i m_i, put **y = η²** and B_i = β_i². Then

    Δ_m(η) = ½ ψ(y),      ψ(y) := Σ_i m_i ln|y − B_i| − W ln(1+y),      y ≥ 0.        (3.1)

Four structural gifts follow, all exact:

**(G1) Δ′_m(0) = 0 automatically**, since Δ_m(η) = ½ψ(η²) is even. So η = 0 is a critical point of
*every* symmetric mixture — no tuning is needed to place a critical point.

**(G2) Nondegeneracy is free and always strict.** Δ′_m(η) = η ψ′(η²), Δ″_m(η) = ψ′(η²) + 2η²ψ″(η²), so

    Δ″_m(0) = ψ′(0) = − Σ_i m_i/B_i − W  <  0     for every m > 0.                      (3.2)

This is an identity. *The measured range Δ″ ∈ [−0.506, −0.069] that the old thickening step relied
on is replaced by a closed form that cannot be positive.*

**(G3) The critical-point equation is LINEAR in y.** With ψ as in (3.1),
ψ′(y) = Σ_i m_i/(y−B_i) − W/(1+y) = N(y)/D(y), D(y) = (1+y)Π_i(y−B_i), and for **two** pairs
(B₁ = G, B₂ = B, weights m₁, m₂):

    N(y) = y·[ m₁(1+G) + m₂(1+B) ]  −  [ m₁B + m₂G + W·G·B ].                          (3.3)

The bracket multiplying y is strictly positive, so ψ has exactly one critical point
y* = [m₁B + m₂G + WGB] / [m₁(1+G) + m₂(1+B)] > 0. A two-symmetric-pair mixture therefore has at
most **two** critical values to check: at y = 0 and at y = y*. That is the entire reason the
construction below closes in closed form, whereas the old asymmetric triple did not.

**(G4) Universal strict concavity on the middle window** (used in §5, step (4c); new, elementary).
For **any** finite positive measure g,

    Δ″_g(η) = − Σ_j w_j/(η − ξ_j)²  −  W·(1 − η²)/(1 + η²)² ,                          (3.2′)

so for |η| < 1 both terms are ≤ 0 and the first is < 0: **Δ_g is strictly concave on every
atom-free subinterval of (−1, 1), for every profile.** The hypothesis |η| < 1 is sharp — a unit
atom at ξ = 10 has Δ″(1.5) = +0.1045 > 0. (3.2) is the special case η = 0. *(Aside, not used: in
window coordinates ξ ∈ (−1,1) ⟺ x ∈ (−½, ½), so −πε″ is strictly concave on the middle half of the
window for every profile. Noted for the librarian; not pursued here.)*


---

## 4. Lemma E1 — an explicit boundary point with a unique, nondegenerate touching maximum. PROVED.

> **Lemma E1.** Let P₁ = S(1/2) and P₂ = S(2), i.e. the four-atom measure
>
>     p₀ = ½·(δ at ξ=−2) + ½·(δ at ξ=−½) + ½·(δ at ξ=+½) + ½·(δ at ξ=+2),
>
> equivalently, in window coordinates, the **equal-drop 4-step staircase** with drops 1/2 at
> t = ±(2/π)arctan(½) = ±0.2951672353… and t = ±(2/π)arctan(2) = ±0.7048327647…
> (note (2/π)arctan(½) + (2/π)arctan(2) = 1 exactly). Then W = 2, M₁ = 0, and
>
>   (a) Δ_{p₀}(η) = ½ ln[ |η²−4|·|η²−¼| / (1+η²)² ];
>   (b) **Δ_{p₀}(η) ≤ 0 for all η ∈ ℝ, with equality if and only if η = 0**;
>   (c) Δ″_{p₀}(0) = **−25/4** exactly;
>   (d) the only other finite critical points are η = ±1, with Δ_{p₀}(±1) = ln(3/4) < 0.
>
> So p₀ lies on ∂𝒫, its active set is the single point {0}, and the touching maximum is
> nondegenerate.

*Proof.* (a) is (3.1) with B₁ = ¼, B₂ = 4, m₁ = m₂ = 1: Δ_{p₀}(η) = ½[ln|η²−¼| + ln|η²−4| − 2ln(1+η²)].

(b) With y = η² ≥ 0, Δ_{p₀} ≤ 0 ⟺ ψ(y) ≤ 0 ⟺ |y−¼|·|y−4| ≤ (1+y)². Set
R(y) := (1+y)² − |y−¼|·|y−4| and check the three sign regimes:

| range | |y−¼||y−4| | R(y) |
|---|---|---|
| 0 ≤ y ≤ ¼ | (¼−y)(4−y) = y² − (17/4)y + 1 | (1+2y+y²) − (y²−(17/4)y+1) = **(25/4)·y** |
| ¼ ≤ y ≤ 4 | (y−¼)(4−y) = −y² + (17/4)y − 1 | **2y² − (9/4)y + 2** |
| y ≥ 4 | (y−¼)(y−4) = y² − (17/4)y + 1 | **(25/4)·y** |

Row 1: R ≥ 0 with equality iff y = 0. Row 2: the quadratic 2y² − (9/4)y + 2 has discriminant
(9/4)² − 16 = 81/16 − 256/16 = **−175/16 < 0** and positive leading coefficient, hence R > 0
throughout. Row 3: R > 0. (The rows agree at the seams: R(¼) = 25/16, R(4) = 25.) Therefore R ≥ 0
on [0,∞) with equality only at y = 0, i.e. Δ_{p₀} ≤ 0 on ℝ with equality only at η = 0. ∎(b)

(c) By (3.2): Δ″_{p₀}(0) = −(1/¼ + 1/4) − 2 = −(4 + ¼) − 2 = −25/4. ∎(c)

(d) By (3.3), N(y) = y[(1+¼) + (1+4)] − [4 + ¼ + 2·¼·4] = (25/4)y − 25/4 = (25/4)(y−1), so y* = 1,
i.e. η = ±1, and ψ(1) = ln(¾) + ln(3) − 2ln 2 = ln(9/16), giving Δ_{p₀}(±1) = ½ln(9/16) = ln(3/4). ∎

**Where the numbers come from (and how far they can be moved).** Take two symmetric pairs with
*reciprocal* half-widths β and 1/β (β < 1) and *equal* masses. Then Δ(0) = ln β + ln(1/β) = 0
automatically — this is what puts the touching point at the origin — and, writing c := β² + β^{−2}
(so c ≥ 2, with c = 17/4 for β = ½), the three rows of the table become

    0 ≤ y ≤ β²    : R = (1+y)² − (β²−y)(β^{−2}−y) = (2+c)·y ≥ 0, equality iff y = 0;
    β² ≤ y ≤ β^{−2}: R = (1+y)² + (y−β²)(y−β^{−2}) = 2y² + (2−c)y + 2;
    y ≥ β^{−2}    : R = (1+y)² − (y−β²)(y−β^{−2}) = (2+c)·y > 0,

so rows 1 and 3 are automatic and the *middle row alone* decides: its discriminant is
(2−c)² − 16 ≤ 0 ⟺ c ≤ 6. Hence:

> **Lemma E1′ (the whole one-parameter family).** For S(β) + S(1/β), 0 < β < 1:
> passive ⟺ c = β² + β^{−2} ≤ 6 ⟺ β ≥ √2 − 1. For √2 − 1 < β < 1 the active set is exactly {0};
> at β = √2 − 1 a second pair of touching points appears at η = ±1 (c = 6, double root); for
> β < √2 − 1 the mixture has gain.

**Symmetry remark (why the construction is exact rather than lucky).** With y = η², the base
point satisfies the identity **ψ(1/y) = ψ(y)**, i.e. Δ_{p₀}(1/η) = Δ_{p₀}(η) = Δ_{p₀}(−η):

    |1/y − 4|·|1/y − ¼| / (1 + 1/y)²  =  |y − ¼|·|y − 4| / (1 + y)² .

In window coordinates ξ ↦ 1/ξ is t ↦ 1 − t (since arctan ξ + arctan(1/ξ) = π/2), so p₀ is invariant
under the Klein four-group generated by t ↦ −t and t ↦ 1 − t — which is exactly why its atoms sit at
t = ±u, ±(1−u). The critical points η ∈ {0, ±1, ∞} are precisely the fixed points of that group, and
that is *why* the critical set is small enough to check by hand. (Contrast the asymmetric triple of
the superseded step (E), which has no such symmetry and therefore no closed-form critical set.)

β = ½ gives c = 17/4 = 4.25, comfortably inside the passive range — the construction is not
knife-edge in β, only in the reciprocity constraint (which is what pins Δ(0) = 0).

---

## 5. Lemma E2 — the sweep and the thickening, in one implicit-function step. PROVED.

This single lemma does what §4 step (E) of `ns_answer.md` did in two pieces (envelope window +
thickening), and does it analytically.

> **Lemma E2.** Fix n ≥ 3 and pairs P₁ = S(½), P₂ = S(2), and P₃,…,P_n = balanced pairs P(a_j,b_j)
> subject to
>
>  **(H1)** the 2n atoms are pairwise distinct (so β_j, γ_j ∉ {½, 2} and the β's, resp. γ's, distinct);
>  **(H2)** β_j > 1 and γ_j > 1 for j ≥ 3 (equivalently a_j, b_j > ½);
>  **(H3)** Σ_{j≥3} (β_j − γ_j)/(β_j γ_j) ≠ 0 — e.g. β_j > γ_j for every j ≥ 3.
>
> Then there are t₁ > 0 and real-analytic functions η(·), r(·) on (−t₁, t₁) with η(0) = r(0) = 0 such
> that for every t ∈ (0, t₁) the mixture
>
>     m(t) := (1 + r(t))·P₁ + P₂ + t·(P₃ + … + P_n)        [ weights (1+r(t), 1, t, …, t) ]
>
> lies in ∂𝒫 **with all n weights strictly positive**, and its active set is the single point
> {η(t)}. Moreover
>
>     ṙ(0) = Δ_u(0)/ln 2 > 0,        **η̇(0) = (4/25)·Σ_{j≥3} (β_j − γ_j)/(β_j γ_j) ≠ 0**,
>
> where u := P₃ + … + P_n. Consequently η is strictly monotone near 0 and
> **A* ⊇ η((0,t₁)) ⊇ an open interval.**

*Proof.*

**(1) The implicit function theorem.** Let m(t,r) := p₀ + t·u + r·e₁ (coordinates
(1+r, 1, t, …, t)) and define, on the open set where η is not an atom,

    Φ(η, r; t) := ( Δ_{m(t,r)}(η) , ∂_η Δ_{m(t,r)}(η) ) ∈ ℝ².

Φ is real-analytic in (η, r, t). By Lemma E1, Φ(0,0;0) = (Δ_{p₀}(0), Δ′_{p₀}(0)) = (0, 0). The
Jacobian in the unknowns (η, r) at (0,0;0) is

    ⎡ Δ′_{p₀}(0)   Δ_{P₁}(0) ⎤   ⎡ 0       −ln 2 ⎤
    ⎣ Δ″_{p₀}(0)   Δ′_{P₁}(0)⎦ = ⎣ −25/4    0    ⎦ ,   det = −(25/4)·ln 2 ≠ 0,

using Δ_{P₁}(0) = ½ln¼ − 0 = −ln 2 and Δ′_{P₁}(0) = 0 (P₁ symmetric, (G1)). The IFT gives t₁ > 0 and
real-analytic η(t), r(t) with η(0) = r(0) = 0 solving Φ(η(t), r(t); t) ≡ 0 — i.e. η(t) is a critical
point of Δ_{m(t)} at which Δ_{m(t)} vanishes.

**(2) The two derivatives.** Differentiating Φ₁ ≡ 0 at t = 0:
Δ_u(0) + Δ′_{p₀}(0)·η̇ + Δ_{P₁}(0)·ṙ = 0, and Δ′_{p₀}(0) = 0, so ṙ(0) = −Δ_u(0)/Δ_{P₁}(0) =
Δ_u(0)/ln 2. Differentiating Φ₂ ≡ 0: Δ′_u(0) + Δ″_{p₀}(0)·η̇ + Δ′_{P₁}(0)·ṙ = 0, and Δ′_{P₁}(0) = 0,
so **η̇(0) = −Δ′_u(0)/Δ″_{p₀}(0) = (4/25)·Δ′_u(0)**. By Lemma B, Δ′_{P_j}(0) = (β_j−γ_j)/(β_jγ_j),
so η̇(0) = (4/25)Σ_{j≥3}(β_j−γ_j)/(β_jγ_j) ≠ 0 by (H3).
For a single pair, Δ_{P}(0) = [γ ln β + β ln γ]/(β+γ), which is > 0 when β, γ > 1; so (H2) gives
Δ_u(0) > 0 and hence **ṙ(0) > 0**, whence r(t) > 0 for all small t > 0. (This sign is used in step
(4); it is a convenience, see §6.)

**(3) All weights positive.** m(t) has weights (1 + r(t), 1, t, …, t). For t ∈ (0, t₁) with t₁ small,
1 + r(t) > 0 and t > 0, so **m(t) lies in the open orthant** — this *is* the thickening, obtained for
free as part of the same IFT solve rather than as a separate perturbation.

**(4) Δ_{m(t)} ≤ 0 globally, with equality only at η(t).** Let A be the set of all 2n atoms,
H = [min A, max A] its hull. By (H1)–(H2) every atom satisfies |ξ| ≥ ½, so the open interval
(−½, ½) is atom-free. Fix ρ := ¼ (any ρ < ½ works) and set Ω := H ∖ (−ρ, ρ), a compact set. (K is reserved
for ker M in §7.)

 *(4a) Outside H.* All weights of m(t) are > 0, so by Lemma B(ii), Δ_{m(t)} < 0 on ℝ ∖ H.

 *(4b) On Ω.* Δ_{p₀} is upper semicontinuous on the compact Ω (continuous where finite, → −∞ at
 ±½, ±2) and, by Lemma E1(b), strictly negative there; hence ε := −max_Ω Δ_{p₀} > 0. (Explicitly,
 for ρ = ¼ the candidates for max_Ω are the endpoints ±ρ, the interior critical points ±1, and the right/left ends
 of H; Δ_{p₀}(¼) = −0.2123398365… exceeds Δ_{p₀}(±1) = ln(¾) = −0.2876820725…, and the tail
 satisfies Δ_{p₀}(η) ≤ Δ_{p₀}(¼) for 2 < |η| ≤ 4 because the identity Δ_{p₀}(1/η) = Δ_{p₀}(η) of the
 symmetry remark in §4 maps (2, 4] into [¼, ½), on which Δ_{p₀} is decreasing (its only critical
 points are 0 and ±1); so **ε = −Δ_{p₀}(¼) = 0.2123398365…** for any configuration whose atoms lie
 in |ξ| ≤ 4.) Because r(t) > 0 and Δ_{P₁} < 0 everywhere (S(β) with β = ½ < 1 satisfies
 |η²−¼| ≤ 1+η² strictly for all η), we get (1+r)Δ_{P₁} ≤ Δ_{P₁}, so

     Δ_{m(t)} = (1+r)Δ_{P₁} + Δ_{P₂} + t Δ_u  ≤  Δ_{p₀} + t Δ_u  ≤  −ε + t·C_u   on Ω,

 where C_u := max(0, sup_Ω Δ_u) < ∞ by Lemma B(iii). Hence Δ_{m(t)} ≤ −ε/2 < 0 on Ω for
 0 < t ≤ min(t₁, ε/(2C_u)).

 *(4c) On [−ρ, ρ], ρ = ¼.* This interval contains no atom (all atoms have |ξ| ≥ ½ > ρ) and
 ρ < 1, so by **(G4)** Δ_{m(t)} is **strictly concave on [−ρ, ρ] for every t ≥ 0** — no smallness in
 t, and no estimate on Δ″_{P₁}, Δ″_u, is needed. For t small, η(t) ∈ (−ρ, ρ) by continuity of η(·)
 with η(0) = 0, and Δ_{m(t)}(η(t)) = 0 with Δ′_{m(t)}(η(t)) = 0 by construction. A strictly concave
 function's interior critical point is its unique maximum on the interval, so Δ_{m(t)} ≤ 0 on
 [−ρ, ρ] with equality only at η(t).

 Combining (4a)–(4c): Δ_{m(t)} ≤ 0 on ℝ with equality exactly at η(t). So m(t) ∈ ∂𝒫, in the open
 orthant, with A(m(t)) = {η(t)}.

**(5) The interval.** η is real-analytic with η(0) = 0 and η̇(0) ≠ 0, so it is strictly monotone on
some (−t₂, t₂), and η((0,t₂)) is an open interval with endpoint 0. Every point of it lies in A*. ∎

**What just happened to the two outstanding items.** *Lemma E* is step (5) together with (4):
the active points of genuine open-orthant boundary mixtures fill an interval, with no grid and no
envelope inversion. *The thickening step* is step (3): the "add mass δ of the remaining n−3 pairs
and rescale the bad block to re-touch the boundary" is exactly the pair (t, r(t)) produced by the
IFT, and the nondegeneracy it needed is Δ″_{p₀}(0) = −25/4 — an identity from (3.2), not a
measurement. The uniform continuity claim "θ′(θ,δ) → θ uniformly as δ → 0" is now the analyticity
of η(·) at t = 0.

---

## 6. Which hypotheses are load-bearing, and what breaks without each

| Hypothesis | Load-bearing? | What breaks if dropped |
|---|---|---|
| **Reciprocity** β·β′ = 1 with equal masses (here ½ and 2) | **Yes, exactly.** | Δ_{p₀}(0) = ln β + ln β′ ≠ 0, so p₀ is either strictly inside 𝒫 (Δ(0)<0) or has gain. *Checked:* replacing S(2) by S(3) gives Δ(0) = +0.4055 → gain. This is the one exactly-tuned equation in the construction, and it is linear in the masses, so it is satisfiable in an (n−1)-parameter set of ways. |
| **c = β²+β^{−2} < 6**, i.e. β ∈ (√2−1, 1) | **Yes.** | At c = 6 a *second* pair of active points appears at η = ±1 (the middle-row discriminant vanishes) and the active set is no longer a single point; for c > 6 the mixture is not passive. *Checked:* β = 0.3 (c = 11.20) gives sup Δ = +0.417. β = ½ (c = 4.25) is well inside. |
| **Symmetry** of P₁ and P₂ (a = b) | Yes, but only for *convenience of proof*. | Symmetry buys three things at once: Δ′(0) = 0 for free (G1), Δ″(0) < 0 for free (G2), and a *linear* critical-point equation (G3). Drop it and every one of those becomes a transcendental condition to be certified — which is precisely why the original asymmetric triple needed a grid. |
| **(H3)** Σ_{j≥3}(β_j−γ_j)/(β_jγ_j) ≠ 0 | **Yes.** | η̇(0) = 0: the active point does not move at first order and the argument produces no interval. It is not vacuous — if every added pair is symmetric then Δ_{m(t)} is even for all t, η(t) ≡ 0, and this family alone gives A* = {0}. Satisfied by any choice with all β_j > γ_j. |
| **(H2)** β_j, γ_j > 1 for j ≥ 3 | **No** — convenience only. | It gives Δ_u(0) > 0 hence r(t) > 0, which makes the estimate in (4b) one line. If r(t) < 0 the same conclusion holds but (4b) needs the standard extra step of isolating σ-neighbourhoods of P₁'s atoms, where the singular term (1+r)·½ln|η∓½| with coefficient ≥ ½ dominates all other terms. Nothing else changes. |
| ρ = ¼ in step (4c) | No | Any ρ < ½ works (atom-free) and any ρ < 1 is concave by (G4); the only role of ρ is to separate the concave cap from the compact remainder. |
| **(H1)** distinct atoms | **Yes** — but it is Lemma A′'s hypothesis, not this construction's. | Lemma A′ fails for coincident atoms (two identical pairs give a nontrivial null combination), and the final contradiction needs Corollary A″. |
| n ≥ 3 | Yes | Theorem 2 uses n = k+2, so n ≥ 3 whenever there is at least one measurement (k ≥ 1); k = 0 is trivial. |

---

## 7. Theorem 2, restated with its label corrected

> **Theorem 2 (no finite battery of linear measurements decides passivity). PROVED.**
> Let φ₁,…,φ_k : (−1,1) → ℝ be arbitrary functions, ℓ_i(g) = ∫φ_i dg, and let F : ℝᵏ → {PASSIVE,
> GAIN} be an arbitrary decision rule. Then F∘ℓ does not decide passivity: there exist finite
> positive atomic g_P (passive) and g_G (with gain), both with M₁ = 0 and the **same atom set**, with
> ℓ(g_P) = ℓ(g_G).

*Proof.* Take n = k + 2 and choose the pairs P₁ = S(½), P₂ = S(2), P₃,…,P_n as in Lemma E2, so that
(H1)–(H3) hold. This is possible for every k ≥ 1: take (γ_j, β_j) = (1 + 1/j, 1 + 4/(3j)) for
j = 3,…,n. Both sequences are strictly decreasing in j and lie in (1, 3/2], so all 2n atoms are
distinct and none equals ½ or 2 (H1); both exceed 1 (H2); and β_j − γ_j = 1/(3j) > 0 (H3). Let M ∈ ℝ^{k×n} be the battery matrix M_{ij} = ℓ_i(P_j) and K = ker M, so
dim K ≥ n − k = 2.

Suppose F∘ℓ were correct on the mixture cone C = {Σ_j m_j P_j : m ≥ 0}. Let t ∈ (0,t₂) and
p = m(t) ∈ ∂𝒫 ∩ (open orthant) with active point η(t), from Lemma E2. For κ ∈ K and |s| small,
p + sκ has nonnegative coordinates (p is in the open orthant) and ℓ(p + sκ) = ℓ(p); since p is
passive, correctness forces p ± sκ passive, i.e. Δ_{p±sκ}(η(t)) ≤ 0. But Δ_p(η(t)) = 0, so
±s·Δ_κ(η(t)) ≤ 0 for both signs, giving **Δ_κ(η(t)) = ⟨v(η(t)), κ⟩ = 0**. As κ ∈ K was arbitrary,
v(η(t)) ∈ K^⊥ = Row(M), of dimension ≤ k = n − 2.

This holds for every t ∈ (0,t₂), and η((0,t₂)) contains an open interval I containing no atom
(η(t) → 0 and all atoms have |ξ| ≥ ½). By Corollary A″, span{v(η) : η ∈ I} = ℝⁿ. Hence
ℝⁿ ⊆ Row(M), so n ≤ k = n − 2 — contradiction. Therefore no F is correct on C.

*Extraction of the matched witnesses.* If no two elements of C with equal ℓ-readings had opposite
verdicts, then the verdict would be a function of the reading, and defining F on ℓ(C) by that
function (arbitrarily off ℓ(C)) would make F∘ℓ correct on C — impossible. So there are
g_P, g_G ∈ C with ℓ(g_P) = ℓ(g_G), g_P passive and g_G with gain. To make their atom sets equal,
pick h := C₀·P₁ + δ·(P₂ + … + P_n) with δ > 0 and C₀/δ > b/a, where a := −sup_H Δ_{P₁} > 0 and
b := sup_H Σ_{j≥2} Δ_{P_j} < ∞ (both finite by Lemma B(iii) and upper semicontinuity). Then
Δ_h ≤ −C₀a + δb < 0 on H, and Δ_h < 0 off H by Lemma B(ii), so **h is strictly passive with all
coordinates positive**. Now Δ_{g_P + λh} = Δ_{g_P} + λΔ_h ≤ 0 for every λ > 0, so g_P + λh stays
passive; and if Δ_{g_G}(η₀) > 0 then Δ_{g_G + λh}(η₀) > 0 for λ small, so g_G + λh keeps its gain;
and ℓ(g_P + λh) = ℓ(g_G + λh). Both now have every coordinate positive, hence **the same 2n-atom
set**, and only the weights differ. ∎

**Relation to the old proof skeleton.** The proof above is self-contained. Step (A)/(B) of
`ns_answer.md` §4 (setting up M, K and the mixture cone) is the first paragraph; step **(C)**
("interior boundary points have a finite active point") is **no longer needed at all** — Lemma E2
exhibits the boundary point *and* its active point explicitly, so the compactness argument that
produced one abstractly can be deleted; step **(D)** is the two lines beginning "For κ ∈ K"; step
**(E)** is Lemma E2. **No step of this proof is numerically certified.**

### Status

| Item | Label |
|---|---|
| Lemma A′ (independence of pair gaps) | **proved** (new real-variable proof; supersedes the harmonic-continuation proof) |
| Lemma B (single-pair gap derivative) | **proved** (unchanged; algebra re-derived) |
| Lemma E1 (explicit boundary point, unique nondegenerate touch) | **proved** |
| Lemma E1′ (the β ∈ (√2−1,1) family; bifurcation at c = 6) | **proved** |
| Lemma E2 (active points sweep an interval; thickening) | **proved** |
| **Theorem 2** | **proved** — no "modulo", no numerically-certified ingredient |
| Theorem 1, Cor 2.1–2.3, Prop 3 | unchanged (Prop 3 still leans on inherited Thm 6.1 at `conjecture`) |

---

## 8. Numerical verification (checks, not evidence)

Scripts: `research/results/f1_lemmaE_closure/e2_sweep.py` and `e3_checks.py` (raw outputs
`e2_output.txt`, `e3_output.txt`). Everything below is a
*check on the algebra*; no number is load-bearing for the proof. Two independent evaluators are used
(a fresh mpmath implementation written from the definition, and the pre-existing exact node test in
`results/fable_ns_answer/common.py`), and the harness carries controls that must fail.

### 8.1 Lemma E1 — the base point

| quantity | predicted (closed form) | measured | residual |
|---|---|---|---|
| W, M₁ of p₀ | 2, 0 | 2.000000000000000, 0.0 | 0 (exact) |
| Δ_{p₀}(0) | 0 exactly | 0.0 | 0 |
| Δ″_{p₀}(0) | −25/4 = −6.25 exactly | −6.25 | 0 |
| Δ_{p₀}(1) | ln(3/4) = −0.287682072452 | −0.287682072452 | < 1e−12 |
| Δ′_{p₀}(1) | 0 (critical point) | 0.0 | 0 |
| closed form ½ln[|η²−4||η²−¼|/(1+η²)²] vs direct atom sum | identical | — | **4.56e−16** (max over 8 η) |
| sup Δ_{p₀}, exact node test (`common.py`, independent code path) | 0 at η = 0 | **−3.331e−16 at η = +0.000000000** | machine zero |
| sup Δ_{p₀}, dense scan (270k points, step 1.85e−3) | ≈ ½Δ″(0)·η² at the nearest grid point | −1.32031e−6 at η = −6.5e−4 | ½·6.25·(6.5e−4)² = **1.3203e−6** — the scan's own residual reproduces Δ″(0) = −25/4 to 5 digits |

Two independent evaluators agree (control C1: fresh mpmath implementation vs `common.py`'s
`Atomic.Delta`, max difference **8.88e−16** over 8 η values), and the node test is a different
*algorithm* (real roots of the degree-(2n+1) polynomial Q), not merely a different implementation.

### 8.2 Controls that must fail — and do

| control | what it tests | result |
|---|---|---|
| **C5** reciprocity broken: S(½)+S(**3**) | that the exact tuning β·β′ = 1 is load-bearing | Δ(0) = **+0.40546511** = ln(3/2) ≠ 0 → not a boundary touch |
| **C6** bifurcation β = √2−1 | Lemma E1′'s threshold c = 6 | β = 0.30 (c = 11.20): sup Δ = **+4.165e−1** → GAIN; β = 0.41421356 (c = 6.0000): sup Δ = **+8.10e−9** at η = +1 (the truncated decimal sits just below √2−1, i.e. just on the gain side — the double root at η = ±1 is reproduced); β = 0.5 and 0.9: sup Δ = −3.3e−16, −3.7e−16 at η = 0 → passive |
| **C4** move r off the IFT solution (n = 3, t = 1e−2) | that m(t) is *exactly* on ∂𝒫 and the node test is sensitive | r−1e−3 → sup Δ = **+6.931473e−4** (GAIN); r−1e−5 → **+6.931473e−6**; r → **−3.2e−16** (BOUNDARY); r+1e−5 → **−6.931473e−6**; r+1e−3 → **−6.931473e−4** (passive). The response rate is 0.6931473 per unit r, versus the closed-form prediction −Δ_{P₁}(η\*) = 0.693147270 (and −ln 2 = 0.693147181; the 9e−8 offset is ½Δ″_{P₁}(0)η\*² = 8.97e−8 — matched) |
| **C2** mirror the added pair (β < γ) | the *sign* of the sweep direction, predicted by η̇(0) = (4/25)Σ(β_j−γ_j)/(β_jγ_j) | η(1e−2) = +1.8943418e−4 → **−1.8943418e−4**; signs opposite, magnitudes equal to 9 digits |
| **C3** symmetric added pair (β = γ = 1.3) | that (H3) is not vacuous — with Δ′_u(0) = 0 the sweep must die | η(t) = **0.0 exactly** at t = 1e−3 and 1e−2; r(t) still solves (r/t = 0.3785116233, matching Δ_u(0)/ln 2 to **6.7e−37**) |
| **V6 sharpness** of (G4) | that "|η| < 1" in universal concavity is not decorative | max Δ″ over 300 random positive measures × 199 η ∈ (−1,1): **−4.03e−3 < 0**; but a unit atom at ξ = 10 gives Δ″(1.5) = **+0.1045 > 0** |

C3 and C4 are the two that would catch a broken harness: C3 forces a predicted *zero* out of the
same code that produces the nonzero sweep, and C4 predicts the *rate* at which the boundary
condition is violated in closed form.

### 8.3 Lemma E2 — the continuation (V4, V5)

Newton solve of Φ(η, r; t) = (Δ_m(η), Δ′_m(η)) = 0 at 40 digits; independently, the exact node test
on the resulting m(t).

**n = 3**, added pair (β₃, γ₃) = (1.4, 1.2). Predicted η̇(0) = (4/25)(0.2/1.68) = **0.01904761905**,
ṙ(0) = Δ_u(0)/ln 2 = **0.3656770618**.

| t | η(t) | η(t)/t | r(t)/t | node-test sup Δ | \|argmax − η(t)\| | min weight |
|---|---|---|---|---|---|---|
| 1e−4 | 1.90465714e−6 | 0.019046571 | 0.3656772 | −3.14e−16 | 1.7e−21 | 1e−4 |
| 1e−3 | 1.903714768e−5 | 0.019037148 | 0.3656787 | −2.92e−16 | 3.4e−21 | 1e−3 |
| 1e−2 | 1.894341811e−4 | 0.018943418 | 0.3656933 | −3.20e−16 | 1.4e−19 | 1e−2 |
| 3e−2 | 5.621514371e−4 | 0.018738381 | 0.3657253 | −3.48e−16 | 6.5e−19 | 3e−2 |

Richardson extrapolation of η(t)/t to t → 0: **0.01904761888** vs predicted **0.01904761905** —
agreement to **9 significant figures** (relative residual 9e−9). Relative error of r(t)/t at
t = 1e−4: **4.5e−7**.

**n = 6**, added pairs (1.15, 1.1), (1.35, 1.25), (1.7, 1.55), (2.4, 2.2). Predicted
η̇(0) = **0.03097435761**, ṙ(0) = **2.437530799**. Richardson extrapolation: **0.03097434978**
(relative residual 2.5e−7); r(t)/t = 2.437531 at t = 1e−4 (relative residual **1.8e−7**). Node test
at every t: sup Δ = −(2.5…3.9)e−16 with argmax matching η(t) to 4.8e−18, and **all six weights
strictly positive** (min weight = t).

So for both configurations, and every t tested, m(t) is (i) in the open orthant, (ii) exactly on
∂𝒫 to machine precision, (iii) with its unique active point at the Newton-predicted η(t), and
(iv) η(t)/t converges to the closed-form η̇(0) to 7–9 digits. **The active point moves, at the
predicted rate.**

### 8.4 Ancillary checks

- **(V2) explicit ρ:** max Δ″_{p₀} on [−ρ, ρ] = **−6.25** for ρ = 0.25, 0.40, 0.45 — i.e. the
  maximum of Δ″_{p₀} over the whole interval is attained at η = 0. So ρ = ¼ is admissible with room
  to spare; and by (G4) any ρ < ½ works for *any* t.
- **(V3) ε:** max_Ω Δ_{p₀} = **−0.2123398365** for both configurations (Ω = H ∖ (−¼, ¼)), attained at
  η = ±¼, so ε = 0.2123398365. The tail bound is exactly R = 4: Δ_{p₀}(4) = Δ_{p₀}(¼) to 12 digits,
  by the inversion symmetry.
- No FFT, no `linspace` grid, and no production `ht_derivative` call is used anywhere in this
  verification, so the endpoint-duplication hazard (`grid="linspace"`, auditor 2026-08-19) and the
  `tan(π/2)` overflow trap cannot arise: all atoms are specified directly in the ξ coordinate.

### 8.5 The three claims e2 does not cover (script `e3_checks.py`)

- **Lemma E1′'s three rows, in exact rational arithmetic** (`fractions.Fraction`, no floating point):
  all **90/90** residuals over 6 rational β × 5 rational y are **exactly 0**. Since each row residual
  is a polynomial in y of degree ≤ 2, three distinct y already force the identity in y for that β.
  *Control:* replacing the row-1 coefficient 2+c by 3+c makes 24/30 residuals nonzero (the 6 zeros
  are at y = 0, where the two formulas coincide — as they must). The threshold is confirmed
  exactly: c = 6 ⟺ β⁴ − 6β² + 1 = 0 ⟺ β = √2 − 1 (residual −6.7e−16); β = 2/5 < √2−1 gives
  c = 6.410 → GAIN, β = 1/2 gives c = 4.250 → passive.
- **The strictly-passive ballast h of §7** exists as claimed: with a = −sup_H Δ_{P₁} = 0.143844 and
  b = sup_H(Δ_{P₂}+Δ_{P₃}) = 0.949098, every tested C₀/δ gives sup Δ_h < 0 (e.g. −3.04 at
  C₀/δ = 9.90). Then g_P + λh has sup Δ < 0 for **every** λ tested (0 → 10), while g_G + λh keeps
  its gain only for small λ (positive at λ = 0.01, negative by λ = 0.1) — exactly the asymmetry the
  proof predicts. *Control:* a mixture dominated by the bad pairs (C₀/δ = 0.01) gives
  sup Δ_h = **+0.942 → GAIN**, as required. *Honest caveat:* the bound C₀/δ > b/a is a sufficient
  existence certificate, not a sharp threshold — sub-threshold ratios also happened to be passive
  here, and the document does not claim sharpness.
- **Corollary A″** on a short interval I = [−0.2, 0.2] with 40 samples: singular values of
  [Δ_j(η_l)] are (6.511, 0.205, 0.077) — full rank 3, smallest **7.7e−2**. *Control:* duplicating a
  pair (violating (H1)) collapses the smallest singular value to **8.8e−18**, i.e. the check is
  sensitive to exactly the hypothesis it depends on.


---

## 9. Parsimony check (V6)

- **Absorbs:** `ns_answer.md` §4 step (E) in its entirety — the asymmetric triple, the grid-certified
  window [1.45, 2.96], the 152-point contiguity claim, the measured Δ″ range, and the separately
  stated thickening argument. Four informal ingredients → one lemma with a two-line Jacobian.
- **Replaces:** Lemma A's proof (complex-analytic → real-variable). Statement unchanged, so the
  corpus statement count does not grow.
- **Corpus count:** statements *before* — Lemma A, Lemma B, step (E) window (numerical), thickening
  (hypotheses stated) = 4 ingredients, of which 2 unproved. *After* — Lemma A′, Lemma B, Lemma E1,
  Lemma E2 = 4 ingredients, of which **0 unproved**, and Lemma E1′ comes free as a description of
  the whole one-parameter family rather than a separate result. Net: the count is flat and the
  unproved count goes 2 → 0.
- **New physical object worth keeping:** the equal-drop four-step staircase at
  t = ±0.2951672…, ±0.7048327… is an *exactly known point of the passive boundary*: ε″ ≥ 0 with its
  only interior zero at the window centre x = 0 (plus the two structural endpoint zeros ε″(±1) = 0). It is a natural
  calibration profile for any numerical passivity convention (P0): the correct verdict is
  "passive, margin exactly 0 at x = 0", and it has an exact closed form
  ε″(x) = −(1/π)·½ln[|ξ²−4||ξ²−¼|/(1+ξ²)²], ξ = tan(πx/2).

## 10. Splice text for `ns_answer.md` §4

Replace the paragraph beginning "**(E) The active points sweep an interval.** (Lemma E,
constructive.)" and the whole "**Status note.**" paragraph by:

> **(E) The active points sweep an interval.** (Lemma E2 of `lemmaE_closure.md`, proved.) Take
> P₁ = S(½) and P₂ = S(2), the symmetric balanced pairs with atoms ξ = ±½ and ξ = ±2. Their sum p₀
> satisfies Δ_{p₀}(η) = ½ln[|η²−4||η²−¼|/(1+η²)²] ≤ 0 with equality exactly at η = 0 (a quadratic
> inequality with discriminant −175/16), and Δ″_{p₀}(0) = −25/4 exactly. Applying the implicit
> function theorem to (Δ_m(η), ∂_ηΔ_m(η)) = 0 in the unknowns (η, weight of P₁) along the direction
> that switches on the remaining n−2 pairs gives a real-analytic family m(t) ∈ ∂𝒫 with all weights
> positive, unique active point η(t), and η̇(0) = (4/25)Σ_{j≥3}(β_j−γ_j)/(β_jγ_j) ≠ 0. The active
> points therefore sweep an open interval, and the thickening to the open orthant is the same IFT
> step.

and in §9's status table set Thm 2 to **proved**, and in §10 delete the bullet "Lemma E's
all-analytic closure (routine but real work) is open".

## 11. What a hostile referee can still demand

1. ~~**The unwritten ρ.**~~ **Discharged.** An earlier draft of step (4c) shrank ρ "if necessary"
   to keep Δ″ negative. (G4) removes it: Δ″_g < 0 on (−1,1) for *every* positive measure g, so
   ρ = ¼ is admissible for every t with no smallness condition, and the constant is explicit.
2. **The size of t₁.** The proof gives no explicit radius of validity for the IFT. Only existence is
   used, so this is not a gap, but a referee wanting a *constructive* witness pair would need it.
   §8 measures the linear regime instead.
3. ~~**The final witness-pair extraction.**~~ **Discharged** in §7 by the explicit strictly-passive
   ballast h = C₀P₁ + δ(P₂+…+P_n) with C₀/δ > b/a: adding λh preserves passivity *exactly*
   (Δ_{g+λh} = Δ_g + λΔ_h ≤ 0 with no smallness) and preserves gain for λ small. What a referee can
   still ask is the converse bookkeeping: that the two witnesses can additionally be made to have
   *prescribed* atom positions, which is not claimed.
4. **Inherited scaffolding.** Theorem 2 as a whole still rests on the master identity
   π ε″ = −Δ_g (C1, audited) and on the equivalence "passive ⟺ Δ ≤ 0". Prop 3 (not used here) still
   leans on Wave-3 Thm 6.1 at rung `conjecture`. Neither touches Lemma E2.
5. **Scope of the negative result.** Theorem 2 speaks about *finite atomic* profiles. A referee could
   ask for the smooth-profile version. The witnesses are staircases and can be mollified, but the
   present proof does not carry out that mollification — the honest statement is that Theorem 2 is
   about the finitely-described class, where by Theorem A a two-sided criterion *does* exist; its
   force is that the criterion cannot be compressed to k readings, uniformly in n.
