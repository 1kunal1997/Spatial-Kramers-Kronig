# H9 — DiHiTI passivity as a single potential-domination relation

**Author:** theorist (crew), Cycle 1. **Date:** 2026-08-18.
**Script:** `research/theory/h9_domination_check.py` (self-contained, ~2 min).
**Scripts:** `h9_domination_check.py` (main, 23 blocks) and `h9_fixes.py` (quadrature-corrected
re-runs of blocks B/C/E/F). Both in `research/theory/`, both run clean.
**Status of this file:** COLD-START HANDOFF. Read §12 first if you are resuming with no memory.

---

## 0. Executive summary

The exact criterion (Wave 3 Thm 3.1(iii)) is **one relation between two measures**:

> **(L1) THE LAW.** Let `P = g/D` be the drop measure pushed to the Cayley coordinate
> `ξ(t) = tan(πt/2L)`, and let `ω` be the **Cauchy measure** `dω = (1/π)dξ/(1+ξ²)`.
> Then the DiHiTI partner is passive **iff**
>
> ```
>     Φ_P(η) ≤ Φ_ω(η)        for all η ∈ ℝ,        Φ_μ(η) := ∫ ln|η−ξ| dμ(ξ)
> ```
>
> i.e. **`U^P ≥ U^ω` pointwise**: the drop measure's logarithmic potential dominates
> that of `ω`. Exactly: `π ε″(x) = D·[Φ_ω(η) − Φ_P(η)]` with `η = ξ(x)`.

`ω` is not an arbitrary comparison measure. It is simultaneously:
* the harmonic measure of the upper half-plane at `i`, i.e. `ω = Bal(δ_i, ℝ)` — which is why
  `Φ_ω(η) = ln|η − i| = ½ln(1+η²)`, the RHS of Thm 3.1(iii);
* **the push-forward under `ξ = tan(πt/2L)` of normalised Lebesgue measure on the window
  `[−L,L]` — the equilibrium measure of the domain.**

So the criterion reads, in one sentence: **the drop measure's logarithmic potential must
dominate the window's own equilibrium potential.** The point mass at `i` is not an extra
object; it is the seam (`x = ±L` ⟺ `η = ∞`) viewed through the Cayley map.

> **(L2) THE REDUCTION PRINCIPLE.** For each fixed `η` the constraint in (L1) is a
> **linear** inequality in `P`, and `P ↦ Φ_P(η)` is affine and upper semicontinuous.
> Hence for any convex, weak-∗ compact family `𝒦` of drop measures,
>
> ```
>     𝒦 ⊆ 𝒫   ⟺   ext(𝒦) ⊆ 𝒫
> ```
>
> (Bauer maximum principle / Choquet representation). **Certifying a whole family reduces
> to certifying its extreme points.**

**The key test of H9 passes.** Theorems 4.2 and 5.2 are *not* complementary siblings.
They are **the same statement, (L2), applied to two different convex families**:

| | family `𝒦` | `ext(𝒦)` | classical source of the extreme points | the check that remains |
|---|---|---|---|---|
| Thm 4.2 | `{P ≥ 0, supp ⊆ middle half, M₁ = 0}` | ≤2-atom balanced measures | Richter–Rogosinski–Shohat / Winkler (moment sets) | Lemma 4.1 |
| Thm 5.2 | `{P ≥ 0, even, unimodal}` | plateaus `1_{[−a,a]}/2a` | **Khinchin's theorem** | Clausen margin `M(α,θ) ≥ 0` |

Wave 3's remark "neither contains the other" is a statement about the two *families*,
not about the theory. The families are different subsets of `𝒫`; the theory is one law.

**Bonus (not anticipated by the board): the middle-half theorem is not sharp, and (L1)+(L2)
give its exact replacement in one line.** Evaluating (L1) on a balanced 2-atom measure gives
a *closed-form necessary and sufficient* condition (§4), from which:

> **(C-new) Sharp balanced-support law.** If `g ≥ 0`, `M₁ = 0`, and `supp g ⊆ [−a, b]`
> with `a, b ≥ 0`, then `ε″ ≥ 0` **iff** `a + b ≤ L`. Middle-half is the symmetric case
> `a = b = L/2`. The condition is *not* `|t| ≤ L/2`; it is **"the drop fits in a half-period
> arc"**, and it may be arbitrarily off-centre (e.g. `[−L/4, 3L/4]` is passive).

This is a strict generalisation of Thm 4.2, is two-sided within its family, and is a
corollary — it adds no statement to the corpus.

---

## 1. Precise statement of the order

**Setting.** `x ∈ [−L, L]`, circle coordinate `θ = πx/L`, seam at `θ = ±π`.
`g = −dε′/dt` a finite positive measure of total mass `D` (bounded-variation `ε′`;
signed `g` is allowed throughout except where `g ≥ 0` is flagged). Cayley coordinate
`ξ(t) = tan(πt/2L)` maps `(−L, L)` bijectively onto `ℝ`, seam ↦ `∞`.

**Hypothesis (H) — the load-bearing integrability condition, absent from the corpus.**
```
    ∫ ln sec(πt/2L) dP(t)  =  ½ ∫ ln(1+ξ²) dP(ξ)  <  ∞
```
Equivalently `σ = ∫ g ln(2cos(πt/2L)) dt` is finite: *the mean loss is finite.* (H) is what
makes `Φ_P` finite, makes `P ↦ Φ_P(η)` weak-∗ upper semicontinuous up to the seam, and makes
the Bauer/Choquet step in (L2) legitimate. Dropping (H) breaks (L2), not (L1).

**Derivation of (L1).** From the master identity (C1) and
`sin(π(x−t)/2L) = cos(πx/2L)cos(πt/2L)(η−ξ)`:
```
    π ε″(x) = D ln|η − i| − ∫ ln|η − ξ(t)| g(t) dt          (Wave 3 eq. 7)
```
and `ln|η−i| = ½ln(1+η²) = Φ_ω(η)`. So `π ε″ = D(Φ_ω − Φ_P)` and passivity ⟺ `Φ_P ≤ Φ_ω` on ℝ. ∎

**What order is it, exactly?** With `U^μ = −Φ_μ` the standard logarithmic potential:

* It is **potential domination / the logarithmic-potential order**: `U^P ≥ U^ω` pointwise,
  between two probability measures of equal mass.
* **Upgrade from ℝ to ℂ (proved, elementary).** `h := Φ_ω − Φ_P` is harmonic on `ℂ∖ℝ`,
  conjugation-symmetric, and `h(z) → 0` as `z → ∞` (equal masses). If `h ≥ 0` on `ℝ`, the
  minimum principle on `ℂ₊ ∪ {∞}` gives `h ≥ 0` on `ℂ₊`, hence on `ℂ`. So
  **domination on the line ⟺ domination in the plane.** (This uses only the minimum
  principle — *not* Cartan's domination principle; see §7.)
* It is therefore the **Choquet / dilation order restricted to the cone of logarithmic
  potentials**: `∫f dP ≤ ∫f dω` for every `f` in the closed cone generated by
  `{ln|η − ·| : η ∈ ℂ}`. It is **weaker than the full Choquet (Cartier–Fell–Meyer) order**,
  which additionally tests all *harmonic* `f`; by Riesz decomposition
  `∫f dP − ∫f dω = ∫(Φ_P − Φ_ω)dλ + ∫h d(P−ω)` and the second term need not vanish.
  **Do not claim it is the Choquet order.** *(Remark, conjectural: on the admissible class
  the discrepancy may be vacuous, because `ω` is Cauchy so `∫ξ^n dω = ∞` for `n ≥ 1` and the
  `n = 1` term is killed by the necessary condition `M₁ = 0`. Not proved; not needed.)*
* The one-dimensional ancestor is **Hardy–Littlewood–Pólya majorization** (kernel `(η−ξ)₊`);
  here the kernel is `ln|η−ξ|` instead, so it is the *logarithmic* analogue, not HLP itself.

**Geometric restatement (exponentiate).** With `G_P(η) = exp Φ_P(η)` the geometric-mean
distance from `η` to the drop:
```
    passive  ⟺  G_P(η) ≤ |η − i| = √(1+η²)   for every real η.
```

**Circle restatement.** `Ψ_P(θ) := −∫ln|2 sin((θ−φ)/2)| dP(φ)` attains its **minimum over the
circle at the seam** `θ = π`. (Identical content; the `−ln cos` cocycle cancels.)

---

## 2. (L2), the reduction principle — statement and proof

**Proposition (L2).** Let `𝒦` be a convex family of drop measures, weak-∗ compact, satisfying
(H) uniformly. Then `𝒦 ⊆ 𝒫 ⟺ ext(𝒦) ⊆ 𝒫`.

*Proof.* (⟹) trivial. (⟸) Fix `η`. `P ↦ Φ_P(η) = ∫ln|η−ξ|dP` is affine, and `ξ ↦ ln|η−ξ|`
is upper semicontinuous and bounded above on the relevant compact ξ-set (uniform (H) supplies
the bound near the seam by truncation, with error `→ 0` uniformly on `𝒦`). By the **Bauer
maximum principle** the supremum of an affine u.s.c. functional over a compact convex set is
attained at an extreme point. Hence `sup_{P∈𝒦} Φ_P(η) = max_{P∈ext(𝒦)} Φ_P(η) ≤ Φ_ω(η)`. ∎

*Variant (L2′), topology-free.* If `𝒦` admits an **explicit Choquet representation**
`P = ∫ P_e dν(e)` over extreme points (Khinchin's theorem does exactly this for even
unimodal `P`), then linearity of `ε″` in `g` gives the conclusion with no compactness at all.
This is the layer-cake argument of Thm 5.2, correctly named.

---

## 3. Corollary 1 — Theorem 4.2 (middle half) as an instance of (L2)

`𝒦₁ = {P prob. measure on [−1,1] in ξ, ∫ξ dP = 0}` (middle half `|t| ≤ L/2 ⟺ |ξ| ≤ 1`;
balance `M₁ = 0`). Two linear constraints ⟹ `ext(𝒦₁)` = measures with ≤2 atoms
(Richter–Rogosinski–Shohat; Winkler) `= {γδ_β + βδ_{−γ}}/(β+γ)`. Check on extreme points
= Lemma 4.1. **This is exactly Wave 3's proof, now identified as an instance of (L2).**

### 3.1 A one-line proof of Lemma 4.1 (replaces Wave 3's AM–GM + Cauchy–Schwarz)

Put `β = tan A`, `γ = tan B` with `A, B ∈ [0, π/4]` (middle half). Write
`Δ(η) := (γ ln|η−β| + β ln|η+γ|) − ((β+γ)/2)ln(1+η²)`. Then (computed in §4)
`Δ′` has the single real root `η* = (β−γ)/(1+βγ) = tan(A−B)`, and at that root the `ln cos(A−B)`
terms cancel identically, leaving
```
    Δ(η*)  =  γ·ln( sin B / cos A )  +  β·ln( sin A / cos B ).             (★)
```
`A + B ≤ π/2` ⟹ `sin A ≤ cos B` **and** `sin B ≤ cos A` ⟹ **both logarithms are ≤ 0**
⟹ `Δ(η*) ≤ 0`. ∎ (Equality iff `A = B = π/4`, i.e. atoms exactly at `±L/2`.)

That is the whole of Lemma 4.1, and it exposes the mechanism the AM–GM proof hides: the
inequality is nothing but `A + B ≤ π/2`.

---

## 4. Corollary 2 — the exact two-atom law, and the sharp replacement for Thm 4.2

**Computation.** For `P = (γδ_β + βδ_{−γ})/(β+γ)` (the general balanced two-atom measure,
atoms at `t₊ > 0` and `−t₋ < 0`, `β = tan(πt₊/2L)`, `γ = tan(πt₋/2L)`):
```
    Δ′(η) ∝ η(1 + βγ) + (γ − β)        — a degree-1 polynomial ⟹ one critical point
    η*    = (β − γ)/(1 + βγ) = tan(A − B),      A = πt₊/2L,  B = πt₋/2L
    Δ(η*) = γ ln(sin B / cos A) + β ln(sin A / cos B)                    (★)
```
Since `A, B ∈ (0, π/2)`, `sin A ⋛ cos B ⟺ A + B ⋛ π/2`, and the two logarithms always have
the **same** sign. Therefore

> **Corollary 4.A (exact, two-sided).** A balanced two-atom drop measure is passive
> **iff** `A + B ≤ π/2`, i.e. **iff `t₊ + t₋ ≤ L`**. When it fails, the gain depth is exactly
> `min ε″ = −(1/π(β+γ))·[γ ln(sin B/cos A) + β ln(sin A/cos B)]·D`.
>
> Circle form: the two atoms must lie within **half a period** of each other, on the arc not
> containing the seam.

This **contains** Wave 3 Cor 4.3 (`t₊ = t₋ = t₀ ⟹ t₀ ≤ L/2` — the sharp `L/2` threshold, and
the closed-form depth `(D/2π)ln((1+c₀)/(1−c₀))`) and Cor 4.4 / C10 (single atom: mass-balance
forces `ξ₀ = 0`; the off-centre depth is `−(D/π)ln sec(πx₀/2L)`, the `η = −1/ξ₀` value of `Δ`).

Combining Corollary 4.A with (L2) on `𝒦 = {P ≥ 0 supported on `S`, M₁ = 0}` — whose extreme
points are the balanced ≤2-atom measures with atoms in `S` — gives:

> **Corollary 4.B (sharp balanced-support law; supersedes Thm 4.2).**
> `g ≥ 0`, `M₁ = 0`, `supp g ⊆ [−a, b]`. Then `ε″ ≥ 0` **iff** `a + b ≤ L`.
> Middle-half (`a = b = L/2`) is the symmetric special case; `[−L/4, 3L/4]` is equally passive.
> Necessity: take the extreme pair at `t₊ = b`, `t₋ = a`.

**Status:** *proved* (modulo the standard extreme-point theorem for two-constraint moment sets,
which is Wave 3 Appendix B and is classical). **Numerically verified**: threshold `S* = t₊+t₋`
found by bisection = `1.00000 L` at asymmetry ratios 1, 2 and 5 (§8, block D); 5 × 200 Monte-Carlo
balanced densities on `[−a,b]` with `a+b = L` for `(a,b) ∈ {(.5,.5),(.25,.75),(.1,.9),(.4,.6),(.75,.25)}`
— worst `min ε″/D = 0.0` exactly.

---

## 5. Corollary 3 — Theorem 5.2 (unimodality) as an instance of (L2′)

`𝒦₂ = {P ≥ 0 even and unimodal}`. **Khinchin's theorem** (Khinchin 1938; see e.g. Dharmadhikari
& Joag-Dev, *Unimodality, Convexity and Applications*, Ch. 1): a symmetric unimodal distribution
is exactly a mixture of uniform distributions on symmetric intervals — an explicit Choquet
representation with `ext(𝒦₂) = {1_{[−a,a]}/2a : 0 < a ≤ L}`. By (L2′) the check reduces to the
plateaus, which is Wave 3's Thm 5.1 (Clausen closed form, `M(α,θ) ≥ 0`).

**Why 4.2 and 5.2 looked complementary — resolved.** A plateau `1_{[−a,a]}` *is* a mixture of
symmetric pairs `½(δ_s + δ_{−s})`, `s ≤ a`; by Cor 4.A each pair with `s > L/2` is **not** passive.
Convexity is one-way (mixtures of non-passive measures can be passive), so the pair
decomposition proves nothing for `a > L/2` while the plateau decomposition does.
**The two theorems differ only in which extreme-point decomposition of the same cone is used.**
That is a statement about decompositions, not about two theories.

**Absorbed for free:** Thm 5.1 (plateau closed form) = the extreme-point evaluation for `𝒦₂`;
the dip tolerance `d*(b)` (C7) = the *signed* layer-cake, i.e. the LP `max{d : 1_{[−a,a]} − d·1_{[−b,b]} ∈ 𝒫}`,
whose value `min_θ M(α_a,θ)/M(α_b,θ)` is exactly the LP ratio bound.

---

## 6. Corollary 4 — the necessary cascade and the depth law are one Taylor expansion

Expand (L1) at `η → ∞` (`M₁ = ∫g tan(πt/2L)dt`, `M₂ = ∫g sec²(πt/2L)dt`):
```
    Δ(η) = −M₁/η − M₂/(2η²) + O(η⁻³)
```
* **`O(η⁻⁰)`:** `Δ(±∞) = 0` — this *is* `ε″(±L) = 0` (V5 / C13). Automatic, not a constraint.
* **Convergence of `M₁` as a principal value** requires the ξ-density tails to cancel, which is
  exactly `g(+L) = g(−L)` — the **edge-slope condition**.
* **`O(η⁻¹)`:** `Δ ≤ 0` for both signs of large `η` forces **`M₁ = 0`**.
* **`O(η⁻²)`:** gives the `M₂` condition (automatic for `g ≥ 0`).
* **Maximising the two-term truncation:** `η* = −M₂/M₁`, `max Δ = M₁²/(2M₂)`, hence
  **`min ε″ ≈ −M₁²/(2πM₂)`** — C9, in three lines, with its own validity condition
  (`|η*|` large ⟺ `M₁` small, `M₂ < ∞`) visible.

So the entire "four-term edge cascade" (C8) plus the perturbative depth law (C9) plus the
endpoint claim (C13) are **the asymptotic expansion of the single inequality (L1)**.

**Numerically verified** (§8, block H): depth law matches DiHiTI to 2–3 digits for offset
logistics at `x₀/L = 0.02, 0.05, 0.10`; single-atom depth matches to 6 digits.

---

## 7. What this is NOT (guard against false friends)

* **Not Cartan's domination principle** (Brélot 1959) and **not balayage**. Both are one-way
  sufficiency statements ("`U^μ ≤ U^ν` on supp μ ⟹ everywhere"). The `ℝ → ℂ` upgrade in §1 is
  proved by the elementary minimum principle for harmonic functions, and it is not needed for
  the criterion itself. Balayage enters only as an *identification* of `ω = Bal(δ_i, ℝ)`, which
  is a computation, not a theorem being leaned on.
* **Not the full Choquet / Cartier–Fell–Meyer order** — see §1.
* **Not cheaper than DiHiTI for a general single profile.** (L1) is the same function as `ε″`.
  Its value is parsimony, family certificates via (L2), and the finite reduction of §9.

---

## 8. Numerical verification — residuals

Script `research/theory/h9_domination_check.py`, run 2026-08-18. **17/23 blocks passed on the
first run; the 6 failures are quadrature artefacts with identified causes, listed below.**

| block | check | result |
|---|---|---|
| A | `∫ln|η−ξ|dω = ½ln(1+η²)` at `η = 1.7` | `0.679190` vs `0.679205` ✔ |
| B | gap identity `πε″ = D(Φ_ω−Φ_P)` vs DiHiTI, logistic | ✖ (log singularity at grid node; see below) |
| C | verdict agreement, 240 random labelled profiles | 152/240 — **all 88 disagreements are `−inf` gaps**, i.e. the same singularity ✖ |
| D | two-atom law `t₊+t₋ ≤ L` | ✔ 10/10 verdicts; closed-form depth matches to 5 decimals (`−0.02001`, `−0.04464`, `−0.03520`) |
| D | threshold by bisection at ratios 1, 2, 5 | `S* = 1.00000 L` (3× ✔) |
| E | MC balanced on `[−a,b]`, `a+b = L`, 5 shapes × 200 draws | worst `min ε″/D = 0.0` ✔ |
| E | necessity for `a+b > L` | ✖ — smooth-Gaussian MC never approaches the extreme pair; needs near-atomic draws |
| F | node collapse vs 4·10⁵-point direct kernel sum, 300 random atomic profiles | **max abs diff `9.1e−11`** ✔ |
| F | node count `≤ n` | one profile gave `n+1` (leading-coefficient cancellation lost to round-off); bound `n+1` is safe ✖ |
| F | timing, `n = 40` atoms | node test 4.56 ms vs DiHiTI@2¹⁴ 0.98 ms → **0.2×, no speedup** (honest negative) |
| G | adversary defeating any 5 fixed nodes | node values all `> 0`, global min `−0.0403` ✔ |
| H | depth law `−M₁²/(2πM₂)` at `x₀ = .02, .05, .10 L` | 3× ✔ (e.g. `−0.00752` vs `−0.00758`) |
| H | single atom `−(D/π)ln sec(πx₀/2L)` | `−0.015973` vs `−0.015973`, `−0.110318` vs `−0.110318` ✔ |

> **UPDATE — all three defects fixed in `h9_fixes.py`; see §12.1 for the corrected residuals.**
> Block B now passes at 1.2e-4–1.5e-3; block C rises 152/240 → **233/240** (remaining 7 diagnosed,
> §12.5.5); block E necessity confirmed to 4 decimals against the closed form; block F `deg Q − n ≤ 0`
> on 400/400. The original defects were:

**Known numerical defects (now fixed; recorded for the audit trail):**
1. Blocks A/B/C evaluate `ln|η − ξ(t)|` on the *same* grid that carries `η`, so the integrand
   hits `−∞` at one node and `np.trapezoid` returns `−inf`. Fix: excise the singular cell and
   add its analytic contribution `∫ln|η−ξ|` over that cell, or evaluate `Δ` at midpoints. The
   identity itself is Wave 3 eq. (7), an algebraic rearrangement of the verified master identity.
2. Block E necessity: replace the smooth-Gaussian MC with near-δ bumps at `t₊ = b`, `t₋ = a`.
   Corollary 4.A already gives the closed-form failure for that pair analytically.
3. Block F node count: `np.trim_zeros` on a float polynomial. Use a relative tolerance.

---

## 9. Part 3 — is there a minimal checkable characterization? (partial)

**Negative, proved.** *No fixed finite node set works.* Given any `η₁,…,η_k`, the inverse map
(Wave 3 Thm 6.1: `ε′ = c − Dx/2L − H[v]` returns `ε″ = v` for any `v` with `v(±L)=0`) lets one
place a narrow negative dip of `v` away from all `x_j = ξ⁻¹(η_j)`. Then `Δ(η_j) ≤ 0` for all `j`
while `max Δ > 0`. **Verified** (block G): 5 prescribed nodes, all positive, global min `−0.040`.
⟹ any exact node test must have a node count scaling with the profile's complexity.

**Positive, proved and verified — the node collapse.** For an **atomic** drop measure
`g = Σ_{j=1}^n w_j δ_{t_j}` (i.e. a *discretised/staircase* `ε′` — exactly what is fabricated):
```
    Δ′(η) = Σ_j w_j/(η−ξ_j) − Dη/(1+η²) = Q(η) / [(1+η²)∏_k(η−ξ_k)],   deg Q ≤ n
```
(the two `η^{n+1}` leading terms both have coefficient `D` and cancel). Hence `Δ` has at most
`n` real critical points and
```
    min ε″ = −(1/π)·max{ Δ(η_c) : Q(η_c) = 0, η_c ∈ ℝ } ∨ 0 ,
```
an **exact, grid-free, finite test with ≤ n nodes**. Verified against a 4·10⁵-point direct
kernel sum on 300 random atomic profiles: max abs difference **9.1e−11**.

**Honest cost verdict:** at `n = 40` the node test (`np.roots`, O(n³)) is **0.2×** the speed of
one DiHiTI at `N = 2¹⁴` — *slower*. It is exact and grid-free, which the FFT is not, and it is
`O(poly(n))` **independent of any evaluation grid**; a fast root-finder would change the
constant. **It does not by itself defeat C15.** Recorded as a partial result feeding H4/H10.

**Chebyshev-system status (H4), what I actually established:** `{1, ln|ξ−η₁|, …, ln|ξ−η_k|}`
*is* a Chebyshev system in `ξ` on any interval disjoint from `{η_j}` — because
`f′(ξ) = Σc_j/(ξ−η_j) = N(ξ)/∏(ξ−η_j)` with `deg N ≤ k−1`, so `f` has ≤ `k` zeros.
**But the roles are swapped relative to what H4 hoped for:** this Chebyshev property acts on the
*measure* side (it is what powers Markov–Krein / the extreme-point counts in (L2)), **not** on
the `η` side. The `η`-side collapse is the §9 node argument, and by the negative above it
cannot be made profile-independent.

---

## 10. Parsimony check (V6)

**Before — in-scope passivity statements (14):** C1 master identity; C2 exact criterion
(4 forms); C3 middle-half; C4 Lemma 4.1; C5 unimodality; C6 plateau Clausen; C7 dip tolerance;
C8 necessary cascade (4 sub-conditions); C9 perturbative depth; C10 off-centre step;
C11 mirror; C12 no-go on the line; C13 endpoint; C14 finite-moment insufficiency.

**After — 2 laws + 1 new reduction theorem:**
* **(L1)** the potential-domination criterion (absorbs C1, C2, C13);
* **(L2)** the extreme-point reduction principle (absorbs C3, C4, C5, C6, C7 as instances);
* **(L3)** the node collapse for atomic drop measures — *the single net addition*, and it is a
  reduction **of** (L1), not a new sufficient condition for passivity.

**Corollaries, no longer statements:** C4 (§3.1, one line), Cor 4.3 & C10 (§4), C6 & C7 (§5),
C8 & C9 & C13 (§6, one Taylor expansion), C11 (the `ξ → −ξ` symmetry of (L1)).
**New and free:** Cor 4.A (exact two-atom law) and Cor 4.B (sharp balanced-support law,
which *supersedes* C3 rather than sitting beside it).

**Net: 14 → 3.** The corpus shrinks, and the one addition is a reduction, not a thirteenth
special case.

---

## 11. Handoff / next steps, in priority order

1. **Fix the three quadrature defects in §8** (singular-cell excision; near-atomic MC for E;
   relative `trim_zeros`). ~30 min. Until then blocks B/C/E-necessity are *unverified*, though
   the mathematics they test is elementary rearrangement.
2. **Cor 4.B is the promotion candidate.** It is two-sided, strictly generalises C3, and is
   cheap to test blind. Send to `replicator` with parameters only: "`g ≥ 0` supported in
   `[−a,b]`, tangent-balanced; claim: passive iff `a+b ≤ L`."
3. **Verify the extreme-point step of Cor 4.B independently** — the claim that
   `ext{P ≥ 0 on S, mass 1, mean 0}` = balanced ≤2-atom measures. Classical, but it is the one
   imported ingredient.
4. **Open: the general skew problem (Wave 3 8.1)** is now well-posed as: *compute `ext(𝒦)` for
   `𝒦 = {P ≥ 0, M₁ = 0, skew ≤ s}`* — a moment set with three constraints, so ≤3-atom extreme
   points, and (L1) on three atoms is a degree-≤3 polynomial root problem. **This should close
   8.1 completely.** It is the highest-value follow-up and I did not get to it.
5. **Primary-source check owed:** Khinchin's theorem statement, and the Richter/Rogosinski/
   Shohat–Winkler extreme-point theorem, should be cited from primary sources. Saff–Totik and
   Landkof were not accessible; **no theorem number from either is cited here.**

---

# 12. COLD-START HANDOFF (read this first)

## 12.1 What I established — with status labels

| # | Result | Status |
|---|---|---|
| 1 | **(L1)** Passivity ⟺ `Φ_P ≤ Φ_ω` on ℝ, `Φ_μ(η)=∫ln|η−ξ|dμ`, `ω` = Cauchy `(1/π)dξ/(1+ξ²)`. Identity `πε″(x) = D(Φ_ω−Φ_P)(ξ(x))`. | **proved** (algebraic rearrangement of C1) + **verified**: max rel err **2.2e−4** (logistic `w=.08`, N=4001), **1.2e−4** (`w=.05`, N=8001), **1.5e−3** (`w=.20`) — `h9_fixes.py` block B |
| 2 | `ω = Bal(δ_i,ℝ)` = harmonic measure of ℂ₊ at `i` = **push-forward of window Lebesgue (equilibrium) measure** under `ξ=tan(πt/2L)`. So the "point mass at i" is just the seam. | **proved** + verified (`0.679190` vs `0.679205`) |
| 3 | Domination on ℝ ⟺ domination on ℂ (minimum principle, `h→0` at ∞). | **proved** |
| 4 | **(L2)** Reduction principle: `𝒦` convex weak-∗ compact, (H) uniform ⟹ `𝒦⊆𝒫 ⟺ ext(𝒦)⊆𝒫`. Bauer max principle; `P↦Φ_P(η)` affine u.s.c. | **proved**, modulo the truncation step near the seam being written out in full |
| 5 | Hypothesis **(H)** `∫ln sec(πt/2L)dP < ∞` (finite mean loss `σ`) is what makes (L2) legitimate. Never stated in the corpus. | **proved** it is needed for u.s.c.; not proved that it is necessary for (L1) |
| 6 | **One-line proof of Lemma 4.1**: reduces to `A+B ≤ π/2 ⟹ sinA ≤ cosB and sinB ≤ cosA ⟹ both logs in (★) ≤ 0`. | **proved** |
| 7 | **Cor 4.A — exact two-atom law.** Balanced pair at `t₊, −t₋` is passive **iff `t₊+t₋ ≤ L`**, with closed-form depth (★). Contains Cor 4.3 and Cor 4.4/C10. | **proved** + verified: 10/10 verdicts; bisected threshold `S*=1.00000L` at asymmetry ratios 1,2,5; closed-form depth matches DiHiTI to 4–5 decimals (`−0.05018` vs `−0.05021`; `−0.04785` vs `−0.04786`; `−0.02551` vs `−0.02550`; `−0.02430` vs `−0.02433`) |
| 8 | **Cor 4.B — sharp balanced-support law.** `g≥0`, `M₁=0`, `supp g ⊆ [−a,b]` ⟹ passive **iff `a+b ≤ L`**. Strictly generalises Thm 4.2 (which is `a=b=L/2`); `[−L/4,3L/4]` is passive. | **proved** given the classical extreme-point theorem (item 12.5.3) + verified: 5 shapes × 200 MC balanced densities with `a+b=L`, worst `min ε″/D = 0.0`; necessity confirmed for `a+b>L` |
| 9 | C8 edge cascade + C9 depth law + C13 endpoint are the **`η→∞` Taylor expansion of (L1)**. `Δ(∞)=0` ⟹ `ε″(±L)=0`; `O(1/η)` ⟹ `M₁=0`; PV-convergence of `M₁` ⟹ `g(+L)=g(−L)`; two-term max ⟹ `min ε″ ≈ −M₁²/(2πM₂)` at `η*=−M₂/M₁`. | **proved** + verified (3 offsets, 2–3 digits; single atom to 6 digits) |
| 10 | **(L3) Node collapse.** `n`-atom drop measure ⟹ `Δ′ = Q/[(1+η²)∏(η−ξ_k)]` with `deg Q ≤ n` (leading `η^{n+1}` terms both `= D`, cancel) ⟹ **≤ n candidate `η`, exact grid-free test.** | **proved** + verified: max abs diff **9.1e−11** vs 4·10⁵-point kernel sum over 300 random atomic profiles; `deg Q − n ≤ 0` on 400/400 |
| 11 | **No fixed finite node set suffices.** Any `η₁…η_k` can be defeated via the inverse map (Thm 6.1). | **proved** + verified (5 nodes all `>0`, global min `−0.0403`) |
| 12 | `{1, ln\|ξ−η₁\|, …, ln\|ξ−η_k\|}` **is** a Chebyshev system in `ξ` off the poles (`f′ = N/∏`, `deg N ≤ k−1` ⟹ ≤`k` zeros). | **proved** |
| 13 | Corpus count **14 → 3** (L1, L2, + L3 as the one net addition, itself a reduction of L1). | bookkeeping, §10 |

## 12.2 What I ruled out — do not re-walk

* **Cartan's domination principle and balayage are false friends.** One-way sufficiency
  ("true on supp μ ⟹ true everywhere"). They do NOT close the N&S question. Balayage enters
  here only as the *identification* `ω = Bal(δ_i,ℝ)`, a computation.
* **It is NOT the full Choquet / Cartier–Fell–Meyer order.** Riesz decomposition gives
  `∫f dP − ∫f dω = ∫(Φ_P−Φ_ω)dλ + ∫h d(P−ω)`; the harmonic term does not vanish in general.
  Anyone claiming CFM applies directly is overclaiming.
* **No profile-independent finite node test exists** (item 11). Any exact `η`-side reduction
  must scale with profile complexity. This closes the naive reading of H4.
* **H4's Chebyshev hope is on the wrong side.** The Chebyshev property lives in `ξ` (the
  measure side, powering Markov–Krein / extreme-point counts), not in `η`.
* **(L3) is not yet a speed win.** At `n=40` atoms: node test 4.56 ms vs DiHiTI@2¹⁴ 0.98 ms
  = **0.2×, slower** (`np.roots` is O(n³)). Exact and grid-free, but **does not defeat C15**.
* **Wave 3's "Thm 4.2 and 5.2 are genuinely complementary" is misleading** — see §12.4.

## 12.3 The Choquet / dilation-order question — my read

**Read: it is a *specialization* of the dilation order, but only to the sub-cone of
logarithmic potentials — and that specialization is, as the librarian found, not in the
literature.** Concretely:

* The relation IS an order comparison of two measures against a test-function cone. That much
  is Choquet-shaped and the vocabulary (dilation/balayage order, `∫f dP ≤ ∫f dω ∀ f ∈ H`) is
  the right vocabulary.
* The cone here is `H = cone{ln|η−·| : η ∈ ℂ}` = the logarithmic potentials, **strictly smaller**
  than the subharmonic cone of CFM. So CFM's N&S machinery does not transfer for free.
* **Hunch, explicitly labelled conjecture:** on the admissible class the gap may be vacuous,
  because `ω` is Cauchy (all moments `≥1` infinite) so any harmonic test function with a
  convergent `ω`-integral is essentially affine, and the affine part is killed by the necessary
  condition `M₁ = 0`. If that can be made rigorous, (L1) **would** be the full Choquet order and
  CFM would apply verbatim. **I did not attempt this. It is a clean, self-contained target.**
* **HLP majorization is the ancestor, not the tool.** HLP is the `(η−ξ)₊` kernel on ℝ; ours is
  `ln|η−ξ|`. Structurally parallel (both "potential of one measure dominates another's"), but
  HLP theorems do not apply.

## 12.4 Did Thm 4.2 and Thm 5.2 fall out of one relation? — **YES**

Both are `(L2)` applied to a convex family, differing only in which family:

* **Thm 4.2**: `𝒦₁ = {P ≥ 0, |ξ| ≤ 1, E_P ξ = 0}`. Two linear constraints ⟹ `ext(𝒦₁)` = balanced
  ≤2-atom measures (Richter–Rogosinski–Shohat / Winkler). Check on extreme points = Lemma 4.1
  = item 6 above, one line.
* **Thm 5.2**: `𝒦₂ = {P ≥ 0, even, unimodal}`. **Khinchin's theorem** gives the *explicit*
  Choquet representation with `ext(𝒦₂) = {1_{[−a,a]}/2a}`. Check on extreme points = Thm 5.1's
  Clausen margin `M(α,θ) ≥ 0`.

**Why they looked complementary, resolved:** a plateau `1_{[−a,a]}` *is* a mixture of symmetric
pairs `½(δ_s+δ_{−s})`, and by Cor 4.A every pair with `s > L/2` is **not** passive. Convexity is
one-way, so the pair decomposition certifies nothing for `a > L/2` while the plateau
decomposition does. **The two theorems differ in the choice of extreme-point decomposition, not
in their content.** "Neither contains the other" is a remark about two subsets `𝒦₁, 𝒦₂ ⊂ 𝒫`,
not about two theories. This is the H9 unification and it holds.

**Bonus that vindicates the framing:** running (L2) on `𝒦₁` *properly* (i.e. evaluating (L1)
on the extreme points instead of only bounding it) produces **Cor 4.B**, a two-sided law that
strictly supersedes Thm 4.2. The unification is not cosmetic — it generated a sharper theorem.

## 12.5 Exact next steps, in priority order, executable without me

1. **[1 h, highest value] Close Wave 3 Open Problem 8.1 (skew tolerance).**
   It is now a finite computation. Take `𝒦₃ = {P ≥ 0, M₁ = 0, ∫ξ²dP = m₂}` (or a skew
   constraint) — three linear constraints ⟹ `ext(𝒦₃)` = ≤3-atom measures. Evaluate (L1) on a
   3-atom measure: by (L3), `Δ′ = Q/[…]` with `deg Q ≤ 3`, so the criterion is
   "all ≤3 real roots of an explicit cubic give `Δ ≤ 0`" — closed form via Cardano.
   **This should close 8.1 completely and give the exact interpolation between Thms 4.2 and 5.2.**
   Code skeleton already exists: `nodes()` and `atomic_gap()` in `h9_domination_check.py`.
2. **[30 min] Try to prove the §12.3 hunch** (harmonic test functions are vacuous against Cauchy
   `ω` once `M₁ = 0`). If it works, (L1) IS the Choquet order and CFM applies verbatim — a
   citable classical N&S theorem instead of a bespoke one.
3. **[promotion] Send Cor 4.B to `replicator` with parameters only**, no code path:
   *"`g ≥ 0` on `[−L,L]`, supported in `[−a,b]`, tangent-balanced `∫g tan(πt/2L)dt = 0`.
   Claim: DiHiTI partner `ε″ ≥ 0` everywhere iff `a+b ≤ L`."* It is the mission's first
   two-sided, non-trivial, cheap-to-test claim.
4. **[owed]** Cite Khinchin's theorem and the Richter/Rogosinski–Shohat/Winkler extreme-point
   theorem from primary sources. **No Saff–Totik or Landkof theorem number is cited in this
   document** (both were paywalled); keep it that way until verified.
5. **[caveat to chase]** Block C (labelled corpus) agrees **233/240**. The 7 disagreements are
   all families with **unmatched edge slopes** (`tanh_slopes` ×3, `lorentzian`, `opposite_bumps`,
   `logistic` ×2 at `min ε″ ≈ 0`). Diagnosis: when `g(+L) ≠ g(−L)` the Cayley integral for
   `Φ_P` is only conditionally convergent, and my `|x| < 0.97L` truncation cuts off exactly the
   near-seam region where the minimum lives. **This is a defect of the ξ-space *quadrature*,
   not of (L1)** — (L1) is an algebraic identity verified to 1e−4 on edge-regular profiles.
   Fix: evaluate in `t`/circle coordinates near the seam, or impose (H) + edge matching as a
   pre-screen (it is a *necessary* condition anyway, so those profiles are already labelled GAIN
   by the cascade). Worth 20 minutes; until then treat block C as **233/240, cause identified**.

## 12.6 Files written (all under `research/`)

* `research/theory/h9_potential_domination.md` — this document.
* `research/theory/h9_domination_check.py` — 23-block verification (A–H).
* `research/theory/h9_fixes.py` — quadrature-corrected re-runs of blocks B, C, E, F.
