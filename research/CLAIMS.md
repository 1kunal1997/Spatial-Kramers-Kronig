# Claim Register

Rungs: `conjecture` → `supported` → `replicated` → `hardened` → `ESTABLISHED`

**All inherited claims enter at `conjecture`.** Wave 2/3 verified them against `scipy`-pipeline
numerics, which counts as *evidence produced by the author* — not as blind replication, adversarial
attack, or referee sign-off. Written down is not the same as gated. Promotion requires this crew's
verdicts. This is not distrust of the prior work; it is the point of the gate.

Note: Wave 3's numbers were produced by the **same pipeline** (`np.gradient` → `scipy.signal.hilbert`
→ `cumulative_trapezoid`) that defines the quantity under test. A shared-pipeline check cannot detect
a convention error in the pipeline itself. `replicator` must use an independent route.

| ID | Claim | Source | Rung | replicator | red-team | auditor | referee |
|----|-------|--------|------|-----------|----------|---------|---------|
| C1 | **Master identity**: ε″ = (1/π)∫g(t)·ln[cos(πt/2L)/\|sin(π(x−t)/2L)\|]dt — DiHiTI is a log potential of the drop density | W2 §1, W3 Prop 1.2 | conjecture | — | — | — | — |
| C2 | **Exact criterion** (Thm 3.1): four equivalent forms — seam-minimum / supporting-line / Carathéodory–Toeplitz PSD | W3 §3 | conjecture | — | — | — | — |
| C3 | **Middle-half sufficiency** (Thm 4.2): g ≥ 0, supp g ⊆ [−L/2, L/2], M₁ = 0 ⟹ ε″ ≥ 0. Constant L/2 sharp | W2 §2, W3 §4 | conjecture | — | — | — | — |
| C4 | **Two-point inequality** (Lemma 4.1): \|η−β\|^γ\|η+γ\|^β ≤ (1+η²)^((β+γ)/2) — the ingredient completing C3 for asymmetric g | W3 Lemma 4.1 | conjecture | — | — | — | — |
| C5 | **Unimodality theorem** (Thm 5.2): g even and unimodal ⟹ passive at **any** width, no support restriction | W3 §5 | conjecture | — | — | — | — |
| C6 | **Plateau Clausen closed form** (Thm 5.1) and margin M(α,θ) ≥ 0 | W3 §5 | conjecture | — | — | — | — |
| C7 | **Dip tolerance** d*(b) = min_θ M(α_a,θ)/M(α_b,θ), matching DiHiTI bisection to 3 digits | W3 §5.1 | conjecture | — | — | — | — |
| C8 | **Necessary cascade**: edge slopes g(+L) = g(−L); M₁ = 0; M₂ ≥ 0; σ ≥ 0 | W2 §3, W3 §3.1 | conjecture | — | — | — | — |
| C9 | **Perturbative gain depth** min ε″ ≈ −M₁²/(2πM₂), valid when g vanishes fast enough at edges that M₂ < ∞ | W2 §3, W3 eq. 9 | conjecture | — | — | — | — |
| C10 | **Off-centre step closed form**: min ε″ = (D/π)·ln cos(πx₀/2L) at the wrap-antipode; passivity is not translation invariant | W2 §1, W3 Cor 4.4 | conjecture | — | — | — | — |
| C11 | **Mirror identity**: ε″_m(x) = −ε″(−x); the mirror of a passive profile is pure gain | W2 §1, W3 Rmk 1.4 | conjecture | — | — | — | — |
| C12 | **No-go on the line** (Prop 2.1): no asymptotically-lossless asymmetric passive profile on ℝ; truncation is forced, with price max ε″ = (D/π)ln(L/w) | W3 §2 | conjecture | — | — | — | — |
| C13 | **ε″(±L) = 0 automatic** in the DiHiTI gauge — mission requirement V5. *Now confirmed by two independent routes: numerical pipeline (numericist) AND operator rank deficiency (red-team). Promotion to `replicated` should be cheap.* | W3 §1 | **supported** | — | — | — | — |
| C14 | **Finite smooth-moment batteries cannot be sufficient** (cosine dressing, m-independent min ε″ ≈ −0.20) | W2 §4 | conjecture | — | — | — | — |
| C15 | ⚠️ **"Nothing beats running DiHiTI"** — no test cheaper than O(N log N) exists | W3 §7 | **NARROWED** | — | **NARROWED** (cycle 1) | — | — |
| C23 | ⭐ **(L1) THE LAW**: passivity ⟺ Φ_P ≤ Φ_ω on ℝ, with ω = Bal(δ_i,ℝ) = harmonic measure of ℂ₊ at i = push-forward of the window's equilibrium measure. Exact identity πε″ = D(Φ_ω − Φ_P) | cycle 1 theorist | **supported** | — | — | — | — |
| C24 | ⭐ **(L2) Thm 4.2 and Thm 5.2 are ONE theorem** — the extreme-point reduction (Bauer/Choquet) applied to two convex families (Richter/Winkler ≤2-atom; Khinchin plateau). They differ only in decomposition. Wave 3's "complementarity" dissolved | cycle 1 theorist | **supported** | — | — | — | — |
| C25 | ⭐⭐ **Cor 4.B — the mission's FIRST TWO-SIDED non-trivial claim.** g ≥ 0, M₁ = 0, supp g ⊆ [−a,b] ⟹ passive **iff a + b ≤ L**. Supersedes Thm 4.2; [−L/4, 3L/4] is equally passive | cycle 1 theorist | ⭐ **REPLICATED** | ✅ **MATCH** (3 routes, blind) | → next | → next | — |
| C26 | **(L3) node collapse**: n-atom drop ⟹ ≤ n candidate η — an exact grid-free test, verified 9.1e−11 on 300 profiles. **But 0.2× speed at n=40; does NOT defeat C15** | cycle 1 theorist | **supported** | — | — | — | — |
| C27 | **No profile-independent finite node set works** (proved negative; 5 nodes all positive, global min −0.0403) | cycle 1 theorist | **supported** | — | — | — | — |
| C28 | ☠️ **DIAGNOSIS FALSIFIED.** (L1) corpus agreement is 233/240 — the *count* stands (auditor reproduced it exactly), but the theorist's explanation is empirically false on both counts. **Cause is OPEN.** | cycle 1 theorist | **KILLED (as a diagnosis)** | — | — | **FALSIFIED** (cycle 2) | — |
| C29 | ⚠️ **OPEN QUESTION, HIGH PRIORITY**: (L1) disagrees with the corpus on 7/240 profiles, and **4 of them are the project's core profiles** (both `logistic` cases, `opposite_bumps`, `lorentzian`). Three possibilities: (a) a real defect in (L1); (b) a defect in how the domination gap is evaluated; (c) **the corpus labels are wrong** — they came from the C19 blind indicator, which cannot resolve near-threshold cases | cycle 2 auditor | **conjecture** | — | — | — | — |
| C21 | ⭐ **"Margin 0" is an artifact.** DiHiTI is exactly linear (6e−15); the passive-set matrix has rows 0 and N−1 identically zero, rank N−2. ε″(±L)=0 is an operator identity, not a cone facet. Interior margin converges to a positive, N-independent ~1e−3·D | cycle 1 red-team | **supported** | — | — | — | — |
| C22 | DiHiTI = **one** Hilbert transform of the ramp-periodized profile (rel. rms 1.6e−4); measured 1.70× over the incumbent baseline | cycle 1 red-team | **supported** | — | — | — | — |
| C16 | **Passive ⇒ stable**: ε″ ≥ 0 ⟹ Im V ≤ 0 ⟹ Im(E) ≤ 0 for every mode, no growing solution | W2 §3 | conjecture | — | — | — | — |
| C18 | **α_c = 0.66623** — the true grid-converged crossover for the Captain's opposite-bumps family, bisected on the *interior* minimum; stable to <1e−5 over N = 10⁴–1.6×10⁵ | cycle 1 numericist | **supported** | — | — | — | — |
| C19 | ⭐ **min ε″ over the whole domain is a blind PASS/FAIL indicator.** It reads exactly 0.000 for *every* passive profile regardless of margin, because the minimum is realized at the boundary, which C13 pins to zero by construction. The interior minimum is the informative quantity | cycle 1 numericist | **supported** | — | — | — | — |
| C20 | g = −dε′/dx for the opposite-bumps family is **not** the rectangular "plateau minus centred dip" of §5.1 — it has three signed regions, so the narrow closed-form d*(b) hypotheses are not met as stated | cycle 1 numericist | **supported** | — | — | — | — |
| C17 | **Analyticity and passivity are independent**: triangle profile has spectral FoM 99.9999987% *and* min ε″ = −1.82 | W2 §3 | conjecture | — | — | — | — |

## C15 is the mission's central target

It is an **assertion, not a theorem** — stated in a verification-summary section, not proved. The
identical pattern (asserted impossibility) was broken wholesale on 2026-08-13, when three of four
"no such condition exists" claims fell within a single session.

Two outcomes are both wins, and the crew must not prefer one:
- **Killed** → V1 (a cheap certificate exists)
- **Proved** → V2 (a real lower bound, publishable)

The failure mode to guard against is *neither* — accepting C15 on the strength of its author's
confidence. It sits at `conjecture` until a proof or a counterexample moves it.
