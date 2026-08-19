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
| C13 | **ε″(±L) = 0 automatic** in the DiHiTI gauge (seam normalization K(±L;t) ≡ 0) — mission requirement V5 | W3 §1 | conjecture | — | — | — | — |
| C14 | **Finite smooth-moment batteries cannot be sufficient** (cosine dressing, m-independent min ε″ ≈ −0.20) | W2 §4 | conjecture | — | — | — | — |
| C15 | ⚠️ **"Nothing beats running DiHiTI"** — no test cheaper than O(N log N) exists | W3 §7 | **conjecture — CONTESTED** | — | — | — | — |
| C16 | **Passive ⇒ stable**: ε″ ≥ 0 ⟹ Im V ≤ 0 ⟹ Im(E) ≤ 0 for every mode, no growing solution | W2 §3 | conjecture | — | — | — | — |
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
