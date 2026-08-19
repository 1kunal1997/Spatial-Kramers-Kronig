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
| C43 | 🚨 **(L1) MAY BE THE VERY RE-REPRESENTATION THE CAPTAIN REJECTED.** Navigator's critique: (L1) is derived as an *identity*, so "passive ⟺ Φ_P ≤ Φ_ω" is ε″ ≥ 0 **in a change of variables** — logically the same move as Carathéodory–Toeplitz, which the Captain dismissed as "merely a re-representation of the HT in Fourier space." The real content is **(L2)**, the extreme-point reduction. **V3 is NOT met**, and H9's "minimal checkable characterization" was never found | cycle 3 navigator | **supported** | — | — | — | — |
| C44 | 🚨🚨 **THE PROJECT HAS NO CERTIFIED NUMERICAL DEFINITION OF "PASSIVE."** Whole-domain min is blind at the boundary (C19); a fixed interior mask is blind at the seam (C39). That undecided convention sits underneath **the corpus labels, C18's α_c bisection, C33's 240/240, and C36's 327/327**. A shared adjudication convention is the same class of common-mode error PROTOCOL.md already warns about | cycle 3 navigator | **supported** | — | — | — | — |
| C23 | ⭐ **(L1) THE LAW**: passivity ⟺ Φ_P ≤ Φ_ω on ℝ, with ω = Bal(δ_i,ℝ) = harmonic measure of ℂ₊ at i = push-forward of the window's equilibrium measure. Exact identity πε″ = D(Φ_ω − Φ_P) | cycle 1 theorist | **supported** | — | — | — | — |
| C24 | ⭐ **(L2) Thm 4.2 and Thm 5.2 are ONE theorem** — the extreme-point reduction (Bauer/Choquet) applied to two convex families (Richter/Winkler ≤2-atom; Khinchin plateau). They differ only in decomposition. Wave 3's "complementarity" dissolved | cycle 1 theorist | **supported** | — | — | — | — |
| C25 | 🚨 **NARROWED (cycle 3). MUST NOT be promoted to `hardened` in its current wording.** The **`iff` is not an `iff`** — the necessity half is FALSE per-profile. Billing as "the mission's first two-sided non-trivial result" was **overreach**, and it does **not** advance V6. ~~Cor 4.B — FIRST TWO-SIDED non-trivial claim.~~ g ≥ 0, M₁ = 0, supp g ⊆ [−a,b] ⟹ passive **iff a + b ≤ L**. Supersedes Thm 4.2; [−L/4, 3L/4] is equally passive | cycle 1 theorist | 🚨 **NARROWED** | ✅ MATCH (sufficiency) | ❌ **NARROWED** | — | — |
| C26 | **(L3) node collapse**: n-atom drop ⟹ ≤ n candidate η — an exact grid-free test, verified 9.1e−11 on 300 profiles. **But 0.2× speed at n=40; does NOT defeat C15** | cycle 1 theorist | **supported** | — | — | — | — |
| C27 | **No profile-independent finite node set works** (proved negative; 5 nodes all positive, global min −0.0403) | cycle 1 theorist | **supported** | — | — | — | — |
| C28 | ⚖️ **SPLIT VERDICT — both parties were partly right.** The stated *mechanism* was false (auditor, K18: truncation irrelevant, only 3/7 edge-slope-unmatched). But the *conclusion* — "a ξ-space quadrature defect, not a defect of (L1)" — is **CORRECT**, now that C34 identifies the real mechanism and C33 achieves 240/240. ~~DIAGNOSIS FALSIFIED~~ (L1) corpus agreement is 233/240 — the *count* stands (auditor reproduced it exactly), but the theorist's explanation is empirically false on both counts. **Cause is OPEN.** | cycle 1 theorist | **KILLED (as a diagnosis)** | — | — | **FALSIFIED** (cycle 2) | — |
| C33 | ⭐⭐ **(L1) corpus agreement is now 240/240.** New evaluator `theory/t3_clausen.py`: drop measure as piecewise-constant cells with **exact** masses w_j = ε′(t_j) − ε′(t_{j+1}); log kernel integrated **exactly** per cell (Clausen antiderivatives); Cl₂ evaluated exactly on the uniform grid by aliasing Σ sin(nθ)/n² into a **trigamma-weighted DFT** (verified to 1e−15 against Catalan). Whole 240-profile corpus in **0.3 s** | cycle 2 theorist | **supported** | — | — | — | — |
| C34 | ⭐ **TRUE mechanism of the old 7/240 gap** — and it is neither party's story. The error is an **η-independent additive DC constant**, O(h), proportional to g(+L)+g(−L): offset/(g(+L)+g(−L)) = **−0.0331…−0.0352 across all 7**, a universal constant over offsets spanning −0.0055→+0.289. Root cause: at t = ±L, `tan(π/2)` evaluates in float to **1.633e16**, so ln\|η−ξ\| is recorded as 37.33 instead of the correct half-cell average 8.17 — that one node supplies **1/3** of the offset, the rest is accumulated trapezoid error on the same log singularity. Halving h halves it (ratios 2.05 across N = 1024…16384) | cycle 2 theorist | **supported** | — | — | — | — |
| C35 | **T1 / Open Problem 8.1 CLOSED — NEGATIVELY. Skew tolerance is ZERO.** Unimodal + M₁ = 0 admits gain for **every** mode c ≠ 0: balance is *vacuous near the seam* because ξ = tan(φ/2) diverges there. Verified against DiHiTI. The theorist also **retracted its own cycle-1 Cardano plan** — unimodal extreme points are **arcs, not atoms**, hence transcendental, not degree-3 | cycle 2 theorist | **supported** | — | — | — | — |
| C29 | ✅ **RESOLVED by C33/C34** (was: open, high priority): (L1) disagrees with the corpus on 7/240 profiles, and **4 of them are the project's core profiles** (both `logistic` cases, `opposite_bumps`, `lorentzian`). Three possibilities: (a) a real defect in (L1); (b) a defect in how the domination gap is evaluated; (c) **the corpus labels are wrong** — they came from the C19 blind indicator, which cannot resolve near-threshold cases | cycle 2 auditor | **conjecture** | — | — | — | — |
| C21 | ⭐ **"Margin 0" is an artifact.** DiHiTI is exactly linear (6e−15); the passive-set matrix has rows 0 and N−1 identically zero, rank N−2. ε″(±L)=0 is an operator identity, not a cone facet. Interior margin converges to a positive, N-independent ~1e−3·D | cycle 1 red-team | **supported** | — | — | — | — |
| C22 | ⚠️ **NARROWED by its own author (cycle 2).** The *identity* is EXACT (4 digits, better than the claimed 1.6e−4). But the **1.70× speedup is WITHDRAWN**: `np.linspace(-L,L,N)` duplicates the endpoint, so `hilbert` saw period 2L·N/(N−1); fixing it makes N−1 = 65535 = 3·5·17·257, FFT-hostile, and the correct version runs at **0.44×**. There is no speedup | cycle 1 red-team | **NARROWED** | — | **self-narrowed** | ⚠️ audit was wrong (common-mode bug) | — |
| C37 | ⚖️ **RESOLVED — NOT UPHELD**, but the underlying fact is TRUE and was worth finding. σ = 0 is a genuine *zero* of the depression profile (−ℋ[φ](0) = 0 to machine precision, for both the replicator's raised cosine and the proof's C^∞ bump); Lemma 1 uses its **minimum**, at σ\* = −0.723. All three reported mismatches trace to **one artifact**: probing at the bump *edge* (σ ≈ −1) rather than the extremum. −ℋ at the edge is 0.388 vs max 0.661 — **ratio 1.70**, which reproduces the κ/π factor 2.6, the 15% drift (at σ\* the derivative vanishes so the value is second-order insensitive; at the edge it is not), and the 1.64× x₀-spread exactly. Measured properly, κ spreads **1.078×, 1.8% RMS**. ~~Original:~~ under the literal reading — bump symmetric about x₀, depression measured *at* x₀ — width-independence is **provably false**: ℋ maps even functions to odd about their own centre, so ℋ[bump](x₀) = **0 exactly**, for any width or amplitude. Prediction **pre-registered before running numerics**; confirmed at δ^1.0015 over **2.5 decades** (375× change, no plateau). An "offset" construction (probe at fixed *relative* position) approaches a plateau but drifts ~15% over 1.5 decades, ~40% overall — outside the pre-registered 2% tolerance. κ/π measured **1.07–1.10 vs claimed 0.418** (factor ~2.6). κ is profile-independent but **not** x₀-independent (~12% RMS, 1.64× across interior). N-independence **confirmed** to 0.0034%, including non-powers of two | cycle 3 replicator | **supported** | — | → adjudication | — | — |
| C25′ | ✅ **What actually survives of Cor 4.B, and it is solid**: **sufficiency** (g ≥ 0, M₁ = 0, supp g ⊆ [−a,b], a+b ≤ L ⟹ passive) plus a **sharpness statement about the box** — the same logical shape as Thm 4.2, *not* a per-profile necessary condition. Verified: 3600 random many-atom balanced measures across 10 boxes, **zero negatives**; the Bauer/extreme-point step checked against its actual hypotheses; **boundary a+b = L is PASSIVE**, marginally, min ε″ = 0 at the interior point x\* = b−a | cycle 3 red-team | **supported** | — | **SURVIVED** | — | — |
| C38 | 🚨 **The necessity half is FALSE.** Counterexample from exact linearity: P = λ·(balanced corner pair at −0.40, +0.90) + (1−λ)·(balanced pair at ±0.20). Both pieces g ≥ 0 with M₁ = 0; support hull **exactly** [−0.40, 0.90] so a+b = 1.30L, **effective support = nominal box** (closing the ambiguity behind the replicator's 76/80). At λ = 0.2, min ε″ = **+9e−9: passive.** Confirmed at N = 2¹⁵/2¹⁷/2¹⁹ and independently by a hired hand (erf smoothing, both grid conventions, non-power-of-two N, own closed form, λ\* ≈ 0.278). **Not marginal**: max passive corner-mass fraction is 26% at a+b = 1.30L, 21% at 1.40L, 8.7% at 1.80L, 0.09% at 1.98L — for *every* a+b up to 2L there exist passive profiles touching both corners | cycle 3 red-team | **supported** | — | — | — | — |
| C39 | 🚨🚨 **THE INTERIOR-WINDOW CONVENTION HAS ITS OWN BLIND SPOT — affects corpus labels, not just C25.** With M₁ ≠ 0 the profile carries strict gain matching C9 to 3–4 digits, **but the gain migrates to the seam**: argmin at x/L = −0.956, −0.9947, −0.99946, −0.99995 for M₁ = 1e−1…1e−4. **A 0.95L interior window reports PASS for every one.** So *both* conventions fail: whole-domain min is blind (C19) because the endpoint is pinned to zero, and a fixed interior window is blind because near-balanced gain hides arbitrarily close to the seam | cycle 3 red-team | **supported** | — | — | → auditor | — |
| C40 | **Threshold depth law**: min ε″ = −(D/2)·sin(πa/L)·(a+b−L)/L, matching analytics to **5 digits** | cycle 3 red-team | **supported** | — | — | — | — |
| C41 | **Non-interval supports buy nothing** — only the interval hull matters. And the Cayley form of Cor 4.A is the *product* ξ₊\|ξ₋\| ≤ 1 | cycle 3 red-team | **supported** | — | — | — | — |
| C42 | At a+b = L exactly, production DiHiTI returns a small **negative** minimum at every finite N, decaying O(h²). Isolated as **quadrature**, not a support violation (width-independent over a 32× sweep at fixed N). *A referee will ask about this* | cycle 3 red-team | **supported** | — | — | — | — |
| C45 | ⭐ **The cycle-1 contradiction is RESOLVED — all three measurements were right, of different things.** Hand B measured **L¹ mass** (∝ δ, so narrower *is* cheaper); the replicator measured Δ **at the centre** (≡ 0 by parity); Lemma 1 measures **L^∞ amplitude** (flat in δ). No one erred on the physics. Hand B's non-monotonic rise is a **window-edge artifact** — the easiest bump position is x₀ = +0.97L, outside the 0.95L scored window — **not** the "under-resolved at 82 grid points" explanation cycle 2 gave, which is **false** | cycle 3 adjudicator | **supported** | — | — | — | — |
| C46 | 🚨 **THE LOWER BOUND MAY NOT BIND ON PHYSICAL PROFILES.** Adjudicator's own sharpening: **any N-independent Lipschitz bound collapses Θ(N) to Θ(1)**, because a flipping bump must eat the margin m₀ ≈ 1e−3·D. Real ε′ profiles are smooth. So V2a's Ω(N) holds for a class admitting arbitrarily narrow features — which may exclude the physically realizable case. *This partially revives H2/N3 (bandwidth, FFT-free O(N)), demoted on weaker grounds* | cycle 3 adjudicator | **supported** | — | — | — | — |
| C47 | ⚠️ **`v2_lower_bound.md` has internal defects found by the adjudicator, not the replicator.** Constants self-contradictory (κ = πC_φ vs πC_φ⁻¹ vs C_φ = 2.39; **the π is spurious — correct is κ = 1/h_φ = 1.254**). Its `TV(p) ≤ 2r` is **vacuous** for bumps (TV = 2a at any width), so §1's "not an artifact of admitting rough inputs" **contradicts its own §3.5**. Theorem 2 needs its intervals **inset by 2kh** from the scored window | cycle 3 adjudicator | **supported** | — | — | — | — |
| C36 | ⭐⭐⭐ **The 3× wall-clock bar is unreachable by Amdahl, for any approach that keeps the FFT.** Compiled C (fused kernels, 0 numpy temporaries) gives **3.34–3.65× on the O(N) legs** — confirming "constants, not information deficit" for the non-FFT part. But `scipy.signal.hilbert` alone is **52% of baseline wall clock at N = 2¹⁴, rising to 61% at N = 65536**, so the Amdahl ceiling — the best possible speedup with the O(N) legs made **free** — is **already below 3× for every N ≥ 16384**, including the mission's own target. End-to-end measured: **1.50×**. Correctness **327/327**, including 147/147 near-threshold | cycle 3 numericist | **supported** | — | — | — | — |
| C30 | ✅ **RESTORED to `supported`, REWORDED and NARROWED.** The MISMATCH was a **First Mate paraphrase error**, not a defect: Lemma 1 reads **min_x** over the interior window; the paraphrase dropped those words, so the replicator measured at the bump centre where parity forces exactly zero. **Standing result: Θ(N) query complexity, tight, for the finite (M+1)-input family in the L^∞/BV ball of radius 1.68·D — CONDITIONAL on the class admitting O(1)-grid-point features.** ~~CONTESTED~~ ~~V2a PROVED — query complexity is Θ(N), tight.~~ **The width-independence at the heart of the proof failed blind replication under the literal reading.** ~~Original:~~ **V2a claimed: query complexity is Θ(N), tight.** A bump of amplitude a and **any** width δ depresses ε″ by −a/(πC_φ), *independent of δ*, because the singular part of the DiHiTI kernel is the Hilbert kernel and ℋ commutes with dilation. Hence Θ(N) disjoint supports with a uniform N-independent amplitude bound (r/D = 1.680, identical to 4 digits at N = 4096/8192/16384; 41/41 positions flip) → pigeonhole → Ω(N); bound holds against the *finite* (M+1)-input family, foreclosing the "permissive input class" objection; Yao → randomized. Baseline reads N ⟹ **tight** | cycle 2 red-team | **supported** (reworded) | ⚖️ mismatch → **adjudicated: paraphrase error** | narrowed | — | — |
| C31 | **V2b — conditional, method-relative barrier.** Both known superlinear techniques are entropy counts of combinatorial multiplicity (Ben-Or counts components; Björner–Lovász–Yao counts faces). A convex polyhedral cone with Θ(N) facets has **1 component** and ≤2^Θ(N) faces, so both are capped at O(N) **by arithmetic**. Element distinctness gets N log N from N! orderings — no analogue here. Morgenstern's Ω(N log N) is for *computing* the DFT in bounded-coefficient linear circuits and does not transfer to deciding one bit. **Explicitly NOT an unconditional lower bound** | cycle 2 red-team | **supported** | — | — | — | — |
| C32 | **B1/B2 tail bounds are SOUND** — the K13 fix is real: forward differences have symbol ratio sinc(θ/2) ∈ [2/π, 1], never zero, unlike centred differences. Verified against the exact FFT tail (48 rows, min ratio 13.9). **B3 is FALSE**, twice and for two different reasons | cycle 2 red-team | **supported** | — | — | — | — |
| C16 | **Passive ⇒ stable**: ε″ ≥ 0 ⟹ Im V ≤ 0 ⟹ Im(E) ≤ 0 for every mode, no growing solution | W2 §3 | conjecture | — | — | — | — |
| C18 | **α_c = 0.66623** — the true grid-converged crossover for the Captain's opposite-bumps family, bisected on the *interior* minimum; stable to <1e−5 over N = 10⁴–1.6×10⁵ | cycle 1 numericist | **supported** | — | — | — | — |
| C19 | ⭐ **min ε″ over the whole domain is a blind PASS/FAIL indicator.** It reads exactly 0.000 for *every* passive profile regardless of margin, because the minimum is realized at the boundary, which C13 pins to zero by construction. The interior minimum is the informative quantity | cycle 1 numericist | **supported** | — | — | — | — |
| C20 | g = −dε′/dx for the opposite-bumps family is **not** the rectangular "plateau minus centred dip" of §5.1 — it has three signed regions, so the narrow closed-form d*(b) hypotheses are not met as stated | cycle 1 numericist | **supported** | — | — | — | — |
| C17 | **Analyticity and passivity are independent**: triangle profile has spectral FoM 99.9999987% *and* min ε″ = −1.82 | W2 §3 | conjecture | — | — | — | — |
| C48 | ⭐ **Theorem 1 (density)**: any test set of positions that is sound AND complete for passivity (over balanced positive atomic g) must be **dense** — a violation can be confined inside any open window (boundary pair + inflation; positivity window shrinks ∝ √s onto the target). Upgrades C27/K17 from "no finite node set" to "no non-dense set". `ns_answer.md` §3, `e1_density.py` (δ=0 control exact at machine precision) | solo Fable 2026-08-19 | **supported** | — | — | **CLEAN** `results/f5_audit_fable/` | — |
| C49 | ⭐⭐ **Theorem 2 (no finite battery)**: for ANY finite set of linear functionals of ε′ (arbitrary kernels: moments, samples, corner masses, mass/M₁/M₂/σ, …) and ANY decision rule on their values, there exist two positive balanced atomic profiles with the SAME atoms, identical readings, opposite verdicts. Proof = saturation/convexity (Lemma D) + active-point interval (Lemma E, numerically certified window [1.45, 2.96], Δ″<0 throughout) + independence via log singularities (Lemma A). Explicit witnesses: k=8 battery **containing the entire known necessary cascade** (match 6e−16, GAIN −7.8e−3 vs PASSIVE, production-corroborated) and k=16 (match 8e−12, GAIN −1.6e−5, production-confirmed at N=2²⁰–2²²). **Kills the "clean finite conditions" program on continuum profiles**; closes V3/V6 downward. `ns_answer.md` §4, `e2_battery.py`, `e4_lemmaE.py`, `witness_k*.npz` | solo Fable 2026-08-19 | **replicated** (witness pair only — see scope note) | **MATCH** `results/c49_blind_replication/` | — | **CLEAN** (exact evaluator; caveat on k=16 *production* corroboration — see C53) | — |
| C50 | **Lemma B (no exterior gain)**: a balanced positive drop measure has Δ < 0 strictly outside the convex hull of its Cayley support (per-pair outer-branch monotonicity, numerator bound γ(1+β²) > 0, + mean-zero pair decomposition). In x-space: ε″ > 0 strictly between the outermost drop and the seam. **Discharges red-team residual risk #3** of `cycle3_c25_hardening/redteam.md` | solo Fable 2026-08-19 | **supported** (proof is elementary and self-contained) | — | — | — | — |
| C51 | **N1 CLOSED — corner mass cannot be a criterion.** λ* is exact per family but ranges **0.030–0.383** for one and the same corner geometry (−0.4,+0.9) as the bulk moves; killer pair: identical corner distribution (fraction 0.25), bulk (±0.2) → PASSIVE (margin 5.3e−3) vs bulk (−0.24,+0.74) → GAIN (−2.0e−2, production-confirmed). Controls 4/4 vs red-team/handA (0.2620/0.2778/0.0872/0.0009). ⚠️ red-team's λ*(a+b) table mixed unrecorded corner splits — its 1.40L/1.94L rows match neither convention tried; re-derive before reuse. `e3_n1_closure.py` | solo Fable 2026-08-19 | **supported** | — | — | — | — |
| C52 | **T2/N2 (CFM/Choquet conjecture) REFUTED.** f = cos(x)e^{−y} is harmonic entire, bounded on ℂ₊, ω-integrable (∫f dω = e^{−1}); P = δ₀ is passive yet ∫f dP = 1 > e^{−1}. The log-potential order is **strictly weaker** than the subharmonic/CFM order; CFM does not apply verbatim. Corrected form: the test cone is exactly the log-potential cone (+constants); sub-logarithmic growth on ℂ is the growth condition that makes the repaired statement true (and nearly tautological). `ns_answer.md` §7 | solo Fable 2026-08-19 | **supported** (counterexample is one line, checkable by hand) | — | — | — | — |
| C53 | ⚠️ **Production `ht_derivative` carries a profile-dependent additive O(h) offset on smoothed staircases** (+3.55e−5 → +1.76e−5 → +0.85e−5 → +0.4e−5 at N = 2¹⁹…2²², s-independent below s ≈ 6e−4) that **cancels exactly in differences** (differential = 3.382e−5 = 2×closed-form at every N, 4 digits). At corpus-historical N ≈ 2048 the scale is ~1e−3 — same order as C21's interior margin. Near-threshold absolute production verdicts are convention-limited. **Feeds P0.** Mechanism unidentified | solo Fable 2026-08-19 | **supported, MECHANISM NOW IDENTIFIED — see refinement below** | — | — | **CLEAN, refined** `results/f5_audit_fable/` | — |
| C54 | **Prop 3 (simplex rigidity)**: the passive cone is a linear image of the nonnegative cone — a lattice/simplex cone whose extreme rays are the unit-gain-atom profiles and whose **unique** extremal representation of any passive g is ε″[g] itself. Explains why CT / potential / Fejér–Riesz all turned out to be re-representations: for this cone there is nothing else to be. Leans on inherited Thm 6.1 (conjecture rung) | solo Fable 2026-08-19 | **supported** (elementary, modulo Thm 6.1) | — | — | — | — |

## C15 is the mission's central target

It is an **assertion, not a theorem** — stated in a verification-summary section, not proved. The
identical pattern (asserted impossibility) was broken wholesale on 2026-08-13, when three of four
"no such condition exists" claims fell within a single session.

Two outcomes are both wins, and the crew must not prefer one:
- **Killed** → V1 (a cheap certificate exists)
- **Proved** → V2 (a real lower bound, publishable)

The failure mode to guard against is *neither* — accepting C15 on the strength of its author's
confidence. It sits at `conjecture` until a proof or a counterexample moves it.

| C55 | ⭐⭐ **Theorem 2 is PROVED OUTRIGHT — the numerically-certified ingredient is gone.** Lemma E1: the symmetric reciprocal-pair base point (atoms ξ = ±½, ±2, weight ½ each; W = 2, M₁ = 0) has Δ(η) = ½ln[\|η²−4\|\|η²−¼\|/(1+η²)²], so passivity reduces to \|y−¼\|\|y−4\| ≤ (1+y)² in y = η² — three elementary rows, middle one of discriminant −175/16 < 0 — giving Δ ≤ 0 with equality **exactly** at η = 0 and nondegeneracy Δ″(0) = −25/4 as an *identity*. Lemma E2: one implicit-function step on (Δ_m, ∂_ηΔ_m) = 0, Jacobian det = −(25/4)ln2 ≠ 0, performs the sweep **and** the thickening together at rate η̇(0) = (4/25)Σ(β_j−γ_j)/(β_jγ_j) ≠ 0. Also Lemma A′: Lemma A reproved in three real-variable lines, so Theorem 2 is now entirely real analysis, and step (C) of the old proof is no longer needed at all. Supersedes the asymmetric triple, the window [1.45, 2.96] and the measured Δ″ range. `theory/lemmaE_closure.md` | theorist-F1 2026-08-19 | **supported** (proof; First-Mate-verified — see note) | — | — | — | — |
| C56 | ⭐ **(G4) −πε″ is strictly concave on exactly the middle half, for EVERY positive drop measure, in one line.** Δ″_g(η) = −Σw_j/(η−ξ_j)² − W(1−η²)/(1+η²)²: both terms are negative iff \|η\| < 1, and η ∈ (−1,1) ⟺ \|t\| < L/2. Verified independently over 300 random measures (max Δ″ = −1.1e−3 inside; **261/300 go positive somewhere outside**, so the middle-half boundary is *sharp*, not an artifact of the bound). Discharged the last soft step of Lemma E2. **Flagged by theorist-F1 as out-of-scope: this may be the structural home of C3 (middle-half sufficiency) — a direct lead for Goal B/V8.** | theorist-F1 2026-08-19 | **supported** | — | — | — | — |

## C49 promotion scope note (First Mate, 2026-08-19)

**What is `replicated`: the witness pair, not the theorem.** The blind replicator was given only the
20 atom positions and two weight vectors (relabelled A/B, mapping sealed in `research/.sealed/`) and
was forbidden the originating directory; `manifest.json` records `forbidden_paths_touched: []`. It
reproduced the property by an independent route — direct x-space evaluation of the log-potential
master identity, no Cayley coordinates, no polynomial node collapse, no FFT — plus a production
`ht_derivative` cross-check on a **cell-centred non-duplicating grid** swept over N ∈ {4001, 6000,
8192, 16001} and smoothing s ∈ {0.02 … 0.002}, 16 combinations with no verdict flip. Its harness
control (balanced pair passive ⟺ a+b ≤ 1) reproduced the threshold exactly.

Three independent routes now agree on the headline number: Fable's exact node test −7.756e−3,
the replicator's analytic route −0.007756 @ x ≈ 0.6311, and the First Mate's own 4-million-point
x-space scan −7.755719e−3 @ x = +0.63111 (negative on 1.78% of the domain; witness B negative
nowhere). The non-power-of-two N values and the fresh grid convention specifically address the
common-mode failure recorded in `PROTOCOL.md` (two implementers reaching for `np.linspace(-L,L,N)`).

**What is NOT certified by this verdict:** Theorem 2 *itself*. A replicator cannot certify a proof.
Lemma E remains numerically certified rather than analytic, and the thickening step remains a
continuity argument stated with hypotheses — that is board item **F1**, dispatched, and victory
condition **V7** turns on it. C49 must not be quoted as a proved theorem until F1 lands and
`red-team` + `auditor` sign it to `hardened`.

## ⚠️ C53 REFINED by the F5 audit (2026-08-19) — it is the linspace endpoint bug, again

The auditor traced C53's "profile-dependent additive O(h) offset in `ht_derivative`" to its cause,
which Fable had left unidentified. It decomposes into two pieces:

1. **The dominant term is NOT a property of `ht_derivative` at all — it is the `grid="linspace"`
   convention.** `np.linspace(-L, L, N)` duplicates the endpoint, so the FFT-based Hilbert transform
   sees period `2L + dx` instead of `2L`. Verified directly, and independently re-confirmed by the
   First Mate: `N·dx − 2L` is **exactly `+dx`** for linspace and **exactly `0`** for the periodic
   grid, at N = 2048, 2049 and 4096. This is the **same defect PROTOCOL.md already names as the
   voyage's most expensive error class** — it produced a bogus 1.70× speedup that survived an
   independent audit because both implementers reached for the same idiom. It reappeared, undetected,
   inside the solo session's own harness.
2. A smaller grid-independent residual (~−8.3e−7 at the tested s), still unidentified, too small to
   threaten any reported verdict.

**Consequence for C49's k=16 witness — narrow margin, no change in truth.** With the shipped default
grid, one power of two below the range Fable quoted (**N = 2¹⁹**, a value used elsewhere in that same
session), production returns **PASS for a witness that genuinely has GAIN**. `grid="periodic"` —
already implemented in the same function, simply not the default — is correct at N = 2¹⁴ and is
~3 orders of magnitude more accurate at matched N (N = 2048: linspace error +8.03e−3, periodic
−3.47e−5). The **"cancels in differences"** half of C53 is robust: it holds to 4 significant figures
under *both* grid conventions at every N from 2¹⁹ to 2²².

**Blast radius: zero change to any claim's truth value.** Every C48/C49 verdict rests on the exact,
grid-free node test, which the auditor independently reconfirmed to 9–10 digits via mpmath. What
changes is how much weight the phrase "production-confirmed" can carry for the k=16 witness, and it
converts C53 from a mystery into a fixable convention error. **Feeds P0 and F2 directly.**

**Standing instruction:** anything in this voyage calling `production_min_eps2` must pass
`grid="periodic"`. The k=8 witness remains the one to quote — three independent routes agree on it.

## C55 verification note (First Mate, 2026-08-19)

Verified independently before recording, since V7 turns on it:
- Base point: exact node test **and** a 3,000,001-point dense grid both give sup Δ = +6.7e−16 at
  η = 0.000000 — machine zero at exactly the predicted location. Δ″(0) = −6.250000 = −25/4 by finite
  differences; Δ_G(0) = −0.693147 = −ln 2, so the Jacobian entry and det = −(25/4)ln2 are right.
- Both controls the theorist proposed *itself* fire correctly: β = 0.3 (c = 11.20 > 6) → sup Δ =
  +4.17e−1 GAIN; broken reciprocity (2 → 3) → Δ(0) = +0.4055 ≠ 0. Its bifurcation prediction
  β = √2−1 lands on c = 6.000 exactly.
- **η̇(0) reproduced on a configuration of my own choosing** (different extra pairs than the
  theorist used): Newton continuation gives η/t → 0.61887 → 0.61911 → 0.61923 as t → 10⁻³ … 2.5×10⁻⁴,
  converging linearly on the predicted **0.61935250**.
- Lemma A′'s argument checked by reading: real-analytic continuation within a component plus log
  blow-up at an atom belonging to exactly one pair. Correct, and genuinely lighter than the
  complex-analytic original.

**Rung discipline:** C55 enters at `supported`, not higher. It is a *proof*, and the First Mate
cannot promote by judgment — PROTOCOL rule 2. **V7 is NOT yet met**: it requires C49 at `hardened`,
which requires `red-team: SURVIVED`, and red-team has not run. The proof is delivered; the gate is
not closed.

**By-product for P0:** the base point is an exactly-known point of the passive boundary — the
equal-drop four-step staircase at t = ±0.2951672…, ±0.7048327…, with closed form
ε″(x) = −(1/π)·½ln[|ξ²−4||ξ²−¼|/(1+ξ²)²], ξ = tan(πx/2), whose correct verdict is "passive, margin
exactly 0, attained only at x = 0". That is a calibration profile any numerical passivity convention
must get right, and it is the first one the voyage has had.
