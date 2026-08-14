# Passivity of sKK profiles: the potential-theory criterion (2026-08-13)

Adversarial follow-up to `handoff.md`. The previous session claimed the
Carathéodory–Toeplitz eigenvalue test is the end of the road: no cheap
sufficient condition, no finite-moment test, no local condition, no scalar
test. **Three of those four claims are now broken; the fourth is broken in
spirit and confirmed only in its precise technical core.** Everything below is
a statement about ε′(x) only — no Hilbert transform, no Toeplitz matrix, no
eigenvalues — and every claim is verified numerically in
`potential_criterion.py` against the project's own `ht_derivative` (DiHiTI).

Throughout: domain [−L, L], circle angle θ = πx/L, **drop density
g(t) = −dε′/dt**, total drop D = ε_b² − 1, and half-tangent coordinate
ξ = tan(πx/2L). "Passive" means the DiHiTI-gauge ε″ ≥ 0 (endpoint loss zero),
i.e. the generator's own label.

---

## 1. The master identity: DiHiTI is a log-potential, not a Hilbert transform

Integrating the periodic conjugate-function kernel by parts once:

    ε″(x) = (1/π) ∫_{−L}^{L} g(t) · ln[ cos(πt/2L) / |sin(π(x−t)/2L)| ] dt.

The HT is gone; ε″ is an explicit **superposition of "step atoms"**. The atom
sitting at t₀ contributes loss at every point closer (in circle distance) to
t₀ than the domain endpoint is, and **gain** at every point farther — the gain
region is an arc centred on the antipode of the step.

*Verified:* reproduces `ht_derivative` on the logistic to 1.2e−4 relative rms
(pure discretisation); the identity ε″_mirror(x) = −ε″(−x) holds to 1e−14.

Equivalent geometric statement (this **is** Carathéodory–Toeplitz, re-read):
ε″ ≥ 0 everywhere ⟺ the logarithmic potential of the signed derivative
measure dν = (dε′/dt)dt on the unit circle attains its **minimum at the
domain endpoint**. In half-tangent coordinates ξ = tan(πx/2L)
with the drop distribution P (normalised, monotone case):

    passive  ⟺  E_P[ ln|η − ξ| ] ≤ ln|η − i|   for all real η,

i.e. **the geometric-mean distance from every real observation point η to the
drop distribution must not exceed its distance to the imaginary unit i.**
Passivity is a statement about where the drops sit, nothing else.

Two corollaries that the old framing never produced:

* **Off-centre step (closed form).** A single sharp drop at x₀ has
  min ε″ = (D/π)·ln cos(πx₀/2L) < 0, located at the antipode x₀ ∓ L.
  Verified to 4 decimals in depth *and* position across x₀ = 0→1.8
  (part A, `A_offcenter_step.png`). A monotone step is passive **only if
  exactly centred** — passivity is not translation invariant, which is the
  *correct* content of "no local condition can exist" (see §5, claim 3).
* **Mirror identity.** ε″_mirror(x) = −ε″(−x): the mirror of any nontrivially
  passive profile is a **pure gain** medium. The handoff's "mirror pair with
  gain −1.89" is just −max(ε″) of the original. Nothing subtle was going on.

## 2. THEOREM (sufficient condition — breaks claim 1)

**If (i) ε′ is non-increasing on [−L, L], (ii) all of its drop occurs in the
middle half |x| ≤ L/2, and (iii) the balance integral vanishes,**

    M₁ = ∫ g(t) · tan(πt/2L) dt = 0,

**then the DiHiTI partner satisfies ε″ ≥ 0.** No Hilbert transform, no
eigenvalue: one sign condition, one support condition, one explicit integral.
Symmetric profiles satisfy (iii) automatically, so *"symmetric drop confined
to the middle half" ⟹ passive* is a corollary (this covers the paper's
logistic: k = 4 on ±2.5 μm has drop width 0.25 μm ≪ L/2 = 1.25 μm).

*Proof.* In ξ = tan(πx/2L) coordinates the support condition is supp P ⊂
[−1, 1] and the criterion is E_P[ln|η−ξ|] ≤ ln|η−i| = ½ln(1+η²) ∀η. Fix η.
The function ψ(ξ) = ln|η−ξ| is concave on each side of ξ = η and → −∞ there,
so its concave envelope over [−1,1] at ξ = 0 is max{ψ(0), ½ln|η²−1|}
(endpoint chord). Both members are ≤ ½ln(1+η²): |η| ≤ √(η²+1) and
|η²−1| ≤ η²+1. Hence the anchor point (0, ½ln(1+η²)) lies on or above the
concave envelope, so there is a supporting line ℓ(ξ) = ½ln(1+η²) + λξ with
ψ(ξ) ≤ ℓ(ξ) on [−1,1]. Averaging against P and using E_P[ξ] = 0 (⟺ M₁ = 0):
E_P[ψ] ≤ ½ln(1+η²) + λ·0. ∎

*Sharpness of the L/2 constant:* a symmetric pair of half-steps at ±x₀ is
passive iff x₀ ≤ L/2 exactly; beyond it the gain appears **at the centre**
with closed-form depth (D/2π)ln[(1+cos θ₀)/(1−cos θ₀)]. Verified: threshold
crossing at x₀/L = 0.500 and depth match to 3 decimals (part B,
`B_pair_threshold.png`). For the symmetric case there is a one-line proof of
the corollary: with c = cos(πt/L), γ = cos(πx/L), passivity reads
∫ln[(1+c)/|γ−c|]dμ(c) ≥ 0, and support in the middle half means c ≥ 0, where
the *integrand itself* is pointwise ≥ 0 for every γ ∈ [−1,1].

*Monte-Carlo:* 400/400 random monotone profiles satisfying the hypotheses
(including deliberately **asymmetric** balanced ones) are passive, worst
min ε″ = +0.000000. Hypothesis violations are far from vacuous: dropping
balance fails 374/400, letting support reach 0.92L fails 397/400 (part C).

*Hypotheses are load-bearing:* symmetric+balanced+middle-half but
**non-monotone** drop densities fail once the interior dip is deep enough
(−0.046 mild, −7.2 deep; part F). Monotonicity cannot be dropped, though the
log kernel gives finite tolerance to mild dips. The support condition is
sufficient-not-necessary: central ballast can rescue mass outside L/2 (the
k = 1 logistic passes with ~8 % of its drop outside).

## 3. Necessary conditions (breaks claim 4, dents claim 2)

Expanding the master identity at the endpoint x = ±L gives a cascade of
**explicit integrals of ε′ alone**, each necessary for passivity:

| order | condition | real-space meaning |
|---|---|---|
| s·ln s | g(+L) = g(−L) | **edge slopes of ε′ must match** — re-derives the equal-slopes constraint of the May-2026 2nd-deriv work as the leading endpoint term |
| s | **M₁ = ∫ g·tan(πt/2L) dt = 0** | drop distribution balanced about the centre in the tan measure |
| s² | M₂ = ∫ g·sec²(πt/2L) dt ≥ 0 | automatic for monotone profiles |
| mean | **σ = ∫ g·ln(2cos(πt/2L)) dt ≥ 0** | σ ≡ π·mean(ε″): mean loss, computed with no HT |

**σ is the orientation-sensitive scalar the previous session said could not
exist.** Under mirroring g(t) → −g(−t), so σ → −σ, while the old
parity/area/slope scores are all blind. Logistic: σ = +1.228 (passive);
mirrored logistic: σ = −1.228, min ε″ = −1.171 (pure gain). Tie broken by one
integral. Note the kernel ln(2cos(πt/2L)) changes sign at |t| = 2L/3: drops in
the outer third of the domain *reduce* mean loss — "area done right".

**Screen performance** (400 generator samples, ground truth = DiHiTI label):
σ ≥ 0 and |M₁| = 0 together catch **180/214 gain cases with 0/186 false
alarms** (old screen: 13/21). Median relative |M₁| is 1e−14 among passive
samples and ≈ 1 among gain samples — the balance condition is essentially a
parity bit for passivity. Adding 7 fixed-point kernel probes of ε″ (each one
O(N) integral of ε′, still no HT) raises the catch to 207/214, at the price of
14 false alarms that are pure trapezoid error at the log singularity (~2 %
accuracy floor; a smarter local quadrature would remove them).

**Near-threshold prediction** (part E): for the logistic+bump crossover the
endpoint expansion predicts gain depth ≈ −M₁²/(2πM₂) with the minimum walking
in from the endpoint. Depth ∝ α² and location are confirmed; the prefactor is
right only to within ×2 because M₂ is log-divergent when g(±L) ≠ 0 (the
project logistic has e^{−kL} ≈ 3e−4 edge tails), making the numeric M₂
grid-sensitive. Quoted honestly as a scaling law, not an asymptotic constant.
It also *explains* the May-2026 observation that "the crossover occurs near
α ≈ 0": any α ≠ 0 breaks M₁ = 0, so the passive set has empty interior along
that perturbation direction.

## 4. What genuinely survives of the old claims

* **The exact criterion does need infinitely many numbers.** Part F: dressing
  the logistic with a·cos(mπx/L) localised near (not at) the edge, then
  re-balancing M₁ exactly, produces min ε″ ≈ −0.20 **independent of m**, while
  ΔM₁, Δσ ≈ 1e−13 and the change in every low-order Legendre moment of ε′
  falls to 1.7e−4 by m = 64 (Riemann–Lebesgue). So for any *fixed finite*
  battery of smooth functionals of ε′ there exist profiles that pass the
  battery and still carry O(0.2) gain. A finite smooth-moment test can be
  necessary (ours is) but never sufficient on the full profile space. That —
  and only that — is the true content of "you need the whole sequence".
* **The equivalence class of the exact test.** The potential inequality at
  observation point η is exactly ε″(x(η)) ≥ 0 at x(η) = (2L/π)arctan η; the
  exact criterion is still "check every point". What changed is that each
  check is now a closed-form integral of ε′ with geometric meaning, which is
  what made the theorem and the moment cascade provable.

## 5. Verdicts on the four claims

1. **"No cheap sufficient condition" — BROKEN.** §2: monotone + middle-half
   support + one balance integral ⟹ passive. Proven, sharp constant,
   400/400, covers the AR-coating family the paper actually uses.
2. **"Needs the entire coefficient sequence" — HALF-BROKEN.** A finite battery
   (slope match, M₁, σ, plus optional probes) is necessary and empirically
   catches 84–97 % of gain cases with zero/near-zero false alarms; the
   sufficient test of §2 reads one moment plus support. But exactness by
   finitely many smooth moments is impossible (§4) — this is the one claim
   whose technical core stands, now with a proof-grade demonstration instead
   of an assertion.
3. **"No local condition can exist" — BROKEN as stated.** The correct
   statement is: no *translation-invariant* local test exists (an off-centre
   step is locally identical to a centred one yet fails, with closed-form gain
   — §1). Once position relative to the domain is admitted as data, the §2
   test is local-plus-one-scalar and decides the monotone family.
4. **"No single scalar test" — BROKEN.** σ = ∫g·ln(2cos(πt/2L))dt is a single
   orientation-sensitive scalar, necessary for passivity, and it alone breaks
   the mirror tie that was offered as evidence of impossibility.

Furthest-broken to least: 1 ≈ 4 (clean kills), 3 (kill with a correction), 2
(survives in refined form).

## 6. Physics take-aways for the letter paper

* Passivity of an sKK coating is **geometry of the index drop**: keep the
  transition inside the middle half of the truncation window and balanced
  (symmetric suffices) and DiHiTI can never produce gain. An off-centre
  transition produces gain of depth (D/π)ln sec(πx₀/2L), which grows without
  bound as the transition approaches the window edge — for the paper's
  D = 1.89 a step at x₀ = L/2 already carries gain ≈ 0.10.
* The failure lives at the **truncation boundary**, exactly where the paper's
  residual-reflection story (truncation leak) already lives: both
  non-idealities of DiHiTI are edge effects of the same periodisation.
* Every DiHiTI profile sits on the passive-cone boundary (margin 0 at the
  endpoint); the necessary cascade (slope match → M₁ → M₂) is precisely the
  sequence of tangency conditions of that boundary contact.

## 7. Files

| file | contents |
|---|---|
| `potential_criterion.py` | parts A–F, all numbers quoted above |
| `../figures/2026Aug13_potential_criterion/A_offcenter_step.png` | closed-form vs numeric off-centre gain |
| `../figures/2026Aug13_potential_criterion/B_pair_threshold.png` | sharp L/2 threshold |
| `../figures/2026Aug13_potential_criterion/D_screen_scatter.png` | (M₁, σ) plane, passive vs gain |

Conventions note: the May-2026 `passivity_constraint_notes.tex` §"mode-by-mode
safety" states the safe-sign rule as bₙ(−1)ⁿ ≤ 0; its own derivation two lines
earlier gives bₙ(−1)ⁿ ≥ 0. Sign typo there, harmless downstream, recorded here
so nobody chases it.
