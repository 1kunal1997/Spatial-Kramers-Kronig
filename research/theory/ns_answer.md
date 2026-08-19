# The two-sided answer: exactly where necessary-and-sufficient conditions exist, and a proof that nothing smaller can

**Solo Fable session, 2026-08-19.** Scripts and raw outputs: `research/results/fable_ns_answer/`
(`common.py`, `e0_controls.py`, `e1_density.py`, `e2_battery.py`, `e3_n1_closure.py`,
`e4_lemmaE.py`, `*_output.txt`, `witness_k*.npz`, `CHECKPOINT.md`). Everything below is labelled
**proved / verified numerically / conjectured**. Setting and notation are those of
`h9_potential_domination.md` (L = 1, Cayley coordinate ξ(t) = tan(πt/2), drop measure g ≥ 0 unless
stated, gap function Δ_g(η) = ∫ln|η−ξ| dg − W·½ln(1+η²), passivity ⟺ Δ_g ≤ 0 on ℝ, and
π ε″(x) = −Δ_g(ξ(x))).

---

## 0. The answer, in four sentences

1. **The necessary-and-sufficient condition on ε′ is one law** — (L1), the weighted mass-balance
   continuum: Δ_g(η) ≤ 0 for every η ∈ ℝ, equivalently ∫ g(t) ln[cos(πt/2L)/|sin(π(x−t)/2L)|] dt ≥ 0
   for every x.
2. **That law collapses to a finite, exact, transform-free test precisely when ε′ is finitely
   described**: for an n-layer staircase, passivity ⟺ the ≤ n+1 real roots of one explicit
   polynomial all give Δ ≤ 0 (Theorem A); for a family cut out by finitely many linear
   constraints, passivity of the whole family ⟺ closed-form checks on few-atom extreme points
   (Theorem B — this is (L2), with a+b ≤ L as the flagship instance).
3. **For profiles as functions (not finitely described), no smaller criterion exists** — and this
   session makes that a set of theorems, not a lament: every open set of test points is essential
   (Theorem 1); **no finite battery of linear measurements of ε′ — moments, samples, corner
   masses, M₁, M₂, σ, anything — feeding any decision rule whatsoever can decide passivity**
   (Theorem 2, the main new result); no rule reading ε′ on a proper sub-window can decide it
   (Corollary 2.3). The structural reason is that the passive cone is a linear image of the
   positive cone with a **unique** extremal representation, and that representation *is* ε″
   (Proposition 3): every complete certificate necessarily re-represents the transform.
4. **Board consequences**: N1 (corner mass) is closed — the corner-mass threshold λ* is exact
   *per family* but ranges over an order of magnitude as the bulk moves (0.030–0.383 for one and
   the same corner geometry), and Theorem 2 forbids any thresholded corner functional from being
   two-sided. N2/T2 (the CFM/Choquet conjecture) is **refuted** by an explicit counterexample
   (§7). Red-team's residual risk #3 (outer-branch negativity) is discharged by Lemma B.

The mission's V3/V6, answered honestly: the definition the Captain asked for **exists and is
(L1) together with its two finite collapses**; the theorems below prove that no shorter
two-sided statement is available, so the search for a "thirteenth condition" that is both finite
and two-sided on general profiles can stop — not out of fatigue, but because it is now a theorem
that it does not exist.

---

## 1. Two lemmas that carry the negative results

### Lemma A (independence of pair gaps). PROVED.

Let P₁,…,P_n be balanced pairs whose 2n atoms are pairwise distinct. If Σ c_j Δ_{P_j}(η) = 0 for
all η in some open interval I ⊂ ℝ, then c = 0.

*Proof.* Each Δ_{P_j} extends to the conjugation-symmetric harmonic function
G_j(z) = Σ_l w_{jl} ln|z−ξ_{jl}| − ½ln|z−i| − ½ln|z+i| on ℂ minus finitely many points. Set
G = Σc_j G_j. G vanishes on I; by conjugation symmetry ∂G/∂y = 0 on I as well, so G has vanishing
Cauchy data on a segment; harmonic functions are real-analytic, and ℂ minus finitely many points is
connected, so G ≡ 0. Near an atom ξ of P_{j₀} shared by no other pair, G(z) = c_{j₀} w ln|z−ξ| + O(1)
with w > 0, which forces c_{j₀} = 0. Repeat for each pair. ∎

Consequence used later: for any interval I, the vectors v(η) := (Δ_{P_1}(η),…,Δ_{P_n}(η)), η ∈ I,
span all of ℝⁿ. *(Numerically corroborated: the sampled matrix [Δ_j(η_l)] has smallest singular
value 8.3 and 6.4 for the two E2 pair systems — nowhere near rank-deficient.)*

### Lemma B (no exterior gain; outer-branch strictness). PROVED.

For a unit-mass balanced pair with atoms ξ = β, −γ (β, γ > 0):

    Δ′(η) = [η(1+βγ) + (γ−β)] / [(η−β)(η+γ)(1+η²)].

On (β, ∞) the numerator exceeds γ(1+β²) > 0 and the denominator is positive, so Δ is strictly
increasing to its limit 0: **Δ < 0 strictly on (β, ∞)**. Symmetrically Δ < 0 strictly on (−∞, −γ).
Since every positive atomic measure with M₁ = 0 is a finite positive combination of balanced pairs
supported on its atoms (mean-zero two-point decomposition in ξ), we get:

> **A balanced positive drop measure satisfies Δ < 0 strictly outside the convex hull of its
> Cayley support.** In x-space: for g ≥ 0 with M₁ = 0, ε″ > 0 strictly between the outermost
> drop and the seam.

Two side effects: (i) this **discharges red-team residual risk #3** in
`cycle3_c25_hardening/redteam.md` — the outer-branch negativity of Cor 4.A is now a monotonicity
proof, not an asymptotic argument; (ii) all violation of passivity by such measures happens inside
one fixed compact η-interval, which is what makes the perturbation arguments below clean.

---

## 2. The positive side: where finite two-sided criteria exist

### Theorem A (per-profile N&S for every finitely-described ε′). PROVED (h9 §9) + re-verified.

Let ε′ be an n-layer staircase (equivalently, g = Σ_{j=1}^n w_j δ_{t_j}, w_j > 0 — every profile
that can be fabricated). Then

> **ε″ ≥ 0 ⟺ Δ(η_c) ≤ 0 at every real root η_c of the explicit polynomial
> Q(η) = (1+η²)·Σ_j w_j Π_{k≠j}(η−ξ_k) − W η Π_k(η−ξ_k), of degree ≤ n+1.**

This is a finite, exact, grid-free, transform-free, two-sided criterion **on the layer data
itself**. It is (L3) of h9, here promoted to the role it deserves: *for the physically realizable
class, the mission's request for "N&S conditions on ε′" is fully met by this test.* Cost O(n³) via
`np.roots` — not faster than the FFT at the paper's N (K-ledger: 0.2× at n = 40), but exact where
the FFT is approximate, and independent of any grid.

Session controls (`e0_controls.py`): threshold of Cor 4.A recovered as a+b = 1.000000000 at three
asymmetries; closed-form gain depths match production `ht_derivative` to 3–4 digits with matching
argmin (−3.520e−2 both; −1.940e−2 both, the red-team λ = 0.5 mixture); C40 depth-law slopes match
analytics to 4 digits.

### Theorem B (family-level N&S). PROVED in h9/(L2); red-team-hardened as C25′.

For families cut by finitely many linear constraints, (L2) reduces passivity of the family to
closed-form checks on ≤ (#constraints)-atom extreme points; Cor 4.A / C25′ (a+b ≤ L, with the
quantifier on the box) is the flagship. Nothing new added here; recorded because the dichotomy
below needs it: **finite description ⟹ finite two-sided criterion, at both the profile and the
family level.**

---

## 3. Theorem 1 — every open set of test points is essential (density). PROVED + VERIFIED.

> **Theorem 1.** Call E ⊆ ℝ *sound and complete* if, for every finite positive atomic g with
> M₁ = 0: [Δ_g ≤ 0 on E ⟺ Δ_g ≤ 0 on ℝ]. Then E is dense in ℝ. Equivalently, in x-space: no
> open sub-interval of the window may be left unchecked — for every open U there is an
> admissible profile whose only gain lies inside ξ⁻¹(U).

*Proof.* Fix η₀ and r > 0; let U = (η₀−r, η₀+r). Let p₀ be the boundary pair with atoms at
−a, b, a = ½ − arctan(η₀)/π, b = ½ + arctan(η₀)/π (so a + b = 1 and the active point is exactly
η₀). By Cor 4.A and Lemma B, Δ_{p₀} < 0 on ℝ∖{η₀} and Δ_{p₀}(η₀) = 0, with Δ_{p₀} strictly
unimodal between its atoms (Δ′ has a single simple zero there). Inflate: p_s = pair with the sum
a + b = 1 + s (inflating whichever side keeps both atoms inside the window). Then
sup Δ_{p_s} > 0 (Cor 4.A necessity; the depth is linear in s by C40). By Lemma B, Δ_{p_s} < 0
outside the middle interval for *every* s. Inside the middle interval Δ_{p_s} is unimodal, so its
positivity set is a single interval (l_s, r_s) around the maximizer, and for any fixed η ≠ η₀,
Δ_{p_s}(η) → Δ_{p₀}(η) < 0 as s → 0; unimodality then forces (l_s, r_s) → {η₀}. Choose s small
enough that (l_s, r_s) ⊂ U: now p_s violates passivity only inside U, and any E with E ∩ U = ∅
declares it passive. Hence a sound-and-complete E meets every open interval. ∎

**Numerics** (`e1_density.py`; control: at s = 0 the sup is 0.000 at η₀ to machine precision, all
three targets η₀ = 1.7, −0.4, 8.0): the positivity interval shrinks onto η₀ like √s — e.g. for
η₀ = 1.7: width 5.36 → 1.46 → 0.74 → 0.39 → 0.22 as s = .1 → .001, with max Δ outside the
interval ≤ −6e−13 (the tail approaching 0 from below, per Lemma B).

Theorem 1 strictly upgrades C27/K17 ("no *finite* node set") to: **no non-dense test set**, and
it kills every "check a sparse/structured subset of positions" proposal in one stroke.

---

## 4. Theorem 2 — no finite battery of linear measurements decides passivity. PROVED (see status note) + VERIFIED.

> **Theorem 2.** Let φ₁,…,φ_k : (−1,1) → ℝ be arbitrary functions, defining linear measurements
> ℓ_i(g) = ∫φ_i dg, and let F : ℝᵏ → {PASSIVE, GAIN} be an arbitrary decision rule. Then F(ℓ(·))
> does not decide passivity: there exist g_P (strictly passive) and g_G (strict gain), both
> finite positive atomic measures with M₁ = 0 **and the same atom set**, with ℓ(g_P) = ℓ(g_G).

Because the φ_i are arbitrary, this covers *simultaneously*: all moments, all Fourier
coefficients, point samples of ε′ (indicator kernels), corner masses, the entire necessary
cascade (mass, M₁, M₂, σ), and any finite combination thereof, post-processed by any function —
including discontinuous, threshold, or learned rules. Since the witnesses share their atom set,
even rules additionally given the support/atom positions are defeated; only the *weights* differ.

*Proof.* Take n = k+2 balanced pairs including the explicit triple
bad₁ = (−0.25, +0.90), bad₂ = (−0.40, +0.80), good = (−0.15, +0.45) of Lemma E below, the others
good pairs with distinct atoms. Let M ∈ ℝ^{k×n} be the battery matrix M_{ij} = ℓ_i(P_j) and
K = ker M, dim K ≥ 2. Suppose F∘ℓ were correct on the mixture cone C = {Σ m_j P_j : m ≥ 0}.
Write Δ_m = Σ m_j Δ_{P_j} and 𝒫 = {m : Δ_m ≤ 0 on ℝ} — a closed convex cone (each constraint is
linear in m).

**(C) Interior boundary points have a finite active point.** If p ∈ ∂𝒫 has all coordinates
positive and sup over the finitely many critical values of Δ_p were < 0, then for all m near p:
Δ_m < 0 outside a fixed compact interval by Lemma B (all weights stay positive), and < 0 inside
it by uniform continuity (weights bounded below keep the atom singularities negative). Then
p ∈ int 𝒫 — contradiction. So some finite η* has Δ_p(η*) = 0.

**(D) Correctness forces K-annihilation.** Correctness of F∘ℓ means 𝒫 ∩ C is a union of fibers
of ℓ; for p ∈ ∂𝒫 in the open orthant, p ± tκ ∈ 𝒫 for all κ ∈ K and small t. For any supporting
functional u of the convex cone 𝒫 at p (⟨u,m⟩ ≤ 0 on 𝒫, ⟨u,p⟩ = 0) this gives ⟨u,κ⟩ = 0. The
vector v(η*) = (Δ_{P_j}(η*))_j is such a supporting functional. Hence
**v(η*) ∈ K^⊥ = Row(M), which has dimension ≤ k = n−2, for every active point η* of every
boundary point in the open orthant.**

**(E) The active points sweep an interval.** (Lemma E, constructive.) For the explicit triple
above and each θ, solve the 2×3 system Δ_m(θ) = 0, ∂_ηΔ_m(θ) = 0 for the mixture weights m(θ)
(1-dim nullspace). Verified (`e4_lemmaE.py`): for every θ on the contiguous window
**[1.45, 2.96]** (152 consecutive grid values, no gaps) the solution is componentwise positive
and the exact node test confirms sup Δ_{m(θ)} = 0 attained at θ (|sup| < 1e−10, |argmax−θ| < 1e−6
— two independent consistency checks). Thickening to the open orthant in ℝⁿ: add mass δ of the
remaining n−3 pairs and rescale the bad block to re-touch the boundary; the touching maximum is
nondegenerate (Δ″ ∈ [−0.506, −0.069] across the whole window, all strictly negative — measured),
so the perturbed active point θ′(θ, δ) is continuous and → θ uniformly as δ → 0;
the image of [1.45, 2.96] under θ′(·, δ) therefore still contains an interval for δ small.

**(Contradiction.)** By (D), v(η) ∈ Row(M) for all η in that interval; by Lemma A the vectors
v(η) on any interval span ℝⁿ, while dim Row(M) ≤ n−2. Contradiction. Hence no F is correct on C,
which is equivalent to the existence of a matched pair with opposite labels (otherwise define
F correctly on the image of ℓ). ∎

**Status note.** Steps (A)–(D) are proved in full. Step (E) is constructive: the envelope window
is verified numerically (not by asymptotics — by the exact node test at each θ), and the
thickening step is a continuity argument stated with its hypotheses (nondegenerate touching
maximum, verified on the window). I label Theorem 2 **proved, with one numerically-certified
constructive ingredient**; an all-analytic version of Lemma E is routine grind (the envelope
weights are explicit rational expressions in Δ_j(θ), Δ′_j(θ)) and is the one item a referee could
demand.

**Explicit witnesses** (`e2_battery.py`, `witness_k8.npz`, `witness_k16.npz`):

| | battery k = 8: mass, M₁, M₂, σ, t…t⁴ | battery k = 16: + t⁵…t⁸, cos/sin(πt), cos/sin(2πt) |
|---|---|---|
| reading match | 6.1e−16 (machine) | 8.3e−12 (float-SVD-limited; see note) |
| M₁ of both witnesses | 0 (machine) | 0 (machine) |
| exact verdicts | GAIN −7.76e−3 vs PASSIVE (margin 2.4e−2) | GAIN −1.60e−5 vs PASSIVE |
| production check | GAIN −7.66e−3 @ same x; PASS | GAIN −1.30e−5 @ same x (N = 2²²); PASS |
| control | non-kernel direction moves readings by 1e−1 | 1.5e−2 |

The k = 8 battery **contains the entire known necessary cascade**. Its two witnesses are
10-pair mixtures on 20 shared atoms; every reading any member of the cascade zoo could take is
identical between them, and one has 0.8% gain.

Note on the k = 16 match: the 16 smooth functionals are numerically near-dependent on the 36
atoms (singular values fall to 3e−16), so the float SVD kernel is only good to ~1e−11 in the
readings; exact-arithmetic matching is what the theorem proves. Related observation worth
keeping: **a rich smooth battery is nearly rank-deficient on clustered atoms** — its effective
information is far less than k readings — and richer smooth batteries force the witness gain
smaller (here −7.8e−3 → −1.6e−5 from k = 8 → 16) but never to zero, since the log kernel is in
no finite smooth span (Lemma A). This is C14's Riemann–Lebesgue intuition, now with exact
matching and arbitrary functionals.

### Corollaries

**2.1 (No moment criterion).** No finite list of moment-type inequalities (each reading finitely
many linear functionals of ε′) is necessary and sufficient — however nonlinear the thresholds.
In particular the necessary cascade cannot be completed to an N&S criterion by finitely many
further scalar conditions. *(This is the formal death certificate for the "clean conditions"
program on general profiles, and the reason every prior attempt produced one-sided statements.)*

**2.2 (N1 closed).** Corner mass is a linear functional; no thresholded corner-mass rule can be
two-sided. Quantitatively (`e3_n1_closure.py`, controls 4/4 against red-team/handA: 0.2620,
0.2778, 0.0872, 0.0009): for the fixed corner geometry (−0.4, +0.9), the exact threshold λ*
ranges from **0.383** (bulk pair ±0.45) down to **0.030** (bulk pair (−0.2, +0.79), i.e. a bulk
near its own threshold) — a 13× swing controlled entirely by where the *bulk* sits. Killer pair:
identical corner distribution (fraction 0.25 on the corner pair), bulk (±0.2) → PASSIVE
(margin 5.3e−3); bulk (−0.24, +0.74) → GAIN (−2.0e−2, production-confirmed at the predicted x).
N1's residue that *is* true and useful: λ*(corner, bulk) is exact and cheap per family — a
2-parameter instance of (L2), not a route to necessity. *(Also: red-team's λ*(a+b) table mixed
corner splits without recording them — the 1.30L row is the asymmetric (−0.4,+0.9), the 1.80L and
1.98L rows are the symmetric splits; the 1.40L/1.94L rows match neither convention and should be
re-derived before anyone reuses them.)*

**2.3 (No localized criterion).** No rule that reads ε′ only on a proper closed sub-region of
the window can decide passivity: Theorem 1 puts the only violation inside any chosen open set,
and the 2.2 killer pair holds the corner region fixed while the bulk flips the verdict.

**Relation to the ledger.** Theorem 2 is the *measurement-class* completion of the family:
C14 (smooth moments, asymptotic, no exact matching) ⊂ Theorem 2 (arbitrary functionals, exact
matching); C27/K17 (finitely many η-tests) ⊂ Theorem 1 (any non-dense η-set); C30/V2a
(coordinate samples, query-counting) is orthogonal — it counts *reads*, Theorem 2 counts
*summaries*. C46's objection (a Lipschitz promise collapses the query bound) does **not** touch
Theorem 2: the witnesses for a given battery are fixed smooth-able profiles — they need not
narrow as the class smooths; only as k → ∞ does the witness gain shrink, which is the correct
and unavoidable limit (a complete battery *is* the transform).

---

## 5. Proposition 3 — why every complete criterion is a re-representation. PROVED (elementary), interpretive.

The map A: g ↦ ε″ is linear and (modulo the 2-dim gauge: constants and the ramp, C21) a
bijection onto {v : v(±L) = 0} (Wave 3 Thm 6.1 — inherited at `conjecture`, numerically
supported by h9 block G). The passive set is 𝒫 = A⁻¹({v ≥ 0}). The nonnegative cone is a
**lattice (simplex) cone**: its extreme rays are the point masses and every element has a
*unique* extremal representation. Linear bijections transport all of this: 𝒫 is a simplex cone,
its extreme rays are the "unit gain-atom" profiles A⁻¹δ_y, and the unique representing measure
of any passive g **is ε″[g] itself**.

Consequence: any certificate of passivity that exhibits g as a positive superposition of
elementary passive profiles — which is what a Choquet/extreme-ray certificate is — necessarily
computes ε″, because the representation is unique. Cones with short membership certificates
(polyhedral: Farkas; spectrahedral: SOS) have *non-simplex* facial structure; this cone is
maximally rigid. Proposition 3 does not by itself forbid exotic certificate formats — Theorems 1
and 2 do the quantified forbidding — but it explains the pattern the mission kept hitting:
Carathéodory–Toeplitz, the potential form, Fejér–Riesz, and the supporting-line form were all
"the transform in other coordinates" because **for this cone there is nothing else to be**.

---

## 6. The α-crossover seed (V4), noted

C18 (α_c = 0.66623) already answers V4's number. The present frame adds the mechanism: the
opposite-bumps family is a line in profile space; its Δ is affine in α; the crossover is the
line's exit through one facet {Δ(η*) = 0} of the simplex cone, which is why the transition is
clean and linear-in-α near threshold (the C40 slope law). The unresolved 0.2667-vs-0.66623
discrepancy flagged in STATE.md is a definition-of-α question between two documents, not
touched here.

---

## 7. The classical-order question (board N2/T2): REFUTED as stated; corrected form. PROVED both parts.

**Refutation of T2.** The conjecture said: against Cauchy ω, harmonic test functions with
convergent integral are "essentially affine", so the log-potential order equals the full
subharmonic (Choquet/CFM) order. Counterexample: f(x,y) = cos(x)e^{−y} is harmonic on all of ℂ
(hence both f and −f are admissible subharmonic test functions), bounded on ℂ₊, and
ω-integrable: ∫f dω = e^{−1} (the Cauchy characteristic function). Take P = δ₀ — passive
(Δ(η) = ln|η| − ½ln(1+η²) < 0). Then ∫f dP = 1 > e^{−1} = ∫f dω. So the subharmonic-cone order
**fails** for a passive profile: the log-potential order is *strictly weaker* than the CFM
order, the gap is not vacuous, and CFM machinery does not apply verbatim. N2 closes negatively.

**What the order actually is.** With F = {c − Φ_μ : μ ≥ 0 finite, conjugation-symmetric}:
passivity ⟺ ∫f dP ≥ (W)·∫f dω for all f ∈ F (one direction is Fubini against (L1); the other
recovers (L1) from f = −ln|·−η|). I.e. the correct test cone is exactly the log-potential cone —
no larger. The oscillatory bounded harmonics (cos x e^{−y} and relatives) are precisely what must
be excluded, and any growth condition that removes them (sub-logarithmic growth *on ℂ*) collapses
the harmonic part to constants; that is the repaired, and now nearly tautological, form of T2.

**Labelled remark (conjectured, for the librarian).** −ln|x−y| is, up to normalization, the
recurrent potential kernel of the symmetric Cauchy process on ℝ (the boundary trace of planar
Brownian motion, whose harmonic measure at i is exactly ω). Orders "U^P ≥ U^ω, equal masses" for
such kernels are the domain of the filling scheme / stopping-distribution theory (Rost;
Baxter–Chacón). If that identification survives a literature check, the physical reading of
passivity is: *the drop measure can be diffused by the Cauchy process into the window's
equilibrium measure before it ever charges the seam*. Pretty; not load-bearing; unverified.

---

## 8. Consequences for the numerical convention (feeds P0)

Found while cross-checking Theorem 2's k = 16 witness (small gain, −1.6e−5): production
`ht_derivative` on smoothed staircases carries a **profile-dependent, additive, common-mode
offset that is O(h)**: at fixed smoothing s = 6e−4 the offset at the witness's argmin is
+3.55e−5 (N = 2¹⁹) → +1.76e−5 (2²⁰) → +0.85e−5 (2²¹) → +0.4e−5 (2²²), while being s-independent
below s ≈ 6e−4 — and it **cancels exactly in differences**: prod(g_P) − prod(g_G) = 3.382e−5 =
2 × 1.691e−5 (closed form) at every N to 4 digits. Two consequences: (i) production PASS/FAIL
on staircase-like profiles is biased by ~O(h) — at the corpus's historical N = 2048 that scale
is ~1e−3, the same order as the celebrated interior margin (C21), so near-threshold corpus
labels should be treated as convention-limited until P0 lands; (ii) differential comparisons
are trustworthy at much lower N than absolute ones. Recorded for the P0 owner.

---

## 9. Status table

| # | Claim | Status |
|---|---|---|
| Lemma A | independence of pair gaps on any interval | **proved** (+ sanity numerics) |
| Lemma B | no exterior gain; outer branches strictly negative; closes red-team risk #3 | **proved** |
| Thm A | staircase node test is per-profile N&S for finitely-described ε′ | **proved** in h9; controls re-run clean (e0) |
| Thm 1 | any sound-and-complete test set is dense | **proved** + verified (e1; δ=0 control exact) |
| Thm 2 | no finite linear battery + any rule decides passivity | **proved modulo numerically-certified Lemma E window**; witnesses explicit (e2, e4) |
| Cor 2.1–2.3 | no moment / corner-mass / localized criterion | follow from Thm 1–2; killer pairs verified (e3) |
| Prop 3 | passive cone is a simplex cone; unique certificate = ε″ | **proved** modulo inherited Thm 6.1 (conjecture rung) |
| §7 | T2/CFM conjecture refuted; corrected order identified | **proved** (counterexample is one line) |
| §7 remark | Cauchy-process / Rost reading | **conjectured**, needs literature check |
| §8 | O(h) production offset, cancels in differences | **verified numerically**, mechanism unidentified |

## 10. What I did NOT do

- No cheaper-than-FFT test (V1) was attempted; nothing here contradicts the V1/V2 ledger.
- Lemma E's all-analytic closure (routine but real work) is open; until then Theorem 2's label
  carries the constructive caveat.
- The 0.66623-vs-0.2667 α_c conflict remains unresolved (not this session's question).
- Prop 3 leans on inherited Thm 6.1 (surjectivity of the inverse map), still at `conjecture`.
