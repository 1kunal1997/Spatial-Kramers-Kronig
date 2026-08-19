# The complexity of deciding DiHiTI passivity: a tight Θ(N) query bound, and a conditional barrier at the log factor

*Red team, cycle 2, 2026-08-19. Written as a standalone document for lifting into a paper.*
*Mission objective V2 (as restated by the Captain, option A): V2a = the Ω(N) query lower bound;*
*V2b = a structural, **conditional** account of why the remaining log factor resists.*

> **Scope statement, binding on every claim below.** §3 proves an unconditional lower bound in an
> explicitly stated *query* model. §4 proves **nothing** about time. §4 is a statement about the
> reach of currently known lower-bound *methods*, and it is stated conditionally throughout. It must
> never be quoted, paraphrased, or captioned as an unconditional time lower bound. Where the honest
> statement is "no known method can do this", the document says exactly that.

---

## 1. Summary

Let ε′ be sampled at N points on the truncation window and let the DiHiTI map produce ε″. The
**passivity decision problem** is: given the N samples, is ε″ ≥ 0 on the interior?

**Result 1 (V2a, §3, unconditional).** In the deterministic adaptive query model, any correct
decider must read Ω(N) samples. The bound survives restriction to a **fixed, N-independent ball**
around a single base profile — radius ≈ 1.5·D in L^∞ and ≈ 3·D in total variation — so it is not an
artifact of admitting unbounded or arbitrarily rough inputs. A Yao argument extends it to
bounded-error randomized algorithms. The two-FFT baseline reads exactly N samples, so **the query
complexity of passivity is Θ(N) and this is tight.**

**Result 2 (the mechanism, §3.2, proved and verified).** The lower bound rests on an exact local law:
a bump of amplitude a and *any* width δ placed at x₀ depresses ε″ near x₀ by

> Δ_min = −a / (π C_shape),  independent of δ,

because the DiHiTI kernel's singular part is the Hilbert kernel, and the Hilbert transform commutes
with dilation. Hence the amplitude needed to flip the label at x₀ is

> **a_min(x₀) = κ · ε″₀(x₀),  κ = π C_shape⁻¹ ≈ 1.31, uniform in x₀ and in N.**

Measured κ = 0.414–0.436 (in units of π) across the whole window, 21 positions, and a_min identical
to four digits at N = 4096/8192/16384. **Narrow perturbations do not become expensive as they narrow
— they become free in L¹ while staying the same size in L^∞.** That is exactly the property a query
lower bound needs, and it is the property that fails for most transforms.

**Result 3 (V2b, §4, conditional).** The remaining log factor separates a Θ(N) query bound from an
O(N log N) time algorithm. It resists for one identifiable structural reason: **every known method
for proving a superlinear lower bound on a geometric decision problem is a multiplicity count** — of
connected components (Ben-Or) or of faces (Björner–Lovász–Yao) — **and the passive set is convex.**
A convex polyhedral cone with Θ(N) facets has one connected component and at most 2^{Θ(N)} faces, so
these methods are capped at Ω(N) here *by arithmetic*, not by our failure to apply them. Getting
Ω(N log N) from them would require 2^{Θ(N log N)} faces, i.e. permutation-like combinatorial
multiplicity (element distinctness gets its N log N from N! orderings). Convexity forbids it.
§4.5 lists the three specific things that would have to change.

**Honest headline.** *Passivity needs Ω(N) queries, and the transform is optimal up to the log
factor. Whether the log factor is necessary is an open problem in complexity theory, not in
photonics — and we can say exactly which structural feature of the passive set puts it out of reach
of every method now available.*

---

## 2. Setup and the query model — stated explicitly

**2.1 The instance.** Fix the window [−L, L], a grid of N points, and the DiHiTI matrix A ∈ ℝ^{N×N}
defined by the pipeline `np.gradient → scipy.signal.hilbert → cumulative_trapezoid → de-tilt`. A is
linear (verified to 6.2e−15, C21) and its rows 0 and N−1 are identically zero, so rank A = N−2 and
ε″(±L) = 0 is an operator identity, not a constraint (C13/C21, mission objective V5).

The **interior index set** is I = {i : |x_i| ≤ 0.95 L}, |I| = Θ(N). The decision problem is

> **PASSIVE(u)**: given u ∈ ℝ^N, decide whether (Au)_i ≥ 0 for all i ∈ I.

Equivalently, membership in the closed convex polyhedral cone

> **K = {u ∈ ℝ^N : (Au)_i ≥ 0, i ∈ I}**,

which has at most |I| = Θ(N) facets and contains a 2-dimensional lineality space (A annihilates
constants and linear ramps — this is DiHiTI's published shape-invariance, K3).

**2.2 The query model.** An algorithm is an adaptive decision tree. At each step it names an index
j ∈ {1,…,N} and receives u_j; the index may depend on all previously received values. After q steps
it outputs PASS or FAIL. Its **query cost** is the worst-case q over the admissible input class.
No restriction is placed on the computation between queries: the model charges only for reading the
input. A lower bound here is therefore *stronger* than a time lower bound would be in the sense that
it is unconditional, and *weaker* in the sense that it cannot see the log factor at all. That
asymmetry is the entire subject of §4.

**2.3 The input class.** Lower bounds against unrestricted ℝ^N are vacuous (an unread coordinate can
be set to ±∞). We therefore prove the bound against a class that is as restrictive as we can make it:

> **𝒰(u₀, r) = { u₀ + p : ‖p‖_∞ ≤ r,  TV(p) ≤ 2r,  supp p ⊆ [−0.95L, 0.95L] }**

for a fixed base profile u₀ and a fixed radius r independent of N. Every member is a bounded,
bounded-variation perturbation of one physically motivated profile. §3.4 reports r ≈ 1.5·D, where
D = u₀(−L) − u₀(+L) is the profile's own drop — so the ball is *comparable to the profile*, not
astronomically large, and its radius does not grow with N.

---

## 3. V2a — the Ω(N) query lower bound

### 3.1 Why the trivial argument is not enough, and what replaces it

The trivial argument ("an unread coordinate can be set adversarially") is correct and worthless: it
proves Ω(N) only against an unrestricted class, and any referee will answer that physical ε′ are
smooth and that smoothness is exactly what a sub-linear test would exploit. The bound has content
only if the hard family stays inside a small, fixed, smoothness-constrained ball. Establishing that
is the work of §3.2–§3.3, and it is where the earlier hand-B probe (M4) pointed: label-flipping
perturbations get **cheaper**, not more expensive, as they narrow.

### 3.2 Lemma (local flip law) — proved, and verified to 5%

> **Lemma 1.** Let u₀ be passive with interior profile ε″₀ = A u₀. Let φ be a fixed C^∞ bump of unit
> height supported in [−1, 1], and let p(x) = a·φ((x − x₀)/δ). Then, for δ small compared with the
> scale on which ε″₀ varies,
>
>   min_x [A(u₀ + p)](x)  =  ε″₀(x₀) − a/(π C_φ) + O(δ),
>
> where C_φ = 1/max_s (−H[φ](s)) depends only on the *shape* of φ and **not on δ, on x₀, or on N**.
> Consequently the flip amplitude is a_min(x₀) = κ ε″₀(x₀) + O(δ) with κ = π C_φ.

*Proof.* Wave 2's master identity (C1) writes DiHiTI as a logarithmic potential of the drop density
g = −du/dx:

  ε″(x) = (1/π) ∫ g(t) ln[ cos(πt/2L) / |sin(π(x−t)/2L)| ] dt  =: (1/π) ∫ g(t) K(x,t) dt.

Linearity of A gives A(u₀+p) = ε″₀ + Δ with Δ(x) = (1/π)∫ g_p(t)K(x,t)dt, g_p = −p′. Integrating by
parts (p is compactly supported inside the window, so no boundary term),

  Δ(x) = (1/π) ∫ p(t) ∂_t K(x,t) dt,
  ∂_t K(x,t) = (π/2L)[ cot(π(x−t)/2L) − tan(πt/2L) ].

The second term is bounded and smooth on supp p, contributing (1/π)·O(‖p‖₁) = O(a δ). In the first,
write cot(πs/2L) = 2L/(πs) + O(s). For x = x₀ + δσ with σ = O(1), the leading part gives

  Δ(x₀ + δσ) = (1/π) ∫ p(t)/(x₀ + δσ − t) dt + O(aδ)  =  −a·H[φ](σ) + O(aδ),

with H the Hilbert transform, **because the change of variables t = x₀ + δτ leaves the integral
invariant: the Hilbert kernel is homogeneous of degree −1 and H commutes with dilation.** Hence
min_σ Δ = −a·max_σ(−H[φ](σ)) = −a/(π C_φ), with no δ dependence. Adding ε″₀(x₀) + O(δ) — the base
profile evaluated on the support, which varies by O(δ) across it — gives the statement. ∎

The δ-independence is the crux and it is a *property of this transform*: it holds because the
singular part of the DiHiTI kernel is exactly the Hilbert kernel. A transform with a milder
singularity (say a bounded kernel) would give Δ_min = O(aδ), narrow bumps would be expensive, and
the lower bound below would fail.

**Numerical verification** (`research/results/cycle2_v2a/v2a_position.py`, N = 2^13, base = logistic
w = 0.05, L = 2.5, compact C^∞ bump on 32 grid points):

| x₀/L | ε″₀(x₀)/D | a_min/D | a_min/(π ε″₀) |
|---|---|---|---|
| ∓0.950 | 9.906e−4 | 1.356e−3 | 0.436 |
| ∓0.760 | 2.328e−2 | 3.051e−2 | 0.417 |
| ∓0.475 | 1.240e−1 | 1.643e−1 | 0.422 |
| ∓0.190 | 3.957e−1 | 5.194e−1 | 0.418 |
| 0.000 | 1.141e+0 | 1.500e+0 | 0.418 |

The ratio is constant to **±3%** while ε″₀ varies over **three orders of magnitude**. That is
Lemma 1, and it also fixes C_φ = 1/0.418 ≈ 2.39 for this bump shape.

**δ-independence, verified directly** (`v2a_hard_family.py`, Gaussian bumps of width σ = c·h with
h = 2L/N, so the *physical* width shrinks by 16× across the table):

| N | a_min/D (c=2) | a_min/m₀ | ‖p‖₁/(DL) | TV(p)/D |
|---|---|---|---|---|
| 2 048 | 8.974e−3 | 8.656 | 3.11e−5 | 1.795e−2 |
| 4 096 | 8.935e−3 | 8.870 | 1.55e−5 | 1.787e−2 |
| 8 192 | 8.962e−3 | 9.028 | 7.76e−6 | 1.792e−2 |
| 16 384 | 8.958e−3 | 9.045 | 3.88e−6 | 1.792e−2 |
| 32 768 | 8.952e−3 | 9.050 | 1.94e−6 | 1.790e−2 |

The amplitude and the total variation are **flat to 0.3% over a 16× range of N**, while the L¹ mass
falls like 1/N. (M4 reported a_min *rising* again below σ/L = 0.02 at a fixed N = 2^14. That is consistent with a
resolution artifact — σ/L = 0.02 is ≈ 82 grid points there, and below it the bump is under-resolved —
but this document does not rely on that reading, and it has not been verified directly. The
experiments above tie σ to h, so the bump is always resolved by the same number of points, and the
flatness is unambiguous.)

### 3.3 Theorem (V2a) and proof

> **Theorem 2 (deterministic).** Fix k ∈ ℕ and let M = ⌊0.95 L/(k h)⌋ = Θ(N) be the number of
> disjoint intervals I₁,…,I_M of width 2kh inside the interior window. There is a base profile u₀
> and a radius r = 1.5·D, independent of N, such that any deterministic adaptive algorithm that
> decides PASSIVE correctly on 𝒰(u₀, r) must make at least **M = Θ(N)** queries.

*Proof.* By Lemma 1 there is, for each j, a perturbation p_j supported in I_j with
‖p_j‖_∞ = a_min(x_j) ≤ κ·max ε″₀ =: r and TV(p_j) ≤ 2r, such that u₀ is PASSIVE and u₀ + p_j is not.
(The uniform bound r is finite and N-independent because ε″₀ is a fixed bounded function; measured
r/D = 1.680 at k = 4, identically at N = 4096, 8192 and 16384 — see §3.4.)

Run the algorithm on u₀. It reads some set Q of at most q coordinates and, being correct, outputs
PASS. If q < M then by pigeonhole some I_j contains no element of Q. Since p_j is supported in I_j,
the inputs u₀ and u₀ + p_j agree on every coordinate the algorithm read; its execution transcript is
identical and it outputs PASS on u₀ + p_j as well. But u₀ + p_j ∈ 𝒰(u₀, r) and is not passive.
Contradiction. Hence q ≥ M = Θ(N). ∎

> **Remark (strongest form).** The proof uses only that the class *contains* the finite family
> {u₀} ∪ {u₀ + p_j : j = 1…M}. The bound therefore holds against that finite family alone — a class
> of M + 1 = Θ(N) explicit inputs, all within distance 1.5·D of one another in L^∞ and 3·D in total
> variation, all agreeing outside a set of measure 2kh. Restricting the admissible inputs further
> can only strengthen the theorem, never weaken it. This forecloses the standard objection that a
> query lower bound is an artifact of a permissive input class.

> **Corollary 3 (randomized).** Any randomized algorithm deciding PASSIVE on 𝒰(u₀, r) with error
> probability ≤ 1/6 must make Ω(N) queries in expectation.

*Proof (Yao).* Let μ place mass 1/2 on u₀ and mass 1/(2M) on each u₀ + p_j. Let T be a deterministic
algorithm making q queries. If T outputs FAIL on u₀ its error under μ is ≥ 1/2. Otherwise it outputs
PASS on u₀, and by the transcript argument it outputs PASS on every u₀ + p_j whose interval it did
not touch — at least M − q of them — so its error is ≥ (1/2)(1 − q/M). For error ≤ 1/6 we need
q ≥ (2/3)M. Yao's principle transfers the bound to randomized algorithms. ∎

> **Corollary 4 (tightness).** The two-FFT baseline reads exactly N samples. Hence the query
> complexity of PASSIVE is **Θ(N)**, and the query model is exhausted: it cannot distinguish the
> baseline from any hypothetical O(N)-time algorithm. Everything remaining is the log factor, and
> that is §4.

### 3.4 The uniform constant, measured

`research/results/cycle2_v2a/v2a_compact.py`, C^∞ bumps supported on exactly 2k grid points,
41 positions spanning the interior window, both signs of a, 45 bisection steps on the amplitude:

| N | k | ε″₀ margin m₀/D | min a/D | **max a/D** | positions that flip | disjoint supports M |
|---|---|---|---|---|---|---|
| 4 096 | 4 | 1.007e−3 | 1.493e−3 | **1.680** | 41/41 | 486 |
| 8 192 | 4 | 9.927e−4 | 1.546e−3 | **1.680** | 41/41 | 972 |
| 16 384 | 4 | 9.903e−4 | 1.481e−3 | **1.680** | 41/41 | 1 945 |
| 16 384 | 8 | 9.903e−4 | 1.323e−3 | 1.500 | 41/41 | 972 |
| 16 384 | 16 | 9.903e−4 | 1.300e−3 | 1.447 | 41/41 | 486 |

**Direct check of Lemma 1's δ-independence at fixed N** (last three rows, N = 16 384): quadrupling
the bump width, k = 4 → 16, moves the uniform bound only 1.680 → 1.500 → 1.447 and the cheapest
position only 1.481e−3 → 1.323e−3 → 1.300e−3 — a 14% drift, converging monotonically, exactly the
O(δ) correction Lemma 1 predicts, against a 4× change in δ that a δ-dependent mechanism would have
made a factor-of-4 change.

Every position flips at every N; the uniform amplitude bound r/D = 1.680 is **identical to four
digits across a 4× range of N**; and M grows linearly in N as required. The cheap positions (near
the seam, a/D ≈ 1.4e−3) are cheap for exactly the reason Lemma 1 gives: ε″₀ is small there.

### 3.5 What the bound does NOT say — the honest boundary

The theorem is sharp about its own limits, and these limits are the reason Task 1 of this cycle
failed the way it did.

1. **It does not apply to band-limited classes.** If the input is promised to be band-limited to
   n ≪ N modes, the hard family dies: a perturbation supported on O(1) grid points is not in the
   class. Then O(n) queries suffice — decimate and transform. **This is exactly the loophole the
   decimated tests exploit**, and it is why they work on physical profiles and fail on cosine
   dressings near the decimation Nyquist. The correct reading is not "decimation beats the
   transform" but "decimation solves a smaller problem". Under a promise of bandwidth n, the
   baseline itself costs O(n log n), and the honest speedup is 1.
2. **It does not certify a bandwidth promise.** Verifying that a given N-vector is band-limited
   requires reading it — Ω(N) again, and in fact bounding the between-sample behaviour of the
   underlying function from samples alone is impossible without an a priori class. This is the
   abstract form of the obstruction that killed this cycle's certificate attempts (F5/F8).
3. **It says nothing about time.** An algorithm may read all N samples in O(N) and still need to
   decide. Every question about the log factor lives strictly beyond this model.

---

## 4. V2b — why the log factor resists. A **conditional, method-relative** barrier

> Restating the scope: nothing in §4 is a lower bound. §4 is a proof that a specific, exhaustive
> list of *known techniques* cannot produce one here, together with the structural reason.

### 4.1 What C15 actually asserts, formalized

Wave 3 §7's "nothing beats running DiHiTI and taking the minimum (O(N log N), two FFTs)" is, read as
a complexity claim, this:

> **(C15-formal)** Deciding membership of u ∈ ℝ^N in the convex polyhedral cone K requires
> Ω(N log N) time.

Equivalently, since (Au)_i is the value at x_i of the trigonometric polynomial whose coefficients are
−i·sgn(m)·û_m: *deciding whether a degree-N/2 trigonometric polynomial, presented by its samples, is
nonnegative requires Ω(N log N).* This is a classical, well-studied shape; the best known algorithm
is O(N log N) and **no superlinear lower bound is known for it in any general model.** C15 is an open
complexity conjecture that its author did not know he was asserting.

### 4.2 The two methods that give superlinear bounds — and why convexity caps both at Ω(N)

For decision problems of the form "is u ∈ W?" with W ⊆ ℝ^N semialgebraic, exactly two families of
technique are known to yield superlinear lower bounds. Both are multiplicity counts.

**(a) Ben-Or / component counting.** In the algebraic computation tree model, deciding membership in
W requires depth Ω(log C(W)) − O(N), where C(W) is the number of connected components of W (or of
its complement).

> *Here:* K is **convex**, so C(K) = 1. The complement of a closed convex set in ℝ^N, N ≥ 2, is also
> connected. Ben-Or therefore yields **nothing beyond the trivial Ω(N)**.
>
> For contrast: element distinctness attains Ω(N log N) by this route precisely because its
> accepting set is a union of **N!** open cones, and log(N!) = Θ(N log N). The N log N *is* the
> combinatorial entropy of the orderings. There is no analogue here; there is one cone.

**(b) Björner–Lovász–Yao / face counting.** In the linear decision tree model, membership in a
polyhedron W requires depth ≥ log₃ f(W), with f(W) the number of faces.

> *Here:* K is a polyhedron with at most |I| = Θ(N) facets, hence f(K) ≤ 2^{|I|} = 2^{Θ(N)}, hence
> log₃ f(K) = **O(N)**. The method is capped at Ω(N) by arithmetic, independent of how cleverly it
> is applied.
>
> To reach Ω(N log N) one would need f(W) = 2^{Θ(N log N)}, i.e. a facet count of Θ(N log N) or a
> non-simple face structure with permutation-like multiplicity. **A convex set defined by Θ(N)
> linear inequalities cannot have it.**

**The barrier, in one line:** *superlinear lower bounds of this type are entropy bounds on
combinatorial multiplicity, and passivity has no multiplicity to count — it is one convex cone with
Θ(N) facets.* This is not a failure of effort. It is a proof that these methods top out at Ω(N),
which §3 already achieves by an elementary argument.

### 4.3 The one known Ω(N log N) for a linear-algebraic task, and why it does not transfer

Morgenstern's theorem gives Ω(N log N) for **computing** the DFT in the linear circuit model with
bounded coefficients (|c| ≤ 1), by a determinant/volume-growth argument. It is the nearest thing in
the literature to what C15 wants. It does not apply, for two independent reasons:

1. **It bounds computing the transform, not deciding a sign.** A passivity decider is not required to
   output ε″; it outputs one bit. No reduction from "compute the DFT" to "decide nonnegativity of one
   trigonometric polynomial" is known, and one should not be expected: the decision problem discards
   almost all the information the transform produces.
2. **It needs the bounded-coefficient restriction.** With unbounded coefficients no superlinear lower
   bound is known for the DFT either. So even the transform-computation route rests on a model
   restriction that a referee is entitled to challenge.

### 4.4 The general-model position

In the RAM or Turing models, no superlinear time lower bound is known for **any** problem in P — nor
for any problem in NP. Proving C15-formal in a general model would therefore be a breakthrough in
complexity theory with no photonic content. The mission is right to have struck the original V2.

### 4.5 What would have to change — three specific routes, stated as open problems

These are the honest content of "why the log factor resists": not that it is impossible, but that
exactly one of the following would have to be supplied.

> **(R1) A reduction.** Show that any algorithm deciding passivity can be converted, with O(N)
> overhead, into one computing the DiHiTI transform (or the DFT) to fixed precision. Then
> Morgenstern's bound transfers within the bounded-coefficient model. *Assessment: the likeliest of
> the three to be attempted and the likeliest to be false — one bit of output is a very weak handle.
> A partial version (deciding passivity for all O(N) cyclic shifts recovers the transform) may be
> reachable and would already be interesting, though it changes the problem.*
>
> **(R2) A non-multiplicity lower-bound method for convex sets.** Every technique in §4.2 counts
> components or faces. A method that instead bounds depth by a *metric* or *entropy* property of a
> convex body — e.g. the number of distinct facet normals needed to approximate K, or the
> Kolmogorov entropy of its boundary at scale equal to the interior margin (~10⁻³·D, C21) — would
> escape the cap. *Assessment: this is a genuine open problem in computational geometry, not
> specific to this application. Nothing in the literature does it.*
>
> **(R3) A model restriction that fits the practice.** Restrict to *bounded-coefficient linear
> decision trees*, the decision analogue of Morgenstern's circuit model. Every algorithm anyone
> actually proposes here — FFT-based, decimated, hierarchical, branch-and-bound — is of this form.
> A Morgenstern-style determinant argument in that model would be a real theorem and would justify
> C15 *for the algorithms people write*, which is all C15 ever needed. *Assessment: the most
> promising of the three, and the one this crew could actually attempt.*

### 4.6 Explicit non-claims

For the avoidance of the exact overclaim MISSION.md warns against, we state:

- We do **not** claim Ω(N log N) time.
- We do **not** claim that no sub-O(N log N) algorithm exists.
- We do **not** claim convexity makes passivity easy; it makes it *hard to prove hard*. These are
  opposite statements and conflating them is the error to avoid.
- The Ω(N) of §3 and any future O(N)-time algorithm are compatible. Indeed §3's bound is what makes
  an O(N)-time algorithm the *right* thing to hunt for, and its absence after seven measured attempts
  (this cycle's `redteam.md` §8) is empirical, not proved.

---

## 5. Combined statement

> **Deciding whether a DiHiTI profile is passive requires reading Ω(N) of its N samples — and this is
> tight, since the two-FFT transform reads N. The remaining gap between Θ(N) queries and O(N log N)
> time is exactly one logarithmic factor. That factor cannot be closed from below by any method now
> known, and the obstruction is identifiable: the passive set is a single convex polyhedral cone with
> Θ(N) facets, whereas every known superlinear lower bound for a geometric decision problem is an
> entropy count of combinatorial multiplicity — components, or faces — of which a convex set has
> none to spare. Closing it would require a reduction from deciding to transforming, a
> non-multiplicity method for convex bodies, or a restriction to bounded-coefficient linear decision
> trees.**

## 6. Open problems

- **O1.** Prove or refute (R3): a Morgenstern-type Ω(N log N) for bounded-coefficient linear decision
  trees deciding membership in K. *This is the crew-sized version of V2b.*
- **O2.** Is there an O(N)-time (not merely O(N)-query) exact decider? §3 permits one; seven measured
  attempts have failed on constants, not on information (cycle-2 `redteam.md` §8). A compiled
  implementation is the untested lever.
- **O3.** Make Lemma 1's O(δ) term explicit, giving a fully self-contained proof of Theorem 2 with no
  numerical input. The two neglected terms are the regular part of ∂_tK (bounded, contributes O(aδ))
  and the variation of ε″₀ across supp p (contributes δ‖ε″₀′‖_∞); both are elementary.
- **O4.** Quantify the boundary of §3.5: for a promise class of bandwidth n, what is the query
  complexity? The construction suggests Θ(n); a matching lower bound would complete the picture and
  formally retire the "decimation beats the transform" reading.

## 7. Evidence index

| File | What it establishes |
|---|---|
| `research/results/cycle2_v2a/v2a_hard_family.py`, `.json` | δ-independence of the flip amplitude; TV and L^∞ flat to 0.3% over N = 2^11…2^15, L¹ → 0 like 1/N |
| `research/results/cycle2_v2a/v2a_compact.py`, `.json` | Uniform N-independent constant r/D = 1.680 with compactly supported bumps; 41/41 positions flip; Θ(N) disjoint supports |
| `research/results/cycle2_v2a/v2a_position.py`, `.json` | Lemma 1's law a_min(x₀) = κ ε″₀(x₀), κ/π = 0.414–0.436 over three decades of ε″₀ |
| `research/results/cycle1_c15_redteam/m4_summary.txt` | Hand B's original locality probe (the lead this document formalizes) |
| `research/results/cycle1_c15_redteam/e2.py` | Linearity of A (6.2e−15) and rank N−2 — the convexity §4.2 turns on |
| `research/results/cycle2_e1a/redteam.md` | The seven failed challenger routes referenced in §4.6 and O2 |
