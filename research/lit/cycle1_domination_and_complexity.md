# Cycle 1 — Front 1/2/3: potential domination, finite reduction, complexity

**Librarian, cycle 1, 2026-08-18.** Scope: is the mission's central object — "the log potential of
the drop measure is dominated by that of the point mass at i" — already a named, solved classical
problem? And are Fronts 2/3 (finite reduction of the inequality family; complexity lower bounds)
already answered?

**Access caveat, stated once, applies throughout:** I do not have full-text access to Saff–Totik
*Logarithmic Potentials with External Fields* or Landkof *Foundations of Modern Potential Theory* —
both are behind paywalls/out of reach of WebFetch (Google Books/Springer link pages only, no
theorem text). Everything below attributed to those two books is therefore a **claim about their
existence and rough content that I could not verify against primary text** — flagged explicitly
where it matters. Everything attributed to Encyclopedia of Mathematics, arXiv papers, or other
freely fetchable sources was read directly (via WebFetch's summarizer, which is a small model
condensing the page — treat as reliable for "what topic this covers" but re-fetch specific
sections yourself before citing a theorem number in the paper).

---

## FRONT 1 — potential domination: named, but the crew's exact setup needs one more step

### What is classically called "domination"

**Confirmed, freely accessible source (Encyclopedia of Mathematics, "Domination"):** the classical
name is the **Cartan domination principle** (also "Cartan's maximum principle"), citing M. Brélot,
*Éléments de la théorie classique du potentiel* (1959). Its statement, as given on that page:

> Let v be a non-negative superharmonic function on **R**ⁿ, n ≥ 3, and U_μ the Newtonian potential
> of a measure μ ≥ 0 of finite energy. If v(x) ≥ U_μ(x) on some set A with μ(complement of A) = 0,
> then v ≥ U_μ everywhere.

**This is NOT what the mission needs, and saying so loudly because it is an easy trap:** as stated,
it is a **one-way propagation theorem** ("true on the support ⟹ true everywhere"), not a
characterization of when one measure's potential dominates another's. It is also stated for n ≥ 3
(Newtonian, not logarithmic, kernel) in this source — the log-kernel (n=2) case is a known
degenerate special case in the classical literature (requires the extra mass condition
ν(mass) ≤ μ(mass), per the same EoM page: "the domination principle may fail in general... holding
only in a weaker sense where μ(mass) ≥ ν(mass)"). **Adjacent, not a match.**

### Balayage: adjacent, not the theorem itself

**Encyclopedia of Mathematics, "Balayage method":** balayage (sweeping-out) of a measure μ from a
domain D onto ∂D produces μ̂ with U_μ̂ = U_μ quasi-everywhere on the boundary and U_μ̂ ≤ U_μ inside D.
The page states "balayage inside the domain results in a decrease of the potential," and separately
that "the balayage principle implies the Cartan domination principle." So balayage is the
**mechanism** behind domination-type results, but the EoM page explicitly does **not** frame it as
a two-sided necessary-and-sufficient comparison test between two given, unrelated measures (a point
mass at i vs. an arbitrary measure on the line) — it answers "what measure results from sweeping,"
not "does measure A's potential dominate measure B's everywhere." **Adjacent.**

### The actual matching structure: Choquet order / dilation theory (Cartier–Fell–Meyer)

This is the most promising lead found, and worth the crew's direct attention, but comes with an
honesty caveat: I could not fetch primary text (Meyer's *Probability and Potentials*, or the
original Cartier–Fell–Meyer 1964 note), only survey-level descriptions.

Multiple sources (Encyclopedia of Mathematics-linked survey, and a Springer chapter "Orderings and
dilations of measures") describe an order on measures: for measures μ, λ (compact convex support),
**μ ≻ λ iff ∫f dμ ≥ ∫f dλ for every continuous convex (equivalently: superharmonic, for potential
kernels) function f.** This is *exactly* the shape of the mission's criterion — "∫ ln|η−ξ(t)| g(t)dt
≤ (D/2)ln(1+η²) for all η" is precisely "the potential kernel evaluated against g is dominated by
the potential kernel evaluated against the point mass, for every choice of observation point" —
i.e. domination against the whole *family* of functions {ln|η − ·|}_η, which for a superharmonic /
subharmonic kernel is equivalent (by a standard approximation argument) to domination against **all**
convex/superharmonic test functions, because {ln|η−ξ|}_η spans (in an appropriate closure) the cone
of superharmonic functions on the interval — this last step is the crew's own claim to verify, not
something I confirmed in a primary source.

**The theorem that characterizes this order:** the survey material attributes to **P. Cartier
(building on work of J.M.G. Fell and P.-A. Meyer)** a general dilation/comparison theorem: for two
measures of **equal total mass** on a compact convex set, μ ≻ λ in the convex order **iff λ is a
"dilation" of μ** (concretely: iff there is a Markov kernel / "balayage" taking μ to λ that only
moves mass toward the boundary/spreads it out in the convex sense — the measure-theoretic
generalization of majorization). One-dimensional special case (also cited on the same page): **Hardy,
Littlewood & Pólya** gave the dilation characterization on the line — this is the classical
**majorization** theory the crew may already know by a different name.

**Verdict on Front 1:** the crew's domination inequality is a specific instance of the **Choquet /
Cartier–Fell–Meyer convex (balayage) order**, restricted to the one-parameter family of log-kernel
test functions rather than all convex functions. That restriction is *not* automatically free — it
needs to be shown that this family is rich enough (a "determining class") for the convex order on
the relevant space, or the crew is only getting a *necessary* condition from general dilation
theory, not sufficient, unless that richness is established. **This is a real, named theorem
($Cartier$–Fell–Meyer order) adjacent to the mission's problem, not a solved instance of it** — I
found no source that already specializes it to the log kernel on an interval vs. a point mass at a
complex point, which is precisely the missing step (and would be new if the crew does it). Do not
claim more than this: nobody I found has already produced "N&S condition for log-potential
domination by a point mass with an explicit finite check," which is what V1/V3 need.

**One-dimensional simple sufficient conditions (crew's actual ask):** the classical
Hardy–Littlewood–Pólya dilation/majorization characterization on the line is likely the correct
frame for "the middle-half and unimodality theorems might be instances of something classical" —
majorization theory has well-known sufficient conditions in terms of support and moments (e.g.
Schur-convexity criteria), and the crew's middle-half theorem (support confined + first-moment
balance ⟹ domination) has the *flavor* of a classical majorization sufficient condition, but I did
not find the specific classical statement pinned to this log-kernel family. Flag for a follow-up
cycle with a fetchable copy of Hardy–Littlewood–Pólya or a modern majorization-theory reference
(e.g. Marshall–Olkin–Arnold, *Inequalities: Theory of Majorization*) — not fetched this cycle.

---

## FRONT 2 — finite reduction of the infinite inequality family

### Semi-infinite LP → finite constraints: real, cited, but answers a different question than V1 needs

Charnes, Cooper & Kortanek, *"Duality in Semi-Infinite Programs and Some Works of Haar and
Carathéodory,"* Management Science 9(2), 209 (1963) — confirmed to exist and be about exactly this
topic (title-level, not primary-text verified). The general classical fact, well documented in the
semi-infinite programming literature (survey chapter also found): **by Carathéodory's theorem, an
optimal solution to an n-variable semi-infinite LP is supported on at most n (or n+1) of the
infinitely many constraints** — i.e., *the optimum*, not *feasibility of a fixed candidate*, is
determined by finitely many active constraints.

**This is the wrong theorem for the mission's exact ask.** The mission needs: "is a *given* ε′
(equivalently, a given g) feasible, i.e. does it satisfy ALL constraints indexed by η ∈ R?" — a
feasibility question, checked once, not an LP being solved for an optimum. Caratheodory-style
finite-support results characterize optimal solutions of an optimization *over* the feasible set,
not a cheap feasibility certificate for one candidate. **Do not import this result as "V1 solved" —
it answers a related but distinct question.** It may still be useful in Front FALLBACK-2 territory
(inverting the problem / building extremal passive profiles), which the mission has explicitly
parked (see MISSION.md non-goals) — so even where it's real, it's off-target for this cycle's
mission.

### Chebyshev/Krein–Markov systems: the right vocabulary, unresolved fit

Karlin & Studden, *Tchebycheff Systems: With Applications in Analysis and Statistics* (1966), and
Krein & Nudel'man, *The Markov Moment Problem and Extremal Problems* (1977) — both real, standard
references (title/existence confirmed, not primary-text verified this cycle). The relevant fact
class: for a genuine Chebyshev (T-)system of dimension n, the **moment space** has extreme points
achieved by measures supported on at most ⌈n/2⌉ or ⌈(n+1)/2⌉ points ("principal representations"),
and there exist **Chebyshev–Markov–Stieltjes inequality** results giving finite pointwise tests.

**Why this doesn't obviously transfer:** the crew's inequality is indexed by η ranging over all of
**R** (infinite-dimensional test family {ln|η−·|}_η), not a finite-dimensional Chebyshev system —
Krein–Markov theory is about moment problems of dimension n (finitely many given moments), which is
the *opposite* structure (the crew has infinitely many constraints and wants to know if finitely
many suffice, not the reverse). The lead worth chasing next cycle: **arXiv:2403.04548, "An
Introduction to T-Systems — with a special Emphasis on Sparse Moment Problems, Sparse
Positivstellensätze, and Sparse Nichtnegativstellensätze"** (2024) — title suggests exactly the
"can positivity be certified by finitely many moments" question in modern language
(Positivstellensatz / sparse SOS framing). Not read this cycle; flagged as the strongest lead for
Front 2, not yet verified.

### Trigonometric moment problem / Carathéodory–Toeplitz: confirms K2, does not add a finite test

Standard fact, multiply confirmed: a sequence is realizable as Fourier coefficients of a
non-negative measure on the circle **iff every Toeplitz matrix formed from it is PSD** (Herglotz /
Carathéodory–Toeplitz theorem). This is exactly Wave 2's K2 (already killed: "Toeplitz eigenvalue
test IS the Hilbert transform in disguise"), and nothing found this cycle weakens that. No finite
truncation of the Toeplitz PSD condition is known to be exact in general — PSD-ness of *every*
leading principal minor as size → ∞ is intrinsically an infinite check unless the underlying measure
is a priori known to be a finite atomic measure (rank-deficient case, Carathéodory–Fejér 1911
factorization) — which is not assumed here. **Confirms rather than breaks K2; front 2 remains open
on this specific route.**

---

## FRONT 3 — complexity

### Property testing / query lower bounds for non-negativity: not found, likely genuinely open in this exact form

Searched directly for "non-negativity testing from samples" — no hit. Found adjacent property-testing
results (convexity testing lower bounds, arXiv:2410.17958; metric-testing bounds,
arXiv:2411.08989) that use the same *methodology* (Yao's minimax principle, communication-complexity
reductions, per Blais–Brody–Matulef) the crew would need to build a lower bound (V2), but **no
existing theorem was found that already proves an Ω(N) query lower bound for deciding "is this
band-limited / sampled function's associated potential/transform non-negative."** This is consistent
with the mission's framing of V2 as an open, publishable target — I found no reason to believe it's
already been done, and no reason to believe it's easy (the methodology is standard and importable,
which is good news for actually attempting the proof).

### Fast multipole for the log kernel in 1D: standard O(N log N), no evidence of a cheaper minimum-only algorithm

Confirmed: 1D FMM / kernel summation for log and similar non-oscillatory kernels achieves the
standard optimal **O(N log N)** (or O(N) with tree methods), matching — not beating — the mission's
existing two-FFT baseline. Searched specifically for "is finding the minimum of a potential cheaper
than evaluating it everywhere" — **found nothing.** This is worth being explicit about: it is not
obviously true that argmin of a 1D potential is easier than the full potential in general (unless
convexity or unimodality is assumed, which is exactly the kind of profile-restricted case the crew
already has as sufficient theorems, not the general case V1 needs). No existing algorithm found;
flag as open, not merely un-searched.

### Fast Toeplitz PSD certification below O(N log N): not found

Existing fast Toeplitz algorithms found are all for **solving** Toeplitz systems (Levinson–Durbin
O(N²), superfast Toeplitz solvers O(N log²N), symmetric-PD solver O(N^1.5 log N)) — none of these
are advertised as, or reduce to, a sub-O(N log N) **certificate of positive semi-definiteness**.
Solving a system is a different (and generally harder-looking) task than certifying PSD-ness; I
found no claim anywhere that PSD certification for a general Toeplitz matrix beats O(N log N). This
null result is itself useful for V2: it's mild circumstantial evidence (absence of a known fast
algorithm despite Toeplitz PSD testing being a well-studied classical problem for decades) — not a
proof, and should not be reported to the Captain as one.

---

## Bottom line for the Captain

1. **Front 1 has a real classical name and theorem to chase — Cartier–Fell–Meyer / Choquet convex
   (dilation) order** — but nobody has already specialized it to "log-potential of a measure on an
   interval vs. point mass at i," which is exactly the mission's ask. This is genuinely open, not
   solved, and is the crew's best next move: try to show the family {ln|η−ξ|}_η is a *determining
   class* for the convex order on the relevant interval (or find the specific classical paper that
   does this — I did not find one, but did not do an exhaustive search of the majorization-theory
   literature, e.g. Marshall–Olkin–Arnold, which is the next cycle's job).
2. **Cartan's domination principle and balayage, by themselves, are NOT what the crew needs** — both
   are one-way/sufficiency statements, not the N&S comparison theorem. Do not cite either as if it
   already answers V1/V3.
3. **Front 2's semi-infinite-LP route answers a different question (optimum structure, not single
   feasibility check)** — do not treat Charnes–Cooper–Kortanek as closing V1. The Krein–Markov /
   T-system machinery is the right vocabulary but doesn't obviously fit an infinite-dimensional test
   family; arXiv:2403.04548 (T-systems / sparse Nichtnegativstellensätze, 2024) is the strongest
   unread lead for next cycle.
4. **Front 3 (complexity) appears to be genuinely open territory** — no existing query-complexity
   lower bound for this exact non-negativity-testing problem, no cheaper-than-full-transform way to
   find a log-potential's minimum, no sub-O(N log N) Toeplitz PSD certificate. This is good news for
   V2's ambition (nobody has already published the impossibility proof the mission wants) and
   confirms the crew is not wasting time chasing something already known — but also means V1's
   speedup target has no known shortcut to borrow from the literature; it would have to be genuinely
   new.
5. **No item on either front kills any current mission hypothesis outright.** Nothing here should be
   added to KILLED.md. The one caution: don't let the crew cite Cartan domination, balayage, or the
   semi-infinite-LP Carathéodory result as if they already deliver V1 or V3 — each is adjacent
   vocabulary, not the theorem itself, per the standard above.
