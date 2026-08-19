# MISSION — Cheap certificates for DiHiTI passivity

**Captain:** Kunal Lakhanpal. **Briefed:** 2026-08-18.

## The problem

The DiHiTI construction (differentiate → Hilbert transform → integrate, `tmm_helper.ht_derivative`)
takes a real permittivity profile ε′(x) on the truncation window [−L, L] and returns its
spatial-Kramers–Kronig partner ε″(x). The paper (JOPT-113210, `sKK-Paper-Overleaf/main.tex`)
guarantees **reflectionlessness**. It does **not** guarantee **passivity**.

For the paper's logistic, ε″ ≥ 0 held *by luck*. Other ε′ profiles
(`theory/figures/2026Apr27`, `theory/figures/2026May07`) produce ε″ < 0 in regions — gain. The
motivating figure `2026May07/opposite_bumps_sweep.png` shows a clean crossover: as bump magnitude α
grows, PASS at α = 0.0/0.3/0.6 → FAIL at α = 1.0 (min ε″ = −0.117) → FAIL at α = 1.5 (−0.302). The
Captain saw a pattern here and could not quantify it. **Quantifying that crossover is the seed of
this mission.**

Why it matters physically: Longhi 2015 (`Papers/Longhi_sKK_Slowly_Decaying_Profiles_2015.pdf` §3)
shows gain in an sKK medium can seed a growing interface mode whose nonlinear saturation *destroys
the reflectionless property itself*. He notes it "deserves further analysis"; nobody has followed up.

## The goal

Find **clean, ideally necessary AND sufficient** conditions on ε′(x) guaranteeing that its DiHiTI
partner is gain-free (ε″ ≥ 0 everywhere), with ε″ = 0 at the endpoints.

**Ranked objectives — pursue in this order. Falling back is a last resort, not a convenience.**

1. **PRIMARY (the bonus, and the real prize).** Conditions that are **computationally cheaper to
   check than computing the DiHiTI itself** (currently two FFTs, O(N log N)). The Carathéodory–
   Toeplitz criterion is exact but is — in the Captain's words — "merely a re-representation of the
   HT in Fourier space, rather than clever conditions." A re-representation of the transform does
   not count as a win.
2. **FALLBACK 1.** Exact necessary-and-sufficient conditions that are *not* computationally cheaper,
   but are structurally clean and give real insight into *which* ε′ are passive.
3. ~~**FALLBACK 2 — LAST RESORT ONLY.** Looser (sufficient-but-not-necessary) conditions. Only after
   1 and 2 have been genuinely exhausted and the exhaustion documented.~~
   **🚩 PROMOTED 2026-08-19 by the Captain — no longer a fallback. See the amendment below.**

---

## ⭐ CAPTAIN'S AMENDMENT — 2026-08-19 (binding, supersedes the ranking above)

The solo Fable session returned a *two-part* answer to the one question: N&S conditions exist
exactly at the finite-description level, and **below that they provably do not exist** (Theorems 1
and 2, `research/theory/ns_answer.md`, claims C48/C49). The Captain has read it and adjusted the
end goal accordingly. **The mission now has two sequential goals, in this order:**

### GOAL A — make the impossibility *exact*
If necessary-and-sufficient conditions on ε′ guaranteeing ε″ ≥ 0 cannot be found — as C49 now
indicates — then the deliverable is a **mathematically rigorous and exact proof of why they
cannot exist**. Not a numerically-certified argument; not "proved modulo an ingredient". A proof.

This makes Fable's outstanding gap the mission's critical path: Theorem 2 currently carries the
label *"proved modulo a numerically-certified constructive ingredient"* (Lemma E's active-point
window, and the thickening step written with hypotheses). Closing that is **Goal A**, and until it
closes the mission has no deliverable at all under the new framing.

### GOAL B — the cleanest, most succinct one-sided conditions
Once A is closed: find the **cleanest, most succinct** set of conditions on ε′ that are
**sufficient-but-not-necessary and/or necessary-but-not-sufficient** for passivity.

**Read this correctly — B is a demand for FEWER statements, not more.** The corpus already holds
roughly a dozen one-sided results (middle-half, unimodality, plateau/Clausen, dip tolerance, the
four-term edge cascade, the depth law, the off-centre step …). Producing a thirteenth is still a
failure. The deliverable is the **minimal spanning set**: the weakest sufficient condition from
which the existing sufficient theorems fall out as *corollaries*, the strongest necessary
condition from which the cascade falls out as *corollaries*, and an honest characterization of
the gap between them. Two statements bracketing the truth, with the zoo collapsing into them.

Goal A is what makes Goal B respectable rather than defeatist: one-sided conditions stop being an
admission of failure the moment it is *proved* that two-sided ones cannot exist.

## Victory conditions — FALSIFIABLE

**V1 (primary).** A decision procedure that, for any ε′ sampled on N points, returns PASSIVE /
GAIN agreeing with `ht_derivative`'s verdict on ≥ 99.9% of a held-out adversarial test set including
near-threshold cases, at asymptotic cost **strictly below O(N log N)**, or O(N) with a measured
wall-clock speedup ≥ 3× over the two-FFT baseline at N = 2^14. ~~Sharpness at the boundary is mandatory: every DiHiTI profile sits *exactly* on the passive cone
boundary (margin 0), so any criterion that is merely approximate misclassifies precisely the profiles
of interest.~~ **🚩 STRUCK 2026-08-18 — this premise is FALSE.** It was an artifact of including the
structurally-zero endpoints (C21/K16). The interior margin is positive and N-independent (~1e−3·D).
Approximate cheap certificates are back in play (H12).

**V2 (RESTATED 2026-08-19 by Captain's decision — option A).** The original V2 asked for a proof
that no sub-O(N log N) procedure exists. Red-team established that this is **unreachable as written**:
it bundles two very different claims, and the second is a *superlinear time* lower bound on a problem
reducing to "is this degree-N/2 trigonometric polynomial nonnegative?" — the same barrier as circuit
lower bounds, and not a problem in this field. The reachable version, which red-team reports as ~90%
established, is:

  **V2a.** Prove the **Ω(N) query lower bound**: any correct passivity decision must read Ω(N) samples
  of ε′. (Hand B supplied the key step — localized perturbations get *cheaper* to hide as they narrow.)

  **V2b.** Characterize precisely **why the remaining log factor resists**: identify what would have to
  change to beat O(N log N), and show that convexity blocks every known superlinear route. This is a
  structural/conditional barrier result, stated honestly as such — **not** an unconditional lower bound,
  and it must never be written up as one.

Together these support the honest headline *"the transform is optimal up to a log factor, and here is
exactly what would have to give for that factor to fall."*

**V2 and V1 are compatible, not alternatives.** A constant-factor win (e.g. E1(a)'s hoped-for 10–30×)
coexists with an Ω(N) asymptotic bound. The best outcome is **both**: a materially faster practical
test, plus a proof that no fundamentally better one exists.

**V3 (fallback 1).** Exact N&S conditions stated as an explicit, checkable property of ε′ — not a
restatement of "compute the transform and look at its sign" — with proof and numerical verification.

**V4 (mission seed, expected early).** A closed-form prediction of the α-crossover in
`opposite_bumps_sweep.png`, matching the observed PASS→FAIL transition between α = 0.6 and α = 1.0
to within a few percent.

**V6 (parsimony — binding on every deliverable).** ~~The final characterization must be **necessary
and sufficient**, and~~ must consist of **one or two** conditions on ε′, with the prior special cases
recovered as corollaries. ~~A result that adds a new one-sided condition to the existing zoo does not
satisfy this mission regardless of its correctness.~~ Parsimony is a hard requirement here, not a
stylistic preference: the Captain is after a *definition*, and a definition is ~~two-sided and~~ short.

> **🚩 AMENDED 2026-08-19 (Captain).** The **two-sidedness** requirement is withdrawn — C49 makes it
> unachievable for continuum profiles, and V6 must not demand what has been proved impossible. The
> **parsimony** requirement is *retained in full* and now binds Goal B: one or two one-sided
> conditions, with the existing zoo recovered as corollaries. Adding a thirteenth sibling theorem
> remains a mission failure. What changed is the *shape* of the target, not the standard.

**V7 (NEW 2026-08-19 — Goal A, now the critical path).** A **mathematically rigorous and exact**
proof that no necessary-and-sufficient condition of the wanted kind exists on general (continuum)
ε′. Falsifiable form: Theorems 1 and 2 of `research/theory/ns_answer.md` carry the label **proved**
with *no* numerically-certified ingredient — specifically, (i) Lemma E's active-point interval is
established analytically rather than by grid certification, (ii) the thickening step is written out
in full with its hypotheses discharged, and (iii) the result survives `red-team`, `replicator` and
`auditor` verdicts to rung `hardened`. Partial credit is not available here: "proved modulo" is
exactly what this victory condition exists to eliminate.

**V8 (NEW 2026-08-19 — Goal B).** A **minimal spanning set** of one-sided conditions: one
sufficient-but-not-necessary condition implying C3 (middle-half), C5 (unimodality) and C6/C7
(plateau/dip) as corollaries; one necessary-but-not-sufficient condition implying the C8 cascade
(edge slopes, M₁ = 0, M₂ ≥ 0, σ ≥ 0) as corollaries; and a quantified statement of the gap between
them. Falsifiable form: each prior one-sided theorem is derived *from* the new statement in one or
two lines, the derivations are checked, and the total count of independent one-sided statements in
the corpus **strictly decreases**. A result that leaves the zoo intact and adds to it fails V8 by
construction.

**V5 (endpoint requirement).** ε″(±L) = 0 must hold. *Status: believed automatic in the DiHiTI
gauge (the seam normalization K(±L;t) ≡ 0 is built into the kernel). This is a claim to verify, not
assume — it enters the ladder like any other.*

## Structural critique of the prior work — the parsimony requirement

Two structural defects, independent of whether any individual result is correct. Both are Captain's
findings (2026-08-18) and both are binding on this crew's output.

**(A) It is scattered.** The corpus is a patchwork of overlapping special cases rather than a theory:
middle-half sufficiency; unimodality sufficiency; the plateau Clausen closed form; the dip tolerance
d*(b); a four-term edge cascade (slope match, M₁, M₂, σ); the perturbative depth law; the off-centre
step closed form; the Carathéodory–Toeplitz PSD test; the Fejér–Riesz parametrization. That is
roughly a dozen statements, each covering a partly-overlapping family.

The tell is stated in Wave 3 itself: Theorems 4.2 and 5.2 are *"genuinely complementary: neither
contains the other."* Two sufficient conditions, neither implying the other, both needed to cover the
profiles of interest — that is the signature of an incomplete theory, not a rich one.

**(B) Almost nothing is necessary AND sufficient.** Every clean result is one-sided:

| Result | Necessary? | Sufficient? |
|---|---|---|
| Middle-half (Thm 4.2) | no | yes |
| Unimodality (Thm 5.2) | no | yes |
| Plateau / dip tolerance | no | yes (within family) |
| Edge cascade, σ ≥ 0, M₁ = 0 | yes | **no** |
| Carathéodory–Toeplitz (Thm 3.1 iv) | yes | yes — but see below |

Exactly **one** statement in the corpus is both, Thm 3.1, and it is the one the Captain rejected as
"merely a re-representation of the HT in Fourier space." So the corpus contains no clean two-sided
characterization at all: it has a transform in disguise, plus a zoo of one-sided approximations to it.

**A definition is necessary and sufficient.** That is what a definition *is*. One-sided conditions are
useful engineering screens; they are not a theory of when DiHiTI is passive.

**What a clean answer looks like.** Newton's laws, not thirty scenarios. **One or two** conditions on
ε′ — two-sided, with the existing special cases falling out as *corollaries* rather than sitting
beside them as siblings. If the crew produces a thirteenth theorem covering a thirteenth family, that
is a failure of the mission even if the theorem is true.

**Promising unification (H9).** The exact criterion in supporting-line form already reads as a single
statement: ∫ln|η−ξ(t)|g(t)dt ≤ (D/2)ln(1+η²) for all real η is precisely *"the logarithmic potential
of the drop measure is dominated by that of the point mass at i"* — a balayage/majorization ordering
on the drop measure. Under that reading, middle-half and unimodality are two sufficient conditions
for **one** domination relation, not two unrelated theorems. Finding the minimal checkable
characterization of that domination is the cleanest available route to a genuine definition.

## What we have so far

Three waves of prior work. **All of it is `conjecture` until this crew's gate says otherwise** —
inherited results do not enter at a high rung merely because they are written down.

**Wave 1 (mid-Apr → mid-May 2026)** — trial and error, no conditions found. Figures in
`theory/figures/2026Apr27`, `2026May07`. Notes: `passivity_constraint_notes.{tex,pdf}` (546 lines,
Fourier mode-by-mode classification, "two failure mechanisms"; **known sign typo** in the
mode-by-mode safety rule — recorded in `KILLED.md`), `derivation_ht_endpoint_slope_constraints.tex`,
`hilbert_transform_notes.tex`, `derivation_2nd_deriv.pdf`.

**Wave 2 (2026-08-13)** — `theory/passivity_2026Aug/potential_theory_findings.md` +
`potential_theory_derivation.pdf` + `potential_criterion.py`. Broke the previous session's four
"impossibility" claims. Master identity: DiHiTI is a **logarithmic potential** of the drop density
g = −dε′/dx, not a Hilbert transform:

    ε″(x) = (1/π) ∫ g(t) · ln[ cos(πt/2L) / |sin(π(x−t)/2L)| ] dt

Sufficient theorem (monotone + drop in middle half + balance M₁ = 0 ⟹ passive, sharp at L/2).
Necessary cascade: edge-slope match, M₁ = 0, M₂ ≥ 0, σ ≥ 0. Screen: 180/214 gain caught, 0 false
alarms.

**Wave 3 (2026-08-18, Fable session)** — `Responsive Notes - Derivation - 081826.pdf` (12 pp),
verified by `verify_passivity.py`. Completed the middle-half theorem for the general asymmetric case
via a two-point inequality (Lemma 4.1); proved a **unimodality theorem** (g even + unimodal ⟹
passive at *any* width, via a Clausen closed form); gave the exact criterion in four equivalent
forms (Thm 3.1); proved a no-go on the infinite line forcing truncation; and inverted the problem
(Thm 6.1: complete parametrization of the passive set; Cor 6.2 Fejér–Riesz).

## The honest gap — what is NOT done

- **The primary objective is OPEN.** Wave 3 §7 concedes: "as a test of a given profile, nothing
  beats running DiHiTI and taking the minimum (O(N log N), two FFTs)." That concession is an
  **assertion, not a theorem** — and this exact pattern (asserted impossibility) was broken once
  already on Aug 13. It is the crew's first target.
- Sufficient conditions (middle-half, unimodality) are **not necessary**. Gap between them is Open
  problem 8.1 (skew tolerance).
- The Carathéodory–Toeplitz form is exact but is the re-representation the Captain rejected.

## Foundational assumptions

If `red-team` kills one of these, that is a Captain checkpoint — not a routine finding.

1. The domain is a **circle**: scipy's FFT-based `hilbert` continues the profile periodically. The
   governing class is Carathéodory on the disk, not Nevanlinna on the line.
2. "Passive" means ε″ ≥ 0 in the DiHiTI gauge, with ε″(±L) = 0.
3. The master identity (log potential of the drop density) is correct — verified to rel. rms
   2.9e−3, quadrature-limited.
4. Truncation is mathematically necessary, not a numerical convenience (no-go, Prop 2.1).

## Non-goals / out of scope

- **The flipped optimization problem is PARKED, not pursued.** Starting from a passive ε″, finding
  its sKK partner, and optimizing reflectance (Thm 6.1, Cor 6.2, `constructor2.py`, the
  budget–performance frontier of Wave 3 §6) is a *generation* problem. This mission is a *decision*
  problem: given ε′, is it passive? These are genuinely different questions, and conflating them is
  precisely how the previous sessions drifted. See `PARKED.md`.
- Not re-deriving the reflectionless property — the paper covers it.
- Analyticity/spectral-FoM machinery measures one-sidedness, **not** passivity. The triangle profile
  has FoM 99.9999987% *and* min ε″ = −1.82. Do not conflate them.

## Starting resources

- **Paper**: `sKK-Paper-Overleaf/main.tex` (JOPT-113210)
- **Core code (read-only)**: `tmm_helper.py` — `ht_derivative`, `logistic`, `eps`, `skk_spectral_fom`
- **Wave 2/3**: `theory/passivity_2026Aug/` — `potential_theory_findings.md`, `handoff.md`,
  `verify_passivity.py`, `potential_criterion.py`, `profile_generator.py` (labelled samples —
  **start here for test data**), `cheap_screen.py`, `periodic_model.py`
- **Wave 1**: `passivity_constraint_notes.tex`, `derivation_ht_endpoint_slope_constraints.tex`,
  `hilbert_transform_notes.tex` (repo root)
- **Motivation**: `Papers/Longhi_sKK_Slowly_Decaying_Profiles_2015.pdf` §3
- **⭐ LABELLED EXAMPLE CORPUS (Captain, 2026-08-18)**: `theory/figures/2026Apr27/` and
  `theory/figures/2026May07/` hold an *abundance* of worked examples beyond the seed figure — dozens
  of profile families with their DiHiTI partners, many carrying PASS/FAIL labels and min ε″ values.
  Wave 1 produced no conditions, but it produced **data**. Treat these folders as the mission's
  labelled test set: mine them for near-threshold cases, counterexample candidates, and validation of
  any proposed criterion. Any crew member short of test cases or ideas should go here first.
- **Figures**: `theory/figures/2026May07/opposite_bumps_sweep.png` (the seed),
  `2026Aug13_potential_criterion/`, `2026Aug12_passive_constructor/`
