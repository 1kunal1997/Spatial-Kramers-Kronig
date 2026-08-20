# STATE OF PLAY — read THIS, not the whole LOG

**Updated 2026-08-19 (second update: solo Fable session).** `LOG.md` is now long append-only
history. **Crew should read this file plus `MISSION.md`, `BOARD.md` and `PROTOCOL.md` — NOT the
full log.** Reading the log end-to-end wastes context and is a hallucination source.

## 🚩 CAPTAIN'S AMENDMENT 2026-08-19 — READ BEFORE ANYTHING ELSE

The Captain has read the solo-session answer and **adjusted the end goal**. Full text in
`MISSION.md` ("Captain's Amendment"), new victory conditions **V7** and **V8**. In brief:

- **GOAL A (V7): ✅ THE PROOF IS DELIVERED (2026-08-19, board F1).** Theorem 2 is now **proved
  outright** — `research/theory/lemmaE_closure.md`, claim **C55**. The numerically-certified Lemma E
  window is gone: an explicit symmetric reciprocal-pair base point (Δ″(0) = −25/4 as an identity)
  plus one implicit-function step that does the sweep and the thickening together. Lemma A reproved
  real-variable, so Theorem 2 is entirely real analysis.
  **V7 IS NOT YET MET:** it requires C49 at `hardened`, which requires `red-team: SURVIVED`.
  Replicator MATCH ✅ and auditor CLEAN ✅ are in; **red-team is the last gate and has not run.**
- **GOAL B (V8, after A):** the **cleanest, most succinct** set of one-sided conditions
  (sufficient-not-necessary and/or necessary-not-sufficient). This is a demand for **fewer**
  statements: the minimal spanning set, with the existing dozen recovered as corollaries. A
  thirteenth sibling theorem is still a mission failure.
- **V6 amended:** two-sidedness withdrawn (C49 proves it unachievable); **parsimony retained in
  full** and now binds Goal B.

## ⭐ 2026-08-20 solo Fable session #2: GOAL B (V8) IS ANSWERED — read `research/theory/minimal_oneside.md`

Short version (handoff: `research/results/fable_goalB/HANDOFF.md`, claims C57–C60):
- **The one-sided corpus collapses 10+ → 2 parents.** Theorem S (sufficient: balanced
  half-period box ⊕ even unimodal, closed under two NEW covariances — Cayley contraction
  and inversion, each a one-line identity) absorbs C3/C25′/C4/C5/C6/C7. Theorem N
  (necessary: passivity on the closed outer half |x| ≥ L/2 + centre) absorbs the C8
  cascade, C9, C13 as its seam jet; σ ≥ 0 = the ω-average of the full law, proved
  independent of N by explicit witness.
- **Proposition NG (new, proved): no finite zoo can ever span the passive cone** — 𝒫 is
  non-polyhedral on every Lemma-E2 atom set. The exact primal mirror of C49, proved by
  the same Lemma E2: one lemma now kills finite batteries (necessary side) AND finite
  generator families (sufficient side). V8's "prove no parent exists" branch, delivered
  for both sides; {S, N} is extendable forever, completable never.
- **The gap is self-similar (the ladder):** Theorem S-τ — inversion-symmetric drops with
  radii in [L/4, 3L/4] are passive, **sharp at L/4** (arctan(√2−1) = π/8), proved from
  C55's Lemma E1′, provably outside S. C3's middle-half motif regenerated on the radius
  axis. A third rung has a closed-form threshold (numerics, 7 digits).
- **Gap quantified (B3):** single symmetric pair — passive ⟺ radius ≤ L/2, passes N ⟺
  radius ≤ 2L/3 (exact); ensembles — N false-passes 1.5%, S certifies 39.8% of passive
  atomic profiles.
- **Killed candidates (do not re-walk):** radial-CDF dominance; "outer+centre ⟹ passive"
  in EVERY symmetry class (805 even counterexamples; a vanishing centre atom defeats any
  point condition — C48's density theorem made concrete).
- **Rung dependencies:** C58/C59 lean on C55 (`supported`, red-team pending) — the F1
  red-team now gates Goal A **and** Goal B. Everything else in the doc is self-contained
  real analysis + machine-zero controls.

Everything below this line predates the amendment and remains factually current.

## ⭐ 2026-08-19 solo session: the ONE QUESTION is answered — read `research/theory/ns_answer.md`

Short version (handoff: `research/results/fable_ns_answer/HANDOFF.md`, claims C48–C54):
- **Two-sided N&S conditions EXIST exactly at the finite-description level**: per-profile for
  every n-layer staircase (the L3 node test — exact, transform-free, ≤ n+1 closed-form checks),
  and per-family via (L2). That is the positive half of the answer.
- **Below that, impossibility is now THEOREMS, not folklore**: any sound-and-complete test set
  of positions must be dense (C48); **no finite battery of linear measurements of ε′ — moments,
  samples, corner masses, the whole necessary cascade — feeding ANY decision rule can decide
  passivity** (C49; explicit matched witness pairs with identical readings and opposite
  verdicts, k = 8 and k = 16). No localized (sub-window) criterion either. Structural reason:
  the passive cone is a simplex cone whose unique extremal representation IS ε″ (C54) — every
  complete certificate re-represents the transform.
- **V3/V6 adjudication**: the "definition" the Captain wanted is (L1) + its two finite
  collapses; provably nothing shorter exists for continuum profiles. The one-sided-zoo era ends
  for a principled reason.
- Byproducts: N1 closed (C51), T2/CFM conjecture refuted outright (C52), red-team residual
  risk #3 discharged (C50), and a P0-relevant production hazard found (C53: O(h) additive
  offset on staircases, cancels in differences).
- **Fragility to close next**: Lemma E of C49 is numerically certified, not yet symbolic —
  board item F1.

## The question

Given a real permittivity profile ε′(x) on [−L, L], the DiHiTI construction returns its partner
ε″(x). **When is ε″ ≥ 0 everywhere (no gain)?** Want: conditions on ε′ that are **necessary AND
sufficient**. Bonus: checkable **faster** than computing the transform and taking the minimum.

## What is genuinely established

| Result | Status |
|---|---|
| The construction is a **logarithmic potential** of the index-drop density; passivity is a comparison of two potentials | audited clean; but see caveat below |
| **Extreme-point reduction** (Bauer/Choquet) — makes families decidable via their extreme points | the real content; **not yet adversarially tested** |
| **Sufficient condition, strengthened**: drop supported in [−a,b] with **a+b ≤ L** ⟹ passive. Supersedes "middle half", which is just the symmetric case a=b=L/2 | survived red-team; sufficiency blind-replicated |
| Endpoint values ε″(±L) = 0 hold **automatically** | verified two independent routes |
| Corrected evaluator agrees with all **240** test profiles | audited clean vs 40-digit reference |
| Captain's crossover **α_c = 0.66623** | ⚠️ possible conflict with a value of 0.2667 seen elsewhere — **UNRESOLVED**, must be reconciled |
| **Skew tolerance is zero** | verified independently |

## What is NOT established (be honest about these)

1. ~~**No necessary-and-sufficient condition exists yet.**~~ **SUPERSEDED 2026-08-19 (solo
   session, C48/C49):** N&S conditions exist exactly at the finite-description level (staircase
   node test; family-level (L2)), and it is now proved — modulo Lemma E's analytic closure (F1)
   — that no finite battery of linear measurements decides passivity for continuum profiles.
   The corner-mass lead (N1) is closed: corner mass is a linear functional, covered by C49.
2. **The potential reformulation may be a re-representation, not a criterion.** It is derived as an
   *identity*, so "passive ⟺ potential comparison" is ε″ ≥ 0 in a change of variables — logically the
   same move as Carathéodory–Toeplitz, which the Captain explicitly rejected. **Present it as a
   frame that enables the extreme-point theorem, not as the answer.**
3. **No cheaper test.** Eight routes died; best honest speedup **1.5×**. Reasons: you must read
   essentially all N samples, and the FFT dominates runtime.
   ⚠️ **Major caveat**: that argument assumes a function class admitting arbitrarily narrow spikes.
   **Any N-independent smoothness bound collapses the lower bound entirely — and real ε′ profiles are
   smooth.** For the physically relevant class the question is **open again**.
4. **"Passive" has no certified numerical definition.** Whole-domain minimum is blind (endpoints are
   pinned to zero by construction); a fixed interior window is blind (gain migrates to the seam as
   balance → 0). This gates every numerical claim. **P0, unfinished, top priority.**

## Known errors to not repeat

- A reported "20/24 accuracy" was measured on a component **that was never actually called**.
- A reported 1.70× speedup was `np.linspace` duplicating an endpoint — and an independent audit
  **reproduced the same bug** because both wrote the same idiom.
- A blind replication was dispatched with a **paraphrased** claim and rigorously tested the wrong
  construction.
- A confident mechanism ("truncation artifact") was asserted and was **entirely wrong**; the real
  cause was `tan(π/2)` overflowing at one grid node.
