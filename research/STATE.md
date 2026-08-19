# STATE OF PLAY — read THIS, not the whole LOG

**Updated 2026-08-19 (second update: solo Fable session).** `LOG.md` is now long append-only
history. **Crew should read this file plus `MISSION.md`, `BOARD.md` and `PROTOCOL.md` — NOT the
full log.** Reading the log end-to-end wastes context and is a hallucination source.

## 🚩 CAPTAIN'S AMENDMENT 2026-08-19 — READ BEFORE ANYTHING ELSE

The Captain has read the solo-session answer and **adjusted the end goal**. Full text in
`MISSION.md` ("Captain's Amendment"), new victory conditions **V7** and **V8**. In brief:

- **GOAL A (V7, critical path):** since N&S conditions appear not to exist, the deliverable is a
  **mathematically rigorous and exact proof of why they cannot**. C49's current label —
  *"proved modulo a numerically-certified constructive ingredient"* — **does not satisfy V7**.
  Close Lemma E analytically, discharge the thickening hypotheses, and take C49 to `hardened`.
- **GOAL B (V8, after A):** the **cleanest, most succinct** set of one-sided conditions
  (sufficient-not-necessary and/or necessary-not-sufficient). This is a demand for **fewer**
  statements: the minimal spanning set, with the existing dozen recovered as corollaries. A
  thirteenth sibling theorem is still a mission failure.
- **V6 amended:** two-sidedness withdrawn (C49 proves it unachievable); **parsimony retained in
  full** and now binds Goal B.

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
