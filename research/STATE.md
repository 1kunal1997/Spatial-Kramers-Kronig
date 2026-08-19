# STATE OF PLAY — read THIS, not the whole LOG

**Updated 2026-08-19, voyage paused.** `LOG.md` is now long append-only history. **Crew should read
this file plus `MISSION.md`, `BOARD.md` and `PROTOCOL.md` — NOT the full log.** Reading the log
end-to-end wastes context and is a hallucination source.

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

1. **No necessary-and-sufficient condition exists yet.** The a+b ≤ L rule is **sufficient only** —
   passive profiles exist with a+b up to 2L. The live lead: necessity may depend on **how much mass
   sits near the corners**, not on where the support ends.
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
