# HANDOFF — solo Fable session 2026-08-19: the N&S question, answered

**Read this first, then `research/theory/ns_answer.md` (the full document with proofs).**
This file is the pickup point: what was accomplished, what is fragile, and exactly what a
successor crew should do next. STATE.md, BOARD.md and CLAIMS.md have been updated to match.

## What the session was asked

FABLE_BRIEF's ONE QUESTION: find conditions on ε′ that are **necessary AND sufficient** for the
DiHiTI partner ε″ ≥ 0 — two-sided, parsimonious, checkable without effectively computing ε″ —
or prove rigorously why that is not possible.

## The answer delivered (both halves)

**Where two-sided finite criteria EXIST — and they do:**
1. **Per profile, for every finitely-described ε′** (n-layer staircase = every fabricable
   profile): passivity ⟺ all real roots of one explicit degree-≤(n+1) polynomial Q give
   Δ ≤ 0. Exact, grid-free, transform-free, two-sided. (This is h9's L3, promoted to its
   proper role; controls re-run clean.)
2. **Per family** cut by finitely many linear constraints: (L2) extreme-point collapse
   (a+b ≤ L / C25′ is the flagship instance).

**Where they PROVABLY DO NOT (new theorems, `ns_answer.md` §3–5):**
- **Theorem 1 (density):** any test set of positions that is sound AND complete must be dense —
  a violation can be confined inside any open window. Upgrades C27/K17 (no finite node set)
  to "no non-dense set". Proof complete; verified (e1).
- **Theorem 2 (no finite battery — the main result):** for ANY k functions φ₁..φ_k and ANY
  decision rule F, the rule F(∫φ₁dg, …, ∫φ_kdg) fails: two positive balanced atomic measures
  exist with the SAME atoms, identical readings, opposite verdicts. Covers all moments,
  samples, corner masses, the full necessary cascade (mass, M₁, M₂, σ), any thresholds/learned
  rules on top. Proof: convexity/saturation forces every active facet normal into the battery's
  k-dim row space (Lemma D), while the active points sweep a continuous interval (Lemma E,
  constructive) and their facet normals span everything (Lemma A, log-singularity + harmonic
  unique continuation). Explicit witnesses built and double-verified for k = 8 (containing the
  ENTIRE known necessary cascade; readings match to 6e−16, verdicts −7.8e−3 GAIN vs PASSIVE)
  and k = 16 (match 8e−12, GAIN −1.6e−5, production-confirmed at N up to 2²²).
- **Corollaries:** no moment-type criterion; no corner-mass criterion (board N1 CLOSED);
  no criterion reading ε′ on a sub-window.
- **Proposition 3 (why every attempt was a re-representation):** the passive cone is a linear
  image of the nonnegative cone — a simplex cone with UNIQUE extremal representation, and the
  representing measure IS ε″. CT, potential form, Fejér–Riesz were all "the transform in other
  coordinates" because for this cone there is nothing else to be.

**V3/V6 adjudication:** the "definition" exists and is (L1) + its two finite collapses;
the theorems prove nothing shorter is available for continuum profiles. The one-sided-zoo era
can be declared over for a principled reason, not fatigue.

## Byproducts (each independently valuable)

- **T2/CFM conjecture (board N2) REFUTED** with a one-line counterexample:
  f = cos(x)e^{−y} is harmonic entire, ω-integrable, and passive δ₀ violates the subharmonic
  order (1 > e^{−1}). The correct test cone is exactly the log-potential cone — CFM does not
  apply verbatim. Corrected statement in ns_answer.md §7; Rost/Cauchy-process reading recorded
  as a labelled conjecture for the librarian.
- **Lemma B (no exterior gain):** balanced positive drops have Δ < 0 strictly outside their
  Cayley support hull — closes red-team residual risk #3 (outer-branch monotonicity) with a
  proof, and localizes all possible violation to one compact interval.
- **Production pipeline hazard (feeds P0):** `ht_derivative` on smoothed staircases carries a
  profile-dependent additive O(h) offset (s-independent, halves per N-doubling; +3.55e−5 at
  N=2¹⁹ for the k=16 witness), which CANCELS in differences (differential matched closed form
  to 4 digits at every N). At corpus-historical N this offset is ~1e−3 — same order as the
  C21 interior margin. Near-threshold absolute production verdicts are convention-limited.
- **Red-team's λ*(a+b) table used unrecorded corner splits** (1.30L row = asymmetric
  (−0.4,+0.9); 1.80L, 1.98L rows = symmetric; 1.40L/1.94L match neither convention I tried) —
  re-derive before reuse.

## Verification state (everything has a control)

e0: evaluator controls — Cor 4.A threshold = 1.000000000 (3 asymmetries), C40 slopes to 4
digits, production agreement 3–4 digits on gain cases, red-team λ* 0.262/0.278 reproduced by an
independent closed-form route. e1: δ=0 pins to (η₀, 0) at machine precision. e2: non-kernel
control moves readings 14 orders above the matched pair's residual. e3: 4/4 λ* controls.
e4: double consistency check per θ (sup=0 AND argmax=θ), nondegeneracy Δ″ < 0 across window.

## Fragilities / what a referee will attack (in priority order)

1. **Lemma E's analytic closure.** The active-point interval [1.45, 2.96] is numerically
   certified (152 grid θ, exact node test at each), not yet proved symbolically. The envelope
   weights are explicit rational expressions in Δ_j(θ), Δ′_j(θ) — closing this is routine
   real-analysis grind. Until then Theorem 2's label is "proved modulo a numerically-certified
   constructive ingredient". **This is the single most valuable next theory task.**
2. The thickening step (boundary points with 3 positive masses → interior of ℝⁿ orthant) is a
   continuity argument written with hypotheses; make it fully explicit.
3. Prop 3 leans on inherited Thm 6.1 (inverse-map surjectivity), still at `conjecture` rung.
4. Theorem 2's witnesses match readings to float precision, not exactly; the exact matching is
   what the proof gives. State it that way whenever quoted.

## Exact next steps for a successor crew

1. **theorist:** close Lemma E analytically (see 1 above). ~half a day of grind.
2. **red-team:** attack Theorem 2 — try to build a battery+rule that beats the construction
   (e.g., exploit the float mismatch 8e−12, or non-atom-supported functionals like sup supp g —
   note the same-atoms property already blocks support-reading rules).
3. **replicator (blind):** give ONLY the k=8 witness parameters from `witness_k8.npz`
   (20 atoms, two weight vectors) + the claim "identical battery readings, opposite verdicts";
   have it recompute both with its own evaluator.
4. **P0 owner:** fold in the O(h) production-offset finding (§8 of ns_answer.md) — it sharpens
   the case that absolute near-threshold production verdicts need the exact node test or an
   extrapolated convention.
5. **librarian:** check the Rost / Cauchy-process potential-kernel identification (§7 remark);
   also whether "no finite linear battery decides membership in a simplex cone with dense
   exposed normals" exists in the convex-geometry literature (it may be publishable standalone).
6. **Captain decision:** whether to fold Theorems 1–2 + Prop 3 into the mission write-up as
   the formal closure of V3/V6 fallback ("exact N&S exists at exactly the finite-description
   level, provably not below").

## File map

- `research/theory/ns_answer.md` — the document (proofs, tables, status labels).
- `research/results/fable_ns_answer/common.py` — evaluators (exact node test + production wrap).
- `e0_controls.py` … `e4_lemmaE.py` + `*_output.txt` — experiments with canonical outputs.
- `witness_k8.npz`, `witness_k16.npz` — the Theorem-2 witness pairs (atoms, both weight
  vectors, readings, active point).
- `CHECKPOINT.md` — session log including dead ends (T2-as-full-Choquet died here first).
- Ledger updates: STATE.md (established/not-established), BOARD.md (N1, N2 closed; new rows),
  CLAIMS.md (C48–C54).
