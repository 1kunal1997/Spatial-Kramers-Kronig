# HANDOFF — solo Fable session, Goal B (V8), 2026-08-20

**Deliverable: `research/theory/minimal_oneside.md`.** Read that first; this file is the
crew-facing status + next steps. Claims registered: C57–C60.

## What was delivered (one paragraph)

V8 is answered in the form the brief allowed for: **two parents plus a proof that two is
where compression stops.** Theorem S (sufficient parent: balanced half-period box ⊕ even
unimodal, closed under new Cayley-contraction and inversion covariances) absorbs C3, C25′,
C4, C5, C6, C7 as 1-line corollaries/evaluations. Theorem N (necessary parent: passivity
on the closed outer half + centre) absorbs the C8 cascade, C9, C13 as its seam jet, with
σ ≥ 0 identified as the ω-average of the full law (and proved independent of N by explicit
witness). **Proposition NG (new, proved): the passive cone is non-polyhedral on every
Lemma-E2 atom set, so no finite list of generator families can ever span it** — the exact
primal mirror of C49, proved by the same Lemma E2. The S↔𝒫 gap is structured: Theorem S-τ
(new, proved from E1′): inversion-symmetric drops with radii in [L/4, 3L/4] are passive,
sharp at L/4 (arctan(√2−1) = π/8) — C3's middle-half motif regenerated on the radius axis;
a further two-scale rung has a closed-form threshold ln(1/β₁)/ln(β₂) (numerics, 7 digits).
Gap quantified: N passes 1.5% false; S certifies 39.8% of passive atomic profiles; exact
bands (L/2, 2L/3] (N side) and [L/4, L/2) (S side).

## Status labels

- **proved:** Lemmas D & I (+corollaries), Theorem N + corollary derivations, Theorem S +
  corollary derivations, Proposition NG, S-τ-disjoint-from-S, the N band 2L/3, σ⊥N.
- **proved modulo C55's rung:** Theorem S-τ, Proposition NG (both lean on Lemma E1′/E2
  of `lemmaE_closure.md`, currently `supported`, red-team pending).
- **verified numerically only:** ladder-witness threshold = τ-balance point (7 digits, one
  family); all ensemble percentages.
- **killed (do not re-walk):** radial-CDF dominance; "outer+centre ⟹ passive" in every
  symmetry class; (inherited) balanced-pair cone, CFM order.

## Next steps for the crew, in priority order

1. **Red-team the F1/C55 chain** (already queued — note it now also gates C58/C59 here;
   the stakes went up, not down).
2. **Red-team C57–C59.** All three are short and self-contained: Lemma D/I are two-line
   algebra + machine-zero controls; NG is one page leaning only on C55's Lemma E2 + A″.
   The most attackable point is NG's Baire/face-descent step — check it.
3. **Referee pass on `minimal_oneside.md` against V8's mechanical test**, then Captain
   adjudication of V8. The count claim to check: 10+ one-sided statements → 2 parents +
   2 no-gos, with S-τ/witness positioned inside the gap analysis, not as new screens.
4. **Cheap closes left open** (§9 of the doc): (a) prove "binding at the centre" for the
   two-scale family (would make the third rung exact, closed form m₂* = ln(1/β₁)/ln β₂);
   (b) mollify NG/Theorem-2 witnesses to smooth profiles (shared caveat); (c) eventually
   merge C49 + NG into one statement ("no finitely-described one-sided theory is
   complete") for the write-up.
5. **P0 note:** the S-τ sector supplies a *one-parameter family* of exactly-known
   boundary profiles (Δ(0) = 0 facet, touching only at the centre for u ∈ (L/4, L/2)),
   generalizing the single C55 calibration point — useful calibration set for the
   numerical passivity convention.

## Files

- `exp1_parents.py/.txt` — candidate kills + controls
- `exp2_gap.py/.txt` — covariance lemma controls, asymmetric N-redundancy kill, √3 band,
  reciprocal gap map (`gapmap_reciprocal.npz`)
- `exp3_even_reduction.py/.txt` — even-case N-redundancy kill (the decisive one)
- `exp4_ladder_gapstats.py/.txt` — ladder witness, S-τ control, gap percentages
- `manifest.json` — seeds, claims, kill list
