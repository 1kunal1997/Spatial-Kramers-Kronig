# F5 audit — `research/results/fable_ns_answer/` — CLAIMS C48–C54

**VERDICT: CLEAN (with one bounded, non-fatal caveat on production corroboration for C49/C53).**

No finding in this audit changes a proof, a theorem statement, or a passivity verdict of any
witness. One finding narrows the safe operating range of the *secondary* (production) evaluator
and refines C53's mechanism claim; it does not touch the primary (exact, grid-free) evaluator that
every C48/C49 verdict actually rests on.

All numbers below were re-derived by an independently-constructed method (never just re-running
`common.py`'s own function a second time) — see `manifest.json` for the list of independent routes
used, and the control stated inline for each number.

---

## C48 — Theorem 1 (density of essential test points). **CLEAN.**

Reran `e1_density.py` verbatim; reproduces its printed numbers exactly (own log:
`e1_output_rerun.txt`-equivalent, not saved to disk but values transcribed below match).

**Lead checked: non-monotone positivity-set width in η-space for η₀=8** (widths 10.30 → 43.46 →
7.79 → 3.41 → 1.86 as δ = .1→.001 — genuinely non-monotone, confirmed by rerun).

- Independent re-expression in x-space (via `t_of_xi`, a function not otherwise exercised by
  `e1_density.py`): 0.290 → 0.0837 → 0.0544 → 0.0305 → 0.0177 — **monotone**, exactly matching the
  audit brief's own diagnosis. Control: this is a coordinate change of the *same* numbers, so
  agreement is expected if and only if the non-monotonicity is a Cayley-stretch artifact rather
  than a real non-convergence; monotone shrinkage in x confirms that.
- Pushed δ down two more decades (δ = 3e-4 → 1e-6) for η₀=8: width resumes monotone shrinkage in
  η-space too (1.86 → 1.00 → 0.575 → 0.315 → 0.182 → 0.099 → 0.057), i.e. the coarse-δ wobble is
  transient, not a failure to converge. Control: the sequence is monotone once δ is small enough
  that both coordinates agree on direction — cross-checks the x-space result above from a second
  angle.
- Read the actual proof (`ns_answer.md` §3): it needs only *pointwise* convergence
  Δ_{p_s}(η) → Δ_{p₀}(η) < 0 for fixed η ≠ η₀ as s → 0, plus unimodality (from Lemma B, proved
  elementary and independently re-verified below under C50). It does **not** require monotone
  shrinkage of the positivity interval. The coarse-δ non-monotonicity, even if it had not resolved,
  would not have broken the proof.
- `while f(eL) > 0: eL -= step; step *= 1.5` bracket-expansion loop: no case found (across all three
  η₀ targets, five δ values each) where it needed more than a handful of steps or showed signs of
  skipping a second sign change. This is consistent with Lemma B's proof of strict unimodality
  between atoms for balanced pairs specifically — the only construction this script uses — so the
  loop's single-sign-change assumption is analytically justified for this family, not merely lucky.

**Verdict: CLEAN.** No numerical hygiene defect found; the flagged non-monotonicity is real but
inert.

---

## C49 — Theorem 2 (no finite battery), witnesses k=8 and k=16. **CLEAN on the exact-evaluator
side. One real, bounded caveat on the *production* corroboration (see also C53 below).**

### `critical_etas` magic-number sensitivity (Lead 1a)

Swept `trim_tol` ∈ [1e-14, 1e-4], `imag_tol` ∈ [1e-11, 1e-3], `atom_tol` ∈ [1e-14, 1e-6] on the
k=16 witness (36 atoms, the stated most-delicate case), independently reimplementing
`critical_etas`/`exact_sup_gap` with tolerances as parameters (not editing `common.py`).

- `imag_tol`, `atom_tol`: **zero sensitivity** anywhere in the tested range (7+ orders of magnitude
  each) — verdict and sup-Δ value identical to all printed digits.
- `trim_tol`: identical result from 1e-14 to 1e-8 (6 orders of magnitude around the default
  1e-12); only degrades at 1e-6 (loses 3 digits, verdict unchanged) and fails outright at 1e-4
  (wrong verdict) — both **4–6 orders of magnitude looser** than the chosen default, far outside
  any plausible float64 noise floor.

**Control:** the default (1e-12/1e-9/1e-12) sits in the middle of a wide, flat safe plateau, not
at an edge. Not fragile.

### Does `polyfromroots` over 36 clustered atoms lose accuracy? (Lead 1b)

Built a **fully independent** evaluator using `mpmath` at 50 decimal digits: closed-form
Δ′(η) = Σ wⱼ/(η−ξⱼ) − Wη/(1+η²), scanned for sign changes on a fine grid, refined each by
high-precision bisection (no polynomial rooting anywhere), then evaluated Δ at each critical point
in 50-digit arithmetic.

| witness | common.py (float64, polyroots) | independent (mpmath, deriv. root scan) |
|---|---|---|
| k16 gG | sup Δ = +5.040215e-05 @ η=0.956590 | sup Δ = 5.04021517805e-05 @ η=0.956590018765 |
| k16 gP | sup Δ = −5.039394e-05 @ η=0.956557 | sup Δ = −5.03939353645e-05 @ η=0.956557248803 |
| k8 gG  | sup Δ = +2.436531e-02 @ η=1.5281 | sup Δ = 0.0243653087674 @ η=1.5281486797 |
| k8 gP  | sup Δ = −2.421095e-02 @ η=1.5177 | sup Δ = −0.0242109517746 @ η=1.5177495497 |

Agreement to 9–10 significant figures on an algorithmically unrelated method. **This is the direct
control for lead #3 ("is k=16 real or float noise?") — it is real**, confirmed independently to
far higher precision than needed to distinguish the two verdicts.

### k=8 battery rank 7, not 8 (Lead 2)

Confirmed: the rank deficiency lives in the mass-vector kernel construction (the particular 5
bad + 5 good pair family spans only a rank-7 image under the 8 functionals as *columns* of the
battery matrix) — it is not a statement that the 8 functionals fail to be read out independently.
Reimplemented the 8 functional readings from scratch, directly off `witness_k8.npz`'s raw atoms
and weights, with fresh hand-written summation code (not calling `apply_battery`):

```
max |readings(gG) - readings(gP)| (absolute) = 7.1e-15
max relative diff = 6.11e-16   (matches common.py's own printed 6.11e-16 exactly)
```

All 8 functionals individually match to float precision, independent of the rank-7 detail; dim
ker = 3 ≥ 2 required either way. **Confirmed harmless, exactly as the brief suspected.**

### Production corroboration of the k=16 witness — the one real finding

This is the substantive result of this audit, found by applying the brief's "vary the
construction" mandate (grid conventions) to the production cross-check rather than only to the
exact evaluator.

`common.py::production_min_eps2` defaults to `grid="linspace"`, i.e. `np.linspace(-L, L, N)` — the
exact idiom flagged project-wide as the historical common-mode bug (endpoint duplication under FFT
periodic continuation; `research/PROTOCOL.md`). `common.py` also offers, but does not default to,
`grid="periodic"` (`x = -L + 2L·arange(N)/N`, no duplicated endpoint).

Swept N with `common.py`'s own `production_min_eps2` (not reimplemented — this part specifically
audits the shipped function) on the actual k=16 GAIN witness (`gG`), `s=6e-4` as used by
`e2_battery.py`:

| N (grid=linspace, the shipped default) | reported min | verdict |
|---|---|---|
| 2^17 – 2^19.58 (131072…786432) | +2e-8 … +1e-9 (boundary quadrature floor) | **PASS — WRONG (true: GAIN)** |
| 2^20 (1048576) | −1.15e-06 | GAIN (correct) |
| 2^21, 2^22 | −9.0e-06, −1.29e-05 | GAIN (correct, converging to exact −1.604e-05) |

With `grid="periodic"` (same `s`, same witness, same code — just the other already-present grid
option):

| N | reported min | verdict |
|---|---|---|
| 2^14 (16384) | −1.74e-05 | GAIN (correct) |
| 2^15 … 2^20 | −1.70e-05 … −1.688e-05 | GAIN (correct throughout, converging to −1.604e-05) |

**Mechanism** (traced by comparing `tmm_h.ht_derivative` output directly against the exact
closed-form profile at fixed x, sweeping N from 2048 to 2^25 with both grids): the discrepancy
decomposes into two additive pieces.

1. An O(h) term specific to `grid="linspace"`, verified directly: `N·dx − 2L` (the period mismatch
   an FFT-based Hilbert transform sees) is exactly `+dx` for linspace and `0.000e+00` for periodic,
   confirmed at N = 2048, 2049, 4096. This term dominates at moderate N, decays roughly like h
   (log-log slope 1.18, doubling ratios 2.06→2.73 across N=2^19…2^23), and crosses zero somewhere
   between N=2^24 and 2^25 in an extended sweep — i.e. it genuinely is the O(h) effect C53
   describes, but it is a **property of the linspace grid choice**, not of `ht_derivative` itself.
2. A smaller, grid-independent residual floor (~−8.3e-7 for this witness/s combination), present
   with *either* grid at large N and not shrinking further — an as-yet-unidentified but small
   algorithm-level bias (plausibly in the erf-smoothing / `cumulative_trapezoid` + linear-detrend
   pipeline). This matches HANDOFF's own "mechanism unidentified" honesty and is not large enough
   to threaten any reported verdict at the N values actually used.

**Consequence, precisely stated:** CLAIMS.md's phrasing — "k=16 … production-confirmed at
N=2²⁰–2²²" — is **accurate as scoped** (it does not claim lower N works) and is not overclaimed.
But the safety margin is thin: **one power of two below the stated range (N=2^19, a value used
elsewhere in this very session's own `e0_controls.py` for a different case) the shipped default
grid gives the wrong verdict for this flagship witness.** This does not change C49's truth (the
exact evaluator, independently reconfirmed above to 9–10 digits, is what the theorem rests on),
but it means the "production-confirmed" language should not be read as robust corroboration —
it is corroboration that only starts working within a factor of 2 of where it was tested, and only
because the shipped default grid convention is the less accurate of the two options already present
in the same file. Switching to `grid="periodic"` (already implemented, unused by default) would
have given a correct, well-converged answer 64× smaller N and with roughly 3 orders of magnitude
better accuracy at matched N (e.g. N=2048: linspace absolute error +8.03e-3, periodic −3.47e-5,
checked over a 5-point window to rule out a single-point fluke).

**"Cancels in differences" claim, checked independently with both grids:** production(gG) −
production(gP) matches the exact closed-form difference to 4 significant figures (ratio 1.0002) at
every N from 2^19 to 2^22, with **either** grid convention. This is the part that actually
underwrites the "same readings, opposite verdicts" comparison used elsewhere, and it is robust.

**Verdict: CLEAN for the theorem and both witnesses (independently reconfirmed to 9–10 digits by
an unrelated method). DEFECTIVE-adjacent for the production corroboration of k=16 specifically**
— narrow margin, root cause identified (linspace grid, the historically-flagged bug pattern),
fix available in the same file (`grid="periodic"`) but not the default. **Blast radius: zero
change to any claim's truth value; changes how much weight "production-confirmed" should carry
for the k=16 witness, and flags a landmine for anyone rerunning this at a smaller N.**

### Dead code (Lead 6)

`common.py:90` `Q = npoly.polymul([1.0, 0.0, 1.0], np.zeros(1))` is immediately overwritten at
lines 95–96. Confirmed dead; zero numeric consequence (the line's output is never read). No other
dead-code instances found nearby in `critical_etas`/`exact_sup_gap`/`exact_min_eps2`. Cosmetic
only.

---

## C50 — Lemma B (no exterior gain). **CLEAN.**

The proof (`ns_answer.md` §1) claims: for a unit-mass balanced pair, on (β,∞) the numerator of Δ′
exceeds γ(1+β²) > 0. Re-derived by hand: numerator N(η) = η(1+βγ) + (γ−β); N(β) = β(1+βγ)+(γ−β) =
β²γ + γ = γ(1+β²) exactly, and N′(η) = 1+βγ > 0 for β,γ>0, so N(η) > γ(1+β²) > 0 for all η>β.
Algebra checks out exactly as stated; no numerical claim requires a separate control beyond this
(elementary, checkable by hand). Independently reused as the basis for confirming C48's unimodality
assumption above.

## C51 — N1 closure (corner-mass cannot be a criterion). **CLEAN.**

`e3_n1_closure.py` rerun verbatim, reproduces its `e3_output.txt` exactly (λ*=0.2620/0.2778 vs
red-team's 0.262/0.278; killer pair PASSIVE/GAIN as claimed).

`lam_star()` uses a dense (400,001-point) grid **uniform in t-space** (via
`tan(π/2·linspace(...))`), a design choice that already avoids the η-space Cayley-stretch artifact
seen in C48. Independently stress-tested the sharpest case in the sweep (corner (−0.4,+0.9), bulk
(−0.24,+0.74), reported λ*=0.0658) with (a) a 10× denser grid and (b) a local
`scipy.optimize.minimize_scalar` refinement seeded near the grid optimum — both agree with the
reported value to 5 significant figures (0.065757). Grid resolution is adequate; the sharp
near-threshold falloff (λ* dropping from 0.2595 to 0.0302 as bulk sum a+b goes from 0.90 to 0.99)
is real, not a resolution artifact.

## C52 — T2/CFM refutation (counterexample). **CLEAN.**

Independently numerically checked both halves of the one-line counterexample:
- ∫cos(x) dω_i(x) with ω_i the Cauchy/Poisson harmonic-measure density (1/π)/(1+x²): numerical
  quadrature gives 0.36777 vs claimed e⁻¹ = 0.36788 (agreement to 3–4 digits; residual is quadrature
  truncation, a scipy IntegrationWarning on this genuinely slowly-decaying oscillatory integral, not
  a defect in the claim — the exact value is the well-known Fourier transform of the Cauchy density
  at unit frequency).
- δ₀ passivity: Δ(η) = ln|η| − ½ln(1+η²) checked on a 2,000,000-point grid over η ∈ [−10⁴, 10⁴];
  max value found is −5.0e−9, i.e. strictly negative everywhere finite, approaching 0 only as
  η→±∞ — confirms δ₀ is passive, as the proof requires. ∫f dδ₀ = f(0) = 1 > e⁻¹ trivially.

Both legs of the counterexample independently confirmed.

## C54 — Simplex rigidity. **No dedicated numerical claim; not applicable to a numerical audit.**

Elementary/interpretive, explicitly leaning on inherited Theorem 6.1 at `conjecture` rung (already
flagged as such in the source document, not something this session's numerics newly assert).
Nothing here to audit beyond what C48/C49/C50 already establish about the uniqueness of the
Δ↔ε″ correspondence.

---

## Coverage summary — what was checked and found clean

- `e0_controls.py`: reran verbatim, all four control blocks (C0.1–C0.4) reproduce exactly.
- `e1_density.py`: reran verbatim; lead #5 traced to ground and shown harmless via an independent
  coordinate transform.
- `e2_battery.py`: k=8 and k=16 witnesses both independently reconfirmed via mpmath (unrelated
  algorithm) to 9–10 digits; rank-7 lead traced and shown harmless via independent readout
  reimplementation; tolerance sensitivity of `critical_etas` swept 6+ orders of magnitude around
  every one of its three magic numbers with no fragility found; dead code confirmed cosmetic; grid
  convention of the production cross-check identified as a real, bounded fragility (see C49 above).
- `e3_n1_closure.py`: reran verbatim; sharpest λ* value independently refined by two different
  methods (denser grid + local optimizer), agrees to 5 digits.
- `e4_lemmaE.py`: reran verbatim; the "contiguous window [1.45, 2.96], no gaps" claim independently
  re-verified at the level of the raw accepted-θ list (not trusting the script's own gap-check
  logic) — literally every grid θ in [1.45, 2.96] (step 0.01, 152 points) is accepted and none
  outside is, confirming the window is genuinely contiguous at the tested resolution. Correctly
  labelled "numerically certified, not proved" in `ns_answer.md` — not overclaimed.
- `witness_k8.npz`, `witness_k16.npz`: contents independently reconstructed and reconfirmed by two
  unrelated routes each (mpmath sup-Δ recomputation; hand-written battery-readings recomputation).

## What was NOT exhaustively covered

- Did not attempt to close Lemma E analytically (board item F1) — out of scope for a numerical
  audit, already correctly labelled as the open item in `ns_answer.md`.
- Did not sweep every possible (a,b) pair combination in `e3_n1_closure.py`'s part (b) table beyond
  the single sharpest case; the other entries were only checked by rerun-reproduction, not
  independently rederived. Low priority — part (b) is illustrative range-of-λ* evidence, not a
  load-bearing witness the way the k=8/k=16 batteries are.
- Did not independently verify the T2/CFM "labelled remark" (Rost/Cauchy-process identification) —
  explicitly marked in the source as conjectured and non-load-bearing.
