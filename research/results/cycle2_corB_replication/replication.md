# Replication of C25 ("Cor 4.B") — blind, independent

**VERDICT: MATCH**

## The claim

On `[-L, L]`, drop density `g(t) = -d eps'/dt >= 0` (eps' monotone non-increasing), balance
condition `M1 = ∫ g(t) tan(pi t / 2L) dt = 0`, support of `g` contained in `[-a, b]`, `a,b > 0`.
DiHiTI (`tmm_helper.ht_derivative`: `u = deps'/dx`, `v = Im(hilbert(u))`,
`eps'' = cumtrapz(v,x) - linear detilt`) produces `eps''(x)`.

Claim: **`eps''(x) >= 0` everywhere `⟺ a + b <= L`.** Includes the asymmetric case,
e.g. `a=L/4, b=3L/4`.

## Blindness statement

Did **not** open: `h9_potential_domination.md`, `h9_domination_check.py`, `h9_fixes.py`,
`"Responsive Notes - Derivation - 081826.pdf"`, `potential_theory_findings.md`,
`verify_passivity.py`, `potential_criterion.py`. Also deliberately did **not** open
`theory/passivity_2026Aug/profile_generator.py` even though it is listed as an allowed Wave-2
resource in `MISSION.md` — built an independent profile generator from scratch instead, to
keep every route maximally independent. Read only: `MISSION.md`, `PROTOCOL.md`,
`tmm_helper.py`.

The one piece of context taken from `MISSION.md` is the *existence claim* that DiHiTI equals a
logarithmic potential of the drop density (the "master identity"), quoted there as background.
I did not use that formula as given — I **re-derived** the periodic/circular Hilbert transform
of a point mass from scratch via Fourier series on the circle (shown below) and only afterward
noticed it has the same log-potential structure. This is disclosed for transparency; the
derivation itself, the sign/scale calibration, and all numerics are original.

## Agreement tolerance (stated before looking at results — see `CHECKPOINT.md`)

- Sign-flip agreement with the `a+b<=L` threshold: required on all constructed cases outside a
  `|a+b-L| < 0.02L` boundary buffer.
- Analytic (route 1) vs. numeric (routes 2/2b, via `ht_derivative`): same sign, converging
  magnitude, as bump width narrows toward the point-mass limit.
- Convergence: interior min `eps''` stable to <1% between `N=16001` and `N=32001`.
- Interior window: full domain minus a small margin at the structurally-pinned endpoints
  `±L` (not a fixed 0.95L cut — see "ruled out" below for why that was corrected).

## Route 1 — analytic (strongest): two-atom limit, derived from scratch

Circular Hilbert transform of a periodic delta, re-derived via Fourier series (period `2L`,
not copied from any Wave-2/3 source):

```
H[delta_P(.-t0)](x) = (1/2L) cot( pi(x-t0)/(2L) )
```

Calibrated sign/scale against `scipy.signal.hilbert` on a narrow test bump:
**median ratio = 1.000663** (0.066% off unity) — confirms both sign (+1) and the `1/(2L)`
normalization independently, before looking at any pass/fail result.

For `g = A0 δ(t-t0) + A1 δ(t-t1)`, `t0=-a`, `t1=+b` (atoms at the box corners — the extremal,
hardest case), the closed form is:

```
eps''(x) = (A0/pi) ln[cos(pi t0/2L)/|sin(pi(x-t0)/2L)|] + (A1/pi) ln[cos(pi t1/2L)/|sin(pi(x-t1)/2L)|]
```

with `eps''(±L) = 0` automatic (no detilt needed for this construction — an independent check
on V5), and `M1=0` solved in closed form: `A1 = -A0 tan(pi t0/2L)/tan(pi t1/2L)`.

**Swept 29 configurations**, 4 families (symmetric, `a=L/4,b=3L/4`, strongly asymmetric
`a=0.05L`, near-full-one-side `a=0.02L`), `(a+b)/L` from 0.6 to 1.4, densely sampled around 1.
Interior minimum found by direct grid search, excluding only a `1e-4 L` margin at `±L` and a
`1e-6 L` neighborhood of the two log-singularities.

**Result: 29/29 agree exactly with `a+b<=L`.** The sign of the interior minimum crosses zero
at `(a+b)/L = 1.0000` to machine precision in every family, e.g.:

| family | a | b | (a+b)/L | min eps'' |
|---|---|---|---|---|
| symmetric | 0.500 | 0.500 | 1.0000 | +1.4e-16 |
| symmetric | 0.501 | 0.501 | 1.0020 | -2.0e-3 |
| asym L/4,3L/4 | 0.250 | 0.750 | 1.0000 | +1.9e-17 |
| asym L/4,3L/4 | 0.251 | 0.751 | 1.0020 | -8.3e-4 |
| strong asym | 0.050 | 0.950 | 1.0000 | +1.8e-12 |
| strong asym | 0.050 | 0.951 | 1.0010 | -7.8e-5 |

Full table: `route1_results.json`. Script: `route1_analytic.py`.

## Route 3 — cross-check: is `scipy.signal.hilbert` really the periodic DHT?

Implemented an independent `O(N^2)` discrete Hilbert transform via explicit cotangent-kernel
circular convolution (matrix multiply, no FFT). Compared to `scipy.signal.hilbert` on random
signals, N=64/128/256: **max relative error 6e-16 to 4e-15** (machine precision). Confirms
`scipy.signal.hilbert` behaves exactly as the standard periodic/circular DHT with no
padding/windowing surprise — validates MISSION.md's foundational assumption 1 and underwrites
routes 1 and 2's use of it. Script: `route3_cross_check.py`.

## Route 2 — numeric, own construction: smooth (non-atomic) profiles through the production code

Built 8 smooth, compactly-supported `g>=0` families from scratch (single Hann bump, off-centre
variants, flat-top plateau, near-atom, 3-bump and random multimodal), `M1=0` enforced exactly
via a **closed-form single compensator bump** (own derivation: `M1` is linear in `g`, so a
compensator placed on the sign-correct side solves `A_comp = -M1_main/I_comp` exactly). Ran
these through **`tmm_helper.ht_derivative`** — the actual production DiHiTI code. Using it here
is not a blindness violation: it *is* the object whose passivity verdict the claim is about,
and the mission explicitly whitelists it as shared infrastructure.

**Sufficiency (`a+b<=L`): 104/104 PASS, zero mismatches**, across all 8 shape families, 3
asymmetry groups. This is a materially broader confirmation than route 1 — it shows
sufficiency holds for genuinely varied smooth shapes, not just the atom limit.

**Necessity (`a+b>L`): initial apparent problem, investigated and resolved (not tuned away).**
76/80 moderate-width `a+b>L` profiles came out PASS at first. Diagnosis: several shape families
(as first coded) did not actually reach the true box corners `-a,+b` — e.g. a compensator
spread over 15% of the box behaves like smeared mass, diluting the log-singularity effect that
drives `eps''<0`, so their *effective* support was narrower than the nominal `[-a,b]` — this is
not a mismatch with the claim, since `a+b<=L` is a **sharpness** threshold (there exists an
extremal failing profile at `a+b>L`), not a claim that every profile with that nominal box
fails.

Confirmed this interpretation directly with a **mass-normalized narrow-bump family**: two Hann
bumps at `t0=-a+w, t1=b-w` (touching the box corners as `w→0`), amplitude scaled `∝1/w` so
total drop mass stays fixed (matching route 1's `A0=1`), grid resolved with `w/dx>=80`
(up to N=1.6e6). As `w→0`, `min eps''` (from `ht_derivative`) **converges toward route 1's
exact analytic values**:

| (a, b) | route-1 exact | route-2b at smallest resolved w |
|---|---|---|
| 0.55, 0.55 (symmetric) | -1.0041e-1 | -9.838e-2 (w=0.001) |
| 0.30, 0.80 (asymmetric) | -4.1032e-2 | -4.022e-2 (w=0.001) |
| 0.05, 0.951 (near-threshold, strong asym.) | -7.791e-5 | -6.228e-5 (w=1e-4, N=1.6e6) — still converging, correct sign reached at w≈2e-4 |

Full tables: `route2_results.json`, `route2b_results.json`,
`route2b_mass_normalized_convergence.json`. Scripts: `route2_numeric.py`,
`route2b_width_limit.py`.

## Convergence in N

Checked at fixed shapes across N=2001/4001/8001/16001/32001 (`convergence_check.json`):
monotonic, smooth convergence, <1% relative change between N=16001 and N=32001 in all three
spot-checked cases (e.g. 1.8614e-5 → 1.8496e-5, 0.63% change). The mass-normalized
width-convergence study (route 2b) additionally demonstrates convergence in the orthogonal
limit `w→0` at fixed, adequately-resolved N (`N ~ 80·(2L/w)`).

## What would have been a genuine MISMATCH (and wasn't)

- Any route-1 (a,b) configuration where the sign of the interior minimum did not flip exactly
  at `a+b=L`. None found in 29 configurations.
- `scipy.signal.hilbert` disagreeing with the from-scratch `O(N^2)` cotangent-kernel DHT beyond
  floating-point noise. Not found (agreement to 1e-15).
- Any `a+b<=L` smooth profile (route 2) coming out non-passive. Not found in 104 cases.
- The mass-normalized narrow-bump limit (route 2b) failing to converge toward route 1's exact
  values, or converging to the *wrong sign*. Not found — all three test points converge with
  the correct sign, two are within 2% of the analytic value at the resolution reached, the
  third (hardest, near-threshold) is converging monotonically and had reached the correct sign
  with ~20% of the target magnitude at the largest grid this session ran (N=1.6e6); it was not
  pushed further only because of diminishing computational returns, not because of doubt about
  the direction of convergence.

## Honest residual gap

The near-threshold, strongly-asymmetric mass-normalized case (`a=0.05, b=0.951`,
`(a+b)/L=1.001`) had NOT fully converged to the route-1 reference value at the finest
resolution run in this session (`w=1e-4`, `N≈1.6e6`: got -6.23e-5 vs. reference -7.79e-5, about
80% of the way there and closing monotonically). This is expected — the margin over threshold
is tiny (0.1%) and the compensator mass is tiny (~0.6% of the main atom's), so far more
resolution is needed to see the asymptotic value cleanly. This does **not** weaken the verdict:
route 1 already gives an exact, machine-precision closed-form answer for this exact
configuration (`-7.79e-5`, negative, correctly classified FAIL), and route 2b's job was only to
independently confirm the *direction* and *approximate scale* of convergence through the
production code — which it did (correct sign reached, magnitude closing in monotonically).

## Files

- `route1_analytic.py`, `route1_results.json` — analytic two-atom closed form (route 1)
- `route2_numeric.py`, `route2_results.json` — smooth-profile sweep through `ht_derivative` (route 2)
- `route2b_width_limit.py`, `route2b_results.json`,
  `route2b_mass_normalized_convergence.json` — width-limit convergence study (route 2b)
- `route3_cross_check.py` — direct O(N^2) DHT cross-check (route 3)
- `convergence_check.json` — N-convergence spot checks
- `CHECKPOINT.md` — full resumable work log with every intermediate finding, including the
  diagnosis process for the initial apparent route-2 discrepancy
- `manifest.json` — this file's provenance
