# CHECKPOINT — replicator, C25 "Cor 4.B" replication

## GOAL
Blind independent replication of claim C25 ("Cor 4.B"):
On [-L,L], g(t) = -deps'/dt >= 0 (eps' monotone non-increasing), balance condition
M1 = INT g(t) tan(pi t / 2L) dt = 0, support(g) subset [-a,b], a,b>0.
CLAIM: eps''(x) >= 0 everywhere  <=>  a + b <= L.
(eps'' produced by DiHiTI = tmm_helper.ht_derivative: u=deps'/dx, v=Im(hilbert(u)),
eps''=cumtrapz(v,x)-linear detilt.)

Must NOT read: h9_potential_domination.md, h9_domination_check.py, h9_fixes.py,
"Responsive Notes - Derivation - 081826.pdf", potential_theory_findings.md,
verify_passivity.py, potential_criterion.py. Have NOT opened any of these. Have read
MISSION.md, PROTOCOL.md, and tmm_helper.py only (all explicitly allowed).

## AGREEMENT TOLERANCE (stated BEFORE looking at results)
- Sign-flip test (pass/fail vs a+b<=L threshold): must agree on ALL constructed cases
  except within a "near-boundary" buffer of |a+b-L| < 0.02*L, where numerical detection
  of a borderline sign (interior min eps'' within ~1e-3 of 0, matching the mission's
  noted ~1e-3*D margin scale) is expected to be ambiguous. Any clear mismatch outside
  that buffer is a MISMATCH.
- For the analytic 2-atom closed form vs the full FFT DiHiTI pipeline (tmm_helper.ht_derivative)
  on a matching smoothed profile: interior-min eps'' agree to 5% relative (they are not
  identical setups -- atoms vs smoothed bumps -- so exact numerical match is not expected,
  only same sign and same order of magnitude).
- Convergence: interior min eps'' from ht_derivative must stabilize to <1% change between
  N=16001 and N=32001 grid points to be trusted as converged.
- Interior window for "interior minimum": [-0.95L, 0.95L] (excludes the structurally-pinned
  zero endpoints per mission's blind-indicator warning).

## PLAN (three independent routes, strongest first)
1. ANALYTIC (route 1): derive circular/periodic Hilbert transform of a point mass (delta
   in g) from scratch via Fourier series on the circle (period 2L). Get closed form
   eps''(x) for TWO-ATOM g = A0*delta(t-t0) + A1*delta(t-t1), t0<0<t1, with M1=0 solved
   in closed form for A1/A0. Sweep t0,t1 (i.e. effectively a,b) across a+b<L, =L, >L and
   check sign of eps'' in the interior directly from the closed form (no FFT, no numerics
   library needed beyond a 1D grid evaluation of the closed-form expression).
2. NUMERIC, own construction (route 2): build many SMOOTH compactly-supported g>=0 profiles
   (Hann bumps, plateaus, multimodal) with support in [-a,b], enforce M1=0 exactly via a
   linear compensator bump solved in closed form, integrate to eps'(x), and run through
   tmm_helper.ht_derivative (the actual production DiHiTI code -- this IS the thing whose
   passivity verdict the claim is about, so using it as ground truth is correct and is not
   "reading the derivation"). Check convergence in N. Sweep a+b/L across <1, =1, >1 with
   several shapes and strong asymmetry (e.g. a=L/4,b=3L/4).
3. CROSS-CHECK (route 3, weaker): implement my own O(N^2) direct summation discrete Hilbert
   transform via the cotangent kernel (NOT scipy.signal.hilbert), verify it agrees with
   scipy's hilbert() to numerical precision on a few test signals. This validates that
   scipy's FFT hilbert() is behaving as the expected periodic/circular Hilbert transform
   (no padding/windowing surprises), giving confidence in route 2's use of ht_derivative.

## ESTABLISHED SO FAR
- **Route 1 (analytic, two-atom limiting case) DONE, MATCHES.** Derived from scratch (Fourier
  series on the circle, period 2L) the closed form for eps''(x) produced by DiHiTI when
  g(t) = A0*delta(t-t0) + A1*delta(t-t1), t0=-a, t1=+b (atoms at the boundary of the support --
  the hardest/extremal case). Full derivation is in the docstring of route1_analytic.py.
  Key intermediate facts, independently derived:
    - H[delta_P(.-t0)](x) = (1/2L) cot(pi(x-t0)/(2L))  [circular Hilbert transform of a periodic
      delta -- classical Fourier-series result, re-derived here, sign/scale calibrated against
      scipy.signal.hilbert() on a narrow test bump: median ratio = 1.000663, i.e. matches to
      6.6e-4 -- confirms both sign (+1) AND the 1/(2L) normalization independently.]
    - eps''_raw(+/-L) = 0 automatically for this construction (BEFORE detilt) -- an independent
      confirmation of V5 (endpoint pinning) for point-mass drop densities, consistent with
      C21/K16's general finding.
    - Closed-form eps''(x) = (A0/pi) ln[cos(pi t0/2L)/|sin(pi(x-t0)/2L)|]
                            + (A1/pi) ln[cos(pi t1/2L)/|sin(pi(x-t1)/2L)|]
      (this has the same log-potential STRUCTURE as the "master identity" quoted in MISSION.md,
      but was derived independently here via Fourier series, not read from any Wave-2/3 file.)
  Swept a,b over 29 configurations across 4 families (symmetric, a=L/4&b=3L/4, strongly
  asymmetric a=0.05L, near-full-one-side a=0.02L) with a+b/L spanning ~0.6 to ~1.4, densely
  sampled near a+b/L=1. Interior min found by direct grid search over the FULL domain minus a
  1e-4*L margin at +/-L and a 1e-6*L neighborhood of the two singular points (log poles, not
  minima). RESULT: **all 29/29 cases agree exactly with the a+b<=L threshold**, and the sign
  change of the interior minimum crosses zero at a+b/L = 1.0000 to machine precision in every
  family tested (e.g. symmetric a=b=0.5: min eps''=1.4e-16 at a+b/L=1 exactly; asymmetric
  a=L/4,b=3L/4: min=1.9e-17 at a+b/L=1 exactly). This is a genuinely independent, essentially
  exact confirmation of BOTH directions of Cor 4.B (sufficiency AND necessity) in the point-mass
  (near-atom) limit, including the strongly asymmetric case the mission flagged (a=L/4,b=3L/4).
  Script: route1_analytic.py. Raw results: route1_results.json.

## RULED OUT
(none yet)

## ESTABLISHED SO FAR (continued)
- **Route 3 (cross-check) DONE, MATCHES to machine precision.** Implemented my own O(N^2)
  discrete Hilbert transform via explicit cotangent-kernel circular convolution (matrix
  multiply, NOT FFT) and compared to scipy.signal.hilbert on random test signals, N=64/128/256.
  Max relative error 6e-16 to 4e-15 (machine precision). Confirms scipy's hilbert() really is
  the standard periodic/circular DHT with no padding/windowing surprises -- validates
  foundational assumption 1 in MISSION.md and underwrites route 2's use of ht_derivative.
  Script: route3_cross_check.py.

- **Route 2 (numeric, own construction) DONE, MATCHES after resolving an initial apparent
  discrepancy.** Built 8 SMOOTH profile families (Hann bumps, plateau, multimodal, near-atom)
  from scratch, M1=0 enforced via a closed-form single compensator bump (own derivation, did
  NOT open profile_generator.py even though it's an allowed Wave-2 resource -- kept this route
  maximally independent). Ran through tmm_helper.ht_derivative (production DiHiTI code -- used
  as ground truth, this is legitimate per mission definition of "passive", not a blindness
  violation). 184 cases across 3 asymmetry groups (symmetric, a=L/4&b=3L/4, strong a=0.05L).
  SUFFICIENCY (a+b<=L): **104/104 PASS, zero mismatches**, across ALL 8 shape families --
  strong broad confirmation that ANY g satisfying the hypotheses with support box a+b<=L is
  passive, not just the extremal/atom case. Script: route2_numeric.py, route2_results.json.

  NECESSITY (a+b>L) initial apparent problem: 76/80 of my moderate-width a+b>L profiles came
  out PASS (did not fail), naively looking like mismatches. INVESTIGATED (not tuned away):
  root cause was that my shape generator's "reach" toward the true corners -a,+b was too
  diffuse/non-adversarial -- e.g. a compensator bump spread over 15% of the box behaves like a
  smeared mass, not a concentrated one, and dilutes the log-singularity effect that drives
  eps''<0. This is NOT a mismatch with the claim -- it reflects that a+b<=L is a SHARPNESS
  threshold (there exists an extremal failing profile at a+b>L, namely the concentrated/atom
  limit, not that literally every profile with that nominal support box fails). Confirmed this
  interpretation directly: built a MASS-NORMALIZED two-narrow-bump family (total drop mass
  held fixed at 1 as bump half-width w->0, positions t0=-a+w, t1=b-w so they touch the box
  corners), with N scaled to keep w/dx >= 80 (properly resolved) up to N=1.6e6. Result: as
  w->0, interior min eps'' from ht_derivative CONVERGES to route 1's exact analytic atom-limit
  values, confirmed for 3 (a,b) pairs:
    a=b=0.55            (symmetric):        w=0.001 gives -9.84e-2  vs route-1 exact -1.004e-1
    a=0.30,b=0.80        (asymmetric):       w=0.001 gives -4.02e-2  vs route-1 exact -4.10e-2
    a=0.05,b=0.951 (near-threshold, strongly asymmetric, a+b/L=1.001):
                                              needed much smaller w (down to 1e-4, with
                                              N up to 1.6e6) because the compensator mass is
                                              tiny (~0.006) and the margin over threshold is
                                              tiny -- w=1e-4 gives -6.23e-5 vs route-1 exact
                                              -7.79e-5, clearly converging (not yet fully
                                              converged at that w, trend is monotonic and closing).
  This is a second, genuinely independent (different shape family, mass-normalization derived
  fresh, run through production ht_derivative rather than the closed-form) confirmation of
  route 1's necessity/sharpness result, and it resolves what initially looked like a
  discrepancy -- I did NOT tune the criterion to force agreement; I diagnosed the actual cause
  (non-adversarial/diffuse construction) and independently re-tested more carefully.
  Scripts: route2b_width_limit.py + inline investigation (commands run directly, results in
  this checkpoint; see also route2b_results.json for the first pass before mass-normalization
  fix was applied -- KEEP for the record, it documents the diagnosis process).

- **Convergence in N checked and passes pre-registered tolerance** (<1% change N=16001->32001)
  for representative route-2 cases (multimodal_3bump a=0.2/b=0.7, plateau_fill a=b=0.45,
  single_bump_offcenter_right a=0.05/b=0.9): all three showed monotonic, smooth convergence
  with <1% relative change between N=16001 and N=32001 (e.g. 1.8614e-5 -> 1.8496e-5 = 0.63%).

## RULED OUT
- Initial "single_bump_offcenter_*" and "near_atom_*" route-2 families as originally coded
  (component support not reaching the true box corners, or compensator too spread out) are
  NOT valid adversarial tests of the a+b>L necessity direction -- they under-report failure
  because their effective support is narrower than the nominal [-a,b] box. Not a bug in
  ht_derivative or in the claim; a limitation of that specific construction, diagnosed and
  worked around via the mass-normalized narrow-bump study (see above).
- Fixed interior-min search window from a flat 0.95L (my first instinct) to "full domain minus
  a tiny margin (~1e-4L to 1e-2L) at the structurally-pinned endpoints" -- the 0.95L window was
  itself clipping real interior structure for profiles with support reaching near +/-L
  (discovered in route 1, applied throughout route 2 before drawing conclusions from route 2).

## CURRENTLY IN PROGRESS
Writing final replication.md and manifest.json. All three routes complete, verdict: MATCH.

## NEXT STEP
Write research/results/cycle2_corB_replication/replication.md (VERDICT + route summaries +
numbers + convergence) and manifest.json, then final heartbeat. This IS the last step of the
task -- nothing further planned unless replication.md review reveals a gap.

## OPEN QUESTIONS / BLOCKERS
None blocking. One soft note for the write-up: the near-threshold strongly-asymmetric
mass-normalized convergence (a=0.05,b=0.951) was still ~20% away from the route-1 reference
value at the smallest well-resolved width I reached (w=1e-4, N=1.6e6) -- trend is clearly
monotonic and closing, and takes much longer to converge than the other two cases because the
compensator mass is tiny and the margin over threshold is tiny, but I did not chase it to full
convergence given diminishing returns (would need N ~10^7-10^8). Report this honestly as
"converging, not fully converged at the resolution reached" rather than overstating it.
