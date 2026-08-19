# CHECKPOINT — replicator, claim C49 (board item F5)

## GOAL
Blind-replicate C49: two atomic drop measures A, B on the same 20 atom positions
(`research/results/c49_blind_replication/witness_input.json`), same total mass,
M1=0 exactly, identical readings on an 8-functional battery, yet one PASSIVE
(eps'' >= 0 everywhere) and one GAIN (eps'' < 0 somewhere). I was NOT told which
label is which. Forbidden to read `research/results/fable_ns_answer/*` or
`research/theory/ns_answer.md` — did not touch either.

## ESTABLISHED SO FAR (all VERIFIED NUMERICALLY, cross-checked two independent routes)

1. **Route 1 (primary, analytic closed form, Rank-1 strength)**: the log-potential
   master identity quoted in `research/MISSION.md` (foundational assumption #3,
   Wave-2 finding) —
   eps''(x) = (1/pi) * sum_j w_j * ln[ cos(pi t_j/2L) / |sin(pi(x-t_j)/2L)| ]
   for an atomic drop measure g = sum w_j delta(t-t_j). This is EXACT for atoms —
   no FFT, no grid, no smoothing, no endpoint trap. Endpoint check: analytic
   eps''(x) -> 0 as x -> +-1 for BOTH weight sets, confirmed to 1e-11 at margin
   1e-6 (control on the formula itself, matches the claimed structural zero).

2. **Route 2 (cross-check, production pipeline)**: built the atomic measure as a
   smoothed decreasing staircase eps'(x) (erf transitions, width s) on a
   cell-centered grid x = -L + 2L*(arange(N)+0.5)/N (NOT np.linspace(-L,L,N) —
   avoids the endpoint-duplication trap flagged in PROTOCOL.md), pushed through
   `tmm_helper.ht_derivative` unmodified. Swept N in {4001, 6000, 8192, 16001}
   (mix of power-of-2 and non-power-of-2) and s in {0.02,0.01,0.005,0.002}.
   Verdicts for BOTH A and B are stable across all 16 (N,s) combinations —
   no flip anywhere. Grid route matches analytic route to 1.2% rel RMS away
   from atoms/edges (residual is the finite smoothing width, as expected).

3. **Harness control (independent of witness data)**: single balanced atomic
   pair at t=-a,+b with M1=0 enforced is PASSIVE iff a+b<=1 — reproduced exactly
   by BOTH routes (symmetric a=b sweep: PASSIVE for a+b in [0.6,0.96], min~0 at
   a+b=1.00 (boundary), GAIN for a+b in [1.04,1.4], magnitude growing correctly;
   asymmetric pairs with M1=0 enforced via w_b=w_a*tan(pi a/2)/tan(pi b/2) also
   respect a+b<=1 exactly). This is the control the dispatch brief asked for.
   PASSED — harness trusted.

4. **Battery functionals** (8 exact atom-sums, no grid at all, no risk of
   tan-overflow since no atom sits at +-1):
   Battery A = Battery B to ~7e-15 max absolute difference (float64 roundoff
   given magnitudes up to ~57). phi1 (mass) = 19.6310337, phi2 (M1) ~ 2e-16
   (i.e. 0 to machine precision) for BOTH. CONFIRMS "identical total mass,
   M1=0 exactly, identical readings on all 8 functionals."

5. **Labels resolved**:
   - **weights_A: GAIN.** Interior min eps'' = -0.007756 at x = 0.6311
     (converged across n=20001..500001 grid density for the min-search;
     converged across all (N,s) in route 2). Negative region spans
     x in [0.613, 0.649] (~1.8% of the interior domain) — a clean, resolved
     dip, not a marginal/noise artifact.
   - **weights_B: PASSIVE.** Interior eps'' >= 0 everywhere (0% of ~2e6 sampled
     interior points negative, full-domain scan). Min approaches 0 only in the
     edge margin (as x -> +-1, structural zero), min = +2.24e-13 at margin 1e-7
     (i.e. numerically 0, positive). Confirmed positive across all (N,s) in route 2.

## RULED OUT
- Whole-domain min (including x=+-1) as an indicator — known-blind, did not use;
  used interior min with atom-exclusion + edge-margin instead, and separately
  verified the edge->0 behavior as a check on the formula, not as the passivity
  metric.
- np.linspace(-L,L,N) grid convention — avoided from the start per PROTOCOL.md
  C22 warning; used cell-centered non-duplicating grid instead.

## CURRENTLY IN PROGRESS
DONE. `replication.md` and `manifest.json` written and complete. Verdict: MATCH.
weights_A = GAIN, weights_B = PASSIVE.

## NEXT STEP
None remaining for this task. If resumed: just report the verdict in
`research/results/c49_blind_replication/replication.md` to the dispatcher —
no further computation needed. (If someone wants MORE rigor: could add a
third, genuinely different discretization — e.g. Chebyshev/spectral collocation
Hilbert transform instead of FFT — but two independent routes plus a passing
closed-form control already meet the "genuinely independent route" +
"convergence is part of replication" bar from the brief.)

## OPEN QUESTIONS / BLOCKERS
None. Both independent routes agree with each other and with the claim as
restated. No mismatch found.

Scripts (all runnable, in `research/results/c49_blind_replication/replication/`):
- `1_analytic_route.py` — closed-form log-potential sum route + battery + interior min scan (PRIMARY)
- `2_diagnostic_scan.py` — fine full-domain scan confirming B has 0% negative interior points, A has a resolved 1.8% negative region
- `3_ht_derivative_route.py` — tmm_helper.ht_derivative cross-check, N/s sweep
- `4_control_test.py` — a+b<=1 balanced-pair sanity control, both routes
