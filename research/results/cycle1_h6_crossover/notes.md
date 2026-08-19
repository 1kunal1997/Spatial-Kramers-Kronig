# H6 — Clausen dip-tolerance vs the opposite-bumps alpha crossover — CHECKPOINT (in progress)

## Status: Task A partially done, hard numbers on disk. Task B (H7) not started — deprioritized per
## coordinator instruction if time runs out.

## What's confirmed and on disk (see manifest_bisection.json)

1. **Profile family identified and reproduced exactly.** Source script:
   `theory/mismatch_table.py`, lines 641-677 (`opposite_bumps_sweep.png`).
   Parameters: nb=1.7, k_steep=4.0, L=2.5 um, N=40001, sigma_b=0.5,
   e_re(x) = logistic(x,k=4,nb=1.7) + alpha*(bump_L(x) - bump_R(x)),
   bump_L = exp(-((x+1)/0.5)^2)*window, bump_R = exp(-((x-1)/0.5)^2)*window,
   window = ((1+cos(pi x/L))/2)^3, both bumps peak-normalized to 1.
   `ht_d1` = tmm_h-equivalent DiHiTI (gradient -> Hilbert -> cumtrapz -> linear de-tilt).
   Script: `repro.py` in this directory.
   Reproduced EXACTLY the Captain's reported numbers:
   alpha=0.0,0.3,0.6 -> min eps''=0.0000 ; alpha=1.0 -> -0.1168 (matches -0.117) ;
   alpha=1.5 -> -0.3017 (matches -0.302). CONFIRMED correct family/parameters.

2. **Symmetry check**: eps''(x) is even to 1e-14 for all alpha (expected: logistic is point-
   antisymmetric about its midpoint, the bump_L-bump_R term is odd, so total eps'-midpoint is odd,
   so g=-deps'/dx is even, so eps'' is even). Minimum is doubly-degenerate at +/-x0.

3. **IMPORTANT SUBTLETY FOUND**: the naive whole-domain min(eps'') is *exactly* 0.0 (to double
   precision, literally 0.0 for float) for all alpha <= ~0.665, NOT because the interior is passive,
   but because the DiHiTI linear de-tilt forces eps''(+/-L) = 0 EXACTLY for every alpha (this is
   the H7 question — looks like strong evidence H7/V5 is TRUE, i.e. automatic, at least under this
   de-tilt implementation — but not yet formally tested across other profile families, see below).
   For small alpha the whole-domain minimum is realized AT the pinned boundary (argmin x = -L
   exactly), masking the interior behavior. The interior local minimum near x=+/-1.3 is POSITIVE
   for small alpha and only crosses zero later.

4. **True (interior) crossover, bisected and grid-converged:**
   alpha_c = 0.666229 (converged over N = 1e4 -> 1.6e5, drift < 1e-5, see
   `manifest_bisection.json` for full N-convergence table). This is the actual onset of gain in
   the interior dip, distinct from the Captain's coarse (0.6, 1.0) bracket.
   Also recorded: alpha at which the *whole-domain* min first crosses the -0.01 PASS/FAIL flag
   threshold used in `mismatch_table.py`: alpha = 0.696321 (this is what a coarse visual read of
   the sweep figure would call "the crossover"; the TRUE onset is earlier, at 0.6662, because the
   flag threshold is not zero).

   **This alpha_c = 0.6662 IS itself a direct, falsifiable answer to part of V4** (a closed-form/
   quantified location of the PASS->FAIL transition the Captain asked for), independent of whether
   the Clausen machinery below reproduces it cheaply.

## What's in progress (NOT yet concluded — do not report as final)

5. Shape of g(x) = -deps'/dx: NOT the idealized rectangular "plateau minus centered dip" of
   Sec 5.1. Diagnostic (`analyze_g_shape.py`) shows, e.g. at alpha=1.0: g(0)=2.64 (elevated
   center, NOT flat plateau), h_max=3.00 at s=0.368 (off-center peak, not at s=0), h_min=-1.95 at
   s=1.07 (a genuine NEGATIVE region — g goes negative, i.e. eps' rises locally — this is a real
   third feature beyond plateau+single-dip), tail -> ~0 at s=L. So the profile has (at least)
   3 signed regions, not 2. Sec 5.1's narrow d*(b) formula (single rectangular dip depth d, width b,
   inside a flat plateau of half-width a) does NOT literally apply to this g — its hypotheses
   (rectangular plateau, single centered dip) are not met.

6. Was mid-way through implementing the GENERAL form mentioned in the task brief: since g(x)=h(|x|)
   is even but not piecewise-constant, use the continuous signed layer-cake
   h(s) = h(L) + int_s^L (-h'(a)) da, i.e. superpose centered-interval indicators 1_{|x|<=a}
   weighted by dnu(a)=-h'(a)da (+ atom h(L) at a=L), and use Sec 5.1's OWN closed form for a
   single indicator (already Clausen-exact per verify_passivity.py) to reconstruct eps''(x) via
       eps''(x) = (L/pi^2) * [ int_0^L M(pi*a/L, theta_x)*(-h'(a)) da + h(L)*M(pi,theta_x) ]
   Script `clausen_general_form.py` in this directory implements this (function
   `eps_pp_general_layercake`) and was running a validation (compare to DiHiTI directly, N=4001,
   O(N^2) loop) when interrupted by the checkpoint request. Job ID bv074oqq2 was backgrounded after
   timing out at 120s — check
   `/private/tmp/claude-501/-Users-kunal-Documents-Spatial-KK-Project/e97d6429-1c03-4ea3-b749-373f9d0efd38/tasks/bv074oqq2.output`
   for its result when it finishes.

   **Expectation (not yet verified numerically):** this general form should match DiHiTI to
   quadrature precision because it is an exact algebraic reformulation (linearity of the log-
   potential in g, Thm 3.1), NOT an independent approximation — so it is expected to predict
   alpha_c essentially exactly (this would satisfy V4's "closed form to a few percent" criterion,
   but should be reported honestly as the EXACT/FALLBACK-1 form, not the cheap/primary-tier dip-
   tolerance formula, since it costs the same O(N) [or worse, O(N^2) as coded] as DiHiTI itself).

## Exact next step if resumed

1. Check the backgrounded job output (bv074oqq2) for the general-form validation numbers.
2. If it confirms near-exact agreement (expected), STATE CLEARLY in the final report: the narrow
   Sec-5.1 d*(b) closed form (idealized rectangular plateau+dip) does NOT apply to this profile
   (hypotheses violated — g has 3 signed regions, not 2, and is not piecewise constant); but the
   GENERAL Clausen/log-potential machinery (Thm 3.1, reparametrized via continuous layer-cake)
   DOES exactly reproduce alpha_c = 0.6662 — at the same O(N) cost as DiHiTI, i.e. it is an exact
   reformulation (Fallback-1 tier: exact N&S, not cheaper), not the "few-percent cheap prediction"
   V4 was hoping for from the narrow closed form.
3. Optionally: attempt a best-fit mapping of the actual g onto the idealized (a,b,d) rectangle
   family (e.g. matching zero-crossings of h(s) for a,b and mean depth for d) and quantify how far
   off the naive d*(b) prediction lands, as a measure of "how much the calculus needs generalizing."
   This was NOT started.
4. If time allows, do H7 (should be quick: sample eps''(+/-L) directly across profile families and
   grid sizes N using tmm_h.ht_derivative, check exact-zero vs O(1/N) discretization-error scaling,
   and check whether the de-tilt step changes anything beyond machine/discretization noise for a
   profile whose *un-detilted* endpoint values are non-trivially different from each other, e.g. an
   asymmetric/off-centre profile with large slope mismatch at +/-L).

## Files on disk in this directory
- `repro.py` — reproduces Captain's reported numbers exactly (CONFIRMED)
- `bisect_alpha_c.py` + `manifest_bisection.json` — the grid-converged true crossover alpha_c
  (CONFIRMED, this is the headline number)
- `analyze_g_shape.py` — g(x) shape diagnostic showing non-rectangular structure
- `clausen_general_form.py` — general Clausen layer-cake reconstruction (implementation done,
  validation run in progress / interrupted)
