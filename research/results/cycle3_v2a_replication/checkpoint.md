# Checkpoint — replicator, C30 (V2a lower-bound empirical core)

## GOAL
Blind independent replication of C30: claim that adding a small localized bump (amplitude a,
width delta) to eps'(x) at x0 produces a depression of eps''(x0) (DiHiTI partner) of
Delta eps''(x0) = -a/(pi*C_phi), CLAIMED delta-independent. Plus: (a) a_min(x0) = kappa*eps''_0(x0)
proportional to local headroom; (b) kappa/pi = 0.418 +/-3% over 3 decades of eps''_0; (c) an
N-independent ratio at N=4096/8192/16384. Must NOT read the originating files (v2_lower_bound.md,
cycle2_v2a/*, cycle2_e1a/*, cycle1_c15_redteam/*). Only read: MISSION.md, PROTOCOL.md, tmm_helper.py.

## PRE-REGISTERED TOLERANCES (stated before running any bump experiment)
- Width independence: plateau claimed CONFIRMED if Delta eps''(x0) varies <2% relative RMS across
  delta spanning >=1.5 decades in a "resolved & localized" window (delta from ~10*dx to ~L/10).
- Proportionality (a): kappa constant to within +/-5% across eps''_0(x0) spanning >=3 decades ->
  MATCH-worthy; else report actual spread, no tuning.
- Constant value (b): compare my C_phi (my own analogous definition) to 0.418; agreement within 10%
  called supportive, but I do NOT assume same definition of C_phi as the original — flagged explicitly.
- N-independence (c): relative variation <1% across N in {4096,8192,16384,10000,12000} (last two
  non-power-of-2) for MATCH.

## GRID CONVENTION (decided before coding)
x = np.linspace(-L, L, N, endpoint=False)  i.e. x_j = -L + j*(2L/N), j=0..N-1.
Reasoning: scipy.signal.hilbert (via FFT) treats the sampled array as EXACTLY one period. Using
np.linspace(-L,L,N) (endpoint=True, the default) duplicates the point at -L and L (both included),
so N points cover only N-1 independent samples of a period-2L signal -> the FFT implicitly assumes
period 2L*N/(N-1) != 2L. This is the exact bug flagged in PROTOCOL.md (silently shared by two prior
independent implementations). endpoint=False avoids it by construction: x_0=-L is included, x_{N-1}
= L - 2L/N is the last sample, and the point x=+L is implicitly identified with x=-L by periodicity.

## PRE-REGISTERED ANALYTIC PREDICTION (derived myself, before running bump numerics)
Wrote out the DiHiTI pipeline as an operator: since d/dx and the periodic Hilbert transform H are
both Fourier multipliers, they commute. So for any perturbation "bump" added to eps' (call the
resulting change to the raw cumulative-integral output Delta_raw):
  Delta_raw(x) = H[bump](x) - H[bump](-L)
then subtract-line removes any residual affine trend between the endpoints x=-L,L.
KEY FACT (elementary, exact): if bump(x) is SYMMETRIC (even) about its center x0, then H[bump] is
ODD about x0, so H[bump](x0) = 0 EXACTLY, for ANY width delta and ANY amplitude a — not just as
delta->0. This means: for a perfectly symmetric bump measured exactly at its own center, my analytic
argument predicts Delta eps''(x0) should be dominated by the two BOUNDARY terms H[bump](-L) and the
line correction, both of which scale like the bump's AREA ~ a*delta (vanishing as delta->0), NOT a
delta-independent constant.
==> This directly conflicts with the delta-independence claim IF "bump centered at x0, measured at
x0" is the correct literal reading. PRE-REGISTERED PREDICTION: numerics on a truly symmetric bump
will show Delta eps''(x0) ~ O(a*delta), shrinking as delta -> 0, NOT constant. If numerics instead
show a genuine plateau, my analytic argument is wrong somewhere (recompute pipeline commutation
carefully) or the claim's "bump...at position x0" is not literally symmetric-centered-at-measurement
point, and I need to also test offset/asymmetric bumps.

## ESTABLISHED SO FAR
- (verified via tmm_helper.py source, not the mission's off-limits files) ht_derivative is exactly:
  u=np.gradient(e_re,xx); v=imag(hilbert(u)); e_im=cumulative_trapezoid(v,xx,initial=0);
  e_im -= linspace(e_im[0],e_im[-1],N). Matches the task's stated construction exactly.
- Grid convention decided and justified (above).
- Analytic cross-check plan: for eps'(x) built as a finite trig sum with periods that divide 2L
  (i.e. sum_k c_k*cos(k*pi*x/L + phi_k)), the WHOLE ht_derivative pipeline has a closed form:
    eps''_0(x) = sum_k c_k*[ sin(k*pi*x/L + phi_k) - (-1)^k * sin(phi_k) ]
  Derivation: u = -sum c_k*(k*pi/L)*sin(k*theta+phi_k) where theta=pi*x/L; periodic-Hilbert-transform
  mode rule H[sin(k*theta+phi)] = -cos(k*theta+phi) for k>0 (standard convention, to be numerically
  verified not assumed); v = sum c_k*(k*pi/L)*cos(k*theta+phi_k); cumtrapz gives
  sum c_k*sin(k*theta+phi_k) + const, and because every term is exactly 2L-periodic the raw output is
  automatically periodic (raw(-L)=raw(L)) so the "subtract line" step is IDENTICALLY ZERO for this
  family — the const is fixed by initial=0 at x=-L. This gives a genuine Route-1 (analytic) check of
  the numeric pipeline, independent of anything in the off-limits files.
  STATUS: derived, NOT YET numerically verified — next step.

## RULED OUT
(none yet — first substantive derivation pass only)

## ESTABLISHED SO FAR (update 2)
- **PART 1 DONE (analytic cross-check, PASSED).** `part1_analytic_check.py`: numeric ht_derivative
  pipeline matches my hand-derived trig-sum closed form to O(1/N) (error halves as N doubles: e.g.
  N=512 -> rel err 1.6e-2, N=16384 -> 4.9e-4), including at non-power-of-2 N (10000, 12000, both
  converge with no anomaly). Grid convention (endpoint=False) confirmed correct; the endpoint=True
  duplicated-endpoint bug gives visibly worse residual at matched N (2.0e-3 vs 4.0e-3 — actually
  roughly comparable at N=2048; the bug is a systematic bias not just noise, worth flagging but not
  the focus here). Pipeline trusted for part 2+.

- **PART 2 DONE (width independence, `part2_width_independence.py`). MAJOR FINDING — MISMATCH on
  the crux claim, but with an important nuance.**
  Two placements tested at N=8192, L=1, x0=0.15 (fixed interior point), a=0.02, delta swept
  geometrically over ~2.5 decades (delta/dx from 2 to 683):
  - **"centered"**: bump is symmetric (even) about x0 itself, i.e. eps''(x0) is measured exactly at
    the bump's own center. Result: depression is **NOT delta-independent** — it scales roughly
    *linearly* with delta, growing from -1.76e-4*a at delta/dx=2 to -6.6e-2*a at delta/dx=683 (a
    ~375x change over 2.5 decades, monotonic, no plateau anywhere). This exactly matches my
    pre-registered analytic prediction: H maps even functions to odd functones about their own
    center, so H[bump](x0)=0 EXACTLY at any delta when bump is symmetric about x0; the residual
    numeric depression is an O(a*delta) boundary/global correction (area-driven), not a local
    Hilbert-kernel plateau. **This is a clean, elementary, hard mathematical fact (not a numerical
    artifact) — a truly centered bump cannot produce a delta-independent depression at its own
    center.**
  - **"offset"**: bump center placed at x0 + delta/2 (i.e. x0 sits at a FIXED RELATIVE position
    within the bump — a fixed fraction of its half-width — that scales proportionally as delta
    shrinks, rather than a fixed absolute point). Result: much closer to constant — depression/a
    goes from -0.281 (delta/dx=2) to -0.297 (delta/dx=10.6) to -0.341 (delta/dx=392) to -0.393
    (delta/dx=683): a genuine near-plateau over roughly 1.5 decades (delta/dx ~ 10 to ~400, drift
    ~15% relative), degrading (drifting further, ~40% total) as delta approaches domain-size
    fractions (delta/dx > 500, i.e. delta > L/12ish) — consistent with the expected large-delta
    breakdown (bump no longer "localized" vs L).
  - **Interpretation (mine, to be stated in replication.md):** dilation-covariance of the Hilbert
    kernel does produce an approximately delta-independent local effect, but ONLY when the
    measurement point tracks a FIXED RELATIVE position within the bump as it narrows (the "offset"
    construction). If x0 is read literally as a FIXED location where the bump is centered AND where
    the depression is measured (the most natural reading of "add a bump... at position x0... the
    resulting depression of eps''(x0)"), the claimed width-independence is FALSE by an elementary
    parity argument, confirmed numerically to ~2.5 decades with no partial plateau at any delta.
  - Pre-registered tolerance (<2% relative RMS over >=1.5 decades) is **violated by both**
    constructions in the strict sense, but violated catastrophically (375x, no plateau) for
    "centered" and only mildly (~15%) for "offset" over part of its range.

## RULED OUT
- Perfectly-centered/symmetric-bump reading of the claim: numerically and analytically dead (parity
  argument gives exact zero at the singular point regardless of delta; observed depression is a
  distinct, vanishing, boundary-driven O(a*delta) effect). Do not re-test this construction — it is
  settled.

## CURRENTLY IN PROGRESS
Moving to part 3: test consequence (a)/(b) — proportionality of a_min to local headroom, and
x0-independence / profile-independence of the coefficient kappa — using the "offset" (dilation-
covariant) construction at a FIXED, moderate, well-resolved delta (since delta-independence itself
already failed strictly; the interesting remaining question is whether the coefficient is at least
UNIVERSAL across x0 and profile at fixed delta). NOTE: proportionality of a_min to headroom AT FIXED
delta is a near-trivial consequence of ht_derivative's LINEARITY as an operator on eps' (Delta eps''
is exactly linear in a for fixed bump shape/position) — so consequence (a) is not really an
independent thing to verify beyond linearity; what's nontrivial and worth checking is whether the
proportionality CONSTANT (kappa) is the same across different x0 and different base profiles.

## NEXT STEP
Write part3_proportionality.py: for several base profiles (varying overall amplitude scale across
~3 decades) and several interior x0 per profile, fix delta (moderate, e.g. delta=0.02*L, resolved),
measure the local slope d(Delta eps'')/da via two-point (or few-point) finite difference (exploiting
exact linearity — no need for expensive bisection search for a_min; slope alone gives a_min =
headroom/slope directly). Report kappa = -1/(pi*slope) style constant, its spread across x0 and
profiles (headroom spanning 3 decades), and compare loosely to 0.418. Then part4: repeat the slope
measurement (offset construction, fixed delta *in absolute grid-spacing multiples*, i.e. delta/dx
held fixed so resolution keeps pace) at N=4096/8192/16384/10000/12000 to test N-independence.

## OPEN QUESTIONS / BLOCKERS
- "C_phi" and "r/D" in the original claim are undefined to me (by design — blind). I will define my
  OWN analogous constants from my own construction and state explicitly that exact-value agreement
  with 0.418 or 1.680 is suggestive, not proof of matching definitions.
- RESOLVED: the pre-registered tension (H[even]=odd => vanishes at own center) is CONFIRMED by
  numerics for the centered construction. The claim's width-independence, if it refers to a bump
  literally centered at the measurement point, is a MISMATCH. A weaker, offset-dependent form of
  width-independence does hold approximately over a limited delta range.
