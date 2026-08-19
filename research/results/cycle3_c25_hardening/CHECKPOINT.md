# CHECKPOINT — red-team, cycle 3, attacking C25 ("Cor 4.B")

## 1. GOAL
Attack C25 for promotion `replicated` -> `hardened`.
C25 as registered in CLAIMS.md / h9_potential_domination.md §4:
  "g >= 0, M1 = INT g tan(pi t/2L) dt = 0, supp g in [-a,b]  ==>  eps'' >= 0  IFF a+b <= L."
Win condition = KILL or NARROW.  Write verdict to `redteam.md` in this dir.

## 2. ESTABLISHED SO FAR

### (a) [verified-numerically] The DiHiTI gauge/kernel (C1 restricted to atoms) is CORRECT.
`e2_gauge.py`. Production `tmm_helper.ht_derivative` on a smoothed unit step at t0
reproduces  eps''(x) = (1/pi) ln[cos(pi t0/2L) / |sin(pi(x-t0)/2L)|]  to max abs err
7e-5 (N=2^14) -> 4.5e-6 (N=2^16), i.e. converging.  Tested t0 = 0, 0.3, -0.6, 0.85.
CONTROL: the same comparison against the kernel WITHOUT the cos(pi t0/2L) numerator is
off by exactly ln cos(pi t0/2L) (3.67e-2 at t0=0.3; 4.63e-1 at t0=0.85) -- so the
gauge factor is genuinely there and is not an artifact.
IMPORTANT (common-mode guard, PROTOCOL): I used an endpoint-EXCLUDED periodic grid
x_j = -L + 2Lj/N, and both power-of-two AND N=40000. The replicator/theorist used
`linspace`-style grids. Same answer => the C22-class linspace bug is not in play here.

### (b) [PROVED, by hand + numerics] Cor 4.A (the two-atom law) is CORRECT.
I re-derived the algebra independently:
  Delta'(eta) numerator = (beta+gamma)[ eta(1+beta*gamma) + (gamma-beta) ]  -- degree 1, ONE root
  eta* = (beta-gamma)/(1+beta*gamma) = tan(A-B),  and it lies in (-gamma, beta) where the
  denominator (eta-beta)(eta+gamma)(1+eta^2) is NEGATIVE => eta* is a local MAX of Delta.
  Outside (-gamma,beta), Delta rises from -inf to 0^- (since M1=0 gives Delta ~ -M2/2eta^2).
  So sup_R Delta = max(Delta(eta*), 0), and
  Delta(eta*) = gamma*ln(sinB/cosA) + beta*ln(sinA/cosB)   [(*) verified symbolically by me]
  A+B <= pi/2  <=>  sinA<=cosB and sinB<=cosA  <=>  both logs <= 0.
So a balanced 2-atom drop is passive iff t+ + t- <= L.  Airtight given (a).

### (c) ⭐⭐ [PROVED + verified] THE LITERAL "iff" IN C25 IS FALSE.  `e1_counterexample.py`.
The necessity half, read per-profile ("supp g in [-a,b] and a+b > L => NOT passive"),
has explicit counterexamples whose EFFECTIVE support equals the nominal box.
  P = lam*(balanced corner pair at -0.40, +0.90)  +  (1-lam)*(balanced pair at -0.20,+0.20)
  a+b = 1.30 L.  supp P = {-0.40,-0.20,+0.20,+0.90}, convex hull = [-0.40,+0.90] EXACTLY.
  Both pieces >= 0 and M1 = 0, so P is admissible.
  lam=1.00 -> min eps'' = -1.004e-1  FAIL
  lam=0.50 -> -1.940e-2               FAIL
  lam=0.20 -> +9.0e-9                 **PASSIVE**  <-- counterexample
  lam=0.01 -> +5.2e-9                 **PASSIVE**
Confirmed through PRODUCTION ht_derivative at N = 2^15, 2^17, 2^19: min eps'' = +5.05e-9,
+5.05e-9, +4.79e-9 (all PASS).  So it is not a closed-form artifact.
NOTE: 20% of the mass sitting exactly on the two corners of a 1.3L box is still passive --
this is not a marginal/boundary-buffer effect.

The TRUE statement (what the theorist's own proof line "Necessity: take the extreme pair
at t+=b, t-=a" actually establishes) is a statement about the BOX, not the profile:
  FOR ALL g (g>=0, M1=0, supp g in [-a,b]) : eps'' >= 0     <=>    a+b <= L.
i.e. a+b<=L is exactly the maximal box for which passivity is automatic.
That is the same LOGICAL SHAPE as C3/Thm 4.2 ("sufficient, constant L/2 sharp") -- so
C25 is NOT the mission's first two-sided per-profile result. It is a sharp sufficiency
theorem with the region generalised from symmetric to asymmetric.  => at minimum NARROWED.

## 3. RULED OUT (do not re-walk)
- "The counterexample is just the nominal-vs-effective support ambiguity the replicator
  already found."  NO: my counterexample's convex hull of the support IS the nominal box,
  atoms sit exactly ON both corners. The replicator's 76/80 had genuinely narrower
  effective support; mine does not.
- "It is a near-threshold numerical artifact." NO: a+b = 1.30L, margin +5e-9 vs the
  corner-pair depth -1.0e-1, stable over 3 grids spanning 16x in N and over two
  independent evaluators.
- Grid/endpoint (linspace) common-mode bug: excluded, see 2(a).

### (d) [verified] Sufficiency (the surviving half) is SOLID.  `e6_sufficiency_stress.py`
3600 random balanced many-atom measures (1..25 atoms) in 10 boxes with a+b <= L: ZERO negatives,
worst min eps'' = +9.1e-11 (grid noise).  Corner-loaded extreme points at a+b = L: min = 0
attained exactly at the predicted interior point x* = b - a.  Violation depth is LINEAR:
min eps'' = -(a+b-L)/2 to 5 digits over eps = 1e-1 .. 1e-6 (symmetric corner pair).

### (e) [proved] The Bauer/extreme-point step is legitimate.
F(g) = min_x eps''(x;g) is a MINIMUM of linear functionals => CONCAVE and upper semicontinuous;
K = {P>=0 on [-a,b], mass 1, INT tan dP = 0} is weak*-compact (needs [-a,b] compact and tan
continuous there, i.e. a,b < L, which a+b<=L supplies except in the degenerate a=0 case).
Bauer's MINIMUM principle => min of F over K is attained at an extreme point; Richter/Rogosinski
=> extreme points have <= 2 atoms.  With t- <= a, t+ <= b we get t-+t+ <= a+b <= L, and Cor 4.A
applies.  So the SUFFICIENCY half of C25 is genuinely PROVED.

### (f) [verified] Hypothesis load-bearing test.  `e5_hypotheses.py`
- g >= 0 : LOAD-BEARING.  287/400 random SIGNED measures with M1=0 supported in the MIDDLE HALF
  [-L/4,L/4] are non-passive; worst min eps'' = -1.795.
- DiHiTI gauge : LOAD-BEARING.  Deleting the cos(pi t/2L) seam normalisation shifts eps'' by
  -INT g ln cos > 0, eps''(+-L) != 0, and the a+b<=L threshold ceases to mean anything.
- M1 = 0 : LOAD-BEARING and a KNIFE EDGE.  Any M1 != 0 gives strict gain, matching C9's
  -M1^2/(2 pi M2) to 3 digits (M1=1e-1..1e-4: -1.097e-3 vs -1.094e-3; -1.3265e-5 vs -1.326e-5;
  -1.3552e-7 vs -1.355e-7; -1.3578e-9 vs -1.358e-9).  ⚠ BUT the induced gain sits in a
  SHRINKING neighbourhood of the seam: argmin = -0.956, -0.9947, -0.99946, -0.99995 as
  M1 = 1e-1..1e-4.  A fixed interior window of 0.95L (which is what the replicator's
  pre-registered protocol used) reports PASS for every one of these.  This is a real hazard for
  every numerical passivity verdict in the project, not just C25.

### (g) [verified] Boundary case a+b = L EXACTLY: PASSIVE (marginally), unambiguously.
Closed form: Delta(eta*) = 0 identically, so min eps'' = 0 attained at the interior point
x* = b - a.  Confirmed at (a,b) = (.5,.5), (.25,.75), (.1,.9), (.02,.98), (.001,.999).
⚠ PRACTICAL CAVEAT (`e10_boundary_smooth.py`): pushed through PRODUCTION ht_derivative the
corner-atom measure at a+b = L returns a small NEGATIVE minimum at every finite N
(-1.5e-9, -9.1e-11, -5.7e-12 at N = 2^15/2^17/2^19, i.e. O(h^2) -> 0).  Test 1 shows this is
QUADRATURE, not a hypothesis violation: at fixed N = 2^18 the value is -2.286e-11 independent
of the smoothing width w over w = 10h..320h (32x).  For genuinely smooth g strictly inside the
box with a+b = L the interior margin is a healthy +1.1e-3 (matching C21's ~1e-3 D).

### (h) [verified] The counterexample regime is NOT marginal -- it covers EVERY box.
`e11_degenerate.py`: bisecting the largest corner mass fraction lam that stays passive, with
the remaining mass in a balanced pair at +-0.1L:
   a+b = 1.30L -> lam <= 0.262   a+b = 1.40L -> lam <= 0.205
   a+b = 1.80L -> lam <= 0.087   a+b = 1.94L -> lam <= 0.0018   a+b = 1.98L -> lam <= 0.0009
So for EVERY box [-a,b] up to the maximum possible a+b = 2L there exist admissible g whose
support touches BOTH corners and which are passive.  Per-profile necessity fails everywhere,
not merely near threshold.
Degenerate a = 0: g>=0 on [0,b] with M1=0 forces g = c*delta_0, whose eps'' = -ln|sin(pi x/2L)|/pi
>= 0 -- trivially passive for any b <= L.  Degenerate b -> L (a = L-b -> 0): PASS down to
a = 1e-8 despite the Cayley weight ratio reaching 2.5e-16.

## 4. CURRENTLY IN PROGRESS (see NEXT STEP)
Attacking the SURVIVING (box-quantified) form on four fronts:
 E5 sensitivity of the verdict to M1 != 0 (is the M1=0 hypothesis a knife edge? where
    does the induced gain live -- suspicion: a shrinking neighbourhood of the seam, which
    would make it numerically invisible);
 E6 adversarial sufficiency stress: many-atom / continuous g in boxes with a+b = L exactly
    (the marginal case) and a+b = L(1-eps);
 E7 drop g>=0 -> confirm it is load-bearing;
 E8 boundary a+b = L exactly: passive or not (Cor 4.A gives Delta(eta*) = 0 EXACTLY, i.e.
    min eps'' = 0 attained at an INTERIOR point => marginally passive; needs confirming
    through the production pipeline, and it interacts with C19/C21).

## 5. NEXT STEP
**DONE — verdict NARROWED written to `redteam.md`.** Hand A returned and CONFIRMED the Attack-1
counterexample independently (erf smoothing, both grid conventions, non-power-of-two N,
own closed form): lam=0.2 passive, sign change at lam*~0.278, grid conventions agree.
Nothing outstanding. Any successor should read `redteam.md`, not this file.

(historical) All planned experiments E1,E2,E5,E6,E7,E8,E9,E10,E11 are DONE and on disk.
Outstanding: hand A (general-purpose subagent) was dispatched to re-verify the E1
counterexample with a DIFFERENT construction (erf/tanh smoothing, both grid conventions,
non-power-of-two N).  When it returns, fold its table into `redteam.md`.
Then write the final verdict: **NARROWED**.

## 6. OPEN QUESTIONS / BLOCKERS
- Does the classical extreme-point theorem (Richter/Rogosinski, <=2 atoms for 2 moment
  constraints) need any hypothesis that fails here?  S = [-a,b] compact and tan continuous
  on S requires a,b < L; a+b<=L already forces that unless a=0,b=L (degenerate: M1=0 with
  g>=0 on [0,L] forces g = c*delta_0, trivially passive).  Believed fine; not yet stressed.
- Is min_x eps'' the right functional at atoms (kernel is +inf at atoms)?  Using excised
  neighbourhoods; smooth-g route 2 has no such issue.
