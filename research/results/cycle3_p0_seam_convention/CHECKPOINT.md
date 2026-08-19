# CHECKPOINT — P0 seam adjudication convention (numericist, cycle 3)

## 1. GOAL
Certify a numerical definition of "passive" for DiHiTI profiles. Both conventions in use are blind:
whole-domain min reads exactly 0 for every passive profile (C19, because eps''(+-L)=0 is an operator
identity C13/C21); a fixed interior mask (0.95L) reports PASS for profiles carrying real gain that
has migrated to the seam as M1 -> 0 (C39). Deliverable: a RECOMMENDED CONVENTION + soundness argument
+ limits + blast-radius numbers on the 240-profile corpus, C18 (alpha_c=0.66623), C33 (240/240).

## 2. ESTABLISHED SO FAR
### (T1) [DERIVED + VERIFIED NUMERICALLY, e1_seam_law.py] The edge expansion — this is the key.
Master identity C1: eps''(x) = (1/pi) INT g(t) ln[cos(pi t/2L)/|sin(pi(x-t)/2L)|] dt, g = -d eps'/dx.
Put x = L - delta, theta = pi t/2L, eps = pi delta/2L. Then sin(pi(x-t)/2L) = cos(theta+eps), so
    eps''(L-delta) = (1/pi) INT g ln[cos theta / |cos(theta+eps)|] dt.
At eps=0 the integrand vanishes IDENTICALLY -> eps''(+-L)=0 is a KERNEL identity (C13 explained).
Taylor: ln cos t - ln cos(t+e) = e*tan t + (e^2/2)sec^2 t + ...  giving
    eps''(L-delta) = (1/pi)[ M1*eps + (M2/2)*eps^2 + ... ],  M1 = INT g tan theta,  M2 = INT g sec^2 theta
and at the other seam eps''(-L+delta) = (1/pi)[ -M1*eps + (M2/2)eps^2 + ... ].
CONSEQUENCES (all closed-form):
  * seam minimum location:  delta*/L = (2/pi)|M1|/M2 -- LINEAR in M1
  * seam minimum depth:     -M1^2/(2 pi M2)          -- QUADRATIC in M1   (= C9, now derived)
  * hence depth = (pi M2/8) * (delta*/L)^2 : depth vanishes QUADRATICALLY in distance to seam.
  * a fixed mask at fraction rho misses gain up to (pi M2/8)(1-rho)^2. At rho=0.95, M2~1.2 this is
    ~1.2e-3 -- the SAME ORDER as the profile's own interior margin. NOT benign.
VERIFIED (e1_seam_law.py, EXACT kernel, continuous minimisation, no grid/FFT/quadrature at all):
  M1 = 1e-1 .. 1e-6, six decades.  x* predicted vs measured: rel err 3e-3 .. 3e-7 (best mid-range).
  depth predicted vs measured: rel err 2.5e-3 .. 2e-8.
  measured log-log slopes: dist ~ M1^0.999, depth ~ M1^1.9999, depth ~ dist^2.0001
  depth / [(pi M2/8) dist^2] = 1.000000 to 6 digits.   -> the law is EXACT, not fitted.
NOT BENIGN: base M2 = 1.1716, and the balanced base profile's own interior margin is +4.63e-3.
Max gain hidden by a mask at rho: rho=0.90 -> 4.60e-3 (LARGER than the margin), 0.95 -> 1.15e-3
(25% of the margin), 0.99 -> 4.60e-5, 0.999 -> 4.60e-7.  So a fixed mask misclassifies at a
magnitude comparable to the quantity being measured, until rho is pushed to ~0.999.
### (T2) [DERIVED] Unequal edge slopes give an eps*ln(1/eps) term that DOMINATES M1.
Edge-region analysis of the same integral: eps''(L-delta) ~ (2L/pi^2)*E*eps*ln(1/eps) + (1/pi)M1*eps,
with E := g(+L) - g(-L); at the other seam the coefficient is -E. So E != 0 => gain at exactly one
seam, at a rate strictly faster than the M1 term. This is C8's first cascade item, with a mechanism.
Also: M1 only exists as a symmetric-cutoff PRINCIPAL VALUE, and it converges iff E = 0.

### (T3) [REASONED, to verify] The corpus is NOT on the fixed-mask convention.
`profile_generator.make()` labels with `e_im.min() >= -1e-3*scale` over the WHOLE domain. But
production `ht_derivative` sets e_im[0] = 0 exactly (cumtrapz initial=0) and e_im[-1] = 0 exactly
(the linear detrend), so the two endpoint nodes carry NO noise and whole-domain min
== min(0, interior-node min). Hence the corpus convention is OPERATIONALLY IDENTICAL to
candidate (1) "drop only the structurally-zero endpoint nodes". That is why the auditor's 8x/16x
refinement found zero flips, and it means C39's trap does NOT bite the corpus. TO VERIFY.

## 3. RULED OUT
- "fixed fractional interior mask" as a convention: DEAD. Its blind spot is IRREDUCIBLE (does not
  shrink with N) and its size (pi M2/8)(1-rho)^2 is of the same order as the interior margin it is
  meant to certify, at every rho a human would pick.

## 4. CURRENTLY IN PROGRESS
E1 done. Next: E2 -- is candidate (1) (drop only the endpoint nodes) sound? Its blind spot should
shrink like h (a node at distance h reads negative whenever h < 2 delta*, i.e. whenever
|M1| > (pi/4)(h/L)M2), so it is refinable -- UNLESS near-seam quadrature noise in the production
pipeline swamps it. That competition is the crux of the whole P0 question.

## 5. NEXT STEP
Write e2_dropends.py: on the C39 atom family, build eps' as a smoothed staircase, push through
PRODUCTION ht_derivative on BOTH grid conventions and through the independent spectral engine, and
measure (i) the detection threshold in M1 as a function of N, (ii) the near-seam noise floor of a
KNOWN-passive profile at the same N. Sound iff (i) falls faster than (ii).

## 6. OPEN QUESTIONS / BLOCKERS
- NEW (First Mate, mid-flight): reconcile alpha_c = 0.66623 (C18) vs the auditor's "established
  critical alpha = 0.2667" for opposite_bumps. Determine which is right and what each refers to.
- C42: at a+b = L the production DiHiTI returns a small NEGATIVE min at every finite N, O(h^2).
  Any recommended convention must NOT read that as gain.
- corpus grid is np.linspace(-2.5,2.5,2048) = DUPLICATED endpoint (the C22 bug grid). Effective
  period 2L*N/(N-1). Must quantify whether corpus labels move under the correct periodic grid.
- when E=0 but g(+-L)=c!=0, M2 = +inf (sec^2 not integrable). What is the true seam rate then? open.
