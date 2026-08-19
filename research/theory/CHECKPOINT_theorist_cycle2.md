# THEORIST — CYCLE 2 RESUMABLE WORK STATE
Updated 2026-08-19 (T3 essentially done). Read this alone to resume.

## 1. GOAL
T3 fix the 233/240 corpus defect + VERIFY the mechanism (C28 is under independent audit).
T1 close Open Problem 8.1 (skew tolerance) via (L2) -> <=3-atom -> Cardano.
T2 (if time) CFM/Choquet conjecture.
V6: stay at 3 statements (L1,L2,L3); new stuff = corollaries only.

## 2. ESTABLISHED SO FAR (this cycle)

### T3 — DONE. Agreement is now 240/240.
New evaluator `research/theory/t3_clausen.py` :: `gap_clausen(e_re, x) -> (eps''_dom, D)`.
Method (singularity-free, no tan() overflow, no ln at a singular point):
 * drop measure treated as piecewise constant on grid cells with EXACT cell masses
   w_j = eps'(t_j) - eps'(t_{j+1});
 * the log kernel is integrated EXACTLY over each cell. All antiderivatives are Clausen Cl2.
 * Cl2 evaluated EXACTLY on the uniform grid theta_k = 2 pi k/M by aliasing the Fourier
   series sum_n sin(n th)/n^2 into a trigamma-weighted DFT:
       Cl2(2 pi k/M) = sum_{r=1}^{M-1} sin(2 pi r k/M) psi_1(r/M)/M^2
   verified: Cl2(pi/3)=1.014941606409654 (ref ...6536), Cl2(pi/2)=0.9159655941772193
   (Catalan, ref ...190).  [PROVED identity + VERIFIED to 1e-15]
 * the -D ln cos(pi x/2L) terms cancel exactly if D := sum_j w_j (they do, by construction).
 * evaluated by one convolution; whole 240-profile corpus in 0.3 s.

RESULT: **240/240** on the same corpus (pg.sample(240, rng=default_rng(7))).
Pointwise vs DiHiTI: median rel err 5.3e-4, p90 4.5e-3. (The 11 "rel=1e9" entries are the
`sine` family where eps''<=0 everywhere so scale=max(eps'')=0; absolute err there is ~1.4e-3
on |eps''|~2. Normalisation artefact, not an error.)

### C28's DIAGNOSIS IS WRONG (this matters — see mechanism tests, t3_mechanism.py)
C28 said: "the 7 disagreements are edge-slope-UNMATCHED families where the Cayley integral is
only conditionally convergent and the 0.97L truncation cuts the near-seam minimum."
Every clause of that is false:
 * TEST 1: sweeping the truncation frac 0.90 -> 0.9999 leaves agreement at **233/240**
   (235 at 0.90, 233 at 0.95/0.97/0.99/0.999/0.9999). Truncation is NOT the cause.
 * TEST 2: the DiHiTI minimum of the 5 GAIN cases is at |x|/L = 0.80-0.91, INSIDE the mask.
   (The 2 logistic cases have their min at the endpoint, = 0 by C13/C21.)
 * TEST 6: 4 of the 7 offenders have g(+L) EXACTLY equal to g(-L) (opposite_bumps 72,
   lorentzian 170, logistic 41, logistic 199). Slope MISmatch is irrelevant.
 * The integral int ln|eta-xi(t)| g dt has only a LOG singularity at t=+-L and is
   ABSOLUTELY convergent whenever (H) holds. "Conditionally convergent" is wrong.

### THE TRUE MECHANISM  [verified numerically]
The old error is an **eta-independent additive constant** (TEST 3: std/|mean| = 0.4-4%),
i.e. a DC error, produced at the two endpoints t = +-L where the t-space integrand
ln|eta - tan(pi t/2L)| has an integrable log singularity. Two facts pin it:
 * TEST 6: offset / (g(+L)+g(-L)) = -0.0331, -0.0335, -0.0331, -0.0331, -0.0332, -0.0349,
   -0.0352 across the 7 profiles -- a **universal constant** over offsets spanning
   -0.0055 to +0.289 and edge sums +0.156 to -8.74. So offset = c(h)*(g(+L)+g(-L)).
 * TEST 5: the offset is exactly O(h): halving h halves it (ratios 2.05,2.05,2.05,2.05
   over N=1024..16384), with c(h) carrying a slow log enhancement.
 * Analytic account of the leading share: at t=+-L, tan(pi/2) evaluates in float to
   1.633e16, so ln|eta-xi| is recorded as 37.33 instead of the correct half-cell average
   1 - ln(pi h/4L) = 8.17 (N=2048). That single node contributes
   (h/2)(g(-L)+g(+L))(37.33-8.17) -> eps'' offset -0.0113*(sum), i.e. **1/3** of the
   observed -0.0331. The remaining 2/3 is accumulated trapezoid error over the near-edge
   cells of the same log singularity. Both shares are O(h) and both are proportional to
   g(+-L). [mechanism verified; the exact constant is not needed and is not claimed]
So: C28's CONCLUSION ("a xi-space quadrature defect, not a defect of (L1)") is CORRECT and
now demonstrated; C28's stated MECHANISM is wrong and must be rewritten.
It is a DC/endpoint-singularity defect proportional to g(+-L), not a truncation or
conditional-convergence defect and nothing to do with edge-slope mismatch.

## 3. RULED OUT
* Truncation |x|<frac*L is not the cause (TEST 1).
* Edge-slope MISmatch is not the cause (TEST 6).
* "Conditional convergence" of the Cayley integral: not a thing here; (H) gives absolute
  convergence.

## 4. CURRENTLY IN PROGRESS
Robustness of the 240/240 (other seeds, larger corpus, near-threshold cases), then a
self-convergence test to attribute the residual 5e-4 (Clausen evaluator vs DiHiTI itself).

## 5. NEXT STEP
Run t3_corpus.py with seeds 1,2,3,11 and n=400 each; then start T1 (skew tolerance).

## 6. OPEN QUESTIONS / BLOCKERS
* Residual rel err ~5e-4 median: is it the new evaluator's O(h) or DiHiTI's own O(h)?
  Not yet attributed. Does not affect verdicts.

## FILES (all under research/theory/)
* t3_clausen.py   -- the fixed evaluator (importable). RUNS.
* t3_corpus.py    -- 240-profile agreement run. RUNS -> 240/240.
* t3_mechanism.py -- TESTS 1-6 above. RUNS.
* t3_diag.py, t3_diag2.py, t3_diag3.py -- diagnosis scratch. RUN.

---
## ADDENDUM (post-coordinator-message) — T3 CLOSED, (L1) EXONERATED

The coordinator relayed the auditor's independent falsification of C28. **The auditor and I
agree on both negative findings** (truncation is not the cause; only 3/7 are slope-unmatched)
— I reached them independently in t3_mechanism.py TESTS 1, 2 and 6. I have gone further and
found the actual cause and fixed it.

The coordinator's "corpus labels may be wrong" hypothesis: **TESTED AND FALSE.** t3_labels.py:
  * relabelling all 240 by the INTERIOR minimum (|x| < 0.97L) instead of the whole-domain
    minimum changes **0 of 240** labels. The corpus labels are sound.
  * with the fixed evaluator, (L1)-interior vs DiHiTI-interior disagree on **0 of 240**.
  * the two `logistic` cases (the paper's own profile) were FALSE ALARMS of the old
    evaluator: DiHiTI interior min = +1.319e-3 / +1.337e-3, (L1) fixed gives
    +1.283e-3 / +1.300e-3 -- both positive, both PASS. The old evaluator's DC offset of
    -6.4e-3 / -5.5e-3 pushed them negative.
  * likewise the 5 GAIN cases: DiHiTI -0.2287, (L1) fixed -0.2324, etc.
So (L1) is not wrong on the logistic, and it is not wrong on opposite_bumps. The defect was
entirely in the cycle-1 xi-space quadrature.

### ROBUSTNESS
2400/2400 agreement over 6 independent seeds x 400 profiles (t3_robust.py).

### ACCURACY ATTRIBUTION (t3_threshold.py)
Against a fine reference at N=131073, on tanh_slopes:
   N      |clausen - ref|      |dihiti - ref|
  2049      1.039e-04            4.697e-03
  4097      2.590e-05            2.402e-03
  8193      6.449e-06            1.208e-03
 16385      1.593e-06            5.845e-04
 32769      3.791e-07            2.590e-04
The Clausen evaluator converges **O(h^2)** (ratio 4.0 per halving); the scipy DiHiTI pipeline
converges **O(h)** (ratio 2.0). The residual ~5e-4 median difference is therefore **DiHiTI's
error, not (L1)'s**. The potential-theory route is the more accurate of the two.

### NEAR-THRESHOLD
opposite_bumps alpha-crossover bisected independently by both routes (N=16385, interior min,
relative tolerance): DiHiTI alpha_c = 0.266719, (L1) alpha_c = 0.266687 -- agree to 1.2e-4.
(NB this differs from C18's 0.66623 because C18 used a different tolerance/interior
definition; the point here is only that the two ROUTES agree, which they do.)

### PROPOSED LEDGER ACTION
C28 should be REWRITTEN, not merely promoted:
  old text: "233/240; 7 disagreements are edge-slope-unmatched; conditionally convergent
             Cayley integral; 0.97L truncation cuts the near-seam minimum"  -- MECHANISM FALSE
  new text: "(L1) corpus agreement is 240/240 (2400/2400 over 6 seeds) once the gap is
             evaluated with an exact per-cell Clausen quadrature. The cycle-1 233/240 was a
             DC quadrature defect at t=+-L, proportional to g(+L)+g(-L) and O(h); it is
             unrelated to truncation and to edge-slope mismatch. Corpus labels verified
             sound (0/240 change under interior-min relabelling)."

---
## T1 — Open Problem 8.1 (skew tolerance). IN PROGRESS.

### Exact statement of 8.1 (extracted from the Wave-3 PDF, verbatim)
"Skew tolerance. Quantify how far an M1-balanced unimodal density may be skewed before
failure (Fig. 2c shows failure exists); conjecturally an interpolation between Theorems 4.2
and 5.2 controlled by the mass outside [-L/2,L/2] weighted by the margin (12)."

### Machinery built (PROVED identities, VERIFIED numerically)
`research/theory/t1_arcs.py` (reference) and `t1_fast.py` (vectorised).
CIRCLE FORM of (L1): phi = pi t/L in (-pi,pi), seam at +-pi, mu = drop measure / D.
    eps''(phi) = (D/pi)[ Phi_mu(pi) - Phi_mu(theta) ],  Phi_mu(t)=int ln|2 sin((t-phi)/2)| dmu
    PASSIVE <=> Phi_mu(theta) <= Phi_mu(pi) for all theta.
    Balance: int tan(phi/2) dmu = 0.
KHINCHIN (layer cake): mu unimodal <=> mixture of UNIFORM ARCS whose level sets are
intervals all containing the mode. With mass + M1 constraints, Dubins-Klee gives
ext = mixtures of AT MOST TWO overlapping arcs. So (L2) reduces 8.1 to a 4-parameter check.
Closed forms (both verified against 4e5-point quadrature to 6e-6 / 4e-9):
    Phi_arc(theta;[p,q]) = [Cl2(theta-q) - Cl2(theta-p)]/(q-p)
    M1_arc([p,q])        = ln[cos(p/2)/cos(q/2)]/((q-p)/2)
Cl2 implemented two ways, both 1e-15: (i) exact trigamma-DFT on the uniform grid
(t3_clausen.py), (ii) the Lewin series Cl2 = th - th ln th + sum zeta(2n) th^(2n+1)
/(n(2n+1)(2pi)^(2n)) (t1_arcs.py / t1_fast.py). Verified vs Gieseking + Catalan.
The whole circle/arc closed form was validated against DiHiTI on 4 built profiles:
max|pred - DiHiTI| = 1.2e-5 .. 1.8e-4 (t1_validate.py).

### RULED OUT / CORRECTIONS TO MY OWN CYCLE-1 PLAN
* **"3 linear constraints => <=3 atoms => degree-3 Q => Cardano" DOES NOT APPLY to the
  unimodal family.** The extreme points of a unimodal family are ARCS (Khinchin), not
  atoms, so Phi'_mu(theta) = sum_k rho_k ln|sin((theta-p_k)/2)/sin((theta-q_k)/2)| is
  TRANSCENDENTAL, not a polynomial. (L3)'s polynomial collapse is for atomic measures only.
  My cycle-1 §12.5.1 prediction of a Cardano closed form was WRONG. Retract it.
* **First search was ill-posed.** Letting an arc endpoint reach the seam (+-pi) makes
  M1 = int tan(phi/2) dmu DIVERGE (logarithmically), so the "balanced" configurations the
  optimiser found (gain 0.32 at phi_m=0) are balanced only in a regularised sense. They
  satisfy (H) (sigma finite) but not M1 finite. This is exactly the cycle-1 remark that M1
  converges only as a principal value, requiring g(+L)=g(-L). The family must therefore be
  posed with support strictly inside (-pi,pi) (equivalently g(+-L)=0), which makes M1
  absolutely convergent.

### REFORMULATED TARGET (this is the well-posed version of 8.1)
For mu unimodal, M1 = 0, supp mu = [P,Q] subset (-pi,pi):
  * Cor 4.B (no unimodality) => passive iff Q - P <= pi.
  * Thm 5.2 (even unimodal, P=-Q) => passive for every Q < pi.
  Find the PASS/FAIL boundary in the (P,Q) plane. That curve IS the skew tolerance.
Extreme points to check: two arcs [p1,q1],[p2,q2] subset [P,Q] that OVERLAP (disjoint arcs
would make the mixture bimodal), balanced.

### NEXT STEP
Run the (P,Q) scan: for each (P,Q) on a grid in (-pi,pi)^2 with P<Q, maximise the gain over
overlapping balanced 2-arc mixtures inside [P,Q] (differential evolution, t1_scan.py has the
optimiser skeleton). Map the zero-gain contour. Then guess + prove the closed form.

---
## T1 RESULT — Open Problem 8.1 CLOSED, with a NEGATIVE answer. [verified numerically]

**THE SKEW TOLERANCE IS ZERO.** For every mode position c != 0 there exist unimodal,
M1-balanced drop densities whose DiHiTI partner has gain. So Thm 5.2's evenness hypothesis
CANNOT be weakened to "unimodal + balanced", and the interpolation Wave 3 conjectured
("controlled by the mass outside [-L/2,L/2]") does not exist as stated.

### MECHANISM (proved, elementary)
M1 = int tan(phi/2) dmu is a xi-MOMENT, and xi = tan(phi/2) blows up at the seam phi=+-pi.
So a mass eps placed within O(eps) of the seam contributes O(1) to M1 but only
O(eps ln(1/eps)) -> 0 to the logarithmic potential Phi_mu. Hence the balance constraint can
be satisfied VACUOUSLY, and in the limit the measure is potential-indistinguishable from the
single atom delta_c, whose gain is ln sec(c/2) > 0 (that is exactly Cor 4.4 / C10, the
off-centre step). Unimodality does not obstruct this: put the atom at the LEFT END of a wide
arc reaching to the seam -- the density is still unimodal (peak at c, plateau to the right).

### THE COUNTEREXAMPLE FAMILY (explicit)
  mu = lam * (narrow arc at c) + (1-lam) * U[c, q],   q -> pi^-,
  lam fixed by lam*tan(c/2) + (1-lam)*(2/(q-c))*ln[cos(c/2)/cos(q/2)] = 0.
Verified against the scipy DiHiTI pipeline AND against the fixed (L1) evaluator (N=65536):
   c/pi   q/pi     lam      M1      gain(closed form)   min eps'' DiHiTI   min eps'' (L1)
  -0.50  0.9000  0.4069  2.2e-03        0.000000           -0.000000        -0.000000
  -0.50  0.9700  0.5398  2.8e-03        0.000000           -0.000000        -0.000000
  -0.50  0.9950  0.6571  3.8e-03        0.090341           -0.028372        -0.028380
  -0.50  0.9990  0.7218  4.8e-03        0.142034           -0.044784        -0.044793
  -0.50  0.9999  0.7812 -2.3e-03        0.186300           -0.058832        -0.058841
  -0.25  0.9950  0.8548 -2.2e-03        0.012973           -0.004261        -0.004264
  -0.75  0.9900  0.3261  4.3e-03        0.108993           -0.034157        -0.034171
  -0.10  0.9990  0.9593  1.0e-03        0.002356           -0.000732        -0.000733
(all three columns agree to 1e-5; gain -> ln sec(c/2) = 0.3466 for c = -pi/2 as q -> pi.)

### WHY THIS IS A COROLLARY, NOT A NEW STATEMENT (V6)
It is (L1) + Khinchin (already used by Thm 5.2 via (L2)) + the C10 off-centre-step formula,
which cycle 1 already showed is the eta = -1/xi_0 value of the same Delta. Corpus count
unchanged at 3.

### CONJECTURED SHARPENING (NOT yet verified)
sup{ -pi*min eps''/D : mu unimodal with mode c, M1 = 0 } = ln sec(c/2), i.e. the off-centre
step is the WORST unimodal balanced profile with that mode. Attained only in the seam limit.
STATUS: conjecture -- the lower bound (>=) is established by the family above; the upper
bound (<=) is not proved and not yet numerically probed.

### NEXT STEP
1. probe the upper bound sup = ln sec(c/2) numerically (search unimodal balanced mu with
   mode c for gain > ln sec(c/2));
2. give the REPAIRED criterion: unimodal + M1=0 + supp bounded away from the seam. Map the
   sharp (P,Q) boundary with the dense 2-arc search (t1_dense.py is the working search;
   t1_pq.py's differential_evolution was too weak and reported spurious all-zeros).

### FILES
t1_arcs.py (reference Cl2 + arc closed forms, self-tested), t1_fast.py (vectorised),
t1_validate.py (arc form vs DiHiTI, 1e-4), t1_dense.py (dense random search -- FINDS the
failures), t1_boundary.py (atom-at-end boundary scan), t1_verify.py (the table above).
WARNING: t1_pq.py and t1_scan.py under-optimise; t1_scan.py's phi_m search is ILL-POSED
(lets arcs touch the seam, where M1 diverges). Do not trust their outputs.
