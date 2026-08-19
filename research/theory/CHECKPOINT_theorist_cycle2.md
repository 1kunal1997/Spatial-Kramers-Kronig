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
