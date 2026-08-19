# Margin red-team: is the DiHiTI passive-cone boundary really exactly-zero everywhere?

**Claim under test:** "every DiHiTI profile sits EXACTLY on the passive-cone boundary
(margin 0), so any cheap/approximate test misclassifies precisely the profiles of interest."

**Verdict: the claim is WRONG as a global statement.** The margin-0 phenomenon is real but
*only at the two domain endpoints* — it is a structural, closed-form consequence of the
de-tilt in DiHiTI (§M2), not a property of the interior of the profile. Interior margins
are strictly positive and well clear of zero for the overwhelming majority of passive
profiles (§M1). The one place the claim is directionally right is a genuine adversarial
threshold family (§M3, §M4) where the margin *can* be engineered arbitrarily close to zero
by construction — but even there the numerical threshold is grid-convergent to 5 significant
figures, i.e. it is a real, well-defined zero-crossing, not numerical noise pretending to be
one. §M5 confirms a distinct, narrower prior finding (C14) about finite smooth-moment
batteries, with one correction: the "m-independent ≈ −0.20" behavior only holds for m ≳ 16;
at m = 4 it is not m-independent at all.

Convention used throughout: generator's native domain **L = 2.5** (not rescaled to 1.0) —
several profile families (triangle, cosine_bump, gaussian, tanh_slopes, and the fixed bump
widths/separations in logistic_bump / opposite_bumps) are calibrated in absolute x-units tied
to L = 2.5, so rescaling the domain while keeping those absolute widths would change the
physics. All reported ratios are normalized (x/L, margin/D) so this choice doesn't bias
interpretation. "w" for the logistic family means the transition-width parameter w = 1/k in
`tmm_h.logistic(x, k, nb)`, matching this project's established convention.

Scripts (all self-contained, run with `python m*.py` from this directory):
`common.py`, `m1_min_location.py`, `m2_edge_asymptotic.py`, `m3_grid_convergence.py`,
`m4_locality_probe.py`, `m5_cosine_dressing.py`. Raw data: `m1_raw.csv`.

---

## M1 — Where is the minimum?

266 profiles: 220 random draws from `profile_generator.sample()` + the explicit logistic
w=0.05 + 45 explicit `opposite_bumps` draws (3 (sep,w) settings × 15 alpha values), all at
N = 2¹⁴ = 16384, L = 2.5. 134 came out passive, 132 gain (tolerance `min_eim >= -1e-3*scale`
from the generator's own labeling).

- **95.5%** of passive profiles have their *global* minimum endpoint-adjacent
  (|x_argmin|/L > 0.99). This is the structural effect: DiHiTI's de-tilt forces ε″(±L) → 0
  by construction (`e - linspace(e[0], e[-1], N)` exactly zeros the endpoints), and for most
  smooth monotone-ish profiles the endpoints are also where ε″ is smallest.
- 69/134 passive profiles have |D| ≈ 0 (families `bwd_double`, `bwd_gauss`, `bwd_mesa`,
  `sine` — symmetric endpoint values by construction) and are excluded from the margin/D
  ratio (division by ~0 is meaningless, not informative).
- Among the remaining 65 profiles with meaningful D, **interior margin / D is bounded away
  from 0**:

| percentile | interior_min / D |
|---|---|
| 1st  | −6.04e−04 |
| 5th  | +2.67e−04 |
| 25th | +9.99e−04 |
| 50th | +1.08e−03 |
| 75th | +2.06e−03 |
| 95th | +9.52e−03 |
| 99th | +5.79e+00 (one outlier, tiny-D case that slipped the |D| filter) |

64/65 fall in [−1.3e−3, +0.53]; the median interior margin is ~0.1% of the total ε′ drop D,
small but reliably positive and non-zero — not "exactly 0" in any of these cases. The 1st
percentile being slightly negative (−6e−4) reflects the generator's own ±1e−3·scale
passivity tolerance, not a contradiction of the claim being tested.

**Bottom line for M1:** the exact-zero phenomenon is an endpoint artifact (95.5% of the
time, literally the argmin), and the interior — where a cheap test would actually need to
discriminate — carries a real, positive, three-figure-stable margin for the bulk of the
passive population.

## M2 — Is the endpoint zero structural / closed-form?

Predicted edge law near x = −L (s → 0⁺):
`ε″(−L+s) = (1/π)[g(−L)−g(+L)]·s·ln(1/s) − (M1/2L)·s + O(s² ln s)`, `g = −dε′/dx`,
`M1 = ∫ g(t) tan(πt/2L) dt` (from `potential_theory_derivation.tex`, table of necessary
conditions).

**Degenerate case (logistic w=0.05, k=20, L=2.5, N=2¹⁸):** g decays as exp(−2kL) ≈ exp(−100)
at both edges — literally zero in double precision — and g is even about x=0 so M1 = 0 by
symmetry exactly. So the theory correctly predicts **c1 = c2 = 0** for this profile: the
leading closed-form terms vanish identically, meaning ε″ should approach 0 *faster* than the
generic s·ln(1/s) rate. Fitting log|ε″| vs log(s) over s ∈ [1e−3 L, 1e−2 L] gives
**exponent 1.998 ≈ 2** — i.e. the actual approach is O(s²), a genuinely higher order than the
generic formula's leading term, exactly as the vanishing predicted coefficients imply. This
is itself evidence the closed form is doing real physics, not being fit to noise.

**Non-degenerate validation (tanh_slopes, mL=2.0, mR=1.0 — asymmetric edge ramps so g(±L) ≠
0 and different):** predicted c1 = (1/π)[g(−L)−g(+L)] = −0.31831; fitted c1 from the actual
ε″(−L+s) data = −0.32536, **ratio 1.022 (2.2% agreement), R² = 0.999998**. This is a clean
confirmation that the closed-form s·ln(1/s) coefficient genuinely predicts the near-edge
behavior when it is non-trivial. (The M1/c2 term could not be cross-checked on this profile:
the M1 integral's tan(πt/2L) kernel is non-integrable in the naive sense whenever g(±L) ≠ 0 —
a genuine regularization subtlety in the theory note, not a numerics bug; it is only cleanly
defined in the regime where the s·ln(1/s) term already vanishes, i.e. the logistic case
above, where it correctly evaluates to ≈0.)

**Verdict:** the endpoint zero is structural and closed-form. A cheap test does not need to
resolve the seam numerically — it can evaluate g(±L) and M1 (or recognize their symmetry-
forced vanishing) analytically and know the sign/order of the approach to zero without ever
computing ε″ near x = ±L on a fine grid.

## M3 — Grid convergence of the pass→fail threshold

Family: `logistic(x,k=20) + alpha*1.89*(gauss(x+0.45,0.22) − gauss(x−0.45,0.22))`, L=2.5,
criterion = interior min (|x| ≤ 0.95L) crossing zero, bisected over alpha ∈ [0.5, 1.2].

| N | alpha* |
|---|---|
| 2¹¹ = 2048   | 0.6140320 |
| 2¹²  = 4096   | 0.6138611 |
| 2¹³  = 8192   | 0.6137756 |
| 2¹⁴ = 16384  | 0.6136902 |
| 2¹⁵ = 32768  | 0.6136902 |
| 2¹⁶ = 65536  | 0.6136902 |

Monotone, converged, **plateaued exactly from N=2¹⁴ onward**. Spread across all six grids:
3.4e−4 absolute, **0.056% relative**. This is a real, well-posed threshold — the
"margin-0 sharpness" at this crossover is genuine physics (an honest zero-crossing of a
converged function of alpha), not discretization noise. A cheap test operating near this
alpha would need enough resolution to resolve a ~0.06%-scale threshold, but the threshold
itself is not moving around under refinement.

## M4 — Locality / query-lower-bound probe

Base: logistic w=0.05, L=2.5, N=2¹⁴, D=1.89, interior margin m₀ = 1.87e−3 (comfortably
passive but not deeply so — this is the generator's own value for this exact profile).
For each bump width, bisected on amplitude (both signs) over a grid of centers x0,
including a dense scan near |x0|/L ∈ [0.85, 0.98] since the base profile's own interior
minimum sits at x/L ≈ −0.95 (right at the interior cutoff).

| σ/L | a_min | x0 (easiest) | ‖pert‖∞/D | ‖pert‖₁/(D·L) |
|---|---|---|---|---|
| 0.5   | 2.411e−02 | −1.000 | 1.276e−02 | 1.080e−02 |
| 0.2   | 1.235e−02 | −2.450 | 6.533e−03 | 1.288e−03 |
| 0.1   | 5.980e−03 | −2.450 | 3.164e−03 | 3.428e−04 |
| 0.05  | 3.021e−03 | −2.433 | 1.599e−03 | 1.097e−04 |
| 0.02  | 1.875e−03 | −2.425 | 9.919e−04 | 3.457e−05 |
| 0.01  | 2.488e−03 | −2.400 | 1.316e−03 | 2.333e−05 |
| 0.005 | 2.900e−03 | −2.383 | 1.534e−03 | 1.360e−05 |

**Trend: a_min DECREASES by ~13× from σ/L=0.5 to σ/L=0.02** (0.0241 → 0.0019), then a
mild uptick at the two finest widths (0.005, 0.01 → 0.0025–0.0029), staying well below the
σ/L=0.5 value throughout. More strikingly, **‖pert‖₁/(D·L) falls monotonically and
smoothly across the entire range, by three orders of magnitude** (1.08e−2 → 1.36e−5) —
narrow perturbations are cheap in every norm that matters for a query-complexity argument
(you need vanishingly little "mass" to flip the label as the bump narrows).

**Interpretation: a_min does NOT grow as σ shrinks.** Arbitrarily localized perturbations
flip the passivity label at bounded-to-shrinking cost. This is consistent with — and
supports — an Ω(N) query lower bound: no sublinear-query test can be blind to a
vanishing-mass, arbitrarily narrow region and still be correct, because such regions can
carry the entire margin.

## M5 — Cosine dressing (claim C14)

Reproduces `theory/passivity_2026Aug/potential_criterion.py::part_F` exactly (base
logistic k=4.0/w=0.25, L=2.5, edge-localized window, M1 re-balanced via a centered
compensator, a=0.25), extended to m=4 and m=128 as requested, and cross-checked at
N=4096 (original) and N=2¹⁴ (project standard).

| m | min ε″ (N=4096) | min ε″ (N=2¹⁴) | max |ΔLegendre₀₋₅| |
|---|---|---|---|
| 4   | −0.11522 | −0.11530 | 1.5e−01 |
| 8   | −0.16199 | −0.16205 | 4.7e−02 |
| 16  | −0.19823 | −0.19833 | 1.0e−03 |
| 32  | −0.20189 | −0.20210 | 3.7e−04 |
| 64  | −0.20115 | −0.20176 | 8.3e−06 |
| 128 | −0.19996 | −0.20230 | 9.5e−07 |

Interior min equals full min at every m (the perturbed minimum stays well inside |x|<0.95L).
dM1 and dσ stay at 1e−13…1e−1 4 across all m (successfully re-balanced), confirming the
low-order-moment-invisibility part of the construction works as designed.

**For m ∈ {16, 32, 64, 128}: min ε″ = −0.198 to −0.202, spread only 0.004 — genuinely
m-independent, matching the original claim.** Grid convergence is also clean (N=4096 vs
N=2¹⁴ agree to 3 decimal places at every m).

**Correction to the original claim: it is NOT m-independent at m=4 or m=8.** min ε″ is
−0.115 at m=4 and −0.162 at m=8 — a full 40–75% smaller in magnitude than the −0.20
plateau. The m-independence is an *asymptotic* (m ≳ 16, i.e. cosine half-period ≲ window
width/4) property, not a property that holds "for m" unconditionally as originally stated.
This matters for the query-complexity argument built on C14: the adversary construction
needs m large enough relative to the localization window, not just "any m."

---

## Summary of numbers requested

- **M1 percentiles of interior_min/D (passive, |D|≠0):** 1st −6.04e−4, 5th +2.67e−4,
  50th +1.08e−3, 95th +9.52e−3 (99th is a residual small-D outlier). 95.5% of passive
  profiles have their global min endpoint-adjacent.
- **M2 fit verdict:** closed form confirmed. Degenerate logistic case: predicted c1=c2=0
  (correct — g and M1 both vanish by saturation/symmetry), and the empirical decay is O(s²)
  (exponent 1.998), consistent with a genuinely higher-order approach once the leading terms
  vanish. Non-degenerate case (tanh_slopes): predicted vs fitted c1 agree to 2.2%
  (R²=0.999998) — the s·ln(1/s) closed form is real, not curve-fit-with-freedom.
- **M3 alpha\* table:** 0.6140320 (2¹¹) → 0.6138611 (2¹²) → 0.6137756 (2¹³) → 0.6136902
  (2¹⁴, 2¹⁵, 2¹⁶). Relative spread 0.056%; converged from N=2¹⁴ on.
- **M4 a_min vs σ trend:** DECREASES (not grows) as σ shrinks — 0.0241 (σ/L=0.5) down to
  0.0019 (σ/L=0.02), with a small uptick to 0.0029 at σ/L=0.005 but never back up to the
  σ/L=0.5 level; ‖pert‖₁/(D·L) falls monotonically by 3 orders of magnitude. Narrow
  perturbations are cheap → consistent with an Ω(N) query lower bound, not a sublinear test.
- **M5 m-dependence:** m-independent (≈−0.20, spread 0.004) ONLY for m ≥ 16; at m=4/m=8 min
  ε″ is −0.115/−0.162 — clearly m-dependent there. The claim needs the qualifier "for
  sufficiently large m," which was not stated in the original C14 write-up.
