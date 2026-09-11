# sKK-as-PML terminator study — figure index

All figures come from **one** COMSOL model (`Comsol Files/skk_master.mph`), so every comparison
shares the same geometry, mesh, port and solver. ~350 solve points.
Data: `comsol_skk/master_data.npz`. Scripts: `master_data.py` → `patch_pml.py` → `patch2.py` → `make_figures.py`.

**Setup:** 2D, s-pol (Ez), λ = 800 nm, substrate n_b = 1.7, period L_x = λ/4 (only diffraction
order 0 propagates), periodic Port on top, Floquet sides, PEC backing. R = `ewfd.Rtotal`.

## Advisor questions → figure

| Question | Figure | Answer |
|---|---|---|
| "Did you hold the mesh constant and sweep the terminators?" | **10** | Yes — R vs element count, 5→1000, all terminators, θ=80° |
| "Did you hold length constant and sweep amplitude?" | **15** | Yes — L=8λ fixed, A swept 1→16. R ∝ A² |
| "Did you hold length constant and sweep width?" | **16** | Yes — L=8λ fixed; γ, w and ℓ each swept |
| "Are all curves at the same thickness AND mesh?" | **07, 13, 14** | Yes — identical geometry and 800+800 elements on every curve |
| "How does it scale with thickness?" | **09** (sKK), **04** (PML: it doesn't) | |
| "Was the mesh constant in fig 9?" | **09** | Not along x, deliberately: N = 400 for L ≤ 8λ, then 50/λ (5000 at L=100λ), holding element size at 0.02λ = 10 per feature. **Same rule for all three profiles**, so vertical comparisons are at identical mesh. Converged: ×2 and ×4 mesh both give 0.00% change. |
| "Which PMLfactor is in fig 9?" | **09** | Both, pf=1 and pf=4. |
| "Did you tune the PML or use the default?" | **02, 03, 03b** | pf = 1, 2, 4 all shown |
| "Was the PML length constant in the element sweep?" | **03, 03b** | Yes, 2λ throughout |
| "Was the mesh constant in the thickness sweep?" | **04** | Yes, 400 elements at every thickness (pf = 1) |

## A–E controlled-variable set → figures

- **A — fixed total thickness, vary mechanism:** 07 (8λ), 13 (2λ), 14 (12λ)
- **B — R vs thickness at fixed angle:** 09 (sKK, θ=80°), 04 (PML)
- **C — stacking ratio:** dropped (stacking never wins; see note below)
- **D — R vs element count:** 10 (all terminators), 03 / 03b (PML vs PMLfactor)
- **E — backing isolated:** dropped as a figure; result stated in text below

## Figures

| # | File | What it shows |
|---|---|---|
| 01 | `01_geometry_schematic.png` | The harness |
| 02 | `02_pml_baseline.png` | PML vs angle for PMLfactor = 1, 2, 4 |
| 03 | `03_pml_R_vs_elements_theta0.png` | PML cost of tuning, θ = 0° |
| 03b | `03b_pml_R_vs_elements_theta80.png` | Same at θ = 80° (the regime that matters) |
| 04 | `04_pml_R_vs_thickness.png` | PML thickness buys **nothing** |
| 05 | `05_profiles_epsilon_real.png` | ε′(y), entry face at left |
| 06 | `06_profiles_epsilon_imag.png` | ε″(y) |
| 07 | `07_comparison_L8.png` | **Headline.** L = 8λ, identical mesh |
| 09 | `09_truncation_effects.png` | R vs thickness at θ = 80°, L = 1→100λ, vs PML pf=1 **and** pf=4 |
| 10 | `10_equal_cost_comparison.png` | **Equal-cost.** R vs elements at θ = 80° |
| 13 | `13_comparison_L2.png` | Same as 07 at L = 2λ |
| 14 | `14_comparison_L12.png` | Same as 07 at L = 12λ |
| 15 | `15_param_sweep_amplitude.png` | Length fixed, amplitude swept |
| 16 | `16_param_sweep_width.png` | Length fixed, width swept |
| 17 | `17_sum_rule_and_gain_theorem.png` | **Why:** sum rule and ∫HT = 0 |
| 18 | `18_gain_terminator.png` | Gain profile + what it does |

## Results not shown as figures

- **Stacking sKK in front of a PML never wins.** With total thickness fixed at 8λ, R rises
  monotonically as thickness is transferred from PML to sKK.
- **Backing test (PEC vs PML behind the same profile):** the logistic improves 4429×
  (leak-limited); the Lorentzian family does not move at all (truncation-limited).

## Caveats to state out loud

1. **Above θ_c = 36° the logistic is past TIR.** Its low R there is evanescent absorption, not
   the sKK mechanism. The Lorentzian family has ε_b on both sides — no step, no TIR — so its
   full 0–90° range is attributable to sKK.
2. Figs 09/10 compare sKK at L = 8λ against a PML at L = 2λ. Fair only because extra thickness
   does nothing for a PML — **show fig 04 first.**
3. Values below ~1e-12 are near the solver/mesh resolution of this model.
4. **At θ = 80°, a tuned PML (pf=4, R = 5.8e-7) beats every sKK profile at every thickness up to
   100λ** — the logistic gets closest at 8.6e-7. The sKK advantage at grazing is against the
   *default* PML (pf=1, R = 1.3e-2), not a tuned one. Say this before being asked.

---

## Grazing-incidence cost study (2026-09-03) — figures 21-25

**All COMSOL.** Scripts: `patch9.py` (264 solves, element sweeps) then `patch10.py` (11 solves,
1-deg angle sweep at converged mesh), then `grazing_figures.py`. Keys `P9_*` and `P10_*`.
Same model, mesh rule, port and solver as everything else in this folder.

Configurations: PML at pf = 1, 4, 10, 20, 50 (2 lam thick); sKK logistic and tapered Lorentzian
at L = 8, 32, 100 lam, self-similar family w = L/80, ell = 2.5w. Angles 0/45/80/85/89 for the
element sweeps, 0-89 at 1 deg for the converged-mesh sweep.

| # | File | What it shows |
|---|---|---|
| 21 | `21_grazing_equal_cost_theta80/85/89.png` | Equal-cost: R vs element budget at three grazing angles |
| 22 | `22_grazing_crossover.png` | Converged R vs angle at 1 deg, plus ratio to the best PML |
| 23 | `23_grazing_saturation.png` | Does extra footprint still buy anything at grazing |
| 24 | `24_pml_measured.png` | Measured PML floor vs PMLfactor, and the mesh cost of tuning |
| 25 | `25_cost_vs_accuracy_theta89.png` | Scoreboard scatter: elements needed vs R delivered at 89 deg |

### Headline: the PML has an angle-dependent floor that NO PMLfactor beats

| theta | best PML R | achieved by | pf=50 R |
|---|---|---|---|
| 80 deg | 7.12e-7 | pf >= 10 | 7.12e-7 |
| 85 deg | 1.30e-5 | pf >= 20 | 1.30e-5 |
| 89 deg | 1.28e-2 | pf = 50 | 1.28e-2 |

Raising the PMLfactor stops helping: pf = 10, 20 and 50 give the SAME R at 80 and 85 deg. This is
the documented grazing-incidence PML failure (Oskooi et al., Opt. Express 16, 11376 (2008)) measured
in this harness. It also means the sKK profile is not chasing an arbitrarily good opponent.

### The mesh cost of tuning a PML falls at grazing

Measured elements to reach its own floor:

| pf | theta=0 | theta=80 | theta=89 |
|---|---|---|---|
| 1 | 52 | 5 | 5 |
| 4 | 249 | 11 | 5 |
| 10 | 544 | 24 | 5 |
| 20 | 544 | 52 | 5 |
| 50 | 2000 | 114 | 5 |

High PMLfactor is very expensive at normal incidence (2000 elements for pf=50, for no gain in R)
and nearly free at grazing. So there is no regime where the PML's tuning knob is priced out.

### Where the sKK profile is competitive

The logistic at L = 32-100 lam tracks the best PML within a factor of a few from ~45-86 deg, and
dips BELOW it in scattered narrow bands (49-57, 65-72, 83-86 deg; best 4.9x at 85 deg, L=100 lam,
2.65e-6 vs 1.30e-5). This is oscillatory interference structure, not a clean angular window.

Two caveats that have to be said out loud:
1. The logistic terminates in air, so the whole 45-86 deg range is past theta_c = 36 deg — that is
   evanescent absorption, not the sKK mechanism.
2. The tapered Lorentzian, which has eps_b on both sides and IS an honest sKK test at any angle,
   loses to the best PML by 1-3 orders at every angle.

Cost: the L = 100 lam logistic needs 148-323 elements and 100 lam of footprint to reach that floor,
against 11-52 elements and 2 lam for the PML.

### Scoreboard at theta = 89 deg (converged mesh)

| terminator | R | elements to floor | footprint |
|---|---|---|---|
| tapered Lorentzian, L = 100 lam | 2.25e-1 | 11 | 100 lam |
| logistic, L = 100 lam | 3.58e-1 | 8 | 100 lam |
| PML pf = 20 | 1.77e-2 | 5 | 2 lam |
| **PML pf = 50** | **1.28e-2** | **5** | **2 lam** |

At 89 deg every terminator is mesh-converged at N = 5, so the residual there is physics-limited,
not discretization-limited: spending mesh buys nothing for anyone.

### Superseded

An earlier version of this section extrapolated the PML to high pf with the law
R = exp(-8*pi*s*cos(theta)), fitted on pf = 1, 2, 4. That law is accurate for pf = 1 (0.017 decades
median over 0-89 deg) but **fails badly above pf ~ 4**: it predicted R = 3.1e-10 at 89 deg for
pf = 50, where the measured value is 1.28e-2. The PML saturates and the law does not. All numbers
in this section are measured; `grazing_study.py` (TMM + law) is kept only as the prediction that
the COMSOL run refuted.

---

## Mesh grading (2026-09-08) — figure 29. Script: `patch13.py`, keys `P13_*`

One configuration (tapered Lorentzian, L = 10 lam, gam = lam/5, ell = 0.5 lam, A = 4, n_b = 1.7),
50 elements across the terminator, three distributions of those same 50 elements.

| # | File | What it shows |
|---|---|---|
| 29 | `29_mesh_grading_r{5,20,100}.png` | R vs angle, uniform vs centre-dense vs entry-dense |

**Result: uniform wins. Grading the mesh makes it worse, at every ratio and every angle.**

R(theta=0), 50 elements (converged reference, 800 uniform elements: 1.87e-7):

| ratio | uniform | centre-dense | entry-dense |
|---|---|---|---|
| 5   | 1.39e-3 | 3.31e-1 | 3.81e-3 |
| 20  | 1.39e-3 | 1.07e-1 | 4.57e-2 |
| 100 | 1.39e-3 | 8.11e-1 | 1.61e-1 |

Centre-dense (the intuitive choice, since eps only varies in the middle) is the worst — up to
580x worse than uniform. The reason: what the mesh has to resolve here is the WAVE, not the
eps feature. The profile is 10 lam long, so ~10 elements per lam are needed everywhere the
field is non-negligible; starving the tails to feed the centre buys resolution where nothing
was under-resolved. Above ~78 deg all distributions coincide — the residual there is
physics-limited, not mesh-limited (consistent with the N=5 convergence seen at 89 deg in the
grazing study).

## Turned logistic in air (2026-09-08) — figures 30, 31. Script: `patch14.py`, keys `P14_*`

eps'' carries the logistic (0 -> 1, lossless -> lossy) and eps' is its Hilbert partner shifted
to eps' = 1 (air) at both ends: the standard sKK logistic with real and imaginary parts
exchanged, which is the same one-sided analytic function multiplied by e^{-i pi/2}. A constant
phase rotation cannot break one-sidedness, so the sKK property is exact, not approximate
(spectral one-sidedness FoM = 100.000%).

| # | File | What it shows |
|---|---|---|
| 30 | `30_air_flip_profile.png` | eps'(y) and eps''(y), L = 10 lam |
| 31 | `31_air_flip_R_vs_angle.png` | R vs angle, L = 2-100 lam, vs PML pf=1 and pf=4 in air |

**Four-way sign/orientation control (the rule from the Aug notes), COMSOL, L = 10 lam:**

| branch | orientation | R(0) | R(80) |
|---|---|---|---|
| eps' = 1 + HT[eps''] | forward | **2.53e-9** | **4.17e-3** |
| eps' = 1 + HT[eps''] | reversed | 4.70e-2 | 6.11e-1 |
| eps' = 1 - HT[eps''] | forward | 5.33e-2 | 1.00 |
| eps' = 1 - HT[eps''] | reversed | 4.70e-2 | 6.11e-1 |

Correct branch = eps' = 1 + HT[eps'']. TMM predicted 2.54e-9 for it; COMSOL gives 2.53e-9.

**Two things this buys.**
1. **The background is now AIR.** The standard logistic needs an index step (n_b -> air) to
   carry its loss, so it only works with a high-index background. This one returns to air at
   both ends, so it terminates a vacuum simulation — the case a PML user actually has.
   NOTE: every other figure in this folder is at n_b = 1.7 and is NOT comparable to 30/31.
2. **No TIR anywhere.** eps' >= 1 everywhere (peaks at 1.92 at L = 10 lam), so nothing is
   evanescent and the full 0-89 deg range is attributable to the sKK mechanism. This removes
   caveat 1 at the top of this file for this profile.

| terminator | R(0) | R(80) | R(89) |
|---|---|---|---|
| turned logistic L=10 lam | 2.53e-9 | 4.17e-3 | 0.600 |
| turned logistic L=32 lam | 2.27e-11 | 3.72e-5 | 0.398 |
| turned logistic L=100 lam | 6.71e-14 | 2.53e-7 | 0.196 |
| PML pf=1 (2 lam) | 1.22e-11 | 1.27e-2 | 0.646 |
| PML pf=4 (2 lam) | 1.83e-18 | 8.07e-9 | 0.178 |

Say out loud: **in air, a tuned PML (pf=4) still wins** at every angle except a tie at 89 deg.
The turned logistic at L = 100 lam beats the DEFAULT PML (pf=1) by 5e4x at 80 deg and 3.3x at
89 deg, at 50x the footprint. R(0) falls as L^-4, matching the logistic truncation law.
