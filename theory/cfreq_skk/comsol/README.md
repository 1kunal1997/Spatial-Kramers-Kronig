# COMSOL time-domain checks (complex-frequency sKK)

Independent check of the time-domain results (TMM × FFT synthesis, `fig_cfreq_feasibility.py`)
with COMSOL's **built-in** "Electromagnetic Waves, Transient" interface (tag `ewt`, Wave Optics)
and its **built-in Drude–Lorentz dispersion model**. Nothing hand-written in the physics.

## B3a: homogeneous 1λ slab, single Lorentz oscillator
- `build_b3a.py` builds the model from scratch through `mph`, solves it, and saves
  `B3a_slab_transient.mph` (with solutions, step T_r/200; gitignored, ~130 MB) and
  `../data/b3a_comsol_dt*.npz`.
- `fig_b3a_comsol.py` → `../figures/B3a_comsol_check.png` (COMSOL vs TMM × FFT).

### What is in the .mph
| Model Builder node | content |
|---|---|
| Global Definitions › Parameters | `eta`, `chi_im` (target Im χ at ω_c), `ratio` = γ/2\|ω″\|, and the design formulas for `gam`, `wp`, `w0` |
| Component › Definitions › Variables | the drive `Ein`: Planck turn-on, then e^{−bτ}·cos or sin(aτ), τ = t − T0 − x/c |
| Geometry | 2-D strip: air (1λ) │ slab (1λ) │ air (1λ), height λ/60 |
| Electromagnetic Waves, Transient | Air (n = 1); Slab: Drude–Lorentz (ε∞ = 1, ω_p = `wp`) with one Lorentz oscillator (f = 1, ω₀ = `w0`, Γ = `gam`); left scattering boundary injects `Ein`; right one absorbs; periodic top/bottom = normal-incidence plane wave |
| Study 1 | Time Dependent to 66 T_r, generalized-α with a fixed step T_r/200; Parametric Sweep over 4 runs: (chi_im, ratio, q) = (−0.3, 0.5, cos/sin) gain slab, (+0.3, 1.5, cos/sin) lossy slab |
| Results | *Exit-face field vs time* (exit-face field and drive); *E_z / drive envelope along the strip* (choose a time; the gain slab grows, the lossy one stays flat); *E_z in the strip* (animate it) |

Pick the run with the "Parameter value" list in each plot's settings. Runs 1 and 2 are the gain slab, runs 3 and 4 the lossy slab.

### Result
COMSOL matches the TMM × FFT curves of B3a panel a, and the difference converges at second
order in the time step:

| step | gain slab | lossy slab |
|---|---|---|
| T_r/100 | 3.7e‑2 | 8.8e‑3 |
| T_r/200 | 9.3e‑3 | 2.2e‑3 |
| T_r/400 | 2.3e‑3 | 5.4e‑4 |

The numbers are the median relative difference of the complex exit field.
