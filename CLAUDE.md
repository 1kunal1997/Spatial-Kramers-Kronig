# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a nanophotonics computational research project investigating the
**spatial Kramers-Kronig (sKK) relations** originally developed by Horsley et al. (2015). The broad goal is to either extend the theory (for example, zero and unit transmission cases by King, Horsley, Philbin in 2017), or find applications of this phenomena, such as anti-reflection (AR) coatings, thermal emission/absorption asymmetry, perfect absorbers, etc. Currently, the focus is AR coatings (found in coating_Hilbert_transform.py), specifically
targeting backside reflection suppression in **mid-IR ellipsometry** on sapphire substrates. The calculations of the transmission, absorption and reflection of multilayer materials/coatings are performed using Transfer Matrix Method (TMM) simulations.

## Running Code

This is a pure Python research project — no build system. Run scripts directly:

```bash
python theory/<script_name>.py          # theory scripts
python experimental/<script_name>.py    # experimental scripts
python -m pytest module_test_tmm_helper.py   # run tests
```

For interactive work, use VS Code interactive mode (#%%) .

## Core Architecture

### Key Modules

- **`tmm_helper.py`** — primary module imported as `tmm_h`. Contains:
  - `generate_n_and_d_v6_symmetry()` — the only discretization function (legacy versions archived)
  - `TRA()`, `TRA_wavelength()`, `TRA_angle()` — unified TMM functions with auto-coherence classification (see below)
  - `_make_c_list()` — auto-generates coherent/incoherent layer classification per wavelength/angle
  - `hilbert_fom_derivative()`, `skk_spectral_fom()` — Spatial-KK Figure of Merit calculations
  - `HT_help()` — Hilbert transform-based logistic profile generation
  - `plot_tra_curves()`, `plot_param_sweep()` — domain-specific plotting helpers
- **`plot_functions.py`** — matplotlib wrappers: `plot_setup()`, `plot()`, `legend()`, `set_size()`
- **`colors.py`** — custom color palette for consistent scientific figures
- **`module_test_tmm_helper.py`** — test suite for `tmm_helper` functions

### Computational Workflow

```
Parameters (A, gam, nb)
    -> generate_n_and_d_v6_symmetry()   # discretize continuous Lorentzian profile
    -> TRA_wavelength() / TRA_angle()   # TMM through all layers
    -> FOM = T^2/A, ASYM = A_LR/A_RL   # metrics
    -> plot via plot_functions.py
```

### Physical Parameters

| Symbol | Meaning |
|--------|---------|
| `A` | Lorentzian amplitude |
| `gam` (gamma) | Lorentzian width (um) |
| `nb` | Background refractive index |
| `delta` | Discretization step size |
| `lam` / `lambda_list` | Wavelength array (um, typically 2-15 um infrared) |
| `threshold` | Coherence classification threshold (default 5, see below) |

### Auto-Coherence Classification

The unified `TRA()`, `TRA_wavelength()`, and `TRA_angle()` functions automatically classify each layer as coherent ('c') or incoherent ('i') using `_make_c_list()`. The criterion is:

- A layer is **incoherent** if `n_real · d · cos(θ_layer) / λ > threshold` (default threshold=5)
- `cos(θ_layer)` is computed via Snell's law from the incidence angle
- Semi-infinite layers (`d=inf`) and first/last layers are always incoherent (tmm convention)

This means thin sKK coating layers (typical max n·d/λ ≈ 0.05) are always coherent, preserving the sKK effect, while thick substrates (n·d/λ ~ 1000s) are automatically incoherent. The `threshold` parameter can be passed to any TRA function to override the default.

Deprecated shims `TRA_inc()`, `TRA_wavelength_inc()`, `TRA_angle_inc()` exist for backward compatibility but simply call the unified versions (ignoring any `c_list` argument).

### Data & Files

- **`Data/`** — computed `.npy`/`.txt` results organized by parameter sweeps
- **`RI/`** — refractive index data for real materials (sapphire, ZnS, graphite, silicon, etc.)
- **`theory/`** — active theory/paper scripts (sKK analysis, coatings, figure generation)
- **`experimental/`** — active experimental scripts (Bruggeman mixtures, EMT comparisons, fab designs)
- **`archive/`** — ~28 previously used scripts, kept for reference but not actively maintained
- **`tmm/`** — vendored TMM package (do not modify)

### Typical Script Pattern

Scripts in `theory/` and `experimental/` include a path preamble so they can import shared modules from root:

```python
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors

# Define parameters
A, gam, nb = 10, 0.01, 2.3

# Generate layer stack
n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.1)

# Compute spectra
lambda_list = np.linspace(2, 5, 100)
T, R, A = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)

# Plot
fig, ax = plot_setup('Wavelength (um)', 'Transmittance')
plot(fig, ax, lambda_list, T, color=colors.blue)
```

File paths to `RI/` and `Data/` use `os.path.join(_PROJECT_ROOT, 'RI', ...)` so scripts work regardless of working directory.

## Active Scripts

### Theory Track (`theory/`)
- **`skk_analysis_consolidated.py`** — self-contained script generating all 10 paper figures. Intentionally re-implements tmm_helper functions for reproducibility.
- **`coating_Hilbert_transform.py`** — applies sKK coating to real sapphire substrate for mid-IR ellipsometry
- **`bulk_window_vs_sKK_coating_truncated_2025Dec17.py`** — ellipsometry benchmark: bulk sapphire vs sKK coating (wavelength + angle sweeps)
- **`sKK_coating_thermal_emission_2025Dec17.py`** — thermal emission: sKK coating in front of sapphire bulk (wavelength + angle sweeps, bulk vs coated comparison)
- **`stack_of_stacks_constant_losses.py`** — stacked sKK coatings with constant absorption
- **`delta_sweep.py`** — effect of discretization step size on reflectance
- **`modify_KK_losses.py`** — effect of loss (Im(n)) scaling on TRA spectra
- **`fig_sinc_leakage.py`**, **`fig_forward_backward.py`**, **`fig_lorentzian_profile.py`** — standalone figure scripts
- **`verify_lorentzian_spectral_fom.py`** — validates spectral FoM k=0 fix

### Experimental Track (`experimental/`)
- **`bruggeman_mixture_search.py`** — searches 2-material Bruggeman mixtures for target (n,k,d)
- **`bruggeman_load_TMM.py`** — loads precomputed nk from mixture search, runs TMM
- **`EMT_bruggeman_vs_stratified_vs_lamellar.py`** — compares 3 EMT methods (Bruggeman, stratified, lamellar) for graphite/sapphire on ZnS in mid-IR
- **`stratified_fab_design_SiAu_visible.py`** — fab team's device: stratified graphite/sapphire on Si+Au in visible/near-IR, computes Ψ/Δ
- **`key_mixed_layers_on_substrate_2025Oct30.py`** — specific Bruggeman mixture test on ZnS substrate

Both tracks share `tmm_helper.py`, `plot_functions.py`, `colors.py` (in root) and the `RI/` data.

## File Relationships

Do not infer supersession, redundancy, or status from filenames. Suffixes like `_clean`, `_v2`, `_new`, `_old` describe the author's intent at the time of naming, not the file's current relationship to others. Always read file content to determine what each file actually does before making judgments about redundancy.

## Communication Style

When giving physics explanations, use Unicode math symbols (epsilon, gamma, lambda, etc.) instead of LaTeX syntax. LaTeX does not render in the terminal and is hard to read raw.

## Dependencies

`tmm`, `numpy`, `scipy`, `matplotlib`
