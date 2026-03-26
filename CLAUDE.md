# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a nanophotonics computational research project investigating the
**spatial Kramers-Kronig (sKK) relations** originally developed by Horsley et al. (2015). The broad goal is to either extend the theory (for example, zero and unit transmission cases by King, Horsley, Philbin in 2017), or find applications of this phenomena, such as anti-reflection (AR) coatings, thermal emission/absorption asymmetry, perfect absorbers, etc. Currently, the focus is AR coatings (found in coating_Hilbert_transform.py), specifically
targeting backside reflection suppression in **mid-IR ellipsometry** on sapphire substrates. The calculations of the transmission, absorption and reflection of multilayer materials/coatings are performed using Transfer Matrix Method (TMM) simulations.

## Running Code

This is a pure Python research project — no build system. Run scripts directly:

```bash
python <script_name>.py
python -m pytest module_test_tmm_helper.py   # run tests
```

For interactive work, use VS Code interactive mode (#%%) .

## Core Architecture

### Key Modules

- **`tmm_helper.py`** — primary module imported as `tmm_h`. Contains:
  - `generate_n_and_d_v6_symmetry()` — the only discretization function (legacy versions archived)
  - `TRA()`, `TRA_wavelength()`, `TRA_angle()` — compute Transmission/Reflection/Absorption (coherent)
  - `TRA_inc()`, `TRA_wavelength_inc()`, `TRA_angle_inc()` — incoherent TMM variants
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

### Data & Files

- **`Data/`** — computed `.npy`/`.txt` results organized by parameter sweeps
- **`RI/`** — refractive index data for real materials (sapphire, ZnS, graphite, silicon, etc.)
- **`archive/`** — ~28 previously used scripts, kept for reference but not actively maintained
- **`tmm/`** — vendored TMM package (do not modify)

### Typical Script Pattern

```python
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
T, R_LR, R_RL, A_LR, A_RL = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)

# Plot
fig, ax = plot_setup('Wavelength (um)', 'Transmittance')
plot(fig, ax, lambda_list, T, color=colors.blue)
```

## Active Scripts

### Theory Track
- **`skk_analysis_consolidated.py`** — self-contained script generating all 10 paper figures. Intentionally re-implements tmm_helper functions for reproducibility.
- **`coating_Hilbert_transform.py`** — applies sKK coating to real sapphire substrate for mid-IR ellipsometry
- **`bulk_window_vs_sKK_coating_truncated_2025Dec17.py`** — ellipsometry benchmark: bulk sapphire vs sKK coating (wavelength + angle sweeps)
- **`coatings_sKK_window.py`** — sKK AR coating on incoherent window substrate
- **`stack_of_stacks_constant_losses.py`** — stacked sKK coatings with constant absorption
- **`delta_sweep.py`** — effect of discretization step size on reflectance
- **`modify_KK_losses.py`** — effect of loss (Im(n)) scaling on TRA spectra
- **`fig_sinc_leakage.py`**, **`fig_forward_backward.py`**, **`fig_lorentzian_profile.py`** — standalone figure scripts
- **`verify_lorentzian_spectral_fom.py`** — validates spectral FoM k=0 fix

### Experimental Track
- **`bruggeman_mixture_search.py`** — searches 2-material Bruggeman mixtures for target (n,k,d)
- **`bruggeman_load_TMM.py`** — loads precomputed nk from mixture search, runs TMM
- **`lamellar_EMT_substrate.py`** — compares Bruggeman, stratified, and lamellar EMT approaches
- **`key_mixed_layers_on_substrate_2025Oct30.py`** — specific Bruggeman mixture test on ZnS substrate

### Pending Review (kept but may have bugs)
- **`coatings_2_TE.py`** — TE polarization analysis (has undefined variable bugs)
- **`single_mixed_layer_on_substrate.py`** — overlaps with lamellar_EMT_substrate.py

Both tracks share `tmm_helper.py` and the `RI/` data. Do not reorganize into subfolders — relative paths to `RI/` files would break.

## File Relationships

Do not infer supersession, redundancy, or status from filenames. Suffixes like `_clean`, `_v2`, `_new`, `_old` describe the author's intent at the time of naming, not the file's current relationship to others. Always read file content to determine what each file actually does before making judgments about redundancy.

## Communication Style

When giving physics explanations, use Unicode math symbols (epsilon, gamma, lambda, etc.) instead of LaTeX syntax. LaTeX does not render in the terminal and is hard to read raw.

## Dependencies

`tmm`, `numpy`, `scipy`, `matplotlib`
