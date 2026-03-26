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
python -m pytest modules/module_test_tmm_helper.py   # run tests
```

For interactive work, use VS Code interactive mode (#%%) .

## Core Architecture

### Key Modules

- **`tmm_helper.py`** (root) — primary module imported as `tmm_h`. Contains:
  - `generate_n_and_d_v6_symmetry()` — converts Lorentzian profile parameters to discrete layer stacks (preferred discretization)
  - `TRA()`, `TRA_wavelength()`, `TRA_angle()` — compute Transmission/Reflection/Absorption
  - `hilbert_fom_derivative()`, `skk_spectral_fom()` — Spatial-KK Figure of Merit calculations
  - `HT_help()`, `generate_n_and_d_coating_logistic()` — Hilbert transform–based profile generation
- **`modules/tmm_helper.py`** — submodule version (separate `.git`); kept in sync with root version
- **`plot_functions.py`** — matplotlib wrappers: `plot_setup()`, `plot()`, `legend()`, `set_size()`
- **`colors.py`** — custom color palette for consistent scientific figures

### Computational Workflow

```
Parameters (A, gam, nb)
    → generate_n_and_d_v6_symmetry()   # discretize continuous Lorentzian profile
    → TRA_wavelength() / TRA_angle()   # TMM through all layers
    → FOM = T²/A, ASYM = A_LR/A_RL    # metrics
    → plot via plot_functions.py
```

### Physical Parameters

| Symbol | Meaning |
|--------|---------|
| `A` | Lorentzian amplitude |
| `gam` (γ) | Lorentzian width (μm) |
| `nb` | Background refractive index |
| `delta` | Discretization step size |
| `lam` / `lambda_list` | Wavelength array (μm, typically 2–15 μm infrared) |

### Data & Files

- **`Data/`** — computed `.npy`/`.txt` results organized by parameter sweeps (47+ subdirectories)
- **`RI/`** — refractive index data for real materials (sapphire, ZnS, graphite, silicon, etc.)
- **`modules/module_test_tmm_helper.py`** — test suite for `tmm_helper` functions
- **`*.mph`** — COMSOL FEM models (1–8 GB each) for simulation validation

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
fig, ax = plot_setup('Wavelength (μm)', 'Transmittance')
plot(fig, ax, lambda_list, T, color=colors.blue)
```

## Project Tracks

This project has two parallel tracks:

1. **Theory** — extending sKK theory, computing FoMs, parameter sweeps, figure generation for papers. Most scripts fall here (e.g., `coating_Hilbert_transform.py`, `skk_analysis_consolidated.py`, `fig_*.py`).

2. **Experimental** — working with experimentalists to physically realize an sKK coating/metasurface. Scripts here deal with Bruggeman effective medium theory (EMT), lamellar EMT, single layers on substrates, material refractive index data, etc. (e.g., `bruggeman_load_TMM.py`, `lamellar_EMT_substrate.py`, `single_mixed_layer_on_substrate.py`, `pre_Bruggeman_design.py`).

Both tracks share `tmm_helper.py` and the `RI/` data. Do not reorganize into subfolders yet — relative paths to `RI/` files would break.

## File Relationships

Do not infer supersession, redundancy, or status from filenames. Suffixes like `_clean`, `_v2`, `_new`, `_old` describe the author's intent at the time of naming, not the file's current relationship to others. Always read file content to determine what each file actually does before making judgments about redundancy.

## Communication Style

When giving physics explanations, use Unicode math symbols (ε, γ, λ, ∂, ∇, ∫, √, ², ₀, etc.) instead of LaTeX syntax. LaTeX does not render in the terminal and is hard to read raw.

## Dependencies

`tmm`, `numpy`, `scipy`, `matplotlib`
