# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a nanophotonics computational research project investigating the
**spatial Kramers-Kronig (sKK) relations** originally developed by Horsley et al. (2015). The broad goal is to either extend the theory (for example, zero and unit transmission cases by King, Horsley, Philbin in 2017), or find applications of this phenomena, such as anti-reflection (AR) coatings, thermal emission/absorption asymmetry, perfect absorbers, etc. The calculations of the transmission, absorption and reflection of multilayer materials/coatings are performed using Transfer Matrix Method (TMM) simulations.

## Running Code

This is a pure Python research project — no build system. Run scripts directly:

```bash
python theory/<script_name>.py          # theory scripts
python experimental/<script_name>.py    # experimental scripts
python -m pytest module_test_tmm_helper.py   # run tests
```

For interactive work, use VS Code interactive mode (#%%) .

### Running consolidated paper figures individually

`theory/skk_analysis_consolidated.py` supports selective figure generation via CLI args. Each figure is a standalone function that takes a shared setup namespace `S`.

```bash
# Run all figures:
python theory/skk_analysis_consolidated.py

# Run specific figures by name:
python theory/skk_analysis_consolidated.py fig1 fig6 task1

# List available figure names:
python theory/skk_analysis_consolidated.py --list

# Save to a custom output directory (e.g. Overleaf):
python theory/skk_analysis_consolidated.py --outdir sKK-Paper-Overleaf/figures fig6 fig7
```

Available figure names: `fig1`–`fig10`, `loss_shapes`, `width_amplitude`, `thick_shapes`, `task1`, `task2`, `task3`.

For VS Code interactive mode: run `S = setup()` first, then call any figure function directly (e.g. `fig_alpha_tradeoff(S)`).

## Core Architecture

### Key Modules

- **`tmm_helper.py`** — primary module imported as `tmm_h`. Contains:
  - `eps(x, a, gam, nb)` — complex Lorentzian ε(x) profile
  - `logistic(x, k, nb, sx=1)` — logistic GRIN ε'(x) profile
  - `ht_derivative(xx, e_re)` — derivative→HT→integrate method for sKK ε''(x) (paper's main contribution)
  - `smooth_gate(eps_re, eps0, sigma)` — smooth tanh gate on ε'(x) to remove lossy-air region
  - `discretize_profile(xx, ee, delta)` — arc-length adaptive discretization of continuous ε(x) into TMM layers
  - `generate_n_and_d_v6_symmetry(gam, a, nb, delta, M)` — symmetric Lorentzian stack (uses `discretize_profile`; `M` controls domain half-width in units of gam, default 2000)
  - `HT_help(k, nb, ..., M)` — logistic sKK stack via derivative→HT→integrate + `discretize_profile` (replaces old Tukey-taper method; `M` controls domain, default 2000)
  - `skk_spectral_fom(x, u, v)` — spectral one-sidedness FoM via derivative→FFT→÷ik method; returns `(fom, k, pwr)` where `pwr = |ε̂(k)|²`; no padding, no normalization
  - `hilbert_fom_derivative(x, u, v)` — real-space Hilbert FoM (derivative-space correlation)
  - `plot_spectral_fom(ax, k, pwr, fom, klim)` — green/red shaded FoM spectrum plot (log scale)
  - `TRA()`, `TRA_wavelength()`, `TRA_angle()` — unified TMM functions with auto-coherence classification (see below)
  - `_make_c_list()` — auto-generates coherent/incoherent layer classification per wavelength/angle
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

## File Relationships

Do not infer supersession, redundancy, or status from filenames. Suffixes like `_clean`, `_v2`, `_new`, `_old` describe the author's intent at the time of naming, not the file's current relationship to others. Always read file content to determine what each file actually does before making judgments about redundancy.

## Communication Style

When giving physics explanations, use Unicode math symbols (epsilon, gamma, lambda, etc.) instead of LaTeX syntax. LaTeX does not render in the terminal and is hard to read raw.

## Git Workflow

Push to the GitHub repo after big changes or at the end of a session. No need to push after every small edit — use judgment.

## Dependencies

`tmm`, `numpy`, `scipy`, `matplotlib`
