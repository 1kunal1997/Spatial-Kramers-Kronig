# sKK-as-PML study

Can a spatial-Kramers-Kronig profile replace a PML as the absorbing terminator of a
simulation domain? Everything here measures one number: the reflectance `ewfd.Rtotal` of a
PEC-backed terminator, versus incidence angle, thickness and element count.

## Pipeline

```
model_build.py ──> Models/skk_swap.mph        template models. Built once. NO solves.
                   Models/skk_stacked.mph     Experiments never save back to them.
                        │
                        │  loaded by
                        ▼
   skk_lib.py  ◄────  exp_<question>.py  ──writes──>  data/<question>.npz
   (profiles,         one experiment,                 results + a `meta` JSON blob:
    model I/O,        one question,                   every parameter, the profile
    solving,          one data file                   files used, the git commit
    data files)            │
                           │  and the profiles it used
                           ▼
                     profiles/<name>.txt   ──┐
                                             ├──>  fig_<question>.py  ──>  figures/*.png
                     data/<question>.npz   ──┘     (imports figstyle.py; no COMSOL)
```

Three rules make the folder auditable:

1. **One definition of everything.** Profiles, meshing, solving and file writing live only in
   `skk_lib.py`. Copy-pasted profile functions are what let `w = L/40`, `w = L/80` and
   `w = lam/5` all get called "the tapered Lorentzian" in the Aug-Sep 2026 runs.
2. **Data files are self-describing.** `data/*.npz` carries a `meta` entry holding every
   parameter, the profile files, and the commit hash. If a figure is questioned, the answer
   is in the data file — you should never have to read the script that made it.
3. **Figure scripts never touch COMSOL.** `figstyle.py` imports matplotlib only, so figures
   redraw anywhere, with or without a licence.

## The two geometries

```
  swap  (2 domains -- the workhorse)          stacked (3 domains -- back-to-back only)
      ---- periodic Port ----                     ---- periodic Port ----
      |   background  n_bg   | L_bg               |   background  n_bg   | L_bg
      |   TERMINATOR         | L_term             |   TERMINATOR         | L_term
      -------- PEC ----------                     |   lower region       | L_low
                                                  -------- PEC ----------
```

In `swap` the PML coordinate system sits on the **terminator** domain. A PML run activates
`pml1` and loads the flat profile (eps = n_bg^2); an sKK run deactivates it and loads a
profile file. Both therefore occupy the same domain with the same `L_term` and `N_term`, so
equal footprint and equal cost are structural, not something a caption has to argue. (In the
Aug-Sep 2026 harness the PML lived in a separate domain at a different thickness, which is
why figure 09's caption had to explain why 8 lam of sKK against 2 lam of PML was fair.)

`stacked` exists only for sKK-in-front-of-a-PML tests, where a third domain is genuinely
needed. Both come from one code path in `model_build.py`, so they cannot drift apart.

## Conventions

| | |
|---|---|
| `y = 0` | the PEC-backed end of the terminator |
| `y = L_term` | the **entry face**, where the wave arrives |
| propagation | in `-y`; angles are measured **in the background medium** |
| `eps_im >= 0` | loss (`DielectricLoss`, positive `epsilonBis`) |
| feature width | a **required** argument — `skk_lib.profile()` has no default width |
| profile sampling | spacing <= min(feature/40, 0.002 lam): at least 10x finer than the finest mesh, so the table is never the limiting discretization |

Wavelength is 800 nm throughout, period `Lx = lam/4` so only diffraction order 0
propagates, s-polarized (Ez) unless a model is built with `--pol p`.

## Profiles

| name | eps' | eps'' | ends at | notes |
|---|---|---|---|---|
| `logistic` | logistic step eps_b -> 1 | Hilbert partner (DiHiTI) | air | needs n_b > 1 to carry loss; past TIR above theta_c |
| `turned_logistic` | Hilbert partner | logistic step 0 -> amp | eps_b, lossy | the pair turned 90 deg; eps' >= eps_b so **no TIR at any angle** |
| `lorentzian` | eps_b + A gam u/D | A gam^2/D | eps_b | exact closed form; truncation-limited as (A gam/L)^2 |
| `tapered` | Lorentzian with a sech taper | A gam^2/D | eps_b | not an exact pair — trades KK mismatch for a smaller truncation step |
| `flat` | n_bg^2 | 0 | — | what the terminator domain holds during a PML run |

Profiles are written to `profiles/*.txt` (`% ` header + `y eps_re eps_im`) and read by the
model's single `i_prof` interpolation function. Two consequences worth having: the model no
longer carries profile data (34 MB -> 0.38 MB), and a figure script can plot exactly what
COMSOL was fed by reading the same file.

## Running things

```bash
python model_build.py                    # both geometries, s-pol. Run once.
python model_build.py --geometry swap --pol p
python exp_angle_vs_terminator.py        # one experiment -> data/angle_vs_terminator.npz
python fig_angle_vs_terminator.py        # -> figures/angle_vs_terminator*.png
```

`exp_angle_vs_terminator.py` + `fig_angle_vs_terminator.py` are the template pair; copy them
for a new question and keep the basenames matching.

## Legacy (Aug-Sep 2026) — kept until migration, then retired

`master_data.py`, `patch_pml.py`, `patch2`–`patch14.py`, `make_figures.py` and
`grazing_figures.py` are the previous harness. They build and use
`../Comsol Files/skk_master.mph` and `master_data.npz`, and together they produce every
figure numbered `01`–`31` in `figures/`, i.e. the entire Sep 1 2026 deck. They are left in
place and working: nothing above touches them, and no result has been re-measured.

Two things to know if you run them:

- **`master_data.py` is destructive.** Unlike every patch, it writes `master_data.npz`
  without merging, so re-running it rebuilds the model *and* wipes ~700 solves of `P2_*`
  … `P14_*` results. Back up the `.npz` first.
- Figures `01`–`20` and `29`–`31` come from `make_figures.py`; `21`–`28` come from
  `grazing_figures.py`. `figures/INDEX.md` maps advisor questions to figure numbers and
  holds the results tables.

The generation before that (Aug 25–31: `build_*.py`, `plot_*.py`, `run_*.py`,
`verify_baseline.py`, `inspect_baseline.py`, `sweep_thickness_nb17.py`,
`taper_ht_analysis.py`, `grazing_study.py`) is in `../archive/comsol_skk_aug2026/`.
