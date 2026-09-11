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

### COMSOL gotchas, all found the hard way

Three properties of a file-source `Interpolation` fail only at solve time, with messages
that never mention the interpolation file:

1. The function is named after the **feature tag**. `funcname` is ignored for a file
   source, so the tags are `p_re` and `p_im`. Wrong tag -> "Unknown function or operator".
2. Use **two two-column files**, not one three-column file. The spreadsheet format reads
   the leftmost columns as arguments, so `y eps_re eps_im` is a function of TWO arguments
   -> "Function called with wrong number of arguments".
3. Set `filename` **before** `struct`, and call `discardData()` before re-pointing at a new
   file. Set the other way round, `struct` silently reverts to `grid`.

`model_build.py` asserts 1 and 3 at build time, and `skk_lib.set_profile` re-asserts them
on every swap.

## Running things

```bash
python model_build.py                    # both geometries, s-pol. Run once.
python model_build.py --geometry swap --pol p
python exp_angle_vs_terminator.py        # one experiment -> data/angle_vs_terminator.npz
python fig_angle_vs_terminator.py        # -> figures/angle_vs_terminator*.png
```

`exp_angle_vs_terminator.py` + `fig_angle_vs_terminator.py` are the template pair; copy them
for a new question and keep the basenames matching.

**Checking a result by hand in the GUI.** `python make_snapshots.py` writes one fully
configured model per experiment to `Models/snapshots/` (parameters, profile files, mesh
grading, PML state; no solution data). Open one, Study 1 → Compute, then Results → Derived
Values → Global Evaluation of `ewfd.Rtotal`, and compare with the same key in the data file:

    python -c "import skk_lib as S; r,m = S.load_result('budget_vs_length'); print(r['kind=turned_logistic__rule=wfix__L=10__N=400'])"

The interpolation functions `p_re`/`p_im` point at `profiles/*.re.txt` / `*.im.txt` by
absolute path; those files are the profile.

## Legacy

`Sep01_legacy/` holds the harness that produced the 2026-09-01 deck — `master_data.py`,
`patch*.py`, the two figure scripts, `master_data.npz`, `skk_master.mph` and figures 01–31
with their `INDEX.md`. Still runnable in place; see its README. Nothing in the new workflow
imports from it. The generation before that is in `../archive/comsol_skk_aug2026/`.

The plan for what to run next is `PLAN.md`.
