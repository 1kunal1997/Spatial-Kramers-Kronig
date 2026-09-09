# comsol_skk, first generation (2026-08-25 to 08-31)

Archived 2026-09-09. Superseded by the `skk_master.mph` harness (`comsol_skk/master_data.py`
+ `patch*.py`), and in turn by the `Models/skk_swap.mph` harness described in
`comsol_skk/README.md`.

Nothing here is referenced by any live script. It is kept because it is where several
results in the Sep 2026 deck were first measured.

| file(s) | what it did |
|---|---|
| `build_skk.py` | Lorentzian terminator built from the PML baseline; scanned amplitude and orientation |
| `build_logistic.py` | first logistic (DiHiTI) terminator |
| `build_nb17.py` | the n_b = 1.7 -> air configuration that became the paper's direction |
| `build_hybrid.py` | PML stretch and sKK material in the same domain (the hybrid that failed) |
| `build_stacked.py`, `build_lorentzian_stacked.py` | sKK in front of a PML |
| `run_tapered.py`, `run_gain.py` | sech-tapered Lorentzian; the exact-HT gain profile |
| `sweep_thickness_nb17.py` | first thickness sweep |
| `verify_baseline.py`, `inspect_baseline.py` | sanity checks on `sKK_PML_baseline.mph` |
| `taper_ht_analysis.py` | what the Hilbert partner of a tapered eps' looks like |
| `grazing_study.py` | TMM + an extrapolated PML attenuation law. **Refuted** by COMSOL: the law predicted R = 3.1e-10 at 89 deg for pf = 50 where the measured value is 1.28e-2. Kept as the prediction that was wrong. |
| `plot_*.py` | one figure script per build script |
| `data/`, `figures/`, `models/` | the `.npz`, `.png` and `.mph` those scripts produced |

**If you ever re-run one:** the path preamble assumes the script sits in `comsol_skk/`, so
`ROOT` and the `Comsol Files` paths resolve one directory too high from here, and the `.mph`
files it wants are now in `models/` beside it. `sKK_PML_baseline.mph`, which most of the
`build_*` scripts load as their starting point, was deliberately left in `Comsol Files/`.
