# Sep01_legacy — the harness behind the 2026-09-01 deck

Everything that produced `Presentations/2026Sep01_sKK_PML.pptx`, moved here on 2026-09-11
and still runnable in place (path preambles were adjusted; nothing else changed).

```
master_data.py      builds skk_master.mph AND runs the first ~110 solves   (destructive: see below)
patch_pml.py        re-measures the PML with a well-resolved companion domain
patch2 … patch14    one appended study each; all load skk_master.mph read-only and merge into master_data.npz
make_figures.py     figures 01–20, 29–31         grazing_figures.py   figures 21–28
master_data.npz     every result, keyed P2_* … P14_*        figures/INDEX.md   question -> figure map + tables
skk_master.mph      the 3-domain model (34 MB: profile tables are embedded in it)
```

**Geometry:** 3 domains — background (n_b = 1.7) | terminator | lower region (PML *or* a
plain spacer) | PEC. The PML lived in the lower region at 2 λ while the sKK profiles lived
in the terminator domain at 8–100 λ, which is why several captions have to argue that the
thickness mismatch is fair. The `swap` model in `../Models/` fixes that. Cross-check
(tapered Lorentzian, L = 10 λ, 800 elements): the new harness agrees with `P13_uniform_N800`
to four digits at 45°, 80° and 89°; at 0° the 2 λ spacer costs a factor 1.8.

**`master_data.py` is destructive.** It writes `master_data.npz` without merging. Re-running
it wipes every `P2_*` … `P14_*` result. Back the `.npz` up first.

Superseded by the workflow in `../README.md`. Nothing here is imported by anything there.
