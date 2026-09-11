# PLAN — what to run next, and why

Written 2026-09-11 from the post-doc's and advisor's comments on the Sep 1 deck. Each item
is one question, one `exp_*.py`, one data file. Angles are **0° and 80°** throughout unless
stated (item 3). Everything is non-dispersive, so λ is not a parameter: only L/λ, w/λ and
the element count matter. λ = 800 nm is a label.

## Two resolutions, not one

Every mesh question below turns on keeping two length scales apart:

| error | what it is | scale it sets | exists in TMM? |
|---|---|---|---|
| **wave resolution** | the FEM field E(y) is piecewise-quadratic on elements of size h; too coarse -> phase error, numerical dispersion, and spurious reflection wherever h changes | h ≲ λ_local/10, λ_local = λ/n_local, **everywhere the field is non-negligible** | **no** — TMM solves each layer exactly, so the paper never met this |
| **profile resolution** | ε is sampled at each element's Gauss points; a feature narrower than h is aliased, which puts spurious spectral content near 2k_y and breaks one-sidedness | h ≲ w/4, **only where ε varies** | **yes** — this is exactly the paper's Fig 7 (N-granularity) effect |

Both must hold. In the flat tails only the wave constraint exists; at the centre both do,
and the tighter one wins. The optimal centre-to-tail element ratio is therefore not a free
choice — it is the ratio of the two scales:

    r* ≈ h_tail / h_centre = (λ/(10 n_tail)) / (w/4)  =  0.4 λ / (n_tail w)

For w = λ/5, n_tail = 1.7: **r\* ≈ 1.2** — uniform is already right, which is what fig 29
(`Sep01_legacy/figures/29_*`) found, and why its r = 20 and r = 100 were 15–80× too
aggressive. For w = λ/50: r\* ≈ 12 — there a centre-dense mesh is genuinely necessary.
The post-doc and advisor are right about the direction; fig 29 was right about the numbers
at the width it tested. Item 2 measures the crossover.

## Parameters and what is held fixed

| symbol | meaning | note |
|---|---|---|
| L/λ | terminator thickness | shared by PML and sKK in the `swap` model |
| w/λ | feature width (γ for Lorentzians) | **required** argument — no default |
| N | elements across the terminator | the cost |
| N/L | elements per λ | wave resolution |
| N·w/L | elements per feature | profile resolution |
| r | centre/tail element ratio | 1 = uniform |
| A, ℓ | Lorentzian amplitude, taper length | family-specific |
| amp | turned-logistic ε″ plateau | family-specific |
| pf | PMLfactor of the opponent | 1 (default) and 4 (tuned) |

Fixing N while sweeping L changes N/L; fixing N/L changes N. Both are legitimate and answer
different questions. **Every experiment states which of {N, N/L, N·w/L} it holds** in its
`meta` block. Two width rules recur and are kept distinct by name: `w = λ/5` (fixed feature)
and `w = L/40` (self-similar — the profile *shape* is identical at every L, so N·w/L = N/40
is constant and only N/L changes).

## A. `exp_budget_vs_length.py` — the advisor's question (item 8)

Fix N. Sweep L. Small L: fine mesh, heavy truncation. Large L: coarse mesh, unresolved wave.
There is an optimum L\*(N); the envelope of R(L\*; N) over N is the **true equal-cost curve**
for an sKK terminator, and the honest thing to put against a PML at the same N — which,
in the `swap` model, is a PML of the *same* thickness.

- N ∈ {100, 200, 400}; L/λ ∈ {0.2, 0.5, 1, 2, 5, 10, 20, 50}
- both width rules; profiles: `turned_logistic`, `tapered`; uniform mesh
- PML pf ∈ {1, 4} at every (N, L)
- L/λ ∈ {0.2, 0.5, 1, 2} are the paper's d/λ window (item 6). Expect them to lose here:
  the paper scored front-face R with a substrate behind; a PEC-backed terminator that does
  not absorb in one pass just reflects off the PEC (Aug: 5e-4 at 2λ vs 6.5e-9 at 6λ).
- Prediction to be wrong about: L\* ≈ Nλ/10 – Nλ/20 (just resolved), and R(L\*) follows
  the truncation law evaluated there.
- ~150 configurations, ~6 min.

## B. `exp_mesh_grading.py` — non-uniform meshes (item 2)

The decisive version of fig 29: a budget where the wave is resolved, mild ratios, and a
width where r\* is far from 1 so the crossover is visible.

- L = 10λ; N ∈ {100, 200, 400}; r ∈ {1, 1.5, 2, 3, 5, 10}
- centre-dense (`symmetric+reverse`) and entry-dense (`reverse`)
- w ∈ {λ/5 (r\* ≈ 1.2), λ/50 (r\* ≈ 12)}; profiles: `turned_logistic`, `tapered`
- If centre-dense wins at λ/50 and loses at λ/5, the r\* formula above is confirmed and
  the mesh rule becomes: set h_centre from w, h_tail from λ, done.
- Follow-up if warranted: `explicit` distribution with node density ∝ a + b·|dε/dy|
  (one line in `skk_lib.set_mesh`).
- ~250 configurations, ~10 min.

## C. `exp_lateral_mesh.py` — lateral elements at steep angles (item 3)

At 89° the lateral wavelength is λ/(n_b sin θ) ≈ 0.59λ; N_x = 5 across L_x = λ/4 is ~12
elements per lateral wavelength. Indirect evidence says it is fine (R at 89° is converged
at N_term = 5, i.e. physics-limited). Measure it anyway.

- N_x ∈ {5, 10, 20}; θ ∈ {80, 85, 89}; `turned_logistic`, `tapered`, PML pf=4 at L = 10λ
- 9 configurations. If R moves by < 1 %, N_x = 5 stays and the number goes in the README.

## D. `exp_discrete_hilbert.py` — DHT on the mesh nodes (item 4)

Today the pair is one-sided as a *continuous* function; COMSOL then samples it and the
sampled profile is not (the paper's Fig 7 point). Alternative: define ε′ on the mesh's own
N nodes and take the discrete Hilbert transform **of that N-point sequence**, so the profile
COMSOL sees is exactly one-sided within its Nyquist band — which contains 2k_y whenever
the wave is resolved. Should matter precisely in the coarse-mesh regime we want to exploit.

- L = 10λ, w = λ/5; N ∈ {50, 100, 200}; `logistic`, `turned_logistic`
- two profile modes: continuous HT sampled at the nodes vs DHT of the node samples
  (needs a `nodes=` option in `skk_lib.write_profile`; the table *is* the profile)
- 12 configurations. Pre-screen with the one-sidedness FoM on the sampled profile — it
  costs nothing and predicts the ordering before a solve is run.

## E. `exp_width_sweep.py` — the rotation (item 7)

Fix L = 10λ and N = 400 (resolved), sweep the shape parameters one at a time:
w/λ ∈ {0.05, 0.1, 0.2, 0.5, 1}; then ℓ/λ ∈ {0.25, 0.5, 1, 2} and A ∈ {1, 2, 4, 8} for the
tapered Lorentzian; amp ∈ {0.5, 1, 2} for the turned logistic. ~25 configurations. Compute
the one-sidedness FoM for every profile alongside R — it is the shape-only ceiling and it is
free.

## Standing decisions

- **Angles:** 0° and 80° (item 1). Per-angle cost is ~0.2 s inside the parametric sweep;
  the cost is per configuration, so a second angle is nearly free and guards against tuning
  for the wrong regime. 90-angle sweeps only for a final figure.
- **Passivity (item 5):** gain is permitted — COMSOL handles it and a PML is unphysical
  anyway. Do not expect it to help: the Aug test gave 2.3e-6 (best gain) vs 6.1e-7 (best
  passive), because the exact Hilbert partner of a profile that returns to its background
  has ∫ε″dy = 0 by the sum rule and cannot absorb; PEC-backed it amplified (R = 1.96). The
  constraint that bites is the sum rule, not passivity.
- **Order:** A → B → C (cheap, closes a question) → D → E. A and B together answer the two
  questions the advisor and post-doc actually asked.

---

## Status 2026-09-11 — all six run (A–E + F stacked), 0°/80°/89°, air background except the paper's logistic

Data in `data/<name>.npz`, figures in `figures/<name>_*.png`, one opened-and-checkable
model per experiment in `Models/snapshots/`. Details in the session summary; one line each:

| exp | data | one-line result |
|---|---|---|
| A budget_vs_length | `budget_vs_length` | L\*(N) ≈ Nλ/10 as predicted; at equal N and equal L the tuned PML (pf=4) wins everywhere; the turned logistic beats the default PML (pf=1) by 2500× at 80° (L=50λ, N=400) |
| B mesh_grading | `mesh_grading` | at w = λ/50, N = 100, centre-dense r=2 is 10× better than uniform; at w = λ/5 nothing to gain — consistent with r\*. **Caveat:** at L = 10λ the truncation floor (2.5e-9) sits where discretization and truncation reflections *cancel*, so raw R is a bad convergence metric here; redo with \|R − R_converged\| |
| C lateral_mesh | `lateral_mesh` | sKK profiles move < 0.6 % from N_x = 5 → 20. **PML pf=4 at 80° moves 33 %** (3.96e-8 → 2.62e-8): the PML's grazing number carries a lateral-mesh uncertainty the sKK numbers do not |
| D discrete_hilbert | `discrete_hilbert` | no difference: the node sequence of the continuous pair is already 99.98–99.9999 % one-sided at N = 50–200; sampling does not break one-sidedness for w = λ/5 |
| E width_sweep | `width_sweep` | turned logistic flat in w for w ≤ λ/2; ε″ plateau **amp = 0.5 beats 1 at every angle** (R(80) 1.3e-3 vs 4.2e-3) — sweep lower; Lorentzian family best at A = 2 in air; turned logistic beats every Lorentzian variant by ≥3 orders at 0° |
| F stacked | `stacked` | **R is independent of what is behind the sKK section** (PML pf=1/4, lossy slab, even lossless air) to 3 digits: the residual is 100 % front-face truncation of the sKK part, so at fixed footprint every λ given to a PML is a λ taken from the sKK's truncation length. Stacking never wins |

Two things learned about method, both of which should change the next runs:
1. **Discretization and truncation reflections interfere.** At L = 10λ, N = 100 the turned
   logistic gives R(0) = 3.6e-11 while the converged answer is 2.5e-9 — coarser looked
   *better*. Any mesh study must score \|R − R_converged\|, never R.
2. The wself bug: `L/40` with L in units of λ gave w in metres. Caught by the L = 10λ
   cross-check between rules (2.56e-9 vs 2.51e-9 after the fix). Keep such cross-checks.
