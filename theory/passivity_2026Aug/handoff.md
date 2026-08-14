# Handoff — sKK passivity investigation (Aug 2026)

> **UPDATE 2026-08-13:** the "no simpler condition exists" conclusion below is
> now largely overturned — see `potential_theory_findings.md` in this folder.
> Integrating the periodic HT by parts turns DiHiTI into a closed-form
> log-potential of the drop density g = −dε′/dx, yielding (i) a proven
> sufficient condition (monotone + drop in middle half + balance integral
> M₁ = 0 ⟹ passive; sharp at L/2), (ii) necessary scalar tests σ ≥ 0 and
> M₁ = 0 with 0 false alarms on 400 generator samples, and (iii) closed-form
> gain for off-centre steps. §5's open target is answered. What survives of
> the old claims is recorded honestly in that file's §4.

Read this before doing anything. Section 4 lists framings that were already
tried and killed; re-deriving them wastes the session.

---

## 1. The question

The DiHiTI method (`tmm_helper.ht_derivative`) builds a spatial-Kramers-Kronig
coating by prescribing ε′(x) and computing its Hilbert partner ε″(x). The paper
guaranteed **reflectionlessness**. It did *not* guarantee **passivity**
(ε″ ≥ 0 everywhere); that held by luck for the profiles actually used.

Original question: *what constraints on ε′(x) guarantee that ε″(x) is gain-free?*

Motivation for caring: Longhi 2015 (`Papers/Longhi_sKK_Slowly_Decaying_Profiles_2015.pdf`,
§3) shows gain in an sKK medium can seed a growing interface mode, and that
nonlinear saturation then *breaks the reflectionless property itself*. He states
the issue "deserves further analysis" and nobody has followed up.

---

## 2. The answer already exists

**Carathéodory–Toeplitz.** The domain is a **circle** (scipy's FFT-based
`hilbert` continues the profile periodically, not as a constant), so the
governing class is Carathéodory on the disk — *not* Nevanlinna on the line.

Criterion, exact, necessary **and** sufficient:

1. Subtract the straight line joining ε′'s two endpoints → periodic remainder.
2. FFT the remainder → coefficients aₘ.
3. Set cₘ = −i·sgn(m)·aₘ for m ≠ 0; c₀ is free (it is the mean of ε″, i.e. a
   uniform background loss = DiHiTI's integration constant).
4. ε″ ≥ 0 everywhere **iff** the Toeplitz matrix T_jk = c_(j−k) is PSD.

Two-line proof: Σ_jk c_(j−k) z_j z̄_k = (1/2L)∫ ε″(x)|P(x)|²dx for the
trigonometric polynomial P with coefficients z. Non-negative ε″ ⇒ integral ≥ 0
⇒ T PSD; ε″ < 0 somewhere ⇒ pick P a Fejér kernel concentrated there ⇒ T not PSD.

Because c₀ sits only on the diagonal, T(c₀) = T(0) + c₀·I, so

    C_g,min = −λ_min(T(0))     = minimum background loss for passivity.

---

## 3. Verified numerical facts (all reproducible from the scripts here)

| fact | number |
|---|---|
| DiHiTI ≡ (subtract endpoint ramp) + (periodic conjugate function) | agree to 3×10⁻⁴ rel. |
| DiHiTI's gauge constant equals the *minimum* passive one: C_g,DiHiTI = C_g,min | 4 decimals, every passing profile |
| Exact identity `min ε″ = C_g,DiHiTI − C_g,min` | every digit |
| ε′ even about a point ⇒ ε″ odd ⇒ mean zero ⇒ gain guaranteed | ‖ε″(x)+ε″(−x)‖ = 3.9×10⁻¹⁵ |
| Unequal endpoints under periodic wrap ⇒ log-divergent gain in the *direct* HT | coefficient (Δε/π)·ln2 = 0.4169 per grid doubling |
| DiHiTI is blind to the endpoint ramp (H[const] = 0) | ε″ identical to 4×10⁻¹⁵ for ramps 0 → 8.1 |
| Constructor reproduces DiHiTI on the logistic | 0.0026 rms on a 1.89 swing |
| Cost of passivity vs allowing gain (best fit to an index step) | ≈2.7× larger residual (setup-specific) |
| Cheap screen (parity / area / slope mismatch) | catches 13/21 gain cases, **0 false alarms** |

**Two structural results worth carrying forward:**

* **Analyticity and passivity are independent.** The triangle profile has
  spectral FoM = 99.9999987 % (essentially perfectly one-sided ⇒ reflectionless)
  *and* min ε″ = −1.82 (substantial gain). All existing FoM machinery measures
  the first property; the whole passivity question is about the second.
* **Passive ⇒ stable, provable in two lines.** For ε″ ≥ 0, Im V = −k₀²ε″ ≤ 0, so
  Im⟨ψ|Ĥ|ψ⟩ = ∫Im(V)|ψ|² ≤ 0, hence Im(E) ≤ 0 for every mode — no growing
  solution can exist. This needs no simulation.

---

## 4. Already attempted — and where each attempt is vulnerable

These are recorded so the same ground isn't walked blindly. Each entry lists what
was found *and* what would overturn it — treat the second half as an attack
surface, not a closed door.

1. **"Passive profiles form a convex cone; find its extreme rays."** True, and it
   is Ivanenko et al., *SIAM J. Appl. Math.* **79**, 436 (2019) — passive
   approximation on a finite union of intervals with a B-spline-generated
   positive measure, plus a density theorem.
   *Overturned by:* showing the spatial problem (unequal endpoint values, hence
   the ramp decomposition of §3) is not covered by their setting.
2. **"The Toeplitz eigenvalue is a cheap test that avoids the Hilbert
   transform."** Multiplying coefficients by −i·sgn(m) *is* the Hilbert
   transform, and the eigenvalue carries the same information as reading
   min ε″ (identity in §3).
   *Overturned by:* a criterion that reads only a finite number of moments or
   low-order coefficients of ε′ — i.e. one that does not need the whole sequence.
3. **"DiHiTI is blind to a linear term, so the on-axis test is unsound."** The
   blindness is real; the mechanism is published in Waters, Hughes, Mobley &
   Miller, *IEEE UFFC* **50**, 68 (2003), which describes differential KK forms
   as having "shape invariance with respect to subtraction constants."
   *Overturned by:* a case where the ramp changes physical reflectance or
   stability while DiHiTI reports no change.
4. **"DiHiTI's output leaks 17 % of its power into the forbidden half-plane."**
   This was wrong — an artifact of taking a raw FFT across the periodic wrap,
   where the sawtooth ramp alone contributes exactly 50 %. The project's
   `skk_spectral_fom` is correct: forbidden power converges to machine zero
   (1.1e−09 → 0.0 as N goes 2048 → 32768), so DiHiTI's (ε′, ε″) is a genuine
   analytic pair. Recorded because the mistake is easy to repeat.
5. **Eigenvalue screening for the Longhi instability.** Did not produce a usable
   answer: (a) Longhi's own demonstration sits at an *exceptional point*
   (A₄ = −20/k₀²) where Im(E) = 0 and growth is algebraic, so eigenvalues are
   blind to it; (b) Dirichlet boundaries trapped his convectively-amplified case
   into a spurious absolute instability (UNSTABLE at domain ±60, "no bound state"
   at ±120).
   *Overturned by:* beam propagation with absorbing/PML boundaries, or a
   pseudospectral/propagator-norm diagnostic that catches algebraic growth.

*Note:* `cheap_screen.py` Part B argues the D-term is physically vacuous because
a coating is "clamped" outside its edges. That reasoning is **superseded** — the
continuation is periodic, not clamped. Part A (the screen itself) is unaffected.

---

## 5. The open target

The exact criterion is known, which makes the valuable prize a cheap
***sufficient*** condition: a property of ε′ checkable without computing λ_min
that *guarantees* ε″ ≥ 0. Nothing rules this out, and it is not in the literature
for spatial profiles.

Where to look: coefficient sequences that are automatically positive-definite —
Fejér kernels, Poisson kernels, and anything of autocorrelation form
cₘ = Σₙ bₙ b*_(n+m) (Fejér–Riesz). The question is which ε′ shapes have
de-ramped Fourier coefficients of one of those forms, and whether that converts
into a real-space statement about ε′.

Success = "if ε′ has properties X, Y, Z then it is passive", provable, and
covering profiles useful as AR coatings (ε_b on one side, 1 on the other). A
partial result covering a restricted but useful family is worth having.

One known difficulty to design around rather than ignore: every DiHiTI profile
sits *exactly* on the boundary of the passive cone (margin = 0.0000, §3), so any
criterion must be sharp there — approximate boundaries will misclassify precisely
the profiles of interest.

---

## 6. Files

**Here (`theory/passivity_2026Aug/`):**

| file | purpose |
|---|---|
| `profile_generator.py` | on-demand labelled (ε′, ε″) samples; `--list`, `--gallery`, `sample(n)`. **Start here.** |
| `constructor2.py` | passive-by-construction optimiser (non-negative basis + ramp) |
| `repair.py` | repairs gain-carrying targets; TMM reflectance cost |
| `cheap_screen.py` | parity/area/slope necessary conditions (see §4 note) |
| `periodic_model.py` | log-divergence scaling; DiHiTI ≡ de-ramp + periodic HT |
| `make_figs.py` | the three constructor figures |

**Elsewhere:** `tmm_helper.py` (`logistic`, `eps`, `ht_derivative`,
`skk_spectral_fom`); `passivity_constraint_notes.tex` in the repo root — 546
lines of prior analytical work with a Fourier mode-by-mode classification and the
"two failure mechanisms", worth reading before redoing that analysis;
`Papers/Longhi_sKK_Slowly_Decaying_Profiles_2015.pdf` §3 for the stability motivation.

**Figures** (`theory/figures/2026Aug12_passive_constructor/`):

| figure | shows |
|---|---|
| `profile_gallery.png` | all 12 generator families at default parameters, each labelled PASSIVE or GAIN |
| `fig1_constructor_validation.png` | constructor reproduces DiHiTI on the logistic (curves overlap) |
| `fig2_repair_gallery.png` | four gain-carrying targets; red shading = gain, black dashed = passive replacement |
| `fig3_price_of_passivity.png` | gain magnitude vs bump amplitude, and the reflectance cost of removing it |
