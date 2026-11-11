# Paper 2 positioning: Li 2020, the proposed outline, and the Kim 2025 review (2026-10-13)

Both PDFs were read in full in the main session; they are in `Papers/`:
- `2020_Li_Virtual_Parity_Time_Symmetry.pdf` = Li, Mekawy, Krasnok & Alù, *Phys. Rev. Lett.* **124**, 193901 (2020).
- `2025_Kim_complex_frequency_excitations_review.pdf` = Kim, Krasnok & Alù, *Science* **387**, eado4128 (2025).

Related reports from the two literature agents: `prior_art_skk.md` and `prior_art_complex_freq.md`.
This file supersedes their description of Li 2020: the agents called the overlap "the same
physics", which is too strong (§1.3).

Conventions: Li and Kim use e^{+iωt}, so a decaying drive has Im ω > 0. We use e^{−iωt},
ω_c = ω_r − i|ω″| = 2π(1 − iη), b = |ω″|. Everything below is in our convention unless quoted.

---

## 1. Li et al. 2020, "Virtual parity-time symmetry"

### 1.1 What the system is

- Two parallel RLC resonators coupled through a capacitor C_c (and a conductance G_c). Each
  resonator is wired to its **own** transmission line (impedance Z₀), so the left resonator owns
  the left port and the right resonator the right port. Fig. 2(a), p.3.
- It is lumped and 0-D: no material, no χ(ω) or ε(ω), no spatial profile, no angle of incidence.
  It works at a single operating frequency ω_PT and is modelled with temporal coupled-mode theory
  (CMT), Eqs. (2)–(6).
- Drive: |s₊⟩ = e^{−σt + jωt}|s₀₊⟩ with **σ = γ₀, the average intrinsic loss of the two
  resonators** (p.2: "with decay rate σ = γ₀, i.e., balancing the average loss γ₀ of H₀").
  This choice is what maps the circuit exactly onto a PT-symmetric (balanced gain/loss) Hamiltonian
  H₀^PT. That mapping is the point of the paper.

### 1.2 Where it overlaps with us (exact locations)

| Location | What they show | Our counterpart |
|---|---|---|
| p.2, Eqs. (1)–(3) | Offsetting the average decay maps a passive lossy dimer onto a PT dimer at complex frequency; the S-matrix obeys the PT relation. | The "co-decaying frame" argument: each oscillator's damping γ → γ − 2b, so a lossless mode gains. |
| p.3, Eq. (4) | Output = quasi-stationary term S\|s₊⟩ + transient term evolving with the eigenvalues of H₀_eff = H₀ + jΓ. | Our forced-plus-free decomposition; the stability criterion. |
| p.3, Eq. (7) and the text after it | "Virtual ATR" (anisotropic transmission resonance): \|S₁₁\|² = 0 at ω_PT, "PT symmetry ensures \|S₁₂\|² = \|S₂₁\|² = 1", \|S₂₂\|² = 4(γ̂_L − γ̂_R)²/ĉ². | r_L ≈ 0, τ = 1, r_R = O(1) (B8). |
| p.3–4, Eq. (8) | Maximise the slowest transient decay for a given \|S₂₂\|. Optimum: δ̂ = −ĉ/2, **γ̂_R = 0 (lossless right resonator)**, ε̂ = γ̂_L + ĉ. At \|S₂₂\| = 1, γ_tran = 4γ₀ against a drive decaying at γ₀. | "Every pole decays faster than the drive"; our lossless dark mode (γ₂ = 0). |
| Fig. 3, p.4 | Circuit time-domain simulation from rest, ω₀ = 2π·50 MHz, γ̂₀ = 0.01. (a) Left incidence: reflection (green) dies out, transmission (blue) locks onto the input. (b) Right incidence: reflection locks onto the input; output power is twice the input, "mimicking gain". | B8e (EIT r_L, r_R, τ settling). |
| p.4–5, Fig. 4 | Virtual CPA-laser. The transient is suppressed by a monochromatic **pre-excitation** (state preparation, Eqs. (9)–(10)), because virtual CPA "requires inputs with the same decay rate as the transient … hence it may be difficult to observe". | Not done by us; an sKK analogue would be a candidate extension. |

**What is shared, stated once.** A lossless resonator coupled to a lossy one acquires virtual gain
under a decaying drive. Whether the transient settles must be checked against the drive decay.
Done right, the result is a settled one-way response with unit transmission. This is real overlap,
and Li 2020 must be cited and discussed in the main text, not only in a reference list.

### 1.3 Where it does not overlap

1. **Their stability margin comes from the ports; ours comes from the material.**
   - With σ = γ₀ (the average intrinsic loss), the **port-less** dimer in the co-decaying frame is
     an exact PT pair. In the unbroken phase used for the ATR, both eigenvalues are real, so both
     modes decay at exactly the drive rate. That is marginal stability.
   - All of the margin (γ_tran = 4γ₀) comes from leakage into the transmission lines through Γ in
     H₀_eff = H₀ + jΓ, with port coupling ε̂ = γ̂_L + ĉ ≈ 6γ̂₀ (Fig. 3 caption). Eq. (8) also
     shows the margin shrinking as |S₂₂| grows: γ_tran → γ₀ as |S₂₂| → ∞. More contrast means
     less stability.
   - **A bulk medium has no port into which a material resonance can leak.** Worse, a material
     pole of ε(ω) at ω_p spawns an infinite accumulation of slab (Fabry–Pérot) poles around it,
     because n(ω) → ∞ there (visible in B7c). So in a material the **oscillators themselves**
     must decay faster than the drive. Our EIT medium does this: material poles at 2.2–2.8× the
     drive decay rate, with zero help from radiation. Li's recipe, used as a constitutive response,
     would sit exactly on the stability boundary.
   - Proposed sentence for the paper: *"Li et al. showed that a lossless resonator coupled to a
     lossy one acquires virtual gain, with the transient margin supplied by leakage into the
     ports. In a bulk medium no such leakage exists, and we show the material's own poles can
     carry the margin."*
2. **Material level against circuit level.** Li never defines χ or ε. Nothing in the paper
   corresponds to:
   - Im χ(ω_c) < 0 together with all material poles below ω_c;
   - the passivity window for a medium (two-line: f₂/f₁ ≤ γ₂/γ₁), or the EIT model being
     passive by construction;
   - the no-go for Long-type single-Lorentz media (Im χ(ω_c) < 0 forces γ < 2b, i.e. a pole
     above ω_c);
   - the pole re-weighting mechanism (Im ∝ 1/(γ − 2b), so the narrow negative-residue term wins).
3. **Different origin of one-wayness.**
   - Li: a single-frequency resonance of a two-port network, from the PT arrangement (gain on one
     side, loss on the other), at one incidence.
   - Ours: the spatial analyticity of the sKK profile ε(y). It works for any target profile
     (simple 4th-order KHP pole, random 6-pole), at every angle with both polarisations (B9), in a
     continuous graded medium, with |τ| = 1 from the King–Horsley–Philbin construction.
4. **Scale of the demonstration.** Li: one circuit at η ≈ 0.01, CMT plus a circuit simulation.
   Ours: frequency-domain transfer-matrix calculations, exact Fourier synthesis and an independent
   two-oscillator FDTD; two targets, two values of η, oblique incidence; structures of 400–500
   layers.

### 1.4 Things in Li worth borrowing

- The **state-preparation** trick (pre-excite monochromatically, then switch to the decaying
  drive, Eqs. (9)–(10)) suppresses the transient instead of out-running it. It could shorten our
  ~12-e-fold settling time and is worth one test.
- Their transmission-line example (Fig. 1, p.2) calls *spatial growth* along a uniformly lossy line
  "virtual gain" when σ > γ₀. A referee may raise it as "uniform virtual gain in a passive
  distributed medium". It is a different notion: there Im ε(ω_c) stays > 0, and the growth comes
  from the complex k of the decaying wave itself. Our definition must be stated explicitly as
  Im χ(ω_c) < 0 of the constitutive response.

---

## 2. Proposed paper: "Spatial Kramers–Kronig media at complex frequency" (working title)

### 2.1 Organising principle

A complex-frequency sKK effect is observable only if the response settles. It settles iff every
pole of the material **and** of the structure lies below the drive line (decays faster than the
drive). This one criterion sorts every target into "observable" and "not observable", and turns
the paper from a catalogue into a map. Li 2020 then becomes one data point (a circuit whose margin
comes from its ports), not the foil.

### 2.2 Outline

**Part I: theory**
1. sKK at complex frequency for both signs of ω″. ε(y, ω_c) is analytic in the appropriate
   y-half-plane, and the KHP unit-transmission sum rule ∫χ dy = 0 still applies, now evaluated at
   ω_c.
2. Which sKK targets need gain:
   - |t| = 1 requires ∫Im χ dy = 0, so Im χ must take both signs.
   - That includes one-way invisibility (KHP PRL 2017), bidirectional invisibility (Longhi OL
     2016), and the gain-requiring generalised Pöschl–Teller profiles (Horsley–King–Philbin J.
     Opt. 2016).
   - Absorbers and zero-R&T profiles do not need gain: King–Horsley–Philbin J. Opt. 2017; Ye 2017
     built a passive one.
3. The stability criterion (forced plus free decomposition; the co-decaying frame).
4. No-go for a bulk single-Lorentz medium (Long-type): Im χ(ω_c) < 0 ⇔ γ < 2b ⇔ the pole lies
   above ω_c. Material poles also seed accumulating slab poles, so the structure diverges (B3, B7).
   Scope it to the **bulk** Im χ, because resonators with radiative loss can evade it (Zeng &
   Zhang 2024; Xue 2026).
5. Passive two-pole media with stable virtual gain:
   - two-line model: exact passivity iff f₂/f₁ ≤ γ₂/γ₁; virtual gain for any γ₁ > γ₂ > 2b;
   - EIT model: passive by construction; lossless dark mode plus strong coupling;
   - the margin M = slowest pole decay / b ≥ 2;
   - lossless media cannot give stable virtual gain.

**Part II: the map (each target gets a frequency-domain prediction and a time-domain verdict)**

| Target | Drive | Material | Status |
|---|---|---|---|
| One-way invisible, \|t\| = 1 (KHP) | decaying | EIT / two-line | **Done** (B7–B9): settles, FDTD-confirmed, angles OK |
| Same target, Long-type | decaying | single Lorentz | **Done**: diverges (B3, B7, B8) |
| Bidirectional invisibility (Longhi OL 2016) | decaying | EIT | To do (cheap: `eit_design` takes any χ(y)) |
| Gain-requiring generalised Pöschl–Teller (HKP 2016) | decaying | EIT | To do; optional |
| Virtual sKK amplifier (\|r_R\| > 1 on purpose) | decaying | EIT | To do; optional; framing only |
| One-sided absorber from lossless media | growing | γ → 0 Lorentz | **Done** (B4–B6), with caveats §2.3 |
| sKK version of state preparation (Li Eqs. 9–10) | decaying | EIT | Optional; could cut the settling time |

**Part III: practicalities**
- Time-domain settling time (~12–13 e-folds of drive decay for the r_L floor; contrast of 1e2–1e3
  after ~6–8 e-folds) and its η-independence.
- Synthetic complex frequency (Kim/Guan route): valid only with no poles between the real axis and
  ω_c. Our passive two-pole designs satisfy this; Long-type designs do not.
- Required material parameters: resonances within ±14 % (η = 0.05) or ±3 % (η = 0.01) of ω_r;
  ω_p² ≤ 0.2 ω_r²; bright Q ≈ 1/(10η). The realisation as an EIT metamaterial is not mapped yet.
- Truncation amplification under decaying drives (B2): the condition η·L/λ ≲ 0.1–0.3.

**Format.** A long-format journal (PRA, PR Applied, J. Opt.) rather than a letter. This suits a
thesis chapter. Most of the machinery exists; each new target is hours, not weeks.

### 2.3 Caveats and disclaimers for the growing-drive (exponentially rising) half

These must be stated plainly in the paper; the growing-drive section is the weaker half.

1. **Novelty is low.** "Virtual absorption" in lossless systems under growing drives is well
   established:
   - Baranov, Krasnok & Alù, Optica 2017 (coherent virtual absorption);
   - Trainiti et al., Sci. Adv. 2019 (elastodynamic);
   - Ra'di, Krasnok & Alù, ACS Photonics 2020 (virtual critical coupling);
   - Marini et al., Opt. Lett. 2020;
   - Rasmussen et al., Adv. Sci. 2023 ("lossless sink");
   - Hinney et al., Nat. Commun. 2024.

   The Kim review gives it a full section (p.4–5, Fig. 4). Only the sKK and graded, all-angle
   version would be new.
2. **It is storage, not absorption (Parseval; B5 finding 1).** For any finite pulse, the
   time-integrated reflected, transmitted and absorbed energies equal the |D(ω)|²-weighted
   real-frequency |r|², |t|² and A. No envelope, whether growing, decaying, or rising then
   decaying, can create net absorption. It only moves energy around in time.
3. **Delayed release (B5 finding 2).** For the near-lossless, under-resolved structure
   (Q = 10⁴, N = 400):
   - reflection during the drive is 7e-7 of the pulse energy;
   - after the drive stops, **46 %** of the energy comes back as delayed reflection (peaking
     ~730 T_r later);
   - a further 10 % comes out as delayed transmission.

   Integrated over the pulse, this is a mediocre mirror. The Kim review describes the same release
   for coherent virtual absorption ("once the exponential growth stops … the accumulated energy is
   released", p.5).
4. **When it works without the release, it is not a complex-frequency effect (B5 findings 3–4).**
   The Theorem-1 design of a Lorentzian sKK is exactly Ye 2017's chirped-resonance medium. Once
   the resonance zone is resolved (Q = 100 at N = 400, or Q = 10⁴ at N = 40000), the structure is
   an ordinary real-frequency one-way absorber: R_L ≈ 1e-6, A ≈ 0.86, with or without growth. The
   growing drive only widens the effective pole depth during the pulse (δγ/(2b + γ) → δ),
   temporarily hiding under-resolution.
5. **"Lossless" is subtle.** With γ → 0 the structure is a forest of narrow resonances that still
   absorbs at real ω through resonance absorption, independently of γ (B4 caveat). A lossless
   *non-dispersive* dielectric stays lossless at any complex frequency: dispersion is required.
6. **Growth-rate limit.** For the Horsley Lorentzian target, ω₀² > 0 limits the growth to
   |η| ≲ 0.03 (B4).
7. **Stability is not the issue here.** A growing drive lies above every passive pole, so the
   quasi-steady state is always reached. The issues are energy return and the unbounded growth
   (the drive must stop sometime). Kim et al. note that complex-frequency signals "must be
   truncated in time to remain physically realizable" (p.4).
8. **Generation.** Arbitrary waveform generators reach growth/decay rates of a few hundred MHz;
   electro-optic modulators reach ~100 GHz in the near-IR and visible (Kim p.4). At optical
   frequencies the synthetic (post-processing) route is the practical one. It needs phase-resolved
   multi-frequency measurements and "extensive postprocessing" (Kim p.4–5).

**Recommended treatment.** One section that states the result honestly: one-way during the drive,
then release, with a Parseval budget. Present it as the completeness half of the map, not as a
headline. The decaying-drive / passive-two-pole half carries the novelty.

### 2.4 Caveats for the decaying-drive half (for completeness)

- Passive, **not lossless**: the bright resonator is lossy, Q ≈ 1/(10η). Stable virtual gain
  cannot come from lossless media.
- Settling needs ~12–13 e-folds of drive decay for the r_L floor, independent of η, so it is slow
  in absolute time at small η (∝ 1/η).
- The r_L floor (1e-5 to 2e-4) is set by frequency-domain truncation of the profile (B2), not by
  the dynamics.
- The design's ω₀(y) has jumps where the root-finder switches branch. They are harmless but look
  alarming (B9b); a continuous branch should be enforced before publication.
- A physical EIT metamaterial realisation, with layer-by-layer tuned bright/dark pairs, has not
  been worked out.

---

## 3. Kim, Krasnok & Alù, *Science* 387, eado4128 (2025): what to know, cite, and point to

### 3.1 Bottom line

The review never mentions sKK, Horsley, spatial Kramers–Kronig relations, graded or one-way graded
media, a material-level Im ε(ω_c) < 0, or multi-pole media engineered for stable virtual gain.
**It does not undercut the sKK combination or the passive-two-pole material result.** It does make
three things common knowledge that we must not claim:
1. the general transient/stability issue;
2. the virtual-PT one-way response (Li 2020, featured in Fig. 5D);
3. virtual absorption with growing drives.

### 3.2 Points to know (with locations)

- **Framework (p.1–2).** Virtual loss = drive matched to a scattering **zero** (growing signal);
  virtual gain = drive matched to a scattering **pole** (decaying signal). Their "virtual gain" is
  a scattering-level notion (outputs larger than inputs), not a constitutive Im ε < 0. Our paper
  must define its usage explicitly to avoid confusion.
- **Eq. (1), p.3.** Single-port CMT: r_qs at complex ω.
- **Eq. (2), p.3: transient decomposition.** r(t) = r_qs + r_ts(t), with
  r_ts(t) = e^{Z(ω)t}·2γ_ex/Z(ω). Quote: complex-frequency excitations "may produce a dominant or
  even diverging transient that may mask the quasi-steady-state response … However, by properly
  tuning the system parameters and the excitation waveform, one can ensure that the
  quasi-steady-state response dominates."
  - In their single-port model the total resonator loss γ_ex + γ_in sets the stability limit. A
    single resonator **with radiative loss γ_ex** can therefore show stable virtual gain whenever
    γ_in < |ω_i| < γ_ex + γ_in. This is consistent with our scoping of the single-Lorentz no-go to
    bulk media.
- **p.4.** Complex-frequency signals are unbounded and "must be truncated in time"; complex
  exponentials are non-orthogonal, so the system "may not necessarily respond solely at that
  frequency". Practicality requires |ω_i| ≪ ω_r (Fig. 3A). Generation limits: arbitrary waveform
  generators up to a few hundred MHz, electro-optic modulators ~100 GHz.
- **p.4–5: synthetic complex frequency** (Fig. 3F–H): sample the spectrum at real frequencies,
  recombine with complex weights. It needs phase-resolved measurements; it is "cumbersome in
  high-frequency settings" and not applicable to nonlinear systems.
- **p.5–6, Fig. 5.**
  - virtual gain beating passivity bounds on scattering (Kim PRL 2022);
  - zero forward scattering;
  - optical pulling force (Lepeshov & Krasnok 2020);
  - **Fig. 5D**: virtual PT ATR (Li 2020): "transmission remains at unity, but reflection is zero
    from one port and large from the other";
  - **Fig. 5E**: transient non-Hermitian skin effect in an acoustic lattice (Gu et al. Nat. Commun.
    2022). A decaying drive amplifies waves travelling away from the source: directional, but a
    lattice effect, not reflectionlessness.
- **p.7, Fig. 6.** Superlens loss compensation (Kim PRX 2023; Guan Science 2023), polariton decay
  length (Guan Nat. Mater. 2024), biosensing (Zeng eLight 2024). All are loss **compensation**
  (drive decay ≤ material loss), not net gain.
- **p.8, Conclusions.** Quote: complex-frequency signals "are temporally bounded and nonorthogonal.
  This means that a system under such excitations is not guaranteed to reach a quasi-steady state.
  However, with proper design, we can access complex zeros and poles…"
- **Ref. 146** is Novitsky PRA 107, 013516 (2023) (virtual inversion in two-level media; prior
  statement of the single-resonance condition).

### 3.3 The outlook we can point to (open problems that match our contribution)

- p.8: "By integrating all-optical excitation schemes, **designing optimal waveforms for
  quasi-steady states, and engineering resonant structures with tailored singularities at complex
  frequencies**, these techniques can push wave phenomena beyond traditional limits…" This is the
  quotable motivation for stability by material design: we engineer the material's singularities
  so that the quasi-steady state is guaranteed.
- p.8: "Emerging platforms such as metasurfaces, polaritonic materials …" and "scaling to optical
  frequencies remains challenging owing to fast temporal modulation requirements. Solutions include
  … high-Q resonances to shift complex-frequency singularities closer to the real axis." Our η = 0.01
  designs (resonances within ±3 % of ω_r) are in that spirit.
- p.8: they mention the Rozanov bound for absorbers as a target for complex-frequency relaxation.
  This is tangential, but sKK absorbers are the textbook thin broadband absorber, so it could be
  worth a sentence in the growing-drive section, with the Parseval caveat.

### 3.4 Citations to take from the review (verified in its reference list)

| Ref # | Citation | Use |
|---|---|---|
| 58 | Li, Mekawy, Krasnok, Alù, PRL 124, 193901 (2020) | Main precursor (§1) |
| 52 | Ra'di, Krasnok, Alù, ACS Photonics 7, 1468 (2020), virtual critical coupling | Growing-drive background |
| 88 | Baranov, Krasnok, Alù, Optica 4, 1457 (2017), coherent virtual absorption | Growing-drive background |
| 99 | Trainiti et al., Sci. Adv. 5, eaaw3255 (2019) | Growing-drive background |
| 100 | Rasmussen et al., Adv. Sci. 10, e2301811 (2023), lossless sink | Growing-drive background |
| 80 | Hinney et al., Nat. Commun. 15, 2741 (2024) | Experimental optical complex frequency (telecom) |
| 55 | Kim, Lepeshov, Krasnok, Alù, PRL 129, 203601 (2022) | Virtual gain beyond passivity bounds |
| 56 | Kim, Peng, Yves, Alù, PRX 13, 041024 (2023) | Loss compensation / superlens |
| 57 | Guan et al., Science 381, 766 (2023) | Synthetic complex frequency; one-sentence stability remark |
| 81 | Guan et al., Nat. Mater. 23, 506 (2024) | Synthetic complex frequency, polaritons |
| 59 | Gu et al., Nat. Commun. 13, 7668 (2022) | Directional virtual-gain effect (transient skin effect) |
| 146 | Novitsky, PRA 107, 013516 (2023) | Single-resonance virtual-gain condition |
| 118 | Ge, Chong, Stone, PRA 85, 023802 (2012) | ATR definition |
| 119 | Lin et al., PRL 106, 213901 (2011) | Unidirectional invisibility (PT, real gain) |

Plus, from outside the review: Long et al. LSA 2026; Guan et al. Nat. Phys. 2026 (β < γ/2
condition); Lalanne & Wu arXiv:2508.10742; Zeng & Zhang, Acta Opt. Sin. 2024; Severs Millard et al.
arXiv:2608.30146; Anisimov et al. PRL 2011 (EIT lineshape); and the sKK canon (Horsley–Artoni–La
Rocca 2015; HKP 2016; KHP PRL 2017; KHP J. Opt. 2017; Longhi EPL 2015 and OL 2016; Ye 2017;
Horsley–Artoni–La Rocca PRA 2016; Horsley–Longhi PRA 2017).

---

## 4. Still unread (to be fetched)

1. Horsley, Artoni & La Rocca, "Reflection of waves from slowly decaying complex permittivity
   profiles", Phys. Rev. A 94, 063810 (2016).
2. Horsley & Longhi, "Spatiotemporal deformations of reflectionless potentials", Phys. Rev. A 96,
   023841 (2017).
3. (optional) Horsley & Longhi, "One-way invisibility in isotropic dielectric optical media", Am.
   J. Phys. 85, 439 (2017).

Titles checked against Crossref on 2026-10-13. None of these is in `Papers/` as of that date. Note
that `Papers/Longhi_sKK_Slowly_Decaying_Profiles_2015.pdf` is Longhi's EPL paper ("Wave reflection
in dielectric media obeying spatial Kramers–Kronig relations", arXiv:1510.01491), **not** item 1.
