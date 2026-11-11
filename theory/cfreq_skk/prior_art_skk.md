# Prior art for complex-frequency (virtual-gain) realisations of sKK media

Librarian report, 2026-10-13. STATUS: FINAL. The resumable-state block is at the end.

Scope: claims 1–4 of `theory/cfreq_skk/FINDINGS.md` (B1–B9), checked against the spatial
Kramers–Kronig (sKK) literature and the complex-frequency / virtual-gain literature.
Convention: e^{−iωt}, ω_c = ω_r − i|ω″|. A "decaying drive" means |ω″| > 0.

Companion files (still valid; not repeated here):
- `research/prior_art_virtual_gain_stability.md`: who already states the condition "the drive must
  decay slower than every pole".
- `research/prior_art_2026Oct12.md`: temporal sKK (Hayran 2021) and the chirped-resonance sKK of Ye 2017.

---

## 0. VERDICT

**(i) Using complex-frequency virtual gain to realise the gain regions of sKK media: not found anywhere.
As far as an extensive search can establish, this is novel.** No paper evaluates an sKK or graded
reflectionless profile at complex frequency, or drives one with a decaying or growing excitation, or
replaces sKK gain regions with virtual gain. Long et al. (LSA 2026) never mention sKK, Horsley, or
reflectionless media; I checked the full text and SI with grep. None of the papers citing Long does
either. Google Scholar full-text phrase searches return **0** results for "virtual gain" + "spatial" +
"Kramers" + "reflectionless". Every hit for "spatial Kramers–Kronig" + "complex frequency" uses
"complex frequency plane" only in the ordinary frequency-KK sense.

Caveat. Claim 1 on its own, in the frequency domain, is exposed to a referee saying "you just
substituted ω → ω_c into Long's Theorem 1". Your own B3 result also shows that single-pole designs
never settle. The paper's positive content therefore has to be claims 3–4, not claim 1.

**(ii) Stable (multi-pole) virtual gain applied to sKK: the sKK application is novel, but the mechanism
only partly is.** A direct precedent for the stabilising principle exists, one level down (a lumped
two-resonator circuit, not a medium):

> **Li, Mekawy, Krasnok & Alù, "Virtual parity-time symmetry", PRL 124, 193901 (2020).**
> Full text read (arXiv:1911.12875).
> - **System.** Two coupled resonators, one **lossy** (γ_L ≈ 2γ₀) and one **lossless** (γ_R = 0),
>   driven at ω + jσ with σ = γ₀. That is, the drive decays at the mean loss.
> - **Mechanism.** In the quasi-stationary state the pair acts as a balanced gain/loss (PT) dimer. They
>   *optimise the design so the slowest transient decays at 4γ₀*, "reasonably larger than the decay
>   rate γ₀ of the excitation" (Eqs. 4 and 8).
> - **Result.** Time-domain circuit simulations show a **virtual anisotropic transmission resonance**
>   (ATR) that settles: S_LL → 0 after a short transient, |S_RL| = 1, |S_RR| = 1.
> - **Physical picture.** They attribute the virtual gain to "reactive energy stored at earlier times in
>   the system".
>
> This is the same physics as your EIT medium (lossy bright oscillator plus lossless dark oscillator),
> and the same output pattern as B8 (r_L ≈ 0, |τ| = 1, |r_R| = O(1)). The differences: it is a
> two-port lumped device, not a susceptibility, not graded, not sKK, and has no angle dependence.
> **Rating: PARTIAL OVERLAP. Cite it prominently and differentiate explicitly.**

What I could not find anywhere, and so appears to be new:
- **(a) The constitutive-level statement.** A passive multi-pole susceptibility can have
  Im χ(ω_c) < 0 with *every* material pole decaying faster than the drive. Contrast this with Long's
  single-pole Theorem 1, where Im χ(ω_c) < 0 forces a pole above ω_c (γ < 2|ω″|).
- **(b) The two-line "broad Lorentzian minus narrower Lorentzian" medium** as a virtual-gain material.
- **(c) Graded, all-angle, unit-transmission (King–Horsley–Philbin) one-way-invisible sKK structures**
  built from passive layers.
- **(d) Time-domain settling of an extended structure** (Fourier synthesis and FDTD, oblique angles),
  where the structural (slab) poles, not only the material poles, set the stability margin.
- **(e) Claim 2 applied to Long's media.** The single-Lorentz no-go is a known general condition
  (drive slower than every pole), but applying it to Long's Im ε < 0 media appears unpublished (see the
  companion file). I found no critique of Long et al.

Two corrections to the task brief that the paper text must also respect:
1. The 2015 Nature Photonics sKK paper is by **Horsley, Artoni & La Rocca**, not "Horsley, King &
   Philbin". Horsley–King–Philbin is J. Opt. 18, 044016 (2016).
2. **sKK profiles do not in general require gain.** Horsley 2015 states that they "do not rely in
   general on the presence of gain", and every experiment is a passive absorber. Gain is required for
   specific targets:
   - unit-transmission profiles (∫χ dx = 0 ⇒ ∫Im χ dx = 0; King–Horsley–Philbin PRL 2017, Eq. 7);
   - bidirectionally invisible profiles (Longhi OL 2016);
   - generalised Pöschl–Teller profiles (Horsley–King–Philbin 2016).

   Phrase the motivation as "unit-transmission / invisible sKK profiles require gain".

---

## 1. Coverage: what was searched, and what could NOT be accessed

**Forward-citation sweeps (titles and abstracts keyword-screened).**
- Semantic Scholar: 204 citers of Horsley 2015.
- OpenAlex: 252 citers of Horsley 2015 (union with Semantic Scholar about 260 distinct records,
  including about 25 spurious DFT/materials hits).
- Other sKK papers:

| Paper | Citers screened |
|---|---|
| Ye 2017 | 72 |
| King–Horsley–Philbin PRL 2017 | 21 |
| Longhi EPL 2015 | 51 |
| King et al. J. Opt. 2017 | 12 |
| Horsley–King–Philbin J. Opt. 2016 | 39 |
| Horsley–Longhi AJP 2017 | 18 |
| Horsley–Longhi PRA 2017 | 11 |
| Philbin 2016 | 5 |
| Jiang et al. LPR 2017 | 51 |
| Hayran 2021 | 27 |

- Complex-frequency side:

| Paper | Citers screened |
|---|---|
| Long 2026 | 1–3 per database; Google Scholar adds Guan Nat. Phys. 2026 |
| Li 2020 (Virtual PT) | 106 |
| Kim–Krasnok–Alù Science 2025 review | 51 |

Scripts and JSON are in the session scratchpad (`s2/fetch_cites.py`, `s2/oa_cites.py`); they are not
needed again.

**Full-text phrase searches.**
- Google Scholar via WebFetch: "spatial Kramers-Kronig" + "complex frequency"; + "virtual gain";
  "Kramers-Kronig" + "virtual gain"; "virtual gain" + "unidirectional invisibility"; virtual gain +
  EIT / dark mode; "virtual parity-time symmetry"; "optical antimatter" + "complex frequency".
- Europe PMC (open-access full text): the same combinations.
- About 15 web searches on the axes (a)–(e).
- arXiv full texts read or grepped: 1911.12875, 2606.30916, 2609.29293, 2010.10758, 2310.00590,
  2106.10975, 1811.04839, 1510.02920, 1607.04085, 1503.00152, 1612.02451 (including its
  supplementary material), 1803.04634, 1909.04017, 2606.07191, 2505.21976, 2604.26628.

**Not accessible (paywall or Cloudflare). A reference to sKK in these cannot be ruled out:**
- Kim, Krasnok & Alù, "Complex-frequency excitations in photonics and wave physics", Science 387,
  eado4128 (2025). This review is the highest residual risk: it could mention graded or one-way media
  as an application in passing. **Check it via the Rice library before submission.**
- Krasnok, "A new knob for resonant photonics: the waveform", OPN, April 2026 (feature; only the
  abstract was visible).
- Horsley, Artoni & La Rocca, PRA 94, 063810 (2016) and Horsley & Longhi, PRA 96, 023841 (2017):
  abstracts only.
- Horsley & Longhi, AJP 85, 439 (2017): title and venue only.
- Lončar et al., Photonics 8, 56 (2021): abstract only (403 on the PDF).
- Marini et al., EPJ Appl. Metamater. (2021): Google Scholar snippet only.
- Long et al. conference versions (FiO 2025; CLEO 2026 FW1E.6; APS Global Physics Summit 2026): not
  read.
- Google Scholar "cited by" for Horsley 2015 itself was not paged; I used Semantic Scholar and OpenAlex
  instead, so a few conference or non-English citers may be unscreened.

---

## 2. Findings by axis

Ratings: **SCOOPS** / **PARTIAL OVERLAP** / **RELATED–MUST-CITE** / **TANGENTIAL**.
"Read" means full text; "abstract" means abstract only.

### (a) sKK at complex frequency, or under decaying/growing excitation: NOTHING FOUND

Closest by vocabulary only:
- **Sweeney, Hsu & Stone, "Theory of reflectionless scattering modes", PRA 102, 063511 (2020),
  arXiv:1909.04017 (read/grep).** Describes one-sided reflection zeros ("R-zeros") of cavities at
  discrete complex frequencies. These are generically unidirectional when flux is not conserved. Cites
  Horsley 2015 only in a list. Does not propose exciting R-zeros with complex-frequency signals.
  **TANGENTIAL** (cavities, not graded media; no drive).
- **Sokol, MSc thesis, VUT Brno (2026), "Electromagnetic wave propagation in time-varying
  metamaterials" (abstract via the DSpace API).** One-way reflection in the *time* domain of dispersive
  time-varying media, linked to causality: a temporal sKK analogue. **TANGENTIAL** (it is the
  Hayran-2021 direction).
- **Staliunas, PRA (2026), arXiv:2606.07191** and **Koutserimpas & Monticone, OME (2024)**: "complex
  frequency plane" appears only as frequency-domain analyticity. **TANGENTIAL**.

### (b) sKK gain regions replaced by virtual gain or another passive mechanism: NOTHING FOUND

- **Long, Catrysse, Han & Fan, LSA 15, 48 (2026), doi:10.1038/s41377-025-02137-w (read, main + SI).**
  - Theorem 1 (single-pole Lorentz–Drude; any ε(ω_c) reachable); complementary media, perfect lens,
    superscattering.
  - No mention of sKK, Horsley, reflectionless or one-way media, stability, or transients.
  - Its future directions name "invisibility cloaking" and perfect absorbers.
  - Citers: Guan et al. Nat. Phys. (2026) and Liu et al. arXiv:2609.29293. Neither critiques Long or
    touches sKK.
  - **RELATED–MUST-CITE** (basis of claim 1).
- **Atomic sKK with real gain (abstracts):**
  - Geng et al., OE 31, 38228 (2023), four-wave mixing;
  - Li et al., OE (2024), doi:10.1364/oe.520454, Δ-type + microwave;
  - Lin et al., Photonics 11, 389 (2024);
  - Geng et al., PLA (2025);
  - Peng et al., PRA (2026), "inversionless gain".

  Pump-driven coherent media with linearly graded control fields give unidirectional reflection
  *amplification*. The gain is real, supplied by the pump fields, so it is active, not virtual.
  **TANGENTIAL**. At most one sentence: "gain in sKK-type atomic media has been supplied by coherent
  pumping".
- **Philbin, "All-frequency reflectionlessness", J. Opt. 18, 01LT01 (2016) (read).**
  - Builds a causal, passive ε(x, ω): analytic in the upper ω half-plane, Im ε > 0 for ω > 0.
  - It is reflectionless at *all* real ω for perpendicularly *exiting* waves.
  - The *entering* version needs gain at all frequencies.
  - Not complex-frequency excitation. **RELATED** (the dispersion-consistent sKK-type design; also a
    case where gain is forced).
- **"Passive PT" uniform-loss offset** (Guo et al., PRL 103, 093902 (2009); Ornigotti & Szameit,
  J. Opt. 16, 065501 (2014)). This is the trivial passive route; see `research/hayran_BplusA_assessment.md`.
  I found no paper applying it to sKK.

### (c) One-way invisibility from passive materials by a time-domain trick

- **Li, Mekawy, Krasnok & Alù, PRL 124, 193901 (2020): PARTIAL OVERLAP** (see §0).
  - Passive two-resonator circuit, decaying drive, quasi-stationary *virtual ATR*.
  - Time domain settles because the transients (4γ₀) outrun the drive (γ₀).
  - The lossy+lossless pair matches your EIT design.
  - It is lumped, two-port, single-frequency and has no angles, so it does not scoop the graded,
    all-angle sKK result.
  - It does pre-empt any claim to be the "first passive one-way invisibility by complex-frequency
    excitation" or the "first stable virtual-gain design".
- **Chen, He, Li, Li, Kou, Lu, Xu & Alù, "Observation of parity-time symmetry for evanescent waves",
  Commun. Phys. 7, 339 (2024) (read).**
  - Passive anti-PT acoustic scatterer; ATR (unidirectional invisibility) for *evanescent* waves at
    real frequency.
  - Mentions complex-frequency virtual gain only as an alternative (citing Li 2020).
  - **TANGENTIAL/RELATED**.
- **Brandstötter, Makris & Rotter, PRB 99, 115402 (2019) (read/grep).**
  - Broadband unidirectional invisibility via constant-intensity waves; pulse propagation shown.
  - Uses real gain/loss in a non-dispersive model; no causality or stability discussion.
  - **RELATED** (one-way-invisible media with pulses, but with gain).
- **Hayran, Chen & Monticone, Optica 8, 1040 (2021).** The temporal-sKK cloak needs real
  (strength-inverted) gain; see the B+A assessment. **RELATED**.

### (d) Dispersive or time-domain sKK analyses: stability or settling

Nobody treats the temporal settling of gain-containing sKK media under a causal dispersive model.

- **Longhi, EPL 112, 64001 (2015), §3 (read).**
  - Gain-containing KK media can host growing interface modes.
  - For the A < 0 double pole A/(x + iξ)², which gives t = 1 on both sides (exactly the King–Horsley–
    Philbin class), there is an exceptional point in the continuum. A seeded interface bound state
    then grows algebraically along y at grazing incidence (beam propagation, Fig. 2).
  - Nonlinear saturation "would modify the effective permittivity… breaking the exact reflectionless
    property".
  - This is a *spatial*, real-frequency instability caused by real gain, not temporal settling.
  - **RELATED–MUST-CITE.** Your real-ω structure has no gain at all, so this instability is absent.
    Say so; it strengthens the case for a passive realisation.
- **Longhi, "Bidirectional invisibility in Kramers–Kronig optical media", OL 41, 3727 (2016) (read).**
  - "Since in such media there is optical gain, some instability issues might arise."
  - Fig. 4: secular growth of an interface mode, suppressed by changing Θ.
  - **RELATED–MUST-CITE**.
- **Horsley, Artoni & La Rocca, PRA 94, 063810 (2016) (abstract only).**
  - Pulse propagation through slowly decaying sKK profiles "where the dissipation or gain is not
    asymptotically negligible".
  - No reflected pulse from one side; also discusses truncation.
  - The only time-domain sKK study by the originators. From the abstract it does not appear to use a
    causal dispersive material model, but this is **unverified (paywall)**.
  - **RELATED–MUST-CITE**.
- **Ye et al., Nat. Commun. 8, 51 (2017) (read).**
  - Dispersive "space–frequency Lorentz" medium, ε(ω, x) = ε∞ − ω_p²/[ω² + iγω − (ω₀ − qx)²].
  - Satisfies KK in ω at every x and is "broadband by nature"; measured in CW.
  - Its "causality/stability" remark refers to gain in transformation optics.
  - **RELATED–MUST-CITE.** FINDINGS item 5 shows that the Theorem-1 design of a Lorentzian sKK *is*
    this chirped-resonance medium; credit Ye explicitly.
- **Szenes et al., IEEE Metamaterials 2026 (abstract).** Plasmonic-nanorod sKK absorber; FDTD with
  12 fs pulses; passive. **TANGENTIAL** (passive time-domain sKK; no gain, no stability question).
- **King, Horsley & Philbin, PRL 118, 163201 (2017), supplementary material (read via arXiv:1612.02451).**
  "Wavenumber dependence" means a non-dispersive ε(x) probed at other k₀. No material dispersion and
  no time domain.

### (e) Multi-pole / EIT / Fano material models used to realise sKK profiles

- **Atomic sKK via multi-level (EIT-regime) responses (full-text grep):**
  - Zhang, Wu, Artoni & La Rocca, OE 29, 5890 (2021): three-level V atoms; a far-detuned driving field
    with linearly varying intensity maps frequency KK onto space.
  - Zheng et al., PRA 107, 013704 (2023): Rydberg dipole–dipole interactions.
  - Wang, Zhang, Du & Wu, PRA 108, 053716 (2023): four-level N → effective Λ; EIT plus linear sKK;
    coherent perfect absorption.
  - Pei et al., OE 31, 14694 (2023): tripod; two-colour; EIT windows.

  All are real-frequency and steady-state, and passive-absorptive apart from the pumped variants in (b).
  None uses complex-frequency excitation or two-pole virtual gain. **RELATED** (prior use of multi-level
  or EIT media as sKK materials; one sentence).
- **Ye 2017** (one Lorentz pole per position) and **Li et al., Commun. Phys. 7, 147 (2024)** (matryoshka
  metamaterial, independent Lorentz ε and μ): single-pole-per-quantity graded resonators.
  **TANGENTIAL/RELATED**.
- **Virtual-gain work using coupled-resonator / plasmon-induced-transparency (PIT) samples:**
  - Guan et al., arXiv:2604.26628 (2026): a PIT metamaterial used as a *test sample* for synthetic
    complex-frequency loss compensation.
  - Marini et al. (2021, snippet only): virtual absorbers that "mimic EIT", with lossless media and a
    growing drive.

  Neither uses the two-pole structure to obtain stable Im χ(ω_c) < 0. **TANGENTIAL**.
- **No paper found** with the two-line passive medium (χ = f₁/D₁ − f₂/D₂, f₂/f₁ ≤ γ₂/γ₁) or the
  EIT-medium design rule as a source of stable virtual gain.

### Stability-condition and dynamic-range literature (adjacent to claims 2–3)

See `research/prior_art_virtual_gain_stability.md` for the full entries:
- Guan et al., Science 381, 766 (2023);
- Lalanne & Wu, arXiv:2508.10742;
- Khurgin et al., arXiv:2607.02211;
- Xue et al., Nat. Commun. 17, 3359 (2026);
- Baranov, Krasnok & Alù, Optica 4, 1457 (2017).

New in this sweep:
- **Liu, Gong, Li, Chen, Rotter & Lin, arXiv:2609.29293 (2026) (read/grep).**
  - Synthesised complex-frequency excitation for zero-forward Kerker scattering.
  - Validity condition (Eq. 8): the drive's "artificial pole" must dominate the inherent scattering
    poles. Shows a transient-to-steady evolution.
  - Cites Long neutrally. **TANGENTIAL/RELATED** (another statement of the pole-dominance condition).
- **Guan, Lin, Chen, Wen, Li & Zhang, "High-order virtual gain…", Nat. Phys. (2026), arXiv:2505.21976.**
  - "The complex frequency wave requires sufficient virtual gain… inevitably triggering rapid signal
    decay to zero before reaching a quasi-static state."
  - **RELATED** to your dynamic-range caveat (12–13 e-folds needed).
- **Lončar, Vuković, Krois & Hrabar, Photonics 8, 56 (2021) (abstract).** "Any parity-time-symmetric
  system is necessarily marginally stable", including unidirectional-invisibility applications. Shown
  with time-domain transients and an RF demonstrator. **RELATED** (motivation: real-gain one-way
  invisibility is marginally stable; the passive virtual route avoids this).

---

## 3. Must-cite list for the paper

**sKK core**
1. Horsley, Artoni & La Rocca, Nat. Photon. 9, 436 (2015). doi:10.1038/nphoton.2015.106
2. Horsley, King & Philbin, J. Opt. 18, 044016 (2016). doi:10.1088/2040-8978/18/4/044016
3. King, Horsley & Philbin, PRL 118, 163201 (2017). doi:10.1103/PhysRevLett.118.163201 (target profiles)
4. Longhi, EPL 112, 64001 (2015). doi:10.1209/0295-5075/112/64001 (gain ⇒ interface instability)
5. Longhi, Opt. Lett. 41, 3727 (2016). doi:10.1364/OL.41.003727 (invisible KK media contain gain; instability)
6. Horsley, Artoni & La Rocca, PRA 94, 063810 (2016). doi:10.1103/PhysRevA.94.063810 (pulses through gain/loss sKK; abstract-verified only)
7. Ye et al., Nat. Commun. 8, 51 (2017). doi:10.1038/s41467-017-00123-4 (chirped Lorentz = the Theorem-1 Lorentzian design)
8. King, Horsley & Philbin, J. Opt. 19, 085603 (2017). doi:10.1088/2040-8986/aa7783 (optional; graded-resonator realisation route)
9. Philbin, J. Opt. 18, 01LT01 (2016). doi:10.1088/2040-8978/18/1/01LT01 (optional; dispersion-consistent passive design)

**Complex-frequency / virtual gain**

10. Long, Catrysse, Han & Fan, LSA 15, 48 (2026). doi:10.1038/s41377-025-02137-w
11. **Li, Mekawy, Krasnok & Alù, PRL 124, 193901 (2020). doi:10.1103/PhysRevLett.124.193901.
    Must be discussed, not just listed.**
12. Kim, Krasnok & Alù, Science 387, eado4128 (2025) (review; content unverified)
13. Guan et al., Science 381, 766 (2023) and Kim et al., PRX 13, 041024 (2023) (synthetic
    complex-frequency route; Guan's one-sentence instability remark)
14. Lalanne & Wu, arXiv:2508.10742 (general observability condition |Im ω| < |Im ω̃_m|)
15. Baranov, Krasnok & Alù, Optica 4, 1457 (2017) (coherent virtual absorption; needed for the B4/B5
    growing-drive results, if they are kept)
16. Optional: Xue et al., Nat. Commun. 17, 3359 (2026); Khurgin et al., arXiv:2607.02211; Liu et al.,
    arXiv:2609.29293; Guan et al., Nat. Phys. (2026); Lončar et al., Photonics 8, 56 (2021); Hayran,
    Chen & Monticone, Optica 8, 1040 (2021) (temporal sKK dual).

## 4. Suggested claim wording

**Safe**
- "To our knowledge, complex-frequency excitation has not previously been used to realise sKK media."
- "We show that single-pole (Long-type) realisations of the gain regions of an sKK medium are
  dynamically unreachable."
- "We identify passive two-pole susceptibilities with stable virtual gain, extending the virtual-PT
  principle of Li et al. (2020) from coupled lumped resonators to a constitutive design rule."
- "We use them to construct graded, all-angle, one-way-invisible sKK media from passive layers, and
  verify their time-domain settling."

**Do NOT claim**
- First passive one-way invisibility or ATR by complex-frequency excitation (Li 2020).
- First identification of the condition that the drive decays slower than every pole (Lalanne & Wu,
  Guan, Li 2020 Eq. 4).
- That sKK media in general require gain.
- That the time-domain behaviour of gain-containing sKK media has never been considered (Longhi 2015
  and 2016 did so spatially; Horsley–Artoni–La Rocca 2016 did it with pulses).

---

## RESUMABLE WORK STATE

1. **GOAL.** Prior-art search on sKK × complex-frequency / virtual gain for claims 1–4 (FINDINGS B1–B9).
   Rate each hit and give a novelty verdict and must-cite list. Do not modify code.
2. **ESTABLISHED.** Everything above. Every item is labelled "read", "grep", "abstract" or "snippet";
   the verdict is in §0.
3. **RULED OUT.**
   - "complex frequency" in Zhang et al. OE 2021 and Ye 2017 means analyticity of the frequency
     response, not excitation.
   - Chen et al. 2024 (evanescent ATR) is real-frequency.
   - Atomic "reflection amplification" papers use real pumped gain.
4. **IN PROGRESS.** Nothing; the search is complete.
5. **NEXT STEP** (optional, for whoever resumes). Obtain through the Rice library:
   - Kim–Krasnok–Alù Science 2025;
   - Horsley–Artoni–La Rocca PRA 94, 063810 (2016);
   - Horsley–Longhi AJP 85, 439 (2017).

   Grep each for "Kramers", "graded", "reflectionless", "one-way", "invisib" and for dispersive models
   in the pulse simulations. These are the only plausible residual scoops.
6. **OPEN QUESTIONS.**
   - Does Li 2020's supplementary material contain a distributed or transmission-line version that
     approaches a "medium"? Main text only was read; the SI was not.
   - Whether Longhi's interface-mode instability (spatial, real gain) has a counterpart at ω_c in your
     virtual-gain structures is a physics question, not a literature one. Nobody has asked it.
