# Prior art: complex-frequency excitation, stability of virtual gain, and stable virtual gain from passive multi-pole media

Librarian, 2026-10-13. STATUS: COMPLETE (the raw log and work state are in the appendices).
This builds on `research/prior_art_virtual_gain_stability.md` and `research/prior_art_2026Oct12.md` (both 2026-10-12).

**Method.**
- About 35 full texts read: pdftotext of arXiv/PMC/open-access PDFs, plus Europe PMC XML.
- Long et al. main text and SI read in full from `Papers/202601_*`.
- Forward citations harvested from Semantic Scholar: Kim–Krasnok–Alù Science review (66), Guan Science 2023 (111), Kim PRX 2023 (47), Long 2026 (3), Lalanne–Wu (1).
- arXiv API sweeps of abstracts: "virtual gain"; "complex frequency" × {excitation, transparency/Fano/coupled resonators/multipole, stability/transient/quasi-steady, Kramers/reflectionless/one-way}; "spatial Kramers".
- Targeted web searches in English and Chinese.

Conventions: e^{−iωt}, ω_c = ω′ − iΓ, Γ = |ω″| > 0. "Read" means I read the passage quoted. Everything else is marked.

---

## 1. VERDICTS

### (i) "Single-Lorentz (Long-type) virtual gain is time-domain unstable." The condition is known; nobody has applied it to Long.

**The condition is already published for a single Lorentzian, and recently.**
- **Guan, Lin, Chen, Wen, (T. Li), S. Zhang, "High-order virtual gain for optical loss compensation in plasmonic metamaterials", Nat. Phys. 22, 489–494 (2026); arXiv:2505.21976.** Theory section, after splitting the truncated-CFW response into target + D_TIS (pole term ∝ e^{−γt/2}) + D_FFL:
  > "To reach the quasi-steady state, it is required that β < γ/2 such that D_TIS attenuates faster than the target signal."
  - For one Lorentzian, β < γ/2 ⇔ Im ε(ω_c) > 0. The sentence therefore already implies that net virtual gain from a single Lorentzian has no reachable quasi-steady state.
  - They never write that corollary. They discuss only "loss compensation" and push β up to 0.499γ.
- **Guan et al., Science 381, 766 (2023); arXiv:2303.16081** (Drude case):
  > "if the virtual gain of CFW is greater than the actual material loss τ > γ, oscillatory behavior will arise, leading to an unstable state for the whole system."
- **Novitsky, PRA 107, 013516 (2023); arXiv:2301.05404, Sec. III, Eq. (8).** In a two-level medium the effective inversion is w_eff = (1 − T₂/τ) w_st, so the sign flips iff τ < T₂.
  - This is Long's SI Eq. S18 (γ < 2Γ) in Bloch language, three years earlier.
  - His Maxwell–Bloch simulations show the resulting virtual gain only as a transient lasting about 0.1 ps.
- **General multi-mode (QNM) statements:**
  - Lalanne & Wu, arXiv:2508.10742, Suppl. §5.1: the drive must decay slower than "the QNM with the longest lifetime".
  - Li et al., PRL 124, 193901 (2020), Eq. (4): the transient evolves with the eigenvalues of H_eff.
  - Xue et al., Nat. Commun. 17, 3359 (2026): single-WGM threshold divergence.
  - Kozlov…Alù, arXiv:2610.09182 (Oct 2026): "provided that no slower decaying poles exist in the proximity of the … zero".
  - Khurgin, Alù, Kozlov, Lalanne, arXiv:2607.02211.
  - Li & Pal, arXiv:2605.12842 (mechanical).
  - Kim et al. (PRX 2023 and PRL 2022) concede there is "no guarantee" of convergence.

**What I could not find anywhere:**
- anyone linking this condition to Long et al.'s Theorem 1, to Im ε(ω_c) < 0 media, or to the antimatter or perfect-lens designs;
- any critique of Long et al. (3 forward citations, none critical).

Long's main text and SI contain no time-domain, transient, stability or multi-pole content. SI §VIII ("finite bandwidth") varies only Re ω, with Im ω held fixed. The main text says "we employ the single-pole Lorentz–Drude model" and calls it "the most general model".

**What the crew can still claim as new:**
- the explicit corollary: every Im ε(ω_c) < 0 member of Long's single-pole family has its material pole, and an accumulation of slab poles, between the real axis and ω_c, so no causal turn-on reaches it;
- the time-domain demonstration (exact synthesis plus FDTD), including Long's own Table-1 perfect lens;
- the bulk effective-medium framing.

The paper must cite Guan Nat. Phys. 2026 and Guan Science 2023 for the single-Lorentz condition, and Novitsky 2023 for virtual inversion.

**WORDING HAZARD.** "Single-resonance virtual gain is never observable" is false for resonators with radiative damping.
- **Zeng & Zhang, Acta Optica Sinica 44, 1026019 (2024)** (Chinese, open access) deliberately drive the bare polarizability into a "gain state" (τ > γ). They keep it non-divergent through radiative loss: "τ < γ + Gω₀".
- **Xue 2026** has virtual gain exceeding the intrinsic loss while staying below the total (intrinsic + coupling) loss, and the response saturates stably.
- So claim (i) must be restricted to the homogeneous medium's own Im χ, where the relevant pole is the material pole. Alternatively, phrase the criterion in terms of the structural poles throughout.

**"Synthetic CF reconstruction is invalid with poles in the strip."** This is known in substance:
- Guan Nat. Phys. 2026 (as above);
- Liu, Gong, Li, H. Chen, Rotter, Lin, arXiv:2609.29293, Eq. (8): residues of the inherent poles must be negligible;
- Lalanne–Wu.

The synthesized response is mathematically identical to the response to the truncated CFW, so this is the same statement as (i), not a separate result.

### (ii) "Stable virtual gain from passive multi-pole (EIT / Lorentzian-hole) media." The mechanism is partly anticipated at resonator level; the medium-level result was not found.

**Essential precursor: Li, Mekawy, Krasnok, Alù, "Virtual parity-time symmetry", PRL 124, 193901 (2020); arXiv:1911.12875.**
- Two coupled RLC resonators are driven at a decaying complex frequency.
- For the virtual ATR they optimise the slowest transient eigenvalue (Eq. 8). The optimum has **γ_R = 0 (a lossless resonator)**, a detuning δ = −c/2, and a lossy partner, giving γ_trans = 4γ₀, "reasonably larger than the decay rate γ₀ of the excitation".
- Time-domain circuit simulations started from rest settle to a quasi-stationary state in which the outgoing power is twice the incident power ("mimicking gain").
- This is the crew's EIT mechanism (lossless dark mode + lossy bright mode, coupled, every eigenmode decaying faster than the drive), at the level of a two-port resonator.
- Must be cited prominently. The "couple a lossless mode to a lossy one" idea cannot be claimed as new.

**Two-pole bright/dark systems have already been driven at complex frequency, but only to compensate loss:**
- PIT metamaterial: Guan Nat. Phys. 2026 and Guan et al. arXiv:2604.26628. The latter treats only the marginal case β = γ and finds a residual damping (N+2)/t.
- Fano antenna/waveguide/molecule: Severs Millard…Oulton, arXiv:2608.30146. Virtual gain is limited to about half the dark-mode loss; "CFWs do not allow a weakly coupled system to become strongly coupled"; the EP position is invariant.
- Coupled Ag bars: Zeng & Zhang 2024.

None of these seeks Im χ(ω_c) < 0. None locates the hybrid poles relative to ω_c. None exploits the unequal re-weighting of poles.

**The two-line medium is textbook.** The "broad Lorentzian minus narrow Lorentzian hole" is the standard weak-pump EIT lineshape: Anisimov, Dowling & Sanders, PRL 107, 163604 (2011), A_EIT = C₊²/(γ₊²+δ²) − C₋²/(γ₋²+δ²). Cite it as background. The novelty lies in how this medium behaves under complex-frequency drive, not in the medium itself.

**What I could not find anywhere:**
- (a) the statement that a sum of positive-strength Lorentz terms whose poles all decay faster than the drive always has Im χ(ω_c) > 0, so that stable virtual gain needs a negative-residue term or coupling to a lossless mode;
- (b) any passive effective medium with Im χ(ω_c) < 0 in which every material pole decays faster than the drive;
- (c) a joint passivity window (f₂/f₁ ≤ γ₂/γ₁) plus stability window (γ₂ > 2Γ) for virtual gain;
- (d) time-domain settling of bulk virtual gain in a slab, from synthesis or FDTD.

On this evidence claim (ii) appears **novel at the medium level**, with Li 2020 as the essential precursor. Expect a referee to call it "virtual PT symmetry inside a unit cell" and pre-empt that.

**Auxiliary claims:**
- **"Growing drives cannot give virtual gain."** This is a classical corollary of Landau & Lifshitz, *Electrodynamics of Continuous Media*, §123: ε is real in the upper half-plane only on the imaginary axis, which together with Im ε > 0 on the positive real axis gives Im ε > 0 throughout the first quadrant. I am citing from memory, so check the page before citing. Long SI §VI says it for one Lorentzian.
- **"Lossless media cannot give stable virtual gain."** Immediate, because all poles lie on the real axis. State it as a remark, not a result.
- **The Parseval/energy point** (FINDINGS item 5.1) is already in Xue 2026, Suppl. Note 6 (E_out/E_in ≤ 1). Khurgin 2026 ("broadening compensation", integrated absorption unchanged) says the same thing.
- **(c) General Herglotz / positive-real conditions.** Nothing in the complex-frequency literature. The natural question is whether the passive class is closed under the weighting h(t) → h(t)e^{Γt} when the weighted kernel is still stable: yes for a single Lorentz term, no for multi-pole media (the crew's result).
  - This is **not** the control-theory notion of strict positive realness, "Y(s−ε) is PR". I checked: the two maps differ (s·χ(s−Γ) versus (s−Γ)·χ(s−Γ)). Do not cite SPR as equivalent.
  - The question looks open.

### (iii) sKK combined with complex frequency: nothing found

This rests on arXiv API sweeps and web searches on 2026-10-12 and 2026-10-13.

Real-frequency relatives:
- Ye et al., Nat. Commun. 8, 51 (2017): space-chirped Lorentz resonance.
- Zhang, Wu, Artoni & La Rocca, Opt. Express 29, 5890 (2021), arXiv:2010.10758: the same mechanism in cold atoms.
- Wang, Zhang, Du & Wu, PRA 108, 053716 (2023): atomic sKK in the EIT regime.

The closest complex-frequency analogue of one-way virtual gain is Gu et al., Nat. Commun. 13, 7668 (2022), the "transient non-Hermitian skin effect": direction-dependent virtual gain in a passive acoustic lattice, observed in a time window, with no stability analysis.

---

## 2. TABLE OF HITS

| # | Citation | What it actually does | Rating | Read |
|---|---|---|---|---|
| 1 | Guan, Lin, Chen, Wen, T. Li, S. Zhang, *Nat. Phys.* 22, 489 (2026), doi:10.1038/s41567-026-03171-0; arXiv:2505.21976 | Single Lorentz: quasi-steady state needs β < γ/2; high-order t^{n−1}e^{−iω̃t} drives; PIT experiment, loss compensation only | **Partial overlap (i)**; related (ii) | arXiv full text; published version paywalled |
| 2 | Guan et al., *Science* 381, 766 (2023), doi:10.1126/science.adi1267; arXiv:2303.16081 | One sentence: τ > γ → "unstable state" (Drude) | **Partial overlap (i)** | arXiv passage |
| 3 | Novitsky, *PRA* 107, 013516 (2023), arXiv:2301.05404 | Two-level medium: virtual inversion iff τ < T₂ (= Long S18); virtual gain seen only as a transient | **Partial overlap (i)**, precursor to Long S18 | Sec. III full |
| 4 | Li, Mekawy, Krasnok, Alù, *PRL* 124, 193901 (2020), arXiv:1911.12875 | Virtual PT: lossless + lossy coupled pair, transient 4× faster than drive, stable over-unity output; state preparation for the virtual laser | **Partial overlap (ii), mechanism** | Full text |
| 5 | Lalanne & Wu, arXiv:2508.10742 (2025) | General QNM condition; Guan/Kim used Im ε = 0 (marginal) | Related, must cite | Main text + Suppl. §5 |
| 6 | Xue…Alù, Wan, *Nat. Commun.* 17, 3359 (2026), doi:10.1038/s41467-026-70123-w | Single WGM: stable below threshold, divergent above; energy bound | Related, must cite | Europe PMC full text (grep) |
| 7 | Zeng & S. Zhang, *Acta Optica Sinica* 44, 1026019 (2024), doi:10.3788/AOS240692 | Bare polarizability in a "gain state" (τ > γ) kept stable by radiative loss, τ < γ + Gω₀; coupled modes | Related, must cite; **wording hazard for (i)** | Full text (Chinese) |
| 8 | Severs Millard…Oulton, arXiv:2608.30146 (2026) | Fano bright/dark + molecule under synthetic CFW; ≤ ~½ dark-mode loss; EP invariant | Related, must cite (ii) | Main text + SI grep |
| 9 | Guan, N. Chen, Lin, Song, Zhu, T. Li, S. Zhang, arXiv:2604.26628 (2026) | Marginal β = γ: γ_eff ≈ (N+2)/t; Hann window; PIT | Related | Full text |
| 10 | Kozlov, Wang, Kim, Thielens, Alù, arXiv:2610.09182 (2026) | Zero-forward scattering; "no slower decaying poles" near the target zero | Related | Main text (SI Notes 4–5 not read) |
| 11 | Khurgin, Alù, Kozlov, Lalanne, arXiv:2607.02211 (2026) | Forced + pole decomposition; damping "compensated or even reversed"; "broadening compensation" | Related | Full text |
| 12 | Liu, Gong, Li, H. Chen, Rotter, Lin, arXiv:2609.29293 (2026) | Synthesis validity via residues (Eq. 8); cites Long | Related | Grep + key passages |
| 13 | Kim, Peng, Yves, Alù, *PRX* 13, 041024 (2023); Kim, Lepeshov, Krasnok, Alù, *PRL* 129, 203601 (2022) | "No guarantee" of convergence; checked in the time domain; claims "no instabilities" | Related | Passages |
| 14 | Anisimov, Dowling, Sanders, *PRL* 107, 163604 (2011) | EIT lineshape = broad minus narrow Lorentzian (= crew's two-line medium) | Background, must cite | Passage |
| 15 | Gu, Gao, Xue, Li, Su, Zhu, *Nat. Commun.* 13, 7668 (2022) | Transient NHSE: one-way virtual gain in a passive lattice, observed in a time window | Related ((iii)-adjacent) | Grep + passages |
| 16 | Zhang, Wu, Artoni, La Rocca, *Opt. Express* 29, 5890 (2021); Wang et al., *PRA* 108, 053716 (2023); Zheng et al., arXiv:2106.10975 | sKK in cold atoms at real ω (chirped light shift; EIT regime) | Related (iii) | Abstracts + grep |
| 17 | Zouros, Loulas, Almpanis, Krasnok, Tsakmakidis, *Commun. Phys.* 7, 283 (2024), arXiv:2308.09513 | "Stable under complex-frequency excitation" means only that the poles are in the lower half-plane; no pole-versus-drive check | Tangential | arXiv HTML (via fetch summary) |
| 18 | Long, Catrysse, Han, Fan, *LSA* 15, 48 (2026), doi:10.1038/s41377-025-02137-w | Target paper. No time domain, transients, stability or multi-pole content | — | Full main text + SI |

**Tangential (checked, no bearing):**
- Archambault, Besbes, Greffet, *PRL* 109, 097405 (2012)
- Rogov & Narimanov, *ACS Photonics* 5, 2868 (2018)
- S. Zhang & Hang, *Nanophotonics* (2026), doi:10.1002/nap2.70125: polynomial-fit continuation, no strip discussion
- Zeng et al., *eLight* (2024), arXiv:2307.09117
- Novitsky & Shalin, *PRA* 108, 053513 (2023)
- R. Ali, arXiv:2107.11519
- Trivedi et al., arXiv:2408.09617 and arXiv:2506.02207
- Tian…Gu, arXiv:2512.17156
- Farhi et al., arXiv:2506.03485
- Maddi et al., *JAP* 137, 234701 (2025)
- Hinney et al., *Nat. Commun.* 15, 2741 (2024): growing drives (OSTI copy, grep only)
- Krasnok & Seletskiy, arXiv:2603.12519: adiabatic tracking via σ_min(A + iω̃I)
- Li & Pal, arXiv:2605.12842 and arXiv:2507.10940
- Mao et al., *Adv. Sensor Res.* (2025)
- Xu…Alù, Basov, arXiv:2506.18224

---

## 3. WHAT THE PAPER MUST CITE AND MUST NOT CLAIM

**Must cite:**
- For (i): Guan Nat. Phys. 2026; Guan Science 2023; Novitsky PRA 2023; Lalanne & Wu 2025; Xue 2026; Kozlov 2026; Khurgin 2026; Zeng & Zhang 2024 (radiatively stabilised gain state).
- For (ii): Li et al. PRL 2020 (virtual PT; mechanism); Millard 2026 and Guan 2026 (two-pole PIT/Fano media under CF, loss-compensation regime); Anisimov 2011 (EIT lineshape).
- Classical: Landau & Lifshitz §123 (growing-drive no-go); Laplace region of convergence.

**Must not claim:**
- that anyone "overlooked" instability in general (Guan, Novitsky and Xue all discuss it);
- that the strip-pole condition is new;
- that coupling a lossless mode to a lossy one to get stable virtual gain is new (Li 2020);
- that single-resonance virtual gain beyond the *intrinsic* loss is never stable (it is stable in radiatively damped resonators: Zeng & Zhang, Xue);
- that the two-line or EIT media are new;
- that the energy budget is unchanged by the envelope as a new insight (Xue Suppl. 6; Khurgin).

**Can claim, with the precise scope stated:**
- Long's Im ε < 0 family is dynamically unreachable, with the explicit pole-location corollary and time-domain proof.
- Passive bulk media (EIT / Lorentzian-hole) give Im χ(ω_c) < 0 with every material and structural pole decaying faster than the drive.
- The exact passivity + stability design window.
- The slab and sKK time-domain demonstrations.
- The re-weighting mechanism (the 1/(γ − 2Γ) enhancement of the narrow negative-residue term).
- Any sKK application.

---

## 4. NOT ACCESSED / UNVERIFIED

These are ranked by risk.

1. **Kim, Krasnok, Alù, "Complex-frequency excitations in photonics and wave physics", Science 387, eado4128 (2025).** Closed access: Semantic Scholar and OpenAlex list no open-access copy, and ResearchGate returns 403. I read only the abstract. This is the most likely single place for a general strip-pole statement, or a remark on multi-resonance media. **Check it through the Rice library before submission.**
2. Published (paywalled) versions of Guan Nat. Phys. 2026 and Guan Science 2023. I read the preprints only; wording may differ.
3. Supplementary material of Li 2020, Kozlov 2026 (SI Notes 4–5) and Xue 2026.
4. Not opened:
   - Ghoshroy, Özdemir, Güney, Opt. Mater. Express 10, 1862 (2020), "virtual gain" in the plasmon-injection sense;
   - Loulas, Psychogiou, Tsakmakidis, Stefanou, Opt. Express 33, 28333 (2025);
   - Long et al., FiO 2025 and CLEO 2026 abstracts;
   - Trainiti et al., Sci. Adv. 2019;
   - Ra'di et al., ACS Photonics 2020;
   - Mazières et al. 2026.
   All are virtual absorption, growing-drive, or abstract-level items.
5. Cited from memory and needing verification: the page in Landau & Lifshitz §123; Alzar, Martinez, Nussenzveig, Am. J. Phys. 70, 37 (2002) (classical-oscillator EIT); the Commun. Phys. volume number (7, 283) for Zouros et al., taken from a reference list.
6. Chinese-language literature: one search (complex frequency + EIT + virtual gain) found only Zeng & Zhang 2024. Coverage is thin.

---

# APPENDIX A — RAW FINDINGS LOG AND WORK STATE (chronological; kept for resumption)

## WORK STATE (for resumption)

1. GOAL. Prior-art search for Kunal's planned paper. Claims to check: (i) single-Lorentz
   (Long et al. LSA 2026 Thm 1) virtual gain needs gamma < 2|w''| so the material pole sits above
   w_c and the time-domain response never settles; synthetic CF reconstruction invalid with poles in
   the strip; (ii) STABLE virtual gain (Im chi(w_c) < 0 with every pole decaying faster than the
   drive) from PASSIVE multi-pole media (EIT bright+dark; broad Lorentzian minus narrow hole,
   f2/f1 <= g2/g1); lossless media can't give stable virtual gain; growing drives can't give
   virtual gain; (iii) combination with sKK. Output: this file + final message.
2. ESTABLISHED SO FAR. Prior sweep exists: `research/prior_art_virtual_gain_stability.md`
   (2026-10-12) covers claim (i) => condition is KNOWN (Guan Science 2023 sentence; Lalanne & Wu
   arXiv:2508.10742; Khurgin et al. arXiv:2607.02211; Xue et al. Nat Commun 2026; Li PRL 2020).
   It did NOT search claim (ii) (multi-pole). `research/prior_art_2026Oct12.md`: no sKK+CF hit.
3. RULED OUT. Long et al. main+SI read in full (pdftotext): NO time-domain, stability, transient,
   or multi-pole content. Main text says "Here, we employ the single-pole Lorentz-Drude model" and
   claims the Lorentz-Drude model is "the most general model" (cites Luo et al. Opt Express 29,
   14345 (2021) for using real-frequency poles at small Im w). SI Sec VI = Eq S13-S18 only
   (gamma < -2w''). SI Sec VIII finite bandwidth: only Re w varied, Im w fixed -> no time domain.
   Discussion: "future directions ... exceptional points and perfect absorbers" (ref 59 Sweeney).
4. IN PROGRESS: NONE. Report written 2026-10-13 (Sections 1-4 above). Historical note (as of ~F26). Core sweep essentially done: Long main+SI; citation harvests of
   Kim review / Guan Science / Kim PRX / Long / Lalanne-Wu; arXiv API sweeps; ~30 full-text reads
   logged as F1-F26 below. Remaining: a few Semantic Scholar keyword searches (bg job), Ghoshroy
   OME 2020 abstract check, then WRITE THE CONSOLIDATED REPORT at top of this file (verdicts
   (i)/(ii)/(iii), table of hits with ratings, what-to-cite, what-not-to-claim).
5. NEXT STEP (for a successor). (a) Read Kim-Krasnok-Alu Science 2025 review via the Rice library
   and check it for strip-pole and multi-resonance statements; update Section 1 if needed. (b) Check
   the published Guan Nat. Phys. 2026 wording. (c) Verify the Landau-Lifshitz Sec. 123 page.
6. OPEN QUESTIONS. Paywalled: Kim-Krasnok-Alu Science 2025 review; Hinney 2024; Trainiti 2019.

## RAW FINDINGS LOG (appended as found; consolidated into report at end)

F1. **Guan, Lin, Chen, Wen, Zhang (S. Zhang, HKU), "High-order virtual gain for optical loss
compensation in plasmonic metamaterials", arXiv:2505.21976 (v1 28 May 2025); published Nat. Phys.
(2026), nature.com/articles/s41567-026-03171-0.** READ full arXiv text (pdftotext).
- Theory section, single Lorentzian eps_L = 1 - wp^2/(w^2 + i w g - w0^2 - g^2/4), truncated CFW
  e^{-i w~ t} theta(t), w~ = w - i beta. Residue decomposition D = eps(w~)e^{-iw~t} + D_TIS + D_FFL,
  D_TIS ~ e^{-g t/2} (material pole), D_FFL ~ 1/(w_Delta t) (finite synthesis band).
  QUOTE: "To reach the quasi-steady state, it is required that beta < gamma/2 such that D_TIS
  attenuates faster than the target signal". "When beta approaches gamma/2 ... the ratio is close
  to unity, making loss compensation almost impossible to reach."
  => EXACTLY the single-Lorentz settling condition. Since Im eps(w~) > 0 iff g > 2 beta for one
  Lorentzian, it implies (unstated) that single-pole NET virtual gain (Im eps<0) is unreachable.
  They never state the Im eps<0 corollary; they only talk about "loss compensation".
- HVG drive t^{n-1} e^{-i w~ t}: still needs beta < g/2 (they push beta to 0.499 g at n=4).
- Experiment: plasmon-induced transparency (PIT; bright+dark dolmen, Zhang PRL 2008 design) =
  a TWO-POLE (EIT-type) medium. Goal: recover PIT transparency peak (loss compensation). They do
  NOT do a pole analysis of the coupled system, do NOT claim Im chi<0, do NOT exploit
  re-weighting of poles. Uses beta = 200 cm^-1 for all n. Rating: PARTIAL OVERLAP / MUST CITE
  (single-pole condition stated; EIT medium used under CF but for a different purpose).
- Its refs to chase: Ghoshroy, Ozdemir, Guney, "Loss compensation in metamaterials and plasmonics
  with virtual gain [Invited]", Opt. Mater. Express 10, 1862 (2020); Tetikol & Aksun Plasmonics
  2020; Yang et al. "Enhancing imaging resolution of superlens through transient illumination"
  (2024 proc.); An, Liu, Zhu, Cheng "Complex-frequency calculation in acoustics with real-frequency
  solvers" PRB 111 (2025); Zeng et al. eLight 2023 molecular sensing; Farhi/Stone APA papers.

F2. **"Anisotropic virtual gain and large tuning of particles' scattering by complex-frequency
excitations", Commun. Phys. (2024), doi 10.1038/s42005-024-01772-w, arXiv:2308.09513.** Read via
arXiv HTML (WebFetch summary). Magnetized InSb (Drude + cyclotron) particle. Fig 4(b): "poles
exist only for m=+-2 and only in the upper-half complex plane (exp(iwt)) i.e. our lossy system is
stable under real- and complex-frequency excitations." This is only the passivity statement
(poles in the lower half plane in e^{-iwt}); it does NOT compare pole depth with drive decay.
No v-vs-gamma condition. Rating: TANGENTIAL (and arguably an example of the blind spot).

F3. **Liu, Gong, Li, Chen, Rotter, Lin, "Zero-forward Kerker scattering via synthesized
complex-frequency excitation", arXiv:2609.29293 (2026).** Cites Long 2026. Abstract only so far:
"creating an artificial pole at a complex frequency that prevails over inherent scattering poles".
TODO: read full text for pole-vs-drive discussion.

Semantic Scholar forward citations of Long 2026 (DOI query, 2026-10-13): only 3 — F3, Otsuka &
Yokomizo arXiv:2604.24284 (non-Bloch band theory of nonlinear eigenvalue problems), and Long's own
CLEO 2026 abstract. (Second query by arXiv ID rate-limited.)

F1-pub. Published: Nat. Phys. 22(3), 489-494 (2026) (per HKU hub / search snippet; nature.com
paywalled, redirect to login; published wording NOT checked).

F4. **Guan, Chen, Lin, Song, Zhu, Li (Tao), Zhang (Shuang), "Spectral window engineering for
synthetic wave compensation of plasmonic loss", arXiv:2604.26628 (29 Apr 2026).** READ full text.
Single-pole chi(t) = chi0 e^{-i Omega t - gamma t}; drive theta(t) t^N e^{-(i w0 + beta) t}.
They study ONLY the MARGINAL case beta = gamma (drive decay = pole decay): residual damping
gamma_eff(N,t) ~ (N+2)/t (Eq. 3) -> loss is offset only asymptotically, never reversed. Then
rectangular vs Hann synthesis window (1/t vs 1/t^3 kernels). Experiment: PIT metamaterial modelled
as "superposition of a smooth background and two Lorentzian resonances with distinct frequencies"
(beta = 160 cm^-1); goal is narrowing each resonance, not Im chi<0. No beta > gamma regime, no
pole-reweighting idea. Rating: RELATED-MUST-CITE (marginal-case transient law; same group as F1).
Useful refs from its bibliography to chase:
 - Zhao, Li, Chen, Kim, Li, Xu, Alu, "Virtual Gain for Coherent Perfect Absorption in Ultrathin
   Excitonic Slabs", ACS Photonics 12, 5399 (2025).
 - Trivedi, Madanayake, Krasnok, "Revealing invisible scattering poles with complex-frequency
   signals", J. Appl. Phys. 137, 243103 (2025).
 - Rogov & Narimanov, "Space-Time Metamaterials", ACS Photonics 5, 2868 (2018) (cited as a
   higher-order-waveform ref).
 - Maddi, Poignand, Achilleos, Pagneux, Penelet, "Direct experimental observation of total
   absorption and loss compensation using sound waves with complex frequencies", JAP 137, 234701 (2025).
 - Rasmussen, Rosa, Lewton, Ruzzene, "A Lossless Sink Based on Complex Frequency Excitations",
   Adv. Sci. 10, 2301811 (2023).
 - Huang, Hu, Yang, "Complex frequency detection in a subsystem", Commun. Phys. 9, 84 (2026);
   Huang, Ding, Hu, Yang, "Complex frequency fingerprint", PRB 113, 075128 (2026).
 - Wu et al., "Gain-and-Loss-Free Metamaterials Exhibiting Non-Hermitian Non-Bloch Effects", PRL
   136, 106603 (2026).
 - Chen et al., "Synthetic gain for electron-beam spectroscopy", Nat. Commun. 17 (2026).
 - Cheng & Li, "Complex-frequency waves: beat loss and win sensitivity", LSA 13, 40 (2024) (N&V).
 - Delage et al., "Plasma ignition via high-power virtual perfect absorption", ACS Photonics 10,
   3781 (2023).

F5. Rogov & Narimanov, "Space-Time Metamaterials", ACS Photonics 5, 2868 (2018), arXiv:1801.05107.
READ (grep + intro). Pulse-shape matching + time gating via KK; no complex-frequency poles,
no stability/virtual-gain content. TANGENTIAL.

F6. Kim, Krasnok, Alu, "Complex-frequency excitations in photonics and wave physics", Science 387,
eado4128 (2025). CLOSED access (Semantic Scholar/OpenAlex: no OA copy; ResearchGate 403).
NOT READ. Abstract only. Must be flagged as unverified.

Forward-citation harvest (Semantic Scholar, 2026-10-13) of: Kim review (66), Guan Science 2023
(111), Kim PRX 2023 (47), Lalanne-Wu (1), Long (3). Saved in scratchpad cites/*.tsv (ephemeral).
Shortlist to read (priority order):
 a. Zeng & Zhang, "Complex-frequency analysis of coupled plasmonic systems" (Chinese), Acta Optica
    Sinica (2024)  <- coupled = two-pole, HIGH
 b. Millard et al., "Revealing antenna-molecule coupling using virtual gain from synthetic complex
    frequency analysis" (2026) <- two-pole (Fano), HIGH
 c. Zeng et al., eLight 4 (2024) "Synthesized CF excitation for ultrasensitive molecular sensing"
 d. Novitsky & Shalin, "Virtual perfect absorption in resonant media and their PT-symmetric
    generalizations", PRA 108, 053513 (2023), arXiv:2306.14478
 e. Zhao...Alu, "Virtual gain for CPA in ultrathin excitonic slabs", ACS Photonics 12, 5399 (2025)
 f. Trivedi, Niaz, Alu, Krasnok, "Selective excitation of coupled resonators via CF driving",
    ACS Photonics (2025)
 g. Zhang & Hang, "Loss compensation and CVA via virtual CF wave excitation", Nanophotonics (2026)
 h. Mazieres et al. 2026 "Clarifying energy storage of virtual critical coupling under realistic
    excitation constraints"; Krasnok & Seletskiy 2026 chirped analytic pulses; Tian et al.
    "Synthesis of Complex Frequencies" ACS Photonics 2026 arXiv:2512.17156; Huang-Hu-Yang Commun.
    Phys. 2025; Li & Pal arXiv:2605.12842; "Temporal Fourier optics reveals hidden hybridized
    light-matter states" (2026); Hooper-Capers-Hooper-Horsley PRA 2024.

F7. **Severs Millard, Xiao, Romero-Gomez, Aktas-Vincent, Huang, Phillips, Maier, Kazakova, Oulton,
"Revealing antenna-molecule coupling using virtual gain from synthetic complex frequency analysis",
arXiv:2608.30146 (31 Aug 2026).** READ (pdftotext, main + SI grep). Two coupled oscillators
(bright dimer + dark two-wire waveguide mode, Fano/PIT) + molecule (3-resonator CMT) under synthetic
CFW. Eq. (4) area: complex drive reduces BOTH losses equally, g_{1,2} -> g_{1,2} - tau (their
convention), coupling invariant to first order => EP position invariant; "CFWs do not allow a
weakly coupled system to become strongly coupled through virtual gain or vice-versa". Practical
limit tau/w0 = 6% ~ "approximately half of the loss of the dark waveguide resonance", attributed
to temporal truncation / unphysical sidebands; SI Sec SIV: "Synthetically illuminating with near
zero amplitude causes the process to diverge". They do NOT analyse hybrid-pole positions vs drive,
do NOT seek Im chi<0, do NOT note that coupling to a lossy bright mode stabilises a dark mode
driven past its own loss. Rating: RELATED-MUST-CITE (same two-pole bright/dark system class under
CF, but loss-compensation regime only).

INSIGHT (my reasoning, to verify against Li et al. PRL 2020): the crew's EIT mechanism in the frame
co-decaying with the drive = bright mode loss (g1/2 - G) coupled to dark mode gain (-G) -> a
"virtual PT" dimer. Strong coupling (unbroken phase) gives both hybrids the average rate
(g1/2 - 2G)/2 -> stable iff g1 > 4G. That is EXACTLY the Li-Mekawy-Krasnok-Alu virtual-PT
construction, applied to a bulk susceptibility. => Li 2020 must be read closely; likely
PARTIAL OVERLAP on mechanism.

F8. **Li, Mekawy, Krasnok, Alu, "Virtual Parity-Time Symmetry", PRL 124, 193901 (2020),
arXiv:1911.12875.** RE-READ in full (pdftotext, arXiv v1). UPGRADED from "adjacent" (Oct-12 sweep)
to PARTIAL OVERLAP with claim (ii) mechanism:
- Eq. (4): output = quasi-stationary S|s+> + transient term evolving with eigenvalues of
  H_eff = H0 + jGamma; "the eigenvalues of H_eff represent the decay rates of the transient".
- Virtual ATR design (p.7, Eq. 8): two coupled RLC resonators, drive decay sigma = gamma0; they
  OPTIMISE the slowest transient decay rate and pick delta = -c/2 (detuning), gamma_R = 0
  (LOSSLESS second resonator), eps = gamma_L + c (lossy first resonator + port coupling); get
  gamma_trans = 4 gamma0, "reasonably larger than the decay rate gamma0 of the excitation".
  Time-domain circuit simulation from rest (Fig. 3) reaches the quasi-stationary ATR; from the
  right "total time-averaged power flowing out ... is twice the incident one", "mimicking gain".
  => a passive lossy+lossless coupled pair giving STABLE, SETTLING virtual gain, with the
  pole-faster-than-drive condition explicitly enforced. Same mechanism as crew's EIT medium.
- Virtual laser/CPA (p.9): virtual CPA "requires inputs with the same decay rate as the transient
  of the virtual lasing mode, hence it may be difficult to observe"; they bypass with a
  prepared initial state (monochromatic pre-drive) to kill the transient. "beyond the lasing
  threshold without incurring into instabilities" relies on that preparation.
- TL example (p.3): uniform loss gamma0, drive decay sigma > gamma0 => power grows along z
  ("mimicking gain at any instant"). NOTE (my analysis): distortionless TL has an entire transfer
  function (pure delay x attenuation), so no strip-pole problem; this is virtual gain in Im k,
  not Im eps < 0 of a Lorentz medium. Does not contradict the crew's claim but the paper must
  define "virtual gain" as Im chi(w_c) < 0 of a Lorentz-sum medium to avoid this counterexample.
- What Li 2020 does NOT do: bulk susceptibility / permittivity; Im chi(w_c) < 0; passivity
  conditions on a medium; Long-type no-go; multi-line "hole" medium; sKK.
Rating: PARTIAL OVERLAP / MUST CITE PROMINENTLY. The crew cannot claim the "lossless resonator
coupled to a lossy one gives stable virtual gain" idea as new; they can claim its translation to
an effective-medium Im chi < 0 with all material poles below w_c.

F9. **Zeng Kebo & Zhang Shuang, "Complex-frequency analysis of coupled plasmonic systems
(invited)" [耦合等离激元体系的复频率分析], Acta Optica Sinica 44(10), 1026019 (2024),
doi:10.3788/AOS240692.** READ (Chinese, open PDF researching.cn, pdftotext). Two crossed Ag bars,
eigen-polarisations u, v; CMT polarizability P = -G/(w - w0 + i g/2), t = 1/(1 - i w P).
Sec. 3, Eq. (8): drive w~ = w - i tau/2 -> P~ has effective loss g - tau. QUOTE (translated):
"when tau = g the loss of P~ is 0 ... continuing to increase tau makes P~ enter a GAIN state,
which further narrows the transmission peak ... to keep the physics non-divergent tau cannot
increase without bound; we must ensure |t~(w~0)| < 1, giving the constraint tau < g + G w0."
=> They deliberately put the bare-oscillator (material) pole INSIDE the strip (tau > g) and rely
on RADIATIVE damping (G w0) of the structural mode to keep it decaying faster than the drive. This
is a single-resonance analogue of the crew's "structural margin vs material margin" point, and a
partial counter-example to any over-broad wording of claim (i): "single-Lorentz virtual gain is
never observable" is true for the homogeneous bulk Im chi but NOT for a polarizability in a
radiatively damped resonator. Their stability criterion is heuristic (|t|<1), not a pole
analysis; no time-domain check in what I read. Then two coupled modes (hybrid widths = average).
Rating: RELATED-MUST-CITE (and a wording hazard for claim (i)).

F10. **Lalanne & Wu, "Theory of superlensing with complex frequency illuminations",
arXiv:2508.10742 (2025)** RE-CHECKED (pdftotext). Suppl. Sec. 5.1 / Fig. S5 (three QNM poles of a
gold-nanorod trimer): case (2) decaying drive: "If its damping rate Im(w) exceeds that of the
fundamental mode (the QNM with the longest lifetime), the steady-state regime becomes
unobservable ... However, exceptions may arise — for example, when the fundamental mode is not
excited due to symmetry constraints or selection rules". Main text: Guan 2023 / Kim PRX 2023 used
w -> w - i gamma/2 so Im eps = 0 (marginal); low-kx steady state "may not be observable due to the
dominance of the transient". No passive-medium design, no Im eps<0, no multi-pole material
engineering. Rating: RELATED-MUST-CITE (general multi-pole QNM condition = the stability
criterion the crew uses; the crew's contribution is the material DESIGN that satisfies it).

F11. **Khurgin, Alu, Kozlov, Lalanne, arXiv:2607.02211 (Jul 2026)** RE-CHECKED. Intro: Lorentzian
under CF -> denominator w0^2 - w^2 + G^2 - i w(g - 2G), "dephasing rate ... partially or completely
compensated or even reversed"; transient from finite turn-on needs "careful dispersion
engineering, time gating or post-processing ... when possible"; Eq. (7) forced term + pole
(polariton) term, "Appropriate time gating ... suppress the transient pole contributions". Argue
"broadening compensation" not "loss compensation" (area conserved). Multi-resonance example is
spectral deconvolution, not medium design. Rating: RELATED (cite for the forced+pole decomposition
and the mention of "reversed" dephasing; does not analyse the reversed regime's stability).

F12. **Novitsky, "Tunable virtual gain in resonantly absorbing media", PRA 107, 013516 (2023),
arXiv:2301.05404.** RE-READ Sec. III. Two-level (Maxwell-Bloch) medium, simplified Bloch eqs:
after CW steady state, drive decays as e^{-t/tau}; effective inversion w_eff = (1 - T2/tau) w_st
(Eq. 8): "w_eff will have the sign opposite to the steady-state value w_st, if the decay time is
short enough, tau < T2" => VIRTUAL INVERSION condition = Long's S18 (1/tau > 1/T2 <=> |w''| >
g/2) in two-level language, 3 years before Long. Simulations: virtual gain (output/input ratio
R > 1) appears only as a SHORT TRANSIENT splash after switch-off ("R rapidly (up to 0.1 ps) comes
to unity"); needs tau << tau_p (slab pole). Calls the effect free of "instability-prone gain
media". Does NOT say the strip-pole regime has no reachable steady state, but its data are
consistent with that. Rating: RELATED-MUST-CITE (prior statement of the single-resonance virtual
gain condition; transient-only observation).

F13. Novitsky & Shalin, PRA 108, 053513 (2023), arXiv:2306.14478 — virtual PERFECT ABSORPTION
(growing drive) in two-level absorbing/amplifying slabs + PT bilayer; "quasilasing" at high gain.
TANGENTIAL (growing drives; real gain).

F14. Zeng, Wu, Guo, Guan, ..., Zhang, "Synthesized complex-frequency excitation for ultrasensitive
molecular sensing", eLight 4 (2024), arXiv:2307.09117. Two-mode Drude-Lorentz molecular layer,
uses w~ = w - i gamma_v/2 => eps(w~) REAL (marginal compensation, both modes equal gamma). theta(t)
truncation "to avoid energy divergence"; sidebands. No Im eps<0, no stability analysis.
TANGENTIAL/RELATED.

F15. Zhang Sheng & Hang Zhi Hong, "Loss compensation and coherent virtual absorption via virtual
complex frequency wave excitation", Nanophotonics (2026), doi:10.1002/nap2.70125, PMC13642498.
READ via Europe PMC XML (grep). Method = fit real-frequency response (e.g. k(f)) by a polynomial
(n=2) and substitute complex f (analytic continuation by fitting); rational/Lorentzian fits
mentioned for strong dispersion (SI S1). NO discussion of poles in the strip, stability or
transients. TANGENTIAL (but an example of a reconstruction method that would silently continue
through strip poles). New lead from its refs: Loulas, Psychogiou, Tsakmakidis, Stefanou,
"Analytic theory of complex-frequency-aided virtual absorption", Opt. Express 33, 28333 (2025),
doi:10.1364/oe.563570 (growing drive / virtual absorption -> likely tangential).

F16. **Xue, Zhang, Zhu, Sun, Chen, Alu, Wan, "Lasing-like dynamics with virtual gain driven by
complex-frequency excitations", Nat. Commun. 17, 3359 (2026), doi:10.1038/s41467-026-70123-w,
PMC13066403.** RE-READ via Europe PMC XML (grep). Single WGM mode, add-drop. Virtual gain
"counteracts intrinsic material AND radiation losses"; below threshold (drive decay < total mode
decay) T(t) "after a transient saturates at a quasi-steady level that can far exceed unity ...
stably amplified"; at threshold T grows linearly; past threshold diverges ("the decay rate of the
excitation meets and then exceeds the energy release rate of the resonant mode ... preventing the
system from reaching equilibrium"). Energy: E_out/E_in = 4k1k2/(k(k - w_im)) <= 1 (Suppl. Note 6)
=> total-energy passivity, same content as crew's Parseval point (FINDINGS item 5.1). Single mode
only; no multi-pole, no bulk Im eps. NOTE: like F9, virtual gain may exceed the INTRINSIC
(material) loss and remain stable thanks to radiative (coupling) loss. Rating: RELATED-MUST-CITE.

F17. Archambault, Besbes, Greffet, "Superlens in the time domain", PRL 109, 097405 (2012),
arXiv:1207.1209 (grep). Drives at the surface-wave POLE w_sw (Im eps... eps+1=0 at complex w);
transient vs forced decomposition; resolution improves if pulse ~ surface-wave decay time.
Foundational, TANGENTIAL to stability/multi-pole.

SYSTEMATIC arXiv API sweep (2026-10-13; scripts in scratchpad, ephemeral): abs:"virtual gain" -> 18
hits (all relevant ones listed in this file); abs:"complex frequency"/"complex-frequency" AND
excitation; AND (transparency OR Fano OR "coupled resonators" OR "two resonances" OR multipole);
AND (stability OR transient OR quasi-steady OR quasi-stationary); "temporally decaying"; ->
~200 unique, filtered by hand. No abstract mentions EIT/two-pole MEDIA giving Im eps<0 at w_c.
Still to open from this sweep: 2610.09182 (Kozlov...Kim, "Shadow-free scattering with CF
excitations", Oct 2026); 2107.11519 (R. Ali, "Lighting of a monochromatic scatterer with virtual
gain"); 2609.29293 (Kerker, cites Long). Classical framework leads (for claim (c)):
Skaar, "Fresnel equations and the refractive index of active media", physics/0509112 (PRE 73,
026605 (2006)); Hagenvik, Malema, Skaar, "Fourier theory of linear gain media", arXiv:1407.0187;
Abdelrahman & Gralak "Modal analysis of wave propagation in dispersive media" arXiv:1711.05553;
Ismail & Gralak arXiv:1610.03639 (slab of dispersive medium -> pole accumulation).

F18. **Kozlov, Wang, Kim, Thielens, Alu, "Shadow-free scattering with complex frequency
excitations", arXiv:2610.09182 (6 Oct 2026).** READ (grep + key passages). Lossless high-index
sphere, microwave experiment, targets complex ZERO of forward scattering with decaying drive
("virtual gain"). QUOTES: quasi-steady state reached "provided that no slower decaying poles exist
in the proximity of the forward scattering zero (see Supplementary Notes 4, 5)"; Conclusion: "not
all zeros can be accessed by a realistic excitation, due to limitations on the quasi-steady state
condition. Successful demonstrations require the poles near the real frequency axis to be
sufficiently far away from the desired complex frequencies". => pole-vs-drive condition stated
again (Alu group, Oct 2026), resonator/scatterer context. No medium design, no Im eps<0.
Rating: RELATED-MUST-CITE (most recent statement; SI Notes 4-5 not read).

F19. **Liu, Gong, Li, Chen (Hongsheng), Rotter, Lin, "Zero-forward Kerker scattering via
synthesized complex-frequency excitation", arXiv:2609.29293 (24 Sep 2026).** READ (grep + key
passages). Cites Long 2026 (as virtual-gain ref). Synthesis = contour integral enclosing the
"artificial pole" at w_c and the inherent scattering poles; validity Eq. (8): synthesized ~ ideal
"if the contribution from the inherent scattering poles is negligible compared with the artificial
pole"; observe transient -> steady state. This is a residue-based synthesis validity criterion
(comparable to crew's "contour shift" check). No multi-pole medium design. Rating: RELATED.

F20. R. Ali, "Lighting of a monochromatic scatterer with virtual gain", arXiv:2107.11519 (2021).
Lossless Mie sphere; virtual loss to store then virtual gain at the scattering pole to release.
TANGENTIAL.

F21. **Anisimov, Dowling, Sanders, "Objectively discerning Autler-Townes splitting from
electromagnetically induced transparency", PRL 107, 163604 (2011), arXiv:1102.0546.** READ
(passage). Lambda-system, weak pump region Omega <= (Gab - Gbc)/2: poles purely damped, absorption
"comprises two Lorentzians centered at the origin, one broad and positive and the other narrow and
negative: A_EIT = C+^2/(g+^2 + d^2) - C-^2/(g-^2 + d^2)"; strong pump: ATS = sum of two shifted
Lorentzians. => The crew's "two-line" (broad minus narrow Lorentzian hole) medium IS the textbook
weak-pump EIT lineshape, and the crew's coupled bright/dark oscillator is the classical-oscillator
EIT model (Alzar, Martinez, Nussenzveig, Am. J. Phys. 70, 37 (2002) - NOT opened, cite from
memory, verify). Real-frequency only; nothing on complex frequency. Rating: BACKGROUND-MUST-CITE
(the media are standard; novelty must be placed in their behaviour under CF drive, not in them).

F22. Mao, Ma, Zhang, Cao, Adv. Sensor Res. (2025), doi:10.1002/adsr.202500153 (abstract only):
THz graphene ID-eSRR + synthetic CFW, "Q from 8.54 to 427.96". Loss-compensation regime, single
resonance. TANGENTIAL.

F23. Li & Pal, "Natural frequency estimation using complex-frequency excitations",
arXiv:2605.12842 (2026) (grep): SDOF oscillator, transient ~ e^{(Omega - wn zeta)t}, neglect only
when Omega << wn zeta => single-pole condition again (mechanical). RELATED (minor).
Tian...Gu, "Efficient waveform capture without absorption with synthesis of complex frequencies",
arXiv:2512.17156 (ACS Photonics 2026): growing drives, avoids divergent parts. TANGENTIAL.
Farhi, Hershkovitz, Alu, Suchowski arXiv:2506.03485: EP excitation in time domain; nothing on
strip poles in grep. TANGENTIAL. Maddi et al. arXiv:2503.11748 (JAP 2025): acoustic Helmholtz
resonator, virtual gain compensates viscothermal losses; TANGENTIAL.
No Long/Fan follow-up on time-domain found (arXiv author query 2026-10-13); Long et al. also have
FiO 2025 and CLEO 2026 abstracts with the same title (not opened).

CLASSICAL ANCHORS (my analysis; from memory, VERIFY before citing):
 - Growing-drive no-go for ANY passive medium: Landau & Lifshitz, ECM Sec. 123: in the upper half
   w-plane eps(w) is real only on the imaginary axis; with Im eps > 0 on the positive real axis this
   gives Im eps > 0 throughout the first quadrant => no virtual gain for growing drives (Re w > 0).
   The crew's "growing drives cannot give virtual gain" is therefore a classical corollary; cite LL.
 - "Stable but virtually active": in Laplace variable s, admittance-type passive response Y(s)
   positive-real (PR). Drive at Re s = -G. Virtual gain with all poles left of -G <=> Y(s - G)
   stable but NOT PR. Control theory's STRICTLY POSITIVE REAL notion is exactly "G(s - eps) is PR
   for some eps > 0" (Narendra & Taylor 1973; Taylor 1974 IEEE TCAS; Ioannou & Tao 1987 IEEE TAC;
   Khalil, Nonlinear Systems, SPR definition). First-order (single-pole) PR functions: shifted-
   stable <=> shifted-PR; higher order: not. Caveat: Im chi(w_c) < 0 is not identical to
   Re Y(s_c) < 0 (Y = s chi gives Re Y = w' chi'' - G chi'), so the map is approximate for G << w'.
   ADJACENT (mathematical home for claim (c)); not physics prior art.

F24. sKK + complex frequency: arXiv API re-check 2026-10-13 (abs:"complex frequency" AND
(Kramers OR reflectionless OR one-way OR unidirectional OR invisib); abs:"spatial Kramers") -> NO
paper combining sKK with complex-frequency excitation. Consistent with prior_art_2026Oct12.md (B open).
Closest real-frequency relatives (atomic sKK via resonance chirped in space = Ye-2017 mechanism):
 - Zhang, Wu, Artoni, La Rocca, "Spatial Kramers-Kronig relation and controlled unidirectional
   reflection in cold atoms", arXiv:2010.10758 (PRA 2021; journal ref NOT verified): far-detuned
   control field with intensity linear in x maps frequency-KK to space-KK; passive; real w.
 - Zheng, Zhang, Liu, Zhang, Wu, arXiv:2106.10975 (Rydberg DDI sKK); Wang, Zhang, Du, Wu,
   arXiv:2310.00590 (EIT-regime cold atoms under linear sKK modulation; CPA / chiral phase).
 Rating: RELATED for claim (iii) only (sKK built from resonant/EIT atomic media exists at real w;
 none at complex w; none needs gain).

F25. Gu, Gao, Xue, Li, Su, Zhu, "Transient non-Hermitian skin effect", Nat. Commun. 13, 7668
(2022), arXiv:2212.05787 (grep + passages). Passive acoustic ring-resonator lattice; CF drive gives
direction-dependent virtual gain/loss ("intensity will increase along the propagation direction
when the decay in excitation is faster than the actual decay of the mode in the lattice"); the
quasi-stationary NHSE profile "can remain stable for a finite time window before the signal decays
to the noise level". No pole-vs-drive analysis. RELATED (passive structure, one-way virtual gain,
observed in a time window; conceptually near the crew's one-way virtual-gain sKK but different
system and no stability criterion).

F26. Kim, Peng, Yves, Alu, PRX 13, 041024 (2023), arXiv:2303.17699: "no guarantee that the system
will converge within a finite transient to an output oscillating at the same complex frequency
[31]"; checks by full-wave time domain; also "this effect does not lead to instabilities or noise".
Kim, Lepeshov, Krasnok, Alu, PRL 129, 203601 (2022), arXiv:2210.01004: "we cannot generally expect
that the sphere after a transient reaches a steady-state response ... However, for excitation of
zeros and poles sufficiently close to the real axis ... quasi-steady state". Qualitative only.
RELATED (cite as the community's acknowledgement that convergence is not guaranteed).

CORRECTION to the SPR anchor above (checked 2026-10-13): the textbook SPR definition ("G(s - eps)
is PR for some eps > 0"; Taylor 1974; Ioannou & Tao 1987 give the frequency-domain N&S test:
Hurwitz + Re G(jw) > 0 + high-frequency condition) does NOT map exactly onto "no virtual gain".
With Laplace chi(s) and drive at Re s = -G, the co-decaying system has admittance
Y_G(s) = s * chi(s - G), whereas the SPR shift of Y(s) = s chi(s) is (s - G) chi(s - G). Example:
a single Lorentz Y(s) = f s/(s^2 + g s + w0^2) shifted as in SPR acquires a RHP zero at s = G and
is never PR, although the single Lorentz has NO virtual gain whenever g > 2G. So the right
statement is "is the passive class closed under the exponential weighting h(t) -> h(t) e^{Gt}
(when the weighted kernel is still stable)?": single Lorentz yes, multi-pole no (crew's result).
I found NO physics or math paper posing it that way. Do NOT cite SPR as equivalent; at most as
a loose analogy. LL Sec. 123 anchor (growing drives) stands.
