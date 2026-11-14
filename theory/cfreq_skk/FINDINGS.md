# Idea B feasibility: sKK at complex frequency (2026-10-12)

**Question.** Long, Catrysse, Han & Fan (LSA 15:48, 2026), Theorem 1: under a decaying drive at
ω_c = ω′ − i|ω″|, a passive single-pole Lorentz medium can hit any complex ε(ω_c), including
Im ε < 0. Can we use this to build, from passive materials, sKK profiles that need gain? The
target was King–Horsley–Philbin (PRL 2017) unit-transmission profiles, double or higher poles in
the lower half y-plane. These give R = 0 from the left and t = 1, but ∫Im χ = 0, so they need gain
somewhere.

**Verdict.** It works in the frequency domain and fails in the time domain. The failure is not a
design problem; it follows from causality.

## What was done (all in `theory/cfreq_skk/`)

| file | what |
|---|---|
| `cfreq_lib.py` | s-pol transfer matrix (E, E′/μ) valid at complex ω. Its layer matrix is even in q, so it needs no branch choices. Also: the closed-form Theorem-1 design and exact time-domain synthesis (cos and sin drives combined, so no Hilbert-envelope artefacts). A contour-shifted synthesis extends the usable dynamic range. |
| `sanity_check.py` | Checks: vacuum slab, the Airy formula, the design reproducing its target to 1e-16, passivity on the real axis, and the Lorentzian sKK one-way at real ω (\|t\| matches KHP Eq. 6). |
| `fdtd_check.py` | Independent 1-D FDTD with a Lorentz auxiliary-differential-equation (ADE) update and no Fourier transforms. Reproduces the time-domain verdict. |
| `fig_cfreq_feasibility.py` | Figures B1–B4 (`freq`, `trunc`, `time`, `grow`) → `figures/` |

Units: c = 1, λ_r = 1 (the period T_r = 1), ω_c = 2π(1 − iη). The target profile is a sum of
six random 4th-order poles with a cos² tail window over |y| ≤ 2.5λ_r. Its Im χ reaches −0.73.

## Findings

1. **Frequency domain works (B1).** The realised structure is passive at every real ω (min Im ε
   is 7e-22, i.e. ≥ 0). At ω_c it reproduces the target exactly. At η = 0.05: |r_L|² = 4e-8,
   |r_R|² = 31, |τ − 1| = 2e-3. The gain points are exactly where γ < 2|ω″| (B1d), as in Long's SI
   Eq. S18. The resonators sit within ±3 % of ω_r, with Q ≲ 20 at η = 0.05. At ω_r the same
   structure is an ordinary lossy, weakly asymmetric slab.
2. **Truncation errors are amplified at complex frequency (B2).** Under a decaying drive, an error
   at depth L reaches the front face weighted by about e^{4|k″|L} in power, with |k″| = 2πη/λ_r:
   the incident wave grows going in and the reflected wave grows coming back. With 1/y² tails
   and a sharp cut, η = 0.1 and L = 20λ give |r_L|² ~ 1e8. Higher-order poles (KHP still gives
   t = 1) and a smooth window bring this down to 1e-10–1e-8 at L = 4λ. This is manageable, and it
   is the trade-off a paper would have to state: η·L/λ ≲ 0.1–0.3.
3. **Time domain: virtual gain never settles (B3, FDTD-confirmed).**
   - **Homogeneous 1λ slab.** Targets with Im ε(ω_c) > 0 converge to τ(ω_c) to 1e-7–1e-13. Every
     target with Im ε(ω_c) < 0 diverges relative to the drive, whatever the turn-on (smooth C∞,
     up to 40 periods) and whatever member of Long's one-parameter family is used: zero periods
     within 5 %. FDTD gives 0.036 → 1.17 → 2.06 → 4.7 → 50 → 243 at t = 10 … 60, against Fourier
     synthesis at ~2 at t = 30 and ~250 at t = 60.
   - **Cause.** Im ε(ω_c) < 0 needs γ < 2|ω″|. Each Lorentz term with γ > 2|ω″| has Im > 0 at ω_c,
     so for a sum of Lorentz terms this "needs" is strict. That puts material poles, and an
     accumulation of slab poles (B3b), between the real axis and ω_c. Their free ringing decays
     slower than the drive.
   - **Equivalent picture.** In the frame co-decaying with the drive (fields × e^{|ω″|t}) each
     oscillator's damping becomes γ − 2|ω″| < 0, i.e. it is unstable. The "steady state" is the
     particular solution of an unstable system, and no causal turn-on reaches it.
   - **sKK realisation (B3c).** |r_L|/|E_inc| never approaches the predicted 2e-4. It sits at
     O(1), the same as right incidence. A passive-at-ω_c sKK control (Horsley Lorentzian,
     γ = 1.5·2|ω″|) converges exactly to its predicted |r_L| = 3.2e-3 and |r_R| = 0.43.
   - **Long et al.'s own perfect lens (B3d).** Using their Table 1 media in 1-D at normal
     incidence, the frequency-domain result is t = 0.994, r = 0.004. In the time domain |r| ≥ 0.07
     at all times, and the response diverges at the predicted rate, set by a strip pole at
     Im ω = −0.289ω_r versus ω_c at −0.3185ω_r.
   - **Prior art.** The general condition is known: the drive must decay slower than every mode
     (Lalanne & Wu, arXiv:2508.10742; one sentence in Guan et al., Science 2023; Khurgin et al.
     2026; Xue et al. 2026). Applying it to Long's Im ε < 0 media seems to be unstated. See
     `research/prior_art_virtual_gain_stability.md`.
4. **The stable flip side: growing drive plus lossless media (B4).** With η < 0 (an exponentially
   growing drive) every pole lies below ω_c, so the steady state is always reached. Theorem 1
   then gives only Im ε(ω_c) > 0 targets, but it reaches them with γ → 0, i.e. lossless Lorentz
   media.
   - **Result.** A γ = 1e-4 structure built for a Horsley Lorentzian sKK absorber is stable and
     converges to |r_L| = 9.6e-4 against |r_R| = 0.089, with |τ| = 0.40: one-way reflectionless
     "virtual absorption" with no material loss.
   - **Truncation now helps.** Errors from depth are suppressed, the reverse of finding 2.
   - **Caveats.** At real ω the structure is a forest of narrow resonances. Even with γ → 0 it
     absorbs at real ω by resonance absorption, so "lossless" is subtle there. Its ω0² > 0
     condition limits growth rates to |η| ≲ 0.03 for this profile.

## What this means for idea B

- **The original pitch** (passive realisation of gain-requiring sKK, unit transmission) is not
  physically observable. A unit-transmission sKK needs ∫Im χ = 0, hence virtual gain somewhere,
  hence a pole in the strip, hence no steady state.
- **What survives:**
  - **(a)** An sKK version of coherent virtual absorption: lossless media, growing drive, one
    side, every angle, any sKK profile. This is stable and checked here. The time-reversed
    release and its direction have not been looked at.
  - **(b)** A short paper or comment showing that Long et al.'s Im ε < 0 media (antimatter,
    perfect lens) are dynamically unreachable. The general condition is known, so only the
    application is new; it is also adversarial toward a published paper.

## Not done / caveats

- 1-D only, s-pol / normal incidence (oblique angles not run in the time domain).
- Single-Lorentz-pole media only. A multi-pole "virtual gain" without strip poles is ruled out for
  Lorentz sums (each term with γ > 2|ω″| has Im > 0 at ω_c).
- State preparation (a pre-pulse that loads the exact steady state) would in principle work, but
  any error grows at rate |ω″| − Γ. Not explored.
- B4's release dynamics and direction, the chirped lossless medium (sKK at every growing ω), and
  the Kim–Krasnok–Alù Science review were not checked.

## B+A (Hayran 2021 + complex frequency): `research/hayran_BplusA_assessment.md`

Verdict: drop as its own paper; fold ~3 paragraphs into B. The decaying-envelope trick flips an
oscillator's DAMPING sign, never its oscillator-STRENGTH sign. Hayran's cloak needs strength-sign
gain, because it sits in the Landau–Zener regime. The passive section therefore absorbs again
(amplitude 3.8e-4 against a target of ~1). The complex-k dual penalises exactly the time-reflected
channel. Space-time sKK is already in Horsley & Longhi, PRA 96, 023841 (2017). Checks:
`research/theory/ba_checks.py`.

## Item 5: release after a growing drive, and angles (`exp_release_angle.py` → `data/release_angle.npz`, `fig_release_angle.py` → B5, B6)

**Setup.** The B4 structure: a Horsley Lorentzian target (A = −0.1, δ = 0.15λ, cos² window
|y| ≤ 2λ) built with Theorem 1 at η = −0.02. The pulse: a 10 T_r C∞ turn-on, 60 T_r of
e^{+2π·0.02 t} growth, then a 2 T_r C∞ turn-off. Exact Fourier synthesis over a 16384 T_r window.

1. **Energy is set by the real-frequency spectra (Parseval).** For any finite pulse into a linear,
   time-invariant structure, the total reflected, transmitted and absorbed energies equal the
   |D(ω)|²-weighted averages of the ordinary real-frequency |r|², |t|² and A. Every time-domain
   total matched this to 3 digits. An exponential envelope therefore cannot create or remove net
   absorption; it only moves energy around in time. The "envelope as a loss knob" framing is wrong
   in the energy sense.
2. **Near-lossless and under-resolved (Q = 10⁴, N = 400) is the B4 case.** Reflection during the
   drive is 7e-7 of the pulse energy, as predicted, so the one-way window is real. After the drive
   stops, 46 % of the energy comes back as delayed reflection, peaking about 730 T_r later. A
   further 10 % comes out as delayed transmission, and 31 % is eventually dissipated by the tiny γ.
   Integrated over the pulse, this is a mediocre mirror.
3. **The Theorem-1 design of a Lorentzian sKK is exactly a chirped-resonance medium.**
   - **Analytic.** ω₀²(y) = a² − b² − γb − a(2b + γ)·y/δ is linear in y, ω_p² = a(2b + γ)|A|/δ
     times the window, and γ is uniform. This was verified to 1e-14 (B5d). It is Ye et al. 2017's
     medium.
   - **At any frequency** with Im(ω² + iγω) > 0, the medium's ε(y, ω) is a Lorentzian in y with
     residue A, fixed for every ω, and pole depth δ_ω = δ·Im(ω² + iγω)/(a(2b + γ)).
   - **Consequences.**
     - At every real ω in the resonance band it is a one-way sKK absorber with the same |τ|.
     - The condition Im(ω² + iγω) > 0 at a decaying ω_c is exactly γ > 2|ω″|, the stability
       condition. The "handedness flip" for γ < 2|ω″| therefore lives entirely in the
       unobservable regime.
   - **At real ω the pole depth is δ·γ/(2b + γ)**, i.e. 3.7e-4λ at Q = 10⁴. The absorption is
     resonance absorption; its strength does not depend on γ, but N must resolve that depth.
   - **Numbers (B5c).** For Q = 10⁴: N = 400 gives R_L ≈ 0.47, N = 4000 gives 7e-3, and
     N = 40000 gives 1e-6. Q = 10³ is resolved at N = 4000, and Q = 100 at N = 400. Once
     resolved, the real-frequency band (±10 %) gives R_L ≈ 1e-6 and A ≈ 0.86, the same values as
     the ω_c prediction.
4. **Resolved budgets (B5b).** At Q = 10⁴/N = 4000, Q = 10³/N = 400 and Q = 100/N = 400, the
   pulse is 83–84 % absorbed, 16 % transmitted, and ≤ 0.8 % reflected in total. Absorption
   happens whether or not the drive grows, because the structure is a real-frequency absorber. So
   the growing drive's only effect is to widen the effective pole depth during the pulse,
   δγ/(2b + γ) → δ. That temporarily hides under-resolution, and the energy returns afterwards.
5. **Angles (B6).** At ω_c the realised structure reproduces the ideal target at every angle, s
   and p. Time-domain checks at 45° (s and p) and 70° (s) settle onto the prediction during the
   drive. |r_L|² ≤ 2e-6 up to 45° (p even lower, 5e-8 near 45°), then rises: 2e-4 at 60°, 3e-3 at
   70°, 5e-2 at 80°, 0.7 at 89°. The resolved real-frequency structure (Q = 100) shows the same
   curve, so the grazing rise is the profile's finite length (smaller k_y = k₀cosθ makes the
   window look abrupt), not the complex frequency. Release budgets at 45°/70° behave like 0°:
   delayed reflection 0.54/0.61 for the under-resolved Q = 10⁴ structure.
6. **Not fully understood (small).** For Q = 10⁴/N = 4000, the 0.7 % delayed reflection is
   emitted very late (t₅₀ ≈ 7400 T_r). The energy totals are Parseval-exact, so this does not
   affect the budget.

**Bottom line for item 5.** The release does undo the "virtual" reflectionlessness in the energy
budget whenever the structure is not already a real-frequency absorber. When it is (resonance
zone resolved), no special envelope is needed: graded low-loss resonators already give a broadband
one-way sKK absorber (Ye 2017's mechanism, recovered here exactly through Theorem 1).

## B7–B9: stable virtual gain from passive two-pole media → one-way invisible sKK (2026-10-12)

Code: `cfreq_lib.py` (new section: `eit_shape`, `eit_design`, `eps_eit`, `eit_poles`,
`fano2_design`, `eps_fano2`, `rt_model`), `exp_stable_gain.py` → `data/stable_gain.npz`,
`fdtd_twopole.py` → `data/fdtd_twopole.npz`, `fig_stable_gain.py` → B7, B8, B9.

**Idea.** B3 fails because a single Lorentz term has Im χ(ω_c) < 0 only if γ < 2|ω″|, i.e. its pole
rings longer than the drive decays. That is a property of single-pole media, not a law. In two-pole
media the decaying drive re-weights the two poles differently, so Im χ(ω_c) < 0 is possible with
every pole decaying faster than the drive.
- **EIT medium.** A lossy bright oscillator (γ₁ = 10|ω″|) carries the polarisation and is
  stiffness-coupled (κ = 6ω_r|ω″|) to a LOSSLESS dark oscillator (γ₂ = 0, detuned −0.2|ω″|).
  It is passive by construction. In the frame co-decaying with the drive the dark mode has gain
  and the bright mode loss; strong coupling shares the loss, so every hybrid mode stays damped.
  Margin = slowest material-pole decay / |ω″| ≥ 2.
- **Two-line medium.** χ = f₁/D₁ − f₂/D₂ at a common ω₀, with γ₁ = 10|ω″| > γ₂ = 4|ω″|.
  Exact real-axis passivity condition: f₂/f₁ ≤ γ₂/γ₁. Hole depth fixed at 0.9 of that limit.
- **Design per layer.** ω₀ sets the phase of χ(ω_c), the oscillator strength sets its size, and
  the smallest-strength root is kept. Every phase of χ(ω_c) is reachable, the design is exact to
  1e-14, and every layer is passive (min Im ε over real ω ≥ 0).
- **Strength needed.** η = 0.05: ω₀ ∈ [0.86, 1.14]ω_r, ω_p² ≤ 20|χ| (≤ 0.2ω_r² for the sKK
  target), bright Q ≈ 2. η = 0.01: ω₀ ∈ [0.97, 1.03]ω_r, ω_p² ≤ 4|χ| (≤ 0.04ω_r²), bright Q ≈ 10.
  Everything scales with |ω″|.
- **Why lossless media can't do it.** Lossless media can never give stable virtual gain, because
  every mode must decay faster than the drive. A growing drive cannot give virtual gain at all:
  it shifts the balance between the two lines the wrong way.

**B7 (before/after): homogeneous 1λ slab, ε(ω_c) = 1 − 0.3i, η = 0.05.** All three media give the
identical τ(ω_c) = 2.41. Ratio |τ(t)|/|τ(ω_c)| at t = 20/40/60/100 T_r:

| Medium | t = 20 | 40 | 60 | 100 |
|---|---|---|---|---|
| Single Lorentz (B3) | 1.17 | 4.7 | 243 | 4e4 |
| EIT | 1.053 | 1.001 | 0.9996 | 1 |
| Two-line | 0.89 | 1.04 | 1.000 | 1 |

FDTD (complex fields, coupled ADEs, λ/400) reproduces all three, including 242.8 at t = 60 for the
single Lorentz. In the complex-ω map the single-Lorentz slab has its material pole (−0.025ω_r) and
slab poles above ω_c. For EIT the slab poles sit at Im ω ≈ −0.07ω_r (1.4|ω″|, below ω_c) and the
material poles at ≤ −0.11ω_r. The structural margin (1.4), not the material margin (2.2), sets the
settling rate.

**B8: layered one-way-invisible sKK (unit-transmission KHP, needs gain at ω_c).**
- **Simple target.** χ = 0.5δ⁴/(y + iδ)⁴, δ = 0.2λ, cos² window, L = 4λ, N = 400 (λ/100).
  - Ideal at η = 0.05: |r_L|² = 1.6e-9, |r_R|² = 2.1, |τ − 1| = 5e-4.
  - A double pole was rejected: its truncated 1/y² tail leaves ∫χ ≈ 2A/Y, giving |τ − 1| = 2–6 %.
- **Random target.** The B1 profile (six 4th-order poles, |χ| ≤ 0.8, L = 5λ, N = 500).
- **Time-domain results** (synthesis with contour at 0.9|ω″|), read at t = 150 T_r (η = 0.05) or
  800 T_r (η = 0.01). Each value matches its prediction to 3 digits:

| Case (EIT and two-line agree) | |r_L| | |r_R| | |τ| |
|---|---|---|---|
| simple, η = 0.05 | 4.04e-5 | 1.45 | 1.0000 |
| simple, η = 0.01 | 1.18e-5 | 0.529 | 1.0000 |
| random, η = 0.05 | 2.09e-4 | 5.58 | 1.0005 |
| random, η = 0.01 | 4.47e-5 | 1.49 | 1.0004 |

- **Single-Lorentz versions diverge in every case.** At η = 0.01 they first plateau, because
  Long's κ-rule gives every non-gain layer γ = 2|ω″| exactly, a pole at the drive's own decay
  rate. Gain layers then take over.
- **FDTD, simple target at η = 0.05.**
  - EIT settles to |r_L| = 3.5e-5 and |τ| = 1.000 by t ≈ 40 T_r.
  - Two-line settles to 2.4e-5 by t ≈ 50 T_r.
  - The offset from 4.0e-5 is FDTD's own discretisation.
  - Single Lorentz reaches 16.5 at t = 60, against 17.2 from synthesis.
- **Checks on the synthesis.**
  - Contour shifts of 0, 0.5, 0.9 and 0.99|ω″| all agree wherever each is resolved, so no
    structural pole lies in the strip.
  - The late-time blow-ups seen with small shifts are the double-precision floor: they hit
    r_L, r_R and τ at the same moment and move with the shift.
- **Complex-ω map of r_R.** The single-Lorentz structure has a ring of poles around ω_c (−0.025 to
  −0.075ω_r). The EIT structure has none above −0.10ω_r.

**B9: angles (EIT realisation, simple target, η = 0.05).**
- The realised structure equals the ideal staircase at ω_c for every θ, s and p.
- |r_L|² ≤ 4e-6 out to 85° (s) and ≤ 3e-6 (p).
- |r_R|² is O(1) up to about 60° (s). For p it has a Brewster-like dip, 1e-5 at 45°.
- Time-domain checks at 45° s, 45° p and 70° s settle to the predictions (e.g. 70° s:
  |r_L| = 5.72e-4 and |r_R| = 0.398, both predicted and observed).

**Costs and caveats.**
1. **Passive, not lossless.** The dark resonators are lossless; the bright ones must be lossy
   (Q ≈ 1/(10η)). At real ω the structure is an ordinary lossy, passive slab (B8b, ε″ ≤ 0.21).
2. **Dynamic range.** Reaching the |r_L| floor takes about 12–13 e-folds of drive decay at both
   η (t ≈ 40 T_r at η = 0.05, ≈ 200 T_r at η = 0.01). The number of e-folds is set by the
   structural margin, not by η. One-way contrast |r_R/r_L| ~ 1e2–1e3 appears after about 6–8
   e-folds. Synthetic complex-frequency measurement (Kim–Alù; Guan et al.) is the practical
   route. It needs the response analytic between the real axis and ω_c, which these designs
   now satisfy and Long-type designs do not.
3. **Floor and settling.** |r_L| is limited by the frequency-domain truncation floor (window plus
   e^{4|k″|L}, B2), not by the time domain. Settling time scales as 1/η.
4. **Prior art.** Not checked, at Kunal's call (the sKK literature is thin).

## Conventions (checked 2026-10-14, at the advisor's request)

- **Time convention: e^{−iωt}, the same as Long et al.**
  - Long's main text says "Using the e^{−iωt} time convention", just before Eq. 2.
  - The PDF text extraction drops the minus glyph. Eq. 6 confirms it: 1 + ω_p²/(ω₀² − ω² − iωγ) is passive only under e^{−iωt}.
- **The drive decays.** With ω = ω′ + iω″, e^{−iωt} = e^{−iω′t}·e^{ω″t}, which decays iff ω″ = Im ω < 0.
  - Ours: ω_c = 2π(1 − iη), so η > 0 is a decaying drive and η < 0 a growing one.
  - Long's Theorem 1 assumes ω″ < 0. Their numerics state "temporally decaying wave… Im[ω_c] < 0", and SI §VI (after Eq. S18) says ω″ < 0 "corresponds to a temporally decaying wave".
- **Li 2020 and the Kim–Krasnok–Alù review use e^{+iωt}.** There a decaying wave has Im ω > 0. Flip signs when comparing.

## Long et al.'s printed Theorem-1 equations have a sign error (2026-10-13)

- **Where.** Eqs. (11), (13), (16), repeated in the SI, have the wrong sign on every (C − 1) term.
- **Effect.** Plugging the printed solution into Eq. (6) gives ε(ω_c) = 2 − C + iD instead of C + iD. The existence proof is unaffected.
- **Corrected forms**, with c ≡ C − 1, verified numerically:
  - M = [c, −1, ω″c + ω′D; D, 0, ω″D − ω′c]
  - **b** = [c(ω′² − ω″²) − 2ω′ω″D; D(ω′² − ω″²) + 2ω′ω″c]
  - x_p = [ω′² + ω″², 0, −2ω″]
  - x_n = [ω′c − ω″D, ω′(c² + D²), D]
  - x = [ω₀², ω_p², γ] = x_p + α x_n.
- **Relation to `lorentz_design`.** It is this family with α = κ = ½κ_max, shared by all layers, where κ_max = 2|ω″|/max(−Im χ) keeps every γ > 0.
  - x_p alone is a degenerate anchor: zero strength, and 0/0 at ω_c.

## B1s: single-Lorentz (Long-type) realisation of the SIMPLE unit-transmission target (`fig_cfreq_feasibility.py freq_simple`)

This was the frequency-domain-only figure for advisor update #1.

**Target.** The KHP 4th-order pole χ(y) = 0.5δ⁴/(y + iδ)⁴·W(y).
- δ = 0.2λ_r is a length.
- W is a cos² window: 1 for |y| ≤ 1, falling to 0 at |y| = 2.
- 400 layers of λ/100.
- Im χ is odd, reaching ±0.389.
- ∫χ ≈ −1.6e-4, so |τ| ≈ 1.
- Why 4th order: KHP PRL 118, 163201, p. 2, after Eq. 6, says that poles of order ≥ 2 give zero reflection and perfect transmission. A truncated double pole leaves |τ − 1| ≈ 2–6 % (its 1/y² tail gives ∫χ ≈ 2A/Y); 4th order gives 5e-4.

**Panels.**
- **a:** the target at ω_c, with gain shaded.
- **b:** the same layers at real ω_r: passive and lossy.
- **c:** the η sweep, 0.005–0.12, redesigned at each η.
- **d:** γ/2|ω″|, ω₀/ω_r and 10ω_p/ω_r per layer.

**Numbers.**
- Min Im ε over real ω is 1e-21, i.e. ≥ 0.
- At real ω_r: |r_L|² = 0.0125, |r_R|² = 0.0084, |τ|² = 0.695.
- At η = 0.05 (ω_c): |r_L|² = 1.6e-9, |r_R|² = 2.1, |τ − 1| = 5.0e-4.
- Centre layer: γ/2|ω″| ≈ 1, ω_p ≈ 0.18ω_r, ω₀ ≈ 1.03ω_r.

## B1z: the η → 0 limit is SINGULAR, not smooth (`fig_cfreq_feasibility.py eta_zero`)

**Question (advisor).** As η → 0, ω_c → ω_r and |τ(ω_c)| stays 1. Must there then be real gain at real frequencies?

**Answer.** Only at η = 0 exactly. For every η > 0 the redesigned structure is passive at every real ω, and the η → 0 limit is discontinuous.
- **a, b.** ε(y, ω_r) is the same for η = 0.05, 0.01 and 0.002.
  - It converges to the fixed passive profile χ_r = |χ|²/(χ* − 2iM), with M = max(−Im χ), so Im χ_r ≥ 0.
  - The target's gain dip at y ≈ −0.1 becomes the biggest loss peak.
- **c, d.** Each layer's Lorentz line narrows in proportion to η, and never goes negative at any real ω.
  - Plotted against (ω − ω_r)/|ω″|, all η collapse onto one curve.
- **Why it is singular.**
  - **The scalings.**
    - κ = ½κ_max ∝ |ω″|, so ω_p² = ω_r κ|χ|² ∝ η.
    - γ = 2|ω″| + κ Im χ ∝ η.
    - ω₀² − ω_r² ∝ η.
  - **Consequence.**
    - The distance from ω_r to ω_c is |ω″| ∝ η.
    - The linewidth is also ∝ η.
    - So ε still changes by O(1) between ω_r and ω_c however small η is.
    - Continuity (ε(ω_c) → ε(ω_r)) would hold for one fixed material, but each η has its own material.
- **At η = 0** the family degenerates to γ = 0 and ω_p = 0, i.e. vacuum. Long's window 0 < γ < 2|ω″| closes, and only real gain remains.
- **e.** |τ(ω)|² over real ω is 1 except in a lossy dip of width ~η at ω_r, with |τ|² ≈ 0.69 at every η.
  - Away from ω_r it is 1 only because the medium vanishes (ω_p → 0).
- **f.** As η runs from 1e-4 to 0.1:
  - |τ(ω_c) − 1| = 5e-4 is flat (the truncation floor).
  - 1 − |τ(ω_r)|² ≈ 0.31 is flat.
  - γ_max ∝ η and ω_p,max ∝ √η.
- **Caveat.** Small η needs Q ~ 1/η, and single-Lorentz virtual gain never settles in time (B3). So η → 0 is a limit of the frequency-domain design, not an operating regime.

## B3a confirmed in COMSOL (2026-11-12)
COMSOL 6.4's built-in "Electromagnetic Waves, Transient" interface (`ewt`) with its built-in
Drude–Lorentz polarization reproduces B3a for the two solid curves. Model and scripts are in
`comsol/` (see the README there). Result:
- Gain slab (γ/2|ω″| = 0.5): B3a quantity 243 at t = 60 T_r, same as TMM × FFT.
- Lossy slab (1.5): 1.000.
- The complex exit field converges to TMM × FFT at second order in the time step. Median
  difference: 3.7e‑2 / 9.3e‑3 / 2.3e‑3 (gain slab) and 8.8e‑3 / 2.2e‑3 / 5.4e‑4 (lossy slab)
  at T_r/100, /200, /400.
- Figure: `figures/B3a_comsol_check.png`.

The `temw` / `teew` / `ewfd` choice: `teew` (Time Explicit) has no dispersive materials. `ewfd`
and `emw` are frequency domain. `ewt` takes any number of Drude–Lorentz terms, including a
negative oscillator strength (checked), so it also covers the two-line medium of B7.

## TO REVISIT (Kunal): why every pole must decay faster than the drive (2026-11-13)
Reminder: come back to this and understand it quantitatively. It is why Long-type single-Lorentz
media fail.

**Derivation.**
1. Linear response, e^{−iωt} convention: y(t) = (1/2π) ∫ H(ω) X(ω) e^{−iωt} dω. Here H is
   t(ω) or r(ω) of the structure and X is the drive spectrum. TMM × FFT computes exactly this.
2. For t > 0, e^{−iωt} → 0 in the lower half ω-plane, so the contour can be closed there. The
   residue theorem turns y(t) into a sum over the poles in the lower half-plane: a pole at ω_n
   contributes (residue) × e^{−iω_n t}.
3. A switched-on drive e^{−iω_c t} has X(ω) ∝ 1/(ω − ω_c). That pole gives H(ω_c) e^{−iω_c t},
   the frequency-domain answer. Each pole ω_n of H gives c_n e^{−iω_n t}
   = c_n e^{−i(Re ω_n)t} e^{−|Im ω_n| t}, a free oscillation. The turn-on shape changes the
   amplitudes c_n only, never the exponents.
4. Dividing by the drive:
   y/x = H(ω_c) + Σ c_n e^{−(|Im ω_n| − |ω″|)t} e^{−i(Re ω_n − ω′)t}.
   This settles to H(ω_c) iff |Im ω_n| > |ω″| for every pole n of the structure.
5. Same statement from the ODE: D(ω_n) = 0 is exactly the condition for the undriven
   P̈ + γṖ + ω₀²P = 0 to have the solution e^{−iω_n t}. A pole is a frequency at which the
   system oscillates with no input.
6. Single Lorentz: the pole is at −iγ/2 and Im ε(ω_c) ∝ (γ − 2|ω″|). Gain at ω_c therefore puts
   the pole above the drive, so it diverges.

## Material poles are necessary, not sufficient: thick virtual-gain slabs "lase" (2026-11-13)
- **Material poles** are the poles of ε(ω), a bulk property: −iγ/2 for a single Lorentz;
  −0.10 and −0.25 (two-line) or −0.110 and −0.140 (EIT), in units of ω_r.
- **Slab poles** are the poles of t(ω) of the finite structure, i.e. the bright dots in the
  B7 maps. They are Fabry–Pérot modes and they set the time domain. They pile up against each
  material pole: there |ε| → ∞, so n·L/λ → ∞ and the FP orders m = 0, 1, 2, … run off to
  infinity there. Checked for the 1λ EIT slab (2026-11-14): the chain of bright dots ending at the
  material pole 1.0651 − 0.1098i has m = 0, 1, …, 9 exactly, with |ε| = 0.26 → 17.6 along it.
  The material pole is an accumulation point of slab poles, which is why a material pole above
  the drive always means failure.
- **How to read the maps.** Bright dots are slab poles, the only things that ring. × marks a
  material pole: it is not a pole of t and not a mode of the slab. Near it the faces become
  near-perfect mirrors (|ε| huge), so the background is dark (t ≈ 0), and that high-finesse
  cavity has dense, sharp resonances (the bright chain). Dark regions are just |t| < 1, not modes.
  The sKK figure of merit (τ(ω_c) = 1, r_L(ω_c) = 0) is the value at the ★ only; a |t| = 1
  contour elsewhere in the plane means nothing.
- **Criterion.** "Every pole of the structure (t and r share them) strictly below the drive
  line" is necessary AND sufficient for settling. The material-pole condition is the part of it
  that does not depend on L.
- **Highest slab pole vs thickness** for a homogeneous slab with ε(ω_c) = 1 − 0.3i,
  η = 0.05, drive at −0.05:

| medium | L = 1λ | 2λ | 3λ | 4λ | 6λ |
|---|---|---|---|---|---|
| two-line | −0.061 | **−0.050** | −0.044 | −0.040 | −0.035 |
| EIT | −0.070 | −0.052 | −0.042 | −0.036 | −0.028 |
| single Lorentz | −0.026 | −0.025 | −0.025 | −0.025 | −0.025 |

- Above about 2λ the two-pole slabs have a Fabry–Pérot pole above the drive. In the frame
  that decays with the drive, the slab is a gain medium (ε″(ω_c) < 0) between two weak mirrors.
  Once the round-trip gain beats the mirror loss, it lases. Threshold estimate (corrected
  2026-11-14): a pole sits on the drive line when |r′(ω_c)|² |e^{2i n(ω_c) ω_c L/c}| = 1, with
  the ABSOLUTE spatial growth Im(n ω_c) = −1.25 per λ, not the |τ| = 2.41 relative to vacuum.
  With |r′|² = 0.0054 this gives L_th = ln(1/|r′|²)/(2·1.25) = 2.09λ, consistent with the table.
  An earlier estimate of "near 3λ" used |τ| and was wrong.
- **1λ vs 5λ, time domain (exp_slab_poles.py → B7_{eit,fano2}_thick.png):**
  - At 5λ, EIT has 6 slab poles above the drive (highest 1.0074 − 0.0313i) and two-line has 5
    (0.9941 − 0.0369i), by the argument principle.
  - |t/t_free| grows to 1.0e5 (EIT) and 4.1e3 (two-line) by t = 100 T_r.
  - The frequency-domain value τ(ω_c) = 0.074 is never reached: above threshold r′²e^{2iφ}
    dominates the FP denominator and τ(ω_c) is a meaningless analytic continuation.
  - Numerics: near a material pole at 5λ the TMM product loses precision (bright "blobs" where
    |t| ~ 1e-8). exp_slab_poles.py uses the closed form t = (1−r′²)e^{iφ}/(1−r′²e^{2iφ}), written
    so no exponential grows; it is checked against 50-digit mpmath.
- **FP order (1λ):** m = 0, 1, 2, … along each chain, integers to 1e-3 (B7 panel d labels m = 1).
  - EIT: m = 1 at 0.9749 − 0.0713i.
  - Two-line: m = 1 at 0.9857 − 0.0605i, the highest pole, spiralling into the pole at 1.002 − 0.100i.
  - The absolute index carries the face-reflection phase convention (principal log); the +1 step
    per dot is what matters.
- |ε(ω)| maps next to |t(ω)|: B7_{eit,fano2}_eps_map.png. Bright = material poles (×),
  dark = zeros of ε.
- Implication: stability is a structure property. Check the poles of t and r for the actual
  structure; the material alone isn't enough. B8 (sKK, about 4–5λ) settles in the time domain.
  **Pole check (2026-11-14):** for the B8 4λ stack (simple target, η = 0.05) the
  argument principle finds 0 poles above the drive over Re ω = 0.05–4 ω_r, for both EIT and
  two-line. The highest poles are −0.1036 (EIT) and −0.0969 (two-line). These sit right at the
  highest material poles (−0.104 and −0.100), so no Fabry–Pérot "lasing" pole approaches the
  drive. This is consistent with weak, local gain (about 0.2λ wide, ε″ ≥ −0.4) and a one-sided
  reflectionless stack with little feedback. B8 panel d (|r_R| map) shows the same.
  Single Lorentz in B8: highest pole −0.028, above the drive, as expected.
- Time-domain verified for 1λ vs 5λ (above); 2–3λ not run in the time domain.

## TO REVISIT (Kunal): why ω₀ ≠ ω_r exactly in the two-line design (2026-11-14)
Reminder: be ready to answer "why is the oscillator resonance not at the drive frequency?"
- ω_r = Re ω_c is the drive carrier (fixed, unit of frequency). ω₀ is a material parameter chosen
  by the design so that ε(ω_c) hits the target exactly, including ε′(ω_c) = 1, i.e. Re χ(ω_c) = 0.
- One Lorentz term at the complex frequency ω_c = a − ib:
  D(ω_c) = (ω₀² − a² + b² − γb) + i·a(2b − γ), so Re χ(ω_c) = 0 needs ω₀² = a² − b² + γb.
  B3 (γ = b, ratio 0.5) gives ω₀ = a exactly.
- Two lines sharing one ω₀ (fano2: γ₁ = 10b, γ₂ = 4b): the real parts Xₙ = ω₀² − a² + b² − γₙb
  can't both vanish, so the weighted sum must: (f₁/|D₁|²)X₁ − (f₂/|D₂|²)X₂ = 0.
  With f₂/f₁ = 0.36, |D₁| ≈ 8ab and |D₂| ≈ 2ab, this gives ω₀² = a² + 1.74b², so
  ω₀ = 1.0022a at η = 0.05.
- It is an O(η²) correction (it vanishes as b → 0), not a detuning knob.
- EIT and the sKK stacks: each layer's ω₀ is solved in the same way to hit that layer's χ(ω_c), so the ω₀
  spread across layers (B8 EIT: 0.86–1.14 a) is the design, not a mistake.

## Slowest slab mode vs thickness: analytic law (2026-11-14; `exp_slab_poles.py` → B7_pole_vs_thickness.png)
Homogeneous slab in vacuum, ε(ω_c) = 1 − 0.3i, η = 0.05. Here k(ω) = n(ω)ω/c and r(ω) = (n − 1)/(n + 1).
1. **Exact pole condition:** r(ω)² e^{2ik(ω)L} = 1.
   - Modulus part: **L·Im k(ω) = ln|r(ω)|**. Every pole lies on this curve.
   - Phase part: Re k·L + arg r = mπ. It picks discrete points on the curve, one per FP order m.
2. **Envelope.** The top of the modulus curve tracks the exact highest pole to within 0.0005 ω_r
   for L ≥ 0.5λ. Below that, discreteness matters: only a few poles exist.
3. **First-order closed form** (expanding about ω_c):
   - Im ω_top(L) ≈ −|ω″| + v_g [g − ln(1/|r|)/L], with g = −Im k(ω_c), v_g = 1/Re k′(ω_c), r = r(ω_c).
   - Equivalently, the slowest-mode decay rate is κ(L) ≈ v_g[α + ln(1/|r|)/L], evaluated on the real
     axis. This is the textbook cavity photon decay rate (absorption + mirror leakage).
   - Settling ⇔ κ(L) > |ω″|.
   - **Threshold L_th = ln(1/|r(ω_c)|)/g = 2.09λ.** It is the SAME for EIT and two-line, because it
     depends only on ε(ω_c); dispersion only sets v_g (0.17c EIT, 0.11c two-line), i.e. the slope.
   - Exact first crossing: 2.125λ (EIT) and 2.25λ (two-line), on a 0.125λ grid.
4. **Limits.**
   - L → 0: the top pole goes down to the highest material pole (the FP chain's accumulation point).
   - L → ∞: the top pole tends to the top of the curve Im k(ω) = 0. That is 0.000 for EIT, because
     ε″(ω_r) = 0 (a transparency window gives arbitrarily long-lived modes), and −0.010 for two-line.
5. **Meaning: it is mode lifetime vs drive, not gain per se.**
   - g = −Im(nω_c) = a|n″| + n′|ω″| for n″ < 0. Virtual gain at ω_c therefore always gives g > 0 and
     a finite L_th.
   - Even a gainless nondispersive slab has g = n′|ω″| > 0. For ε = 2, L_th = 3.97λ, exactly
     equal to its pole law Im ω = ln|r|/(nL).
   - Any cavity whose photon lifetime exceeds the drive's decay time fails to settle.
   - Stable for all L requires g ≤ 0, e.g. the B3 lossy slab (ε = 1 + 0.3i).
   - Reflectionless structures (|r| → 0) push L_th → ∞. This is consistent with B8 having no FP
     poles near the drive (heuristic for inhomogeneous stacks: there is no single r there).
6. **Why the material pole is the m → ∞ end** (polariton picture):
   - The slab samples the medium's dispersion k(ω) at k ≈ mπ/L.
   - k(ω) → ∞ only at a pole of ε, i.e. the flat, matter-like end of the polariton branch. So
     ω_m → ω_material as m → ∞.
   - Large-k modes carry almost all their energy in the oscillators. They decay at the oscillator
     rate and don't feel the faces, so they are L-independent.
   - Low-m modes are light-like: they leak through the faces and feel the virtual gain, so they are
     L-dependent and are the ones that cross the drive.
   - One band of modes, two ends.
