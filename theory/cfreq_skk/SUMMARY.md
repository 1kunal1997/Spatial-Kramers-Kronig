# Complex-frequency sKK: summary of the research direction (Nov 2026)

Short version: we can build one-way invisible sKK structures out of purely passive materials, as long
as we drive them with a decaying wave. The catch is that the materials Long et al. proposed for this
never actually settle in the time domain. We found media that do, and we worked out the general rule
for when a structure settles and when it doesn't.

Code, data and the detailed numbers are in `FINDINGS.md`. Figures are in `figures/`.

---

## 1. The idea: virtual gain lets passive materials do sKK

- **sKK profiles need gain.** A spatial Kramers–Kronig profile ε(x) has no reflection from one side.
  The unit-transmission versions (King, Horsley, Philbin 2017) also have t = 1. But the integral of
  ε″ over the profile is zero, so some regions have to have ε″ < 0, i.e. gain. That's the reason
  nobody builds them.
- **Complex frequency.** Drive the structure with a wave that decays in time:
  E ∝ e^{−iω_c t}, with ω_c = ω_r − i|ω″|. We use |ω″| = 0.05 ω_r (η = 0.05) throughout.
- **Virtual gain.** Under this drive, a passive material evaluated at ω_c can have Im ε(ω_c) < 0.
  Long et al. (LSA 2026, Theorem 1) showed that a single passive Lorentz oscillator can hit any
  complex ε(ω_c) you want, including gain.
- **Our plan.** Use that to realise the gain regions of an sKK profile with passive materials, and
  get one-way invisibility (r_L = 0, t = 1) at ω_c.

## 2. The problem: it doesn't settle in the time domain (B3)

- **The frequency-domain answer only shows up after the transients die out.** When you switch the
  drive on, the output is the driven response at ω_c plus the structure's own free ringing. Each
  pole ω_n of the structure rings like e^{−|Im ω_n| t}. The drive itself decays like e^{−|ω″| t}.
- **So the rule is simple.** If any pole decays slower than the drive, its ringing outlives the drive.
  The ratio output/drive then grows forever and never reaches the ω_c value. Settling needs
  **|Im ω_n| > |ω″| for every pole of the structure.**
- **A single Lorentz can't satisfy this and have gain at the same time.** Its pole is at −iγ/2,
  and the sign of ε″(ω_c) is the sign of (γ/2|ω″| − 1). Gain at ω_c needs γ/2 < |ω″|, which is
  exactly the condition for the pole to ring longer than the drive decays. Virtual gain and
  stability are mutually exclusive for this medium.
- **B3a, homogeneous 1λ slab with ε(ω_c) = 1 − 0.3i.**
  - The gain slab (single Lorentz) should settle at |τ| = 2.41. Instead it keeps growing: about
    243× its expected value at t = 60 T_r.
  - The lossy slab (ε(ω_c) = 1 + 0.3i) settles fine.
  - The same holds for the full sKK stacks (B3, B8): every single-Lorentz version diverges.
- **Checked three independent ways:**
  - our TMM × FFT synthesis;
  - our own FDTD;
  - COMSOL's built-in transient EM with its built-in Drude–Lorentz model (`B3a_comsol_check.png`).
    COMSOL matches and converges at second order in the time step.
- **What Long et al. missed.**
  - They only worked at ω_c, and their figures are field maps at that frequency.
  - As far as we can tell, they never checked that the material's own ringing dies faster than
    the drive. For their media, it doesn't.
  - Side note: their printed Theorem 1 equations also have a sign error on the (C − 1) terms. The
    existence result is still fine.

## 3. The fix: two-pole media (B7, B8)

- **The trick.** The single-Lorentz problem comes from having one pole. With two coupled poles, the
  decaying drive weights them differently. So you can get Im ε(ω_c) < 0 while every material pole
  decays at least 2× faster than the drive.
- **The two media we use.** Both are passive at every real frequency.
  - **EIT:** a lossy "bright" oscillator coupled to a lossless "dark" one.
  - **Two-line:** a broad Lorentz line minus a narrow one at the same ω₀,
    χ = f₁/D₁ − f₂/D₂ with γ₁ > γ₂.
    - It works only inside a window: (γ₂/2 − |ω″|)/(γ₁/2 − |ω″|) < f₂/f₁ ≤ γ₂/γ₁.
    - The window exists only if the narrow line is the one subtracted.
- **B7, same 1λ slab, same ε(ω_c) = 1 − 0.3i.** EIT and two-line both settle to |τ| = 2.41 within
  about 20–40 T_r. FDTD agrees. (`B7_eit_slab.png`, `B7_fano2_slab.png`)
- **B8, full one-way invisible sKK stack** (4λ, 400 EIT layers).
  - It settles to |r_L| ≈ 4e-5 and |τ| = 1.0000.
  - The single-Lorentz version of the same stack blows up.
  - A random 6-pole target and η = 0.01 also work.
- **B9, angles.** It stays one-way out to about 85°.

## 4. What we learned from B7: material poles vs slab poles

This is where it got more interesting than "just make the oscillators decay fast".

- **Two kinds of poles.**
  - **Material poles** are the poles of ε(ω): the oscillators ringing on their own. They're a
    bulk property and don't depend on the slab thickness L.
  - **Slab poles** are the poles of t(ω) for the actual finite structure: the Fabry–Pérot (FP)
    modes. They depend on ε(ω), L and the interfaces. **These are what the time domain actually
    sees**, and t and r share them.
- **How to read the complex-ω maps** of log|t|:
  - **Bright dots** are slab poles, the only things that ring.
  - **×** marks a material pole. It is not a mode of the slab.
  - **Dark regions** just mean |t| < 1. They aren't modes either.
  - The sKK performance is read only at the ★ (ω_c), not anywhere else on the map.
  - `B7_*_eps_map.png` puts |ε(ω)| next to |t(ω)|. The × land exactly on the bright spots of |ε|.
- **The FP chain ends at the material pole.**
  - Each × has a chain of bright dots running into it, with FP orders m = 0, 1, 2, … → ∞. We
    checked that m comes out an integer at each dot.
  - Why: the slab only keeps modes with k ≈ mπ/L. Bigger m needs bigger k, which needs bigger n.
    The only place n → ∞ is at a material pole.
  - In polariton language, the dispersion k(ω) flattens out at the oscillator resonance, so the
    high-m modes pile up there.
  - Those high-m modes are basically pure oscillator motion. They decay at the oscillator's own
    rate and don't feel the slab faces, which is why they don't care about L.
  - The low-m modes are mostly light. They leak out through the faces and feel the virtual gain,
    so they do care about L.
  - So it's one band of modes with two ends, not two separate families.
- **Why dark and bright sit right next to each other at the ×.**
  - Near the ×, |n| is huge, so each face is a near-perfect mirror. That makes the background
    dark, since almost nothing gets through.
  - A cavity with near-perfect mirrors also has very sharp, closely spaced resonances: the bright
    dots.
  - t → ∞ at a complex ω doesn't mean infinite transmission. It means output with no input,
    i.e. a mode.
  - Both come from the same t = (1 − r²)e^{ikL}/(1 − r²e^{2ikL}): r → 1 makes the numerator small
    and makes the denominator vanish at the resonances. It's a high-finesse etalon.
- **The rule, stated properly.**
  - Every slab pole below the drive line is **necessary and sufficient** for settling.
  - Material poles below the line is **necessary but not sufficient**. It's the L-independent
    part of the same condition: a material pole above the line drags infinitely many slab poles
    above with it.

## 5. Thickness matters: thick virtual-gain slabs "lase"

- **Same media, 5λ instead of 1λ.** Both EIT and two-line blow up (`B7_*_thick.png`). The low-order
  FP poles rise above the drive line, while the chains near the × stay put.
  - EIT has 6 poles above the line and two-line has 5.
  - The frequency-domain value |τ(ω_c)| = 0.074 is never reached.
- **Slowest mode vs thickness** (`B7_pole_vs_thickness.png`). The slowest mode starts at the
  material pole for a thin slab, rises with L, and crosses the drive at about 2.1λ (EIT) and
  2.2λ (two-line).
- **Analytic law.** Every pole satisfies r²e^{2ikL} = 1 exactly. Expanding near ω_c gives

  Im ω_top(L) ≈ −|ω″| + v_g [g − ln(1/|r|)/L]

  with g = −Im k(ω_c), the spatial growth of the wave in the slab, and v_g the group velocity.
  - This is basically a laser threshold equation: growth rate = v_g × (gain − mirror loss per
    length).
  - Threshold: **L_th = ln(1/|r(ω_c)|)/g = 2.09λ.**
  - L_th is the **same for both media**, because it only depends on ε(ω_c). Dispersion only
    changes the slope through v_g (0.17c for EIT, 0.11c for two-line).
  - The exact version (the top of the curve L·Im k = ln|r|) tracks the real poles to within
    0.0005 ω_r.
- **What it really means.**
  - It's about mode lifetime, not gain itself. Any cavity whose modes live longer than the drive
    decays won't settle. Even a lossless ε = 2 slab fails past 3.97λ.
  - Virtual gain at ω_c just guarantees you hit that point at some finite thickness.
- **Why sKK survives this.**
  - A reflectionless structure has |r| → 0, which pushes L_th → ∞.
  - We checked B8's 4λ stack directly. There are no slab poles above the drive (argument
    principle). Its highest poles (−0.104 ω_r for EIT) sit right at the material poles, a 2× margin.
  - So the one-way-invisible property is exactly what protects the sKK stack from this
    instability. For a graded stack this is a heuristic, since there's no single r, but it fits
    what we see.

## 6. A note on the y-axis: τ = t/t_free

Under a decaying drive, the light reaching the exit face left the source one transit time earlier,
when the drive was stronger. So even vacuum gives |t| = e^{|ω″|L/c} > 1 (1.37 for 1λ). We
therefore plot τ = t/t_free, the transmission relative to the same thickness of vacuum, the same way
in the frequency domain and the time domain. τ = 1 is the "invisible" condition. Long et al. only
show field maps, so this is our convention, not theirs.

## 7. Where this stands

**Solid:**
- the B3 failure of single-Lorentz media (three independent solvers);
- the two-pole fix in the frequency domain, the time domain and FDTD;
- the pole rule;
- the thickness law for homogeneous slabs.

**Still to do:**
- time-domain runs at 2–3λ, right at threshold (we only ran 1λ and 5λ);
- a more general stability argument for inhomogeneous stacks;
- a COMSOL eigenfrequency check of the slab poles;
- understanding why ω₀ ≠ ω_r exactly in the two-line design (reminder in FINDINGS);
- bidirectional targets;
- the Horsley papers.

**Prior art:**
- Li et al. 2020 ("virtual PT symmetry") is the precursor we must discuss. Their stability margin
  comes from leakage through the ports of a lumped circuit, while ours has to come from the
  material itself. See `PAPER2_POSITIONING.md`.
- sKK combined with complex frequency appears to be new.
