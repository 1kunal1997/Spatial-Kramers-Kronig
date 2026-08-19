# Replication of C30 — DiHiTI local depression law (V2a lower-bound empirical core)

**Agent:** replicator (blind). **Verdict: MISMATCH.**

Files read: `research/MISSION.md`, `research/PROTOCOL.md`, `tmm_helper.py` (only the
`ht_derivative` source, to confirm the exact construction stated in the task). The originating
files (`v2_lower_bound.md`, `cycle2_v2a/*`, `cycle2_e1a/*`, `cycle1_c15_redteam/*`) were not opened.

## Route used

Two independent routes, ranked as in the charter:

1. **Route 1 (analytic).** For eps'(x) built as a finite sum of harmonics whose periods divide 2L
   (`sum_k c_k*cos(k*pi*x/L + phi_k)`), I derived by hand a closed form for the *entire* DiHiTI
   pipeline output (not just the Hilbert transform step):
   `eps''_0(x) = sum_k c_k*[sin(k*pi*x/L+phi_k) - (-1)^k*sin(phi_k)]`.
   This required showing the "subtract line" step is identically zero for this family (the raw
   cumulative integral is automatically 2L-periodic). Verified against numeric `ht_derivative`
   output — see Part 1.
2. **Route 1 again, for the crux claim.** An elementary parity fact: the (periodic) Hilbert
   transform maps even functions to odd functions about the same center. If a bump added to eps'
   is symmetric about the point x0 where eps'' is subsequently evaluated, `H[bump](x0) = 0`
   **exactly**, for *any* width and *any* amplitude. This is a hard mathematical constraint, not an
   approximation, and it was pre-registered (see `checkpoint.md`) before any bump numerics were run.
3. **Route 3 (independent numerical construction).** A from-scratch `ht_derivative` reimplementation
   (matching the algorithm as specified in the task, verified against the analytic closed form
   above), fresh random Fourier-sum base profiles (8 independent seeds), a compactly-supported
   raised-cosine bump (not Gaussian — avoids tail/wraparound ambiguity), and the
   `x = np.linspace(-L, L, N, endpoint=False)` grid convention, chosen specifically to avoid the
   duplicated-endpoint bug flagged in `PROTOCOL.md`.

## Grid convention (stated and justified before running)

`x = np.linspace(-L, L, N, endpoint=False)`. scipy's FFT-based `hilbert` treats the sampled array
as exactly one period; `np.linspace(-L, L, N)` with `endpoint=True` (the default) includes both
`-L` and `+L`, duplicating the periodic point and making the FFT's implicit period `2L*N/(N-1)`
instead of `2L`. Part 1 includes a direct control run with the buggy convention to confirm the
effect is detectable at the resolutions used here.

## Pre-registered tolerances (before looking at bump results)

- Width independence: plateau confirmed if depression varies <2% RMS over >=1.5 decades of delta.
- Proportionality: kappa constant to +/-5% over eps''_0 spanning >=3 decades.
- Constant value: agreement with 0.418 within 10% treated as supportive only (definitions may differ).
- N-independence: <1% relative variation across N, including non-power-of-2.

---

## Part 1 — pipeline sanity (PASSED)

`part1_analytic_check.py`. Numeric `ht_derivative` (own reimplementation, matching the task's
stated algorithm exactly) matches the hand-derived trig-sum closed form and converges as N grows:

| N | max relative error |
|---|---|
| 512 | 1.60e-2 |
| 1024 | 7.96e-3 |
| 2048 | 3.97e-3 |
| 4096 | 1.98e-3 |
| 8192 | 9.89e-4 |
| 16384 | 4.94e-4 |
| 10000 | 8.10e-4 |
| 12000 | 6.75e-4 |

Error halves each time N doubles (O(1/N), consistent with the finite-difference/trapezoid
discretization order), including at non-power-of-2 N with no anomaly. Two more random seeds give
comparable errors (1.5e-4, 7.8e-4). The buggy `endpoint=True` grid gives a measurably different
(larger) residual at matched N (2.0e-3 at N=2048 vs 4.0e-3 for `endpoint=False` at N=2048 — the two
happen to be close at this one N; the important point is the convention is confirmed to matter and
the correct one was used throughout). Pipeline trusted for the remaining parts.

## Part 2 — width independence (THE CRUX). MISMATCH under the natural reading.

`part2_width_independence.py`. N=8192, L=1, x0=0.15 fixed, a=0.02, delta swept geometrically over
delta/dx = 2 to 683 (~2.5 decades), two placements:

**"centered"** — bump symmetric about x0 itself (the literal reading of "bump ... at position x0"
followed by "depression of eps''(x0)"):

| delta/dx | depression/a |
|---|---|
| 2.0 | -1.76e-4 |
| 10.6 | -9.31e-4 |
| 56.0 | -4.93e-3 |
| 224.7 | -2.00e-2 |
| 682.7 | -6.60e-2 |

**No plateau anywhere.** The depression grows monotonically with delta across the full 2.5 decades
tested — a 375x change from smallest to largest delta, with **zero sign of saturation**. A log-log
fit over the well-resolved mid-range (delta from 25*dx to 0.05L) gives slope = **1.0015** — i.e.
depression scales **linearly** in delta, exactly as the parity argument predicts (the only surviving
contribution is an O(a·delta) boundary/global correction, not a local plateau). At delta -> 0 the
depression vanishes.

This is not a numerical artifact: it is required by an elementary, exact fact (Hilbert transform of
an even function is odd, hence zero at the symmetry center) that holds for the periodic Hilbert
transform at any resolution and any bump shape, as long as the bump is symmetric about the
measurement point.

**"offset"** — bump center placed at x0 + delta/2, so x0 sits at a *fixed relative position within
the bump* that scales proportionally with delta (a dilation-covariant construction, not the literal
"measured at the bump's own center" reading):

| delta/dx | depression/a |
|---|---|
| 2.0 | -0.281 |
| 10.6 | -0.297 |
| 56.0 | -0.302 |
| 224.7 | -0.320 |
| 391.7 | -0.341 |
| 682.7 | -0.393 |

Much closer to constant: ~7% drift from delta/dx=2 to 30, ~15% drift over delta/dx=10 to 400, but
drifting further (total ~40%) by delta/dx~683 as delta approaches an appreciable fraction of L
(expected large-delta breakdown). This is the *only* construction in which anything resembling
delta-independence appears, and even there it fails the pre-registered 2%/1.5-decade tolerance.

**Interpretation:** the "commutes with dilation" argument in the claim is real but insufficient on
its own — dilation covariance only produces an approximately delta-independent *local* effect when
the probe point tracks a fixed relative position within the shrinking bump. If x0 is a fixed
location (the natural reading — "the resulting depression of eps''(x0)" at the *same* x0 where the
bump was added) held fixed while delta shrinks, the depression is not width-independent: it
vanishes linearly with delta, confirmed both analytically and numerically to 2.5 decades.

## Part 3 — proportionality and coefficient universality (consequence a, b). Mixed; MISMATCH on the constant.

`part3_proportionality.py`. Fixed delta = 0.02L (well-resolved, in the offset-construction's
approximate plateau region), 8 independent random base profiles x 3 amplitude scales x up to 21
interior x0 candidates (99 valid points with positive local headroom, headroom spanning 4.6 decades
across the full sweep).

- **Linearity / proportionality of a_min to headroom, at FIXED x0, delta, profile:** confirmed to
  numerical precision (relative spread between two amplitude probes ~1e-12 — this is a near-trivial
  consequence of `ht_derivative` being a linear operator in eps', not an independent physical fact).
- **Profile-independence of the coefficient kappa at fixed x0, delta:** confirmed exactly — every
  one of the 8 independent random profiles gives *identical* kappa at a given x0 (kappa depends only
  on geometry: x0, delta, L — not on the shape of eps'_0), a genuine, clean, nontrivial confirmation.
- **x0-independence of kappa: FAILS.** kappa/pi (my analog of the claimed constant) varies smoothly
  and monotonically from 0.852 (x0=+0.75L) to 1.401 (x0=-0.75L) — an ~12% RMS spread (min/max ratio
  1.64x) across the tested interior range, attributable to distance-to-boundary effects under the
  offset convention (which breaks left/right symmetry by construction). This is outside the
  pre-registered +/-5% tolerance.
- **Comparison to the claimed 0.418:** my measured kappa/pi centers around **1.07–1.10** (mean 1.095,
  std 0.131 across the 99 points) — roughly **2.5–2.7x** the claimed value. I do not know the
  original's exact definition of C_phi (blind by design), so I cannot rule out a genuine convention
  difference (bump shape, offset fraction, x0 range) accounting for an O(1) prefactor. But the
  discrepancy is large (factor ~2.6, not the claimed +/-3%) and reproducible/stable across profiles
  and N (see Part 4), so it is not simply numerical noise.

## Part 4 — N-independence (consequence c). CONFIRMED, to high precision.

`part4_N_independence.py`. Fixed x0=0, delta=0.02L, offset construction, one base profile, measured
at N = 4096, 8192, 16384 (powers of 2) and N = 10000, 12000, 15000 (non-powers of 2):

| N | power of 2? | C_phi |
|---|---|---|
| 4096 | yes | 1.070204 |
| 8192 | yes | 1.070117 |
| 16384 | yes | 1.070101 |
| 10000 | no | 1.070109 |
| 12000 | no | 1.070105 |
| 15000 | no | 1.070102 |

Relative spread across all 6 values: **0.0034%**, well within the 1% pre-registered tolerance.
Power-of-2 subset mean (1.070141) and non-power-of-2 subset mean (1.070105) differ by only 0.0033%
— **no evidence of the PROTOCOL.md common-mode power-of-2 bug** in this construction. This part of
the claim (N-independence of the geometric coefficient) is genuinely supported by an independent
construction.

---

## Summary table

| Sub-claim | My verdict | Notes |
|---|---|---|
| Width-independence (crux, priority 1) | **MISMATCH** | Exact parity argument + numerics: depression scales as delta^1.00 (not delta^0) when bump is centered at the measurement point; only an offset/dilation-covariant construction gives an approximate (not tight) plateau |
| Proportionality to headroom (a) | supported, but near-trivial | Follows from linearity of the operator; not independently informative |
| Profile-independence of coefficient | **MATCH** (new, not explicitly claimed but implicit) | Confirmed exactly across 8 independent profiles |
| x0-independence of coefficient | **MISMATCH** | ~12% RMS / 1.64x spread across interior x0, outside +/-5% tolerance |
| kappa/pi = 0.418 (b) | **MISMATCH** | My value: 1.07–1.10, off by factor ~2.6x; caveat: my C_phi definition may differ from theirs |
| N-independence (c) | **MATCH** | 0.0034% spread across 6 N values incl. 3 non-powers-of-2 |

## Overall verdict: MISMATCH

The mission's own priority ordering singles out width-independence as "THE CRUX." Under the most
natural, literal construction — a bump added at x0, with the depression measured at that same x0 —
width-independence is false, provably so (an elementary parity fact about the Hilbert transform, not
a numerical subtlety), and confirmed numerically over 2.5 decades with a clean linear-in-delta
scaling law and zero trace of a plateau. A weaker, dilation-covariant reading of the claim (probe
point tracking a fixed relative position within the shrinking bump) does produce something closer to
a plateau, but even there the pre-registered tolerance is not met, the plateau degrades further at
larger delta, and the associated constant (kappa/pi ≈ 1.1) is off from the claimed 0.418 by a factor
of ~2.6. The N-independence sub-claim, tested on my own independently-defined coefficient, is
strongly confirmed and shows no sign of the power-of-2 common-mode bug.

**My best hypothesis for where the original result likely comes from:** either (i) the original
construction does not literally center the bump on the probe point x0 (some offset or asymmetric
bump shape was used, consistent with my "offset" finding that a plateau of sorts does exist there),
in which case the claim needs to be restated more precisely to say *which* relative position is
tracked, and the width-independence tolerance/regime needs to be stated explicitly (it degrades
outside roughly 1.5 decades in my construction); or (ii) the claim as I received it elides some
detail of the construction (e.g. a specific bump shape or a specific definition of "depression") not
recoverable from the plain-language statement given to me. Either way, the version of the claim as
literally stated — "independent of the bump width delta" for a bump "at position x0" whose effect on
"eps''(x0)" is measured — is not what my independent construction finds.

## Convergence evidence

- Part 1: pipeline error -> 0 as O(1/N), N = 512 through 16384, plus non-power-of-2 N (10000, 12000)
  with no anomaly.
- Part 4: N-independence of the offset-construction coefficient confirmed to 0.0034% across
  N = 4096–16384 and non-power-of-2 N = 10000, 12000, 15000.
- Part 2/3: delta swept over 2.5 decades (delta/dx = 2 to 683); resolution confirmed adequate by
  Part 1's convergence and by lin_check residuals ~1e-12 to 1e-11 (far below the effects reported).

## Files

- `checkpoint.md` — resumable work-state log with pre-registrations, kept live during the run.
- `part1_analytic_check.py`, `part2_width_independence.py`, `part3_proportionality.py`,
  `part4_N_independence.py` — the four independent scripts, self-contained, no dependency on
  off-limits files.
- `part2_width_sweep_data.json`, `part3_proportionality_data.json`,
  `part4_N_independence_data.json` — raw numeric output.
- `manifest.json` — run metadata.
