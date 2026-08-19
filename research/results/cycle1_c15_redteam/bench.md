# Red-team benchmark: DiHiTI passivity test claim

Claim under attack: *"as a test of a given profile, nothing beats running DiHiTI and taking the
minimum (O(N log N), two FFTs)."*

Machine: Apple M1 (arm64), macOS 26.1. `numpy 2.2.6`, `scipy 1.15.3`.
All times are wall-clock, `timeit.repeat(repeat=20, number=1)`, best-of-20 and median-of-20 reported
in microseconds. Scripts: `bench.py` (profile families + pipelines), `run_all.py` (T1+T2),
`run_t3.py` (T3a/b), `run_t3cd.py` (T3c/d), `run_t3d_adv2.py` / `run_t3d_adv3.py` (T3d adversarial
constructions), `run_t4.py` (T4).

Profiles: nb=1.7, L=1, A = nb²−1 = 1.89.
- P1 logistic, w ∈ {0.15, 0.05, 0.02, 0.005}
- P2 opposite bumps on top of P1(w=0.05), alpha ∈ {0, 0.3, 0.6, 0.8, 0.9, 1.0, 1.5}
- P3 triangle
- P4 random smooth (20 seeded draws)

---

## T1. Baseline timing: `dihiti(u,x).min()`

| N | best (us) | median (us) |
|---|---|---|
| 4096  | 122.5  | 126.3  |
| 8192  | 213.8  | 215.9  |
| 16384 | 423.6  | 426.7  |
| 32768 | 915.4  | 931.9  |
| 65536 | 1827.3 | 1892.4 |

Scaling is close to linear-in-N over this range (not yet N log N dominated — constant-factor FFT
setup and `np.gradient`/`cumulative_trapezoid` overhead dominate at these sizes on this machine).

---

## T2. Candidate A — single Hilbert transform

`onehilb` replaces `np.gradient` → `hilbert` → `cumulative_trapezoid` → detrend with a single
linear-detrend + one `scipy.signal.hilbert` call.

### (a) Agreement with `dihiti` at N=2^14

Relative RMS of `(dihiti − onehilb)/max|dihiti|`, and max abs diff restricted to the interior
`|x| < 0.99 L`:

| profile | rel rms | max abs diff (interior) |
|---|---|---|
| P1 w=0.15 | 3.66e-04 | 3.54e-04 |
| P1 w=0.05 | 2.10e-04 | 3.61e-04 |
| P1 w=0.02 | 1.56e-04 | 3.61e-04 |
| P1 w=0.005 | 1.13e-04 | 3.69e-04 |
| P2 alpha=0.0 | 2.10e-04 | 3.61e-04 |
| P2 alpha=0.3 | 1.62e-04 | 3.71e-04 |
| P2 alpha=0.6 | 1.33e-04 | 3.81e-04 |
| P2 alpha=0.8 | 1.20e-04 | 3.87e-04 |
| P2 alpha=0.9 | 1.14e-04 | 3.91e-04 |
| P2 alpha=1.0 | 1.09e-04 | 3.94e-04 |
| P2 alpha=1.5 | 8.98e-05 | 4.11e-04 |
| P3 triangle | 1.25e-03 | 1.50e-03 |
| P4 (20 draws) | 3.3e-04 – 1.4e-03 | 1.8e-04 – 1.1e-03 |

Agreement is good (≲1e-3 relative) but **not exact** — `np.gradient` (finite-difference derivative)
and the analytic linear-detrend do not coincide bit-for-bit, and the residual is O(1e-4) in absolute
terms at N=2^14.

### (b) Label disagreements

Full-domain sign-of-min and interior-sign-of-min were compared between `dihiti` and `onehilb`:

- **5 of 32 profiles disagree on sign**, in both the full-domain and interior comparisons:
  `P1 w=0.15, P1 w=0.05, P1 w=0.02, P1 w=0.005, P2 alpha=0.0`.
- These are exactly the profiles sitting **at the exact passivity threshold** (nb matches the
  background on one side, so the true interior min is O(1e-5), i.e. essentially zero). Example
  at N=2^14: `dihiti` interior min = +7.69e-5 (PASS) vs `onehilb` interior min = −1.32e-4 (FAIL)
  for P1 w=0.05.
- **This is a real finding, not noise-in-a-safe-direction**: the "verified equivalent" formula
  flips the pass/fail verdict for marginally-passing profiles at working resolution (N=2^14).
  The two formulas are only asymptotically equivalent; near a hard threshold at min≈0, an O(1e-4)
  discretization-scheme difference is enough to change the label. Any speed contender must be
  benchmarked against this instability, not assumed to inherit `dihiti`'s exact verdict.

### (c) Timing at N=2^14

| method | best (us) | median (us) | speedup vs `dihiti` |
|---|---|---|---|
| `dihiti` | 423.3 | 425.9 | 1.00x |
| `onehilb` | 249.0 | 250.6 | 1.70x |
| `onehilb_rfft` (hand-rolled rfft/irfft) | 254.9 | 257.9 | 1.66x (0.98x vs `onehilb`) |

The hand-rolled `rfft`/`irfft` Hilbert transform does **not** beat `scipy.signal.hilbert` on this
machine — the extra Python-level array bookkeeping (building the frequency multiplier, assembling
the full analytic spectrum) eats the FLOP savings from avoiding the full complex FFT. Numerically
it agrees with `onehilb` to machine precision (max abs diff = 0.0 across all 32 profiles, well
under the 1e-12 bar).

**T2 bottom line: ~1.7x speedup from removing the redundant derivative+cumtrapz+detrend steps,
capped there — not a further win from hand-rolling the FFT — and the "verified" equivalence has a
measurable failure mode exactly at the threshold that matters for a passivity test.**

---

## T3. Candidate B — decimation

Ground truth = `onehilb` at N=2^16, interior min over `|x| < 0.99 L`.

### (a) Smallest n matching sign / matching value to 10%

n tried: powers of 2 from 16 to 65536.

| profile | n_sign | n_10pct | GT interior min |
|---|---|---|---|
| P1 w=0.15 | **65536 (never converges below full N)** | 65536 | 4.32e-05 |
| P1 w=0.05 | **65536** | 65536 | 1.14e-05 |
| P1 w=0.02 | **65536** | 65536 | 1.00e-05 |
| P1 w=0.005 | **65536** | 65536 | 9.77e-06 |
| P2 alpha=0.0 | **65536** | 65536 | 1.14e-05 |
| P2 alpha=0.3 | 16 | 1024 | −6.03e-02 |
| P2 alpha=0.6 | 16 | 256 | −1.93e-01 |
| P2 alpha=0.8 | 16 | 128 | −2.84e-01 |
| P2 alpha=0.9 | 16 | 128 | −3.29e-01 |
| P2 alpha=1.0 | 16 | 64 | −3.75e-01 |
| P2 alpha=1.5 | 16 | 64 | −6.05e-01 |
| P3 triangle | 16 | 128 | −6.93e-01 |
| P4 (20 draws) | 16 in every case | 16 – 16384 | −2.4e-03 to −8.3e-01 |

**Bimodal result.** Profiles that fail passivity comfortably (large negative interior min) get the
correct sign from as few as n=16 points. But the profiles that actually sit near the pass/fail
boundary (the whole P1 family and the alpha=0 case, interior min O(1e-5)) **never converge to the
right sign at any n below the full N=2^16** — decimation gives the wrong sign at every tested
coarser resolution. These marginal cases are precisely the ones a passivity screening tool needs
to get right; decimation is unreliable there by construction (aliasing of a near-zero quantity).

### (b) Fourier spectrum decay of periodized p = u + (D/2L)x

n such that tail power beyond mode n drops below 1e-6 / 1e-10 of total, at N=2^16:

| profile | n(1e-6) | n(1e-10) |
|---|---|---|
| P1 w=0.15 | 4 | 7 |
| P1 w=0.05 | 10 | 19 |
| P1 w=0.02 | 23 | 46 |
| P1 w=0.005 | 78 | 172 |
| P2 alpha=0…1.5 | 10 | 20 |
| P3 triangle | 15 | 324 |
| P4 (20 draws) | 5 – 16 | 12 – 192 |

The spectrum itself decays fast (n~10-80 captures nearly all *energy*), but T3(a) shows that fast
spectral decay does **not** translate into a reliable decimated *sign* verdict for marginal
profiles — the passivity sign lives in the last few parts-per-million of the signal near a
near-cancelling integral, which is exactly what gets thrown away by a coarse grid even though the
bulk power is retained.

### (c) End-to-end decimated-test timing at N=2^14 (baseline: `dihiti` best = 422.8 us)

| n | best (us) | speedup vs N=2^14 baseline |
|---|---|---|
| 16   | 14.0 | 30.1x |
| 32   | 14.4 | 29.4x |
| 64   | 15.1 | 27.9x |
| 128  | 16.1 | 26.2x |
| 256  | 18.0 | 23.5x |
| 512  | 22.2 | 19.1x |
| 1024 | 29.5 | 14.4x |
| 2048 | 43.8 | 9.6x |

**≥3x speedup is easily achieved** (by an order of magnitude) at every tested n, including
including the subsampling touch of the full input array. This part of the claim is true — if
decimation is honest about sign, it is much faster.

### (d) Honesty check

- **P2 alpha = 0.8, 0.9 (near-threshold family):** decimation gets the sign right at every tested
  n (16 → 4096). No failures found in this family — the P2 alpha=0.8/0.9 minima (−0.28, −0.33) are
  large enough in magnitude relative to discretization error that decimation is robust here, in
  contrast to the P1/alpha=0 cases in T3(a) whose margin is ~4 orders of magnitude smaller.
- **Adversarial construction 1** (`a·cos(mπx/L)` with m near the decimated-grid Nyquist,
  m ∈ {32,64,128}, a ∈ {0.02,0.05,0.1} added to P1 w=0.05): decimation got every case right. The
  base profile there is already comfortably failing (interior min ≈ −0.04 to −0.19), so this
  construction did not stress a near-threshold case.
- **Adversarial construction 2** (ripple period locked to the decimated-grid spacing, phase zeroed
  at every decimated sample point — i.e. decimation samples the ripple at its own zero crossings
  and is blind to it by construction, verified: ripple value at decimated samples ≤ 3e-12): tried
  on the strongly-failing base profile (already sign = −1 without the ripple) — no sign flip was
  observed because the base case leaves no headroom to demonstrate a flip.
- **Adversarial construction 3** (same phase-locked-ripple trick, but on the barely-*passing* base
  profile P1 w=0.05, GT interior min = +1.14e-5): here the more fundamental problem in T3(a)
  dominates and pre-empts the adversarial construction — **decimation already reports the wrong
  sign (FAIL, min ≈ −1e-2 to −3e-3) for the un-perturbed base profile at every n tested (64 –
  512)**, regardless of the phase-locked ripple. No hand-crafted adversarial input was needed:
  decimation is already unreliable on the natural near-threshold case from T3(a).

**T3 bottom line: decimation delivers the promised ≥3x (really ~10-30x) speedup, but it is not
generally safe.** For profiles with a comfortable margin (|min| ≳ 1e-2 relative to A~1) it gets the
sign right even at n=16-64 and even under an adversarial phase-locked ripple. But for profiles
sitting near the actual pass/fail boundary — arguably the case that matters most for a screening
test — decimation gives the **wrong sign at every resolution below full N**, confirmed independently
in T3(a) and T3(d).

---

## T4. Candidate C — adaptive coarse-to-fine search

Implemented: evaluate `onehilb` on a coarse grid (n0=32), interior-min value; double n and repeat
until the reported minimum stabilizes to 1% relative between successive levels (proxy for a real
hierarchical/branch-and-bound refinement — this is a cheap prototype, not real FMM).

- Verdict matched ground truth (N=2^16, `onehilb`) on **all 32 profiles** — no mismatches.
- Mean total `eps''` evaluations across all levels: 39,876, vs N=65536 (ratio 0.61). For the
  near-threshold profiles (P1 family, alpha=0.0, and several random draws) the adaptive loop never
  stabilizes below full N — same story as T3(a): the marginal cases force full resolution.
- **Wall-clock at N=2^14, baseline `dihiti` best = 562.6 us:**
  - adaptive search: best = 924.0 us → **0.61x — SLOWER than the plain baseline.**
  - one-shot `onehilb` (single FFT at full N): best = 357.7 us.
  - adaptive vs one-shot onehilb: **0.39x — adaptive search is ~2.5x slower than just doing one
    full-resolution FFT.**

**T4 bottom line, bluntly: this loses.** Even though the *evaluation count* is O(N) with a
sub-1 constant (0.61 of N on average), the *wall clock* is worse than the naive one-shot FFT because
each refinement level pays its own FFT setup/dispatch overhead in Python/NumPy, and the levels don't
amortize. Multiple small FFT calls with per-call Python overhead beat one big FFT only when the
overhead-free complexity gap is much larger than it is here (n ≪ N with cheap per-call cost) —
that is not the regime this prototype operates in. This is exactly the "O(N) method with a
terrible constant is a loss" failure the task asked to watch for, and it materializes even before
considering that real hierarchical/FMM machinery would add more overhead, not less, at this problem
size.

---

## Summary

| Candidate | Correctness | Speed | Verdict |
|---|---|---|---|
| A: single Hilbert (`onehilb`) | Agrees with `dihiti` to ~1e-4 abs / ~1e-3 rel, but **flips sign on 5/32 profiles that sit exactly at threshold** | 1.7x | Real but modest win; inherits/exposes a genuine threshold-fragility, does not remove it |
| A+: hand-rolled rfft/irfft | Exact match to `onehilb` (diff = 0) | ~1.0x vs `onehilb` (no further win) | Not worth the complexity on this machine |
| B: decimation | Correct sign for comfortably-passing/failing profiles at n as low as 16; **wrong sign at every n < N for near-threshold profiles** (confirmed in both T3(a) and the T3(d) honesty check, no adversarial construction even needed) | 10-30x at N=2^14 | Speed claim holds; safety claim does not — unsafe exactly where it matters |
| C: adaptive coarse-to-fine | Correct verdict on all 32 test profiles | **0.61x vs baseline, 0.39x vs one-shot FFT — slower** | Negative result: loses on wall clock despite fewer evaluations |

The strongest evidence against the "nothing beats DiHiTI+min" claim is T2: a single Hilbert
transform is a real ~1.7x win with no algorithmic cleverness. The strongest evidence *for* leaving
DiHiTI alone as the default safety check is T3/T4: every attempt to go sub-linear-in-practice
(decimation, adaptive refinement) either gets the wrong answer specifically on the near-threshold
profiles that a passivity screen exists to catch, or is a net wall-clock loss once real overhead is
measured instead of assumed.
