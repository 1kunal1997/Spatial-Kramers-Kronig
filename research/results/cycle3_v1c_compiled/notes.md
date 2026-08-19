# V1-c — compiled DiHiTI kernel: result

**Verdict: V1 NOT achieved. Clean, quantified negative, with a rigorous reason it cannot be
fixed by writing better C.**

## Headline numbers

| | |
|---|---|
| Correctness | **327/327 (100.00%)** verdict match vs. numpy/scipy baseline, incl. **147/147** near-threshold cases (\|min ε″\| < 1e-2). Max abs diff in min ε″: 4.19e-5 (fp/quadrature noise, not a discrepancy). |
| **O(N)-legs-only speedup** (grad + cumtrapz + de-tilt + interior-min, C vs numpy) | **3.34x – 3.65x**, stable across N ∈ {4096, 16384, 65536, 60000(non-pow2)}. |
| **End-to-end speedup** (full pipeline, FFT included, C vs numpy) | **1.50x at N=2^14** (both `logistic` and `lorentzian` test profiles). Range across all N tested: **1.38x – 1.75x**. |
| Compile time | 301 ms, one-time, amortized (a real deployment compiles once, calls many times). |

**The 3x bar is not met.** But this is not a failure of the compiled code — the O(N) legs alone
hit 3.3–3.65x, squarely confirming red-team's "every loss has been a Python constant" hypothesis
*for that part of the pipeline*. The reason the end-to-end number stalls at 1.5x is structural,
not an implementation shortfall, and is quantified below.

## Why: Amdahl's law, not a weak kernel

I decomposed baseline wall-clock into its four stages (`amdahl_diagnostic.py`,
`amdahl.json`) at N=2^14:

| stage | numpy time (ms) | fraction of total |
|---|---|---|
| `np.gradient` | 0.087 | 19% |
| `scipy.signal.hilbert` (FFT) | 0.235 | **52%** |
| `cumulative_trapezoid` | 0.087 | 19% |
| de-tilt + mask + min | 0.040 | 9% |

The FFT alone is over half of baseline wall-clock at N=2^14, and its share only *grows* with N
(60% at N=65536). Per the task brief, the FFT stays in scipy — it is already compiled and (per
cycle1's own prior attempt, see below) not obviously beatable. That means **the ceiling on any
"compile the O(N) legs, leave the FFT alone" strategy is Amdahl's law**:

    ceiling(N) = numpy_total(N) / fft_time(N)

i.e. the speedup you would get if the O(N) legs became *free* (cost → 0). Measured:

| N | FFT % of baseline | Amdahl ceiling (O(N) legs free) | measured end-to-end |
|---|---|---|---|
| 4096 | 42% | **2.37x** | 1.72–1.75x |
| 16384 | 52% | **1.91x** | 1.50x |
| 65536 | 61% | **1.65x** | 1.40–1.43x |
| 60000 | 58% | **1.72x** | 1.38–1.57x |

**This is the load-bearing finding: for N ≥ 16384 — which includes the mission's own target
N=2^14 — the Amdahl ceiling itself is already below 3x, even in the idealized limit where the
O(N) legs cost nothing at all.** No amount of further C tuning on the derivative/cumtrapz/de-tilt/
min-scan can close that gap while the FFT stays untouched. My measured 1.50x sits at 79% of the
1.91x ceiling at N=2^14 — the remaining C code is not leaving much on the table (see "further
headroom" below), but the ceiling itself is the binding constraint.

Reaching 3x end-to-end would require beating `scipy.signal.hilbert` itself. **This was already
tried and killed in cycle 1**: `results/cycle1_c15_redteam/bench.py`'s `onehilb_rfft` (hand-rolled
`rfft`/`irfft` Hilbert transform) gives "~1.0x (no win)" once the full analytic-signal
reconstruction cost is included (`results/cycle1_c15_redteam/bench.md` lines 87, 233) — the
savings from using half-spectrum rfft are eaten by the reconstruction back to a full complex
result. I independently re-checked the raw transform costs in isolation (not the full
reconstruction) and confirm `scipy.fft.rfft` alone is ~1.4x faster than `scipy.fft.fft` alone at
N=16384 — consistent with cycle 1's finding that the *raw* transform has headroom but the
*usable* Hilbert-transform construction does not, once you pay for turning the half spectrum back
into the full one scipy.signal.hilbert needs internally.

## What was built

- `dihiti_fast.c` — two fused C kernels, compiled with `cc -O3 -march=native -shared -fPIC`
  (no numba or Cython installed on this machine — checked first, both `ModuleNotFoundError`):
  - `grad_fast(u, dx, N, out)` — central-difference derivative on a uniform grid, replicating
    `np.gradient`'s default (`edge_order=1`) exactly.
  - `post_min(v, dx, N, lo, hi)` — cumulative-trapezoid integral + de-tilt (subtract the
    `linspace(e[0],e[-1],N)` ramp) + interior-window minimum, **fused into two passes with zero
    numpy temporary allocations**, vs. numpy's ~4 separate array-length passes
    (`cumulative_trapezoid`, `linspace`, subtract, boolean-mask `.min()`).
  - `post_full` — same computation but materializes the full de-tilted array, used only to
    verify the fused kernel against `scipy.integrate.cumulative_trapezoid` pointwise.
- `dihiti_fast.py` — ctypes wrapper; `verdict_compiled(x, e_re, frac)` (candidate) and
  `verdict_baseline(x, e_re, frac)` (numpy/scipy reference, identical arithmetic to
  `tmm_helper.ht_derivative` + `results/cycle1_c15_redteam/bench.py`'s `dihiti()` +
  `interior_min()`, so this is not a strawman baseline).
- `bench.py` — correctness sweep (300 random draws from
  `theory/passivity_2026Aug/profile_generator.py`'s `sample()` across every profile family, plus
  the `opposite_bumps` crossover — the MISSION.md seed figure — at N ∈ {2049, 4001, 8192} ×
  α ∈ {0.0 … 1.5}, the mission's own near-threshold reference) and timing sweep (N ∈
  {2^12, 2^14, 2^16, 60000 (non-power-of-two)} × {logistic, lorentzian}, 41 reps, median + MAD,
  one untimed warm-up call first).
- `amdahl_diagnostic.py` — the four-stage timing decomposition above.
- `manifest.json` — full machine-readable record (params, git SHA, all timing/correctness data,
  the Amdahl decomposition, and a `headline` block).

## Grid convention (K19/C22 hazard, addressed explicitly)

`x = np.linspace(-L, L, N)` for **both** baseline and candidate, always the same array object in
every comparison. K19/C22 killed a prior 1.70x "speedup" that was really a period-duplication
artifact from this exact `linspace` idiom — but that bug specifically poisoned a candidate
(`onehilb`) that *exploits* the periodicity assumption via a ramp-periodization trick. This
candidate does no such thing: it reimplements the *identical* baseline arithmetic (gradient,
cumtrapz, de-tilt, min), fused into fewer passes, and is verified pointwise against the numpy
baseline on the same array on every single test case (327/327 match). A period-error artifact is
structurally impossible here — there is only one grid, used identically on both sides, and the
outputs are checked to agree to ~1e-13–1e-5 absolute, not merely "the verdict came out right."

## Methodology honesty checks

- Timing: `time.perf_counter()`, one untimed warm-up call (JIT-irrelevant here, but this
  eliminates first-call page-fault/cache-cold effects for both sides equally), 41 timed reps per
  configuration, report median and median-absolute-deviation (robust to outlier scheduler
  hiccups). MAD is consistently ≤2% of the median across all configurations — the numbers are
  stable, not noise-dominated.
- Compile time (301 ms) is reported separately and is a one-time cost — realistic use (many
  profiles checked per process) amortizes it to ~0.
- Correctness checked on the SAME grid used for timing, not a separate "looks right" spot check:
  every one of the 327 correctness-sweep profiles gets both a baseline and a compiled verdict,
  compared exactly (PASS/GAIN agreement) and numerically (abs diff in min ε″).
- Near-threshold coverage: 147/327 correctness cases have \|min ε″\| < 1e-2 (close to the passive
  boundary per MISSION.md's warning that every DiHiTI profile sits near it) — all 147 matched.

## Further headroom (not chased — out of scope / already dead)

- `post_min` currently does two full passes over `v` (one to get the total for the de-tilt slope,
  one fused detilt+min) rather than one, because the slope needs the total before any element can
  be de-tilted. A single-pass version would need to store the running cumsum to a reusable
  pre-allocated buffer and do the de-tilt+min in the *same* pass by paying a second, cheaper,
  cache-resident sweep — this is what `post_min` already effectively does (2 passes, 0
  allocations) since re-accumulating from scratch avoids a malloc/free on every call. I do not
  expect a rewrite to change the Amdahl-capped headline (the O(N) legs are already 3.3–3.65x
  faster and are not the bottleneck at N ≥ 16384).
- Beating `scipy.signal.hilbert` itself is the only route left to close the remaining gap to 3x,
  and it is explicitly out of this task's scope ("do not waste time there") and already tried and
  killed in cycle 1 (`onehilb_rfft`, ~1.0x). I would not recommend re-opening it without a new
  idea beyond hand-rolled rfft/irfft, which cycle 1 already exhausted.

## What this result does and does not show

**Does show:** the "every loss is a Python constant" diagnosis was correct for the non-FFT 80–90%
of the pipeline's *code paths* — a straightforward ctypes C kernel, with no algorithmic cleverness,
gets a robust ~3.5x there with zero correctness cost. **Does not show:** that DiHiTI passivity
testing can be sped up 3x end-to-end by any means that leaves the FFT untouched, because the FFT is
not a small fraction of the work — it is over half of it at the mission's own target N=2^14, and
that fraction *grows* with N. The V1-c lever is now exhausted: two cycles, seven challenger
routes, and a compiled implementation have all failed to reach 3x, and this cycle explains
*exactly* why the compiled route in particular cannot, with a number (the Amdahl ceiling) rather
than an assertion.

**Recommendation for the board:** V1-c should be marked closed/negative with this Amdahl-ceiling
argument as the reason, not just "tried, didn't work." If V1 is to be pursued further, the only
remaining lever is beating `scipy.signal.hilbert` itself (already attempted once, failed), which
would need a genuinely new idea, not a retry of rfft/irfft. Absent that, C15 (no cheaper decision
procedure exists in practice) stands, and the mission's honest headline from cycle 2 — "the
transform is optimal up to a log factor, Θ(N) queries proved tight, and now: the wall-clock
constant is Amdahl-capped by the FFT at every N tested" — is complete.
