# RED TEAM — Cycle 2 — TASK 1: E1(a), "move the cascade onto the decimated grid"

**Target: my own cycle-1 proposal.** BOARD.md rated E1(a) "highest value on the entire board,
~20 lines, ~1 hour", predicting 10–30x and the outright death of C15.

## VERDICT: **E1(a) FAILS. Clean negative. C15 stands stronger than it did after cycle 1.**

E1(a) does not merely underperform — its central premise is false. The screen that made cycle 1's
split test look safe is a **full-grid** quantity; on the decimated grid it fails in both directions
at once. And in chasing the repair I found a **grid-convention bug in my own cycle-1 claim C22**
that inflates its headline speedup.

**V1 is NOT achieved.** Best measured configuration on the mission's grid, against the honest best
baseline, is **2.86x with an UNSOUND certificate** — below the 3x bar, and not certifiable.

Files: `e1a.py`, `e1a2.py`, `e1a3.py`, `e1a4.py`, `e1a5.py`, `diag_onehilb.py`, `diag_period.py`,
`diag_grid.py`; raw JSON `e1a2_N65536.json`, `e1a3_N65536.json`, `e1a4_N65536.json`,
`e1a5_N65536.json`. Full resumable state in `CHECKPOINT.md` (findings F1–F8).

---

## 1. The first thing I found was that cycle 1 misreported its own test

`results/cycle1_c15_redteam/split_test.py`: `split_verdict()` returns the decimated minimum and
nothing else. `cascade()` is computed, printed in a table, and **never gates the answer.**

So cycle 1's headline — *"20/24 sign-correct, and the 4 misses are only the adversarial cosines"* —
is the accuracy of **decimation alone**. The cascade contributed **nothing** to accuracy while
costing the O(N) transcendental pass that made the test 0.88x. And the report's claim that "the
cascade catches exactly the adversary that defeats decimation" was **indicated by a side table,
never wired in, never tested**. That was my error and it is the reason E1(a) looked cheap.

## 2. Why E1(a) fails: the screen is a full-grid quantity (F3)

Screen = |g(−L) − g(+L)|/D. Full grid N = 2^16 vs decimated n = 512:

| family | at N = 2^16 | at n = 512 |
|---|---|---|
| logistic w=0.05 | 1.5e−11 | 3.5e−09 |
| logistic w=0.30 | 8.3e−12 | **1.4e−03** |
| bumps, all α | ~1.2e−11 | **2.3e−03 … 5.7e−03** |
| **cos m=257 (adversary)** | **3.98e−01** | **3.5e−09** |
| cos m=512 / 1024 / 2048 | 1.6 / 6.3 / 25 | 2.4e−05 / 9.6e−05 / 3.8e−04 |

The ten-orders-of-magnitude separation cycle 1 leaned on **is a property of the full grid**. Decimate
and it collapses two ways simultaneously:

- **False DEFERs.** A truncation floor ~h_n²·‖g‴‖ ≈ 1e−3·D contaminates every smooth profile, so the
  screen can no longer tell a logistic from an adversary.
- **False ANSWERS.** High-m dressings *alias away*. cos m=257 reads **3.5e−9 on the decimated grid —
  cleaner than the logistic** — sails through the screen, and the decimated verdict is then wrong in
  sign (+4.5e−2 where the truth is −1.9e−2).

The screen is blind to precisely the adversary it exists to catch. This is not an engineering
constant; it is the information-theoretic content of the decimation.

## 3. And no threshold rescues it: the screen is a defer machine (F4)

`e1a2.py`, N = 2^16, **230 profiles** (5 logistic, 10 bumps incl. 3 within 0.3% of the C18 threshold,
200 random smooth, 15 cosine dressings m = 63…4096). Ground truth `dihiti` at N. A DEFER costs a
full baseline run. Sweep n ∈ {128…2048} × τ ∈ {1e−5…∞}:

| variant | n | τ | false answers | defer % | E[t] | vs B1 |
|---|---|---|---|---|---|---|
| Sd (E1a) | 2048 | 1e−4 | **0** | 86.1% | 1014 µs | **1.11x** |
| Sd (E1a) | 1024 | 1e−5 | **0** | 92.2% | 1069 µs | 1.06x |
| Sf (cycle 1, screen wired) | 2048 | 1e−3 | **0** | 73.0% | 1141 µs | 0.99x |
| Sd (E1a) | 512 | ∞ (no screen) | 9 | 0% | 69 µs | 16.3x |

**Every configuration with zero false answers is ≤ 1.11x.** Every configuration above 3x admits false
answers. The mechanism is structural: tightened enough to be safe, the screen defers 73–92% of
inputs, and a defer costs the full baseline, so E[t] → t_baseline by construction.

## 4. I then tried to build the sound certificate cycle 1 asked for (E3). It is unsound — twice.

I was ordered not to repeat the K13 class of error, so I state exactly where this one broke.

**The K13 fix is real.** K13 died because repeated **centred** differences have symbol i·sin θ/h,
which *vanishes* at θ = π, under-estimating the Nyquist modes it must bound. **Forward** differences
do not: the symbol modulus is 2|sin(θ/2)|/h and the ratio to the true |m|π/L is sinc(θ/2) ∈ [2/π, 1],
never zero. Hence, soundly,

> (B1) ‖p^(k)‖₂ ≤ (π/2)^k ‖D_h^k p‖₂ / h^k  
> (B2) S_M := Σ_{|m|≥M}|p̂_m| ≤ (L/π)^k ‖p^(k)‖₂ · √(2/((2k−1)M^{2k−1}))  
> (B3) ‖ε″_dec − ε″‖_∞ ≤ 4 S_M

**B1 and B2 are sound** — I checked the bound against the exact FFT tail on 48 audited rows; the
bound exceeds the truth every time, minimum ratio 13.9. **B3 is false**, and it failed for two
different reasons in succession:

- **First failure (F5, `e1a3.py`, 13/48 violations).** The error of `dihiti` on a coarse grid is not
  aliasing at all — it is the O(h²) truncation error of the pipeline's own low-order operators
  (`np.gradient`, `cumulative_trapezoid`). Logistic w=0.05, M=256: exact spectral tail **4.9e−13**,
  actual decimation error **1.9e−4** — *nine orders* apart. **Cycle 1's "decimation is sign-correct
  at n=64 for every profile" is therefore a statement about coarse-grid quadrature error happening to
  be small and same-signed, not about band-limitation.** It cannot be certified in that form.
- **Second failure (F8, `e1a5.py`).** Repaired by using a spectrally exact engine *and* the corrected
  period (§5). B3 still fails, up to 33/90 profiles, worst ratio 1.2e4. Residual unbounded term: the
  decimated ε″ is minimised over the *n decimated abscissae* while the truth is minimised over N; for
  oscillatory profiles the continuum minimum falls between decimated points, and that gap is bounded
  by no coefficient-tail sum. Closing it needs a zero-padded refinement plus a Bernstein term — which
  puts the cost back.

## 5. Collateral: a grid-convention bug in my own C22 (F7) — C22 must be NARROWED

`np.linspace(-L, L, N)` **duplicates the endpoint**. The ramp-periodized p satisfies p[0] = p[−1], so
handing all N samples to `scipy.signal.hilbert` declares the period to be 2L·N/(N−1), not 2L. Fix:
transform `p[:-1]`.

Consequences, both of which correct cycle 1:

- **C22's algebraic identity is exact, better than claimed.** Corrected full-N `onehilb` gives
  +1.004834e−3 against `dihiti`'s +1.0054e−3 (4 digits); the buggy version gave +9.632965e−4. The
  "rel. rms 1.6e−4" residual, and the 5/32 sign disagreements cycle 1 called "the deepest crack in
  C15" (attack 2.2), are **this bug**, not a property of the transform. *That crack should be
  withdrawn — a second point for C15, after hand B's M3 already withdrew the first.*
- **C22's 1.70x speed win does not survive on the mission's grid.** N−1 = 65535 = 3·5·17·257 is
  FFT-hostile:

| N | dihiti | onehilb(N−1, correct) | onehilb(N, buggy) | grid B dihiti | grid B onehilb (correct) |
|---|---|---|---|---|---|
| 4096 | 127.8 µs | 75.8 µs (1.69x) | 59.5 µs (2.15x) | 110.5 µs | 59.8 µs (1.85x) |
| 16384 | 444.1 µs | 839.7 µs (**0.53x**) | 248.7 µs (1.79x) | 383.7 µs | 248.7 µs (1.54x) |
| 65536 | 2052.1 µs | 4705.9 µs (**0.44x**) | 1197.5 µs (1.71x) | 1774.5 µs | 1199.7 µs (1.48x) |

  (grid B = the proper periodic grid x = −L + 2Lk/N, N samples = one period, power of two.)

> **Narrowed C22:** *DiHiTI equals one Hilbert transform of the ramp-periodized profile — exactly,
> not to 1.6e−4. The 1.5–1.9x speed win requires a periodic grid of FFT-friendly length; on the
> mission's `linspace(-L,L,N)` convention the correct implementation is 0.44x, i.e. slower than the
> incumbent.*

## 6. Honest best number, stated against the honest best baseline

On the mission's grid, the fastest baseline is `dihiti` itself (1754 µs at N = 2^16), because the
corrected `onehilb` is slower there. Best measured split configuration (`e1a5.py`, n = 4096, k = 3):
93.3% certified, 0 false answers on 90 profiles, E[t] = 612 µs = **2.86x**.

- Below V1's 3x bar.
- The "certificate" gating it is **unsound** (§4), so "0 false answers" is empirical luck.
- On grid B the ratio would improve, but so would the baseline; the honest comparison is like-for-like.

## 7. Attacks attempted, and what happened

| # | Attack on E1(a) | Outcome |
|---|---|---|
| B1 | Does the cascade survive decimation? | **E1(a) KILLED.** Screen floor + aliasing; blind to its own adversary (§2). |
| B2 | Is there any (n, τ) both safe and ≥3x? | **No.** 70 configurations swept; safe ⇒ ≤1.11x (§3). |
| B3 | Was cycle 1's accuracy figure attributable to the cascade? | **No** — the cascade was never wired in (§1). |
| B4 | Can a sound tail certificate rescue it? | **No.** B1/B2 sound, B3 false twice, for two different reasons (§4). |
| B5 | Adversarial cosine dressings m = 63…4096 | Decimation misses 4–9 of 15; the decimated screen never catches them (§2, §3). |
| B6 | Near-threshold physical cases (α = 0.6640/0.6662/0.6680 around C18's α_c = 0.66623) | Handled correctly by decimation, but only via the *uncertified* quadrature accident of §4. |
| B7 | Is the "best baseline" really best? | **NARROWED C22** — the 1.70x is an artifact of a duplicated endpoint (§5). |

## 8. Ledger effect

- **E1(a) → KILLED.** Reason: the edge-slope screen is a full-grid quantity; on the decimated grid it
  simultaneously acquires a 1e−3·D truncation floor (false defers on smooth profiles) and aliases the
  adversarial dressings to zero (false answers). No (n, τ) is both safe and ≥3x.
- **H2 (bandwidth not grid) → demote from top of board.** Its only measured win is uncertified, and
  the certificate route is now understood to require bounding a between-sample quantity from samples,
  which is exactly what §4 and the V2a argument say cannot be done.
- **C22 → NARROWED** as in §5. The identity is *exact*; the speed claim is grid-dependent.
- **Cycle-1 attack 2.2 ("the incumbent has no well-defined answer at the threshold") → WITHDRAWN.**
  It was the endpoint-duplication bug. Hand B's M3 had already withdrawn the grid-convergence half.
- **C15 → still `conjecture`, but STRONGER.** Both of cycle 1's live cracks against it are now
  withdrawn, and its practical core has now defeated **six** named challenger routes: H3
  branch-and-bound (0.39x), rfft/irfft (0.98x), the Sobolev certificate (unsound), the cycle-1 split
  test (0.88x), the screened split test (≤1.11x safe), and E1(a) (killed). The narrowed statement
  from cycle 1 stands; the phrase "nothing beats" remains unearned as a *theorem*, but as an
  engineering claim at N ≲ 2^16 in NumPy it is now well supported.

## 9. What would still change my mind

1. **A compiled implementation.** Every loss in this cycle and the last is a Python/NumPy constant,
   not an information deficit. The split test at n = 4096 does ~10⁻² of the baseline's arithmetic and
   still loses. A numba/C version is the one untested lever, and it is cheap.
2. **A certified decimated test built the other way round**: refine the decimated ε″ by zero-padded
   irfft, add a Bernstein term for the between-sample minimum, and use B1/B2 (which ARE sound) for
   the tail. This is the only route I can see to a *sound* sub-baseline test, and I did not run it.
3. **The problem being restated on grid B.** The mission's `linspace(-L,L,N)` convention is the wrong
   discretization of a periodic problem and it is distorting every timing comparison in this project.
   Recommend the Captain adopt x = −L + 2Lk/N.
