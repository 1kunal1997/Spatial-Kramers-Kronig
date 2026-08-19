# V1-c checkpoint 2 — DONE. Clean negative, mechanistically explained (Amdahl ceiling).

## GOAL
(unchanged from checkpoint 1) V1-c: build a compiled implementation of the DiHiTI passivity
test, measure honestly against the numpy/scipy baseline, see if it clears V1's 3x bar at
N=2^14. See checkpoint-1 text below for full framing; not repeated here.

## ESTABLISHED SO FAR (final)
- No numba/Cython available; used ctypes + `cc -O3 -march=native -shared -fPIC` (clang, arm64
  Darwin). `dihiti_fast.c` / `dihiti_fast.py` / `libdihiti.so` all built and working.
- **Correctness: 327/327 (100.00%) verdict match**, incl. 147/147 near-threshold cases
  (\|min ε″\|<1e-2), against `theory/passivity_2026Aug/profile_generator.py`'s labelled corpus +
  the `opposite_bumps` crossover at N∈{2049,4001,8192}×α∈{0.0..1.5}. Max abs diff in min ε″:
  4.19e-5 (fp/quadrature noise).
- **O(N)-legs-only speedup (grad+cumtrapz+detilt+min, C vs numpy): 3.34x–3.65x**, stable across
  N∈{4096,16384,65536,60000}. Confirms red-team's "constants, not information deficit"
  hypothesis for the non-FFT part.
- **End-to-end speedup (full pipeline incl. FFT): 1.50x at N=2^14** (both logistic and
  lorentzian test profiles). Range 1.38x-1.75x across all N tested. **Below the 3x V1 bar.**
- **Amdahl diagnostic (`amdahl_diagnostic.py`, `amdahl.json`) explains WHY, quantitatively, not
  just "it didn't work":** decomposed baseline wall-clock into 4 stages. FFT (`scipy.signal.
  hilbert`) alone is 42% of baseline at N=4096, rising to 52% at N=16384, 61% at N=65536. The
  Amdahl ceiling (speedup if O(N) legs were free) is 2.37x/1.91x/1.65x/1.72x at
  N=4096/16384/65536/60000 respectively -- **already below 3x at every N≥16384, including the
  mission's own target N=2^14, even in the idealized limit.** No further C tuning on the O(N)
  legs can close this; the FFT is the binding constraint.
- **Beating scipy.signal.hilbert is the only remaining lever and it's already dead**: cycle 1
  tried a hand-rolled rfft/irfft Hilbert (`results/cycle1_c15_redteam/bench.py`'s
  `onehilb_rfft`) and found "~1.0x (no win)" once full analytic-signal reconstruction cost is
  included (`bench.md` lines 87, 233). I independently re-confirmed raw `scipy.fft.rfft` is
  ~1.4x faster than raw `scipy.fft.fft` in isolation at N=16384, consistent with cycle 1's
  finding that the *raw* transform has headroom the *usable* construction doesn't retain.
- **manifest.json** written with full params, git SHA, correctness data, timing data, the
  Amdahl decomposition, and a `headline` summary block. **notes.md** written with the full
  narrative writeup (methodology, honesty checks, recommendation for the board).
- Grid convention: `x = np.linspace(-L, L, N)`, identical array for baseline and candidate in
  every single comparison (327/327 checked pointwise). K19/C22's period-duplication bug is
  structurally inapplicable here since this candidate reimplements the IDENTICAL baseline
  arithmetic (not a periodicity-exploiting trick like `onehilb`) -- addressed explicitly in
  notes.md.

## RULED OUT
- Compiled O(N)-legs-only strategy CANNOT reach 3x end-to-end regardless of implementation
  quality, for N≥16384, because the Amdahl ceiling (FFT-bound) is already <3x there. This is a
  quantified, falsifiable ruling-out, not a hunch -- see `amdahl.json`.
- rfft/irfft hand-rolled Hilbert as a way to beat scipy.signal.hilbert: already dead from cycle
  1 (re-confirmed consistent, not re-litigated in depth here since it's explicitly out of this
  task's scope per the task brief "do not waste time there").

## CURRENTLY IN PROGRESS
Nothing -- task complete. All deliverables written: `dihiti_fast.c`, `dihiti_fast.py`,
`bench.py`, `amdahl_diagnostic.py`, `manifest.json`, `notes.md`, plus raw logs `run1.log`,
`amdahl_run.log`, `amdahl.json`. `__pycache__` cleaned up.

## NEXT STEP
For whoever picks up V1 next (likely first-mate/red-team, to decide whether to close V1-c and
V1 on the board, or to open a fresh sub-lever on beating the FFT itself with a genuinely new
idea -- not rfft/irfft, already dead):
1. Read `notes.md` headline + Amdahl table.
2. Decide BOARD.md status for V1-c: recommend "closed — clean negative, Amdahl-capped" rather
   than "open."
3. If the crew wants to keep chasing V1, the only remaining door is beating scipy's FFT itself
   (not the legs around it) -- e.g. a custom radix kernel exploiting N=2^k structure specifically,
   which is a much bigger undertaking and was NOT what this task was scoped to attempt.
4. Otherwise: V1 is now genuinely exhausted (2 cycles, 7 challenger routes, 1 compiled
   implementation, all failed with clear/quantified reasons for each) and the mission's honest
   headline (V2 achieved: Θ(N) query lower bound proved tight, log-factor barrier stated
   conditionally) stands as the final result on this axis.

## OPEN QUESTIONS / BLOCKERS
None. Task complete, no blockers.
