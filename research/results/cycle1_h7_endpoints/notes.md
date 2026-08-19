# H7 — V5 endpoint verification: DONE

## Verdict: V5 SUPPORTED (eps''(+/-L)=0 automatic in the DiHiTI gauge)

Ran `check_endpoints.py`: 6 profile families (logistic at k=1,4,10,30; a symmetric plateau
[a=0.5L]; an off-centre tanh step [x0=0.3L]; and the alpha=1.5 opposite-bumps gain-carrying case
from H6 — deliberately non-passive) x 6 grid sizes N in {2001,...,64001}. Results + full numbers
in `manifest.json`.

- **detilt(+/-L) is float64-exact 0.0** for every case — trivially true by construction (the code
  literally subtracts the linear interpolant between the raw endpoints).
- **The real test is the RAW (pre-detilt) endpoint mismatch.** For smooth profiles (no derivative
  discontinuity: all logistic widths, the gain-carrying case) the raw mismatch sits at the float64
  noise floor (~1e-15) at every N, no trend. For profiles with a sharp interior feature (plateau
  edge, steep off-centre step) the raw mismatch is nonzero but decays cleanly as **O(1/N)**
  (measured: exact halving each time N doubles, e.g. plateau_a0.5L: 8.90e-4 -> 4.45e-4 -> 2.23e-4
  -> 1.11e-4 -> 5.56e-5 -> 2.78e-5 across N=2001..64001) — a quadrature/Gibbs artifact from the
  FFT-Hilbert + cumtrapz steps near a discontinuity, extrapolating to exactly 0.

**Conclusion**: the raw (un-detilted) DiHiTI kernel already satisfies K(+/-L;t)=0 in the continuum
limit. The de-tilt genuinely is "a no-op that only absorbs discretization error" — for smooth
profiles there is nothing to absorb; for profiles with sharp features it absorbs a vanishing O(1/N)
artifact, not a real O(1) physical effect. No tested family (including a genuinely non-passive,
gain-carrying one) broke this. V5 is free — it does not need to be imposed as an extra constraint.

**Caveat**: only V5 itself (endpoint value) was tested, not whether a nonzero raw mismatch at finite
N could shift OTHER quantities (e.g. min(eps'') used for the H6 crossover) by the same O(1/N)
amount before it vanishes — worth keeping in mind for near-threshold alpha_c determinations at low
N, though H6's own N-convergence check (drift <1e-5 from N=1e4 to 1.6e5) already shows this is not
an issue for that result.

## Files
- `check_endpoints.py` — script, self-contained, run and results captured
- `manifest.json` — full per-family/per-N numbers + headline verdict
