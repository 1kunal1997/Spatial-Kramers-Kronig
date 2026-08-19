# handA — independent check: is eps'' >= 0 for the 4-atom lam-family?

Files: `handA_staircase.py` (primary), `handA_lamscan.py` (threshold + 2nd evaluator),
`handA_results.json`. No existing file was modified.

## Construction (deliberately different from the rest of this directory)

* Smoothing: **Gaussian-CDF (erf) step**, `eps'(x) = -sum_i w_i * 0.5*(1+erf((x-t_i)/(s*sqrt2)))`,
  default s = 0.01 (not raised cosine, not tanh). s-sweep 0.0025 ... 0.02 also run.
* Grids (both tested, stated explicitly):
  * `periodic`: `x_j = -L + 2L*j/N`, j = 0..N-1 — right endpoint **excluded**, spacing 2L/N.
  * `linspace`: `np.linspace(-L, L, N)` — endpoint duplicated under the FFT.
* N = 1024, 4096, 16384 (span 16x) plus **10000 and 6250** (non-powers of two).
* eps'' from the production `tmm_helper.ht_derivative`.
* Second, fully independent evaluator (hand-derived, `handA_lamscan.kernel`):
  `eps''(x) = (1/pi) sum_i w_i ln[ cos(pi t_i/2L) / |sin(pi(x-t_i)/2L)| ]`.
* Reported minimum: over `|x| <= 0.9999 L` excluding `|x - t_i| <= 0.06` at each step
  (exclusion radius swept 0.02 ... 0.15 — result unchanged).

## M1 check

`M1 = sum_i w_i tan(pi t_i / 2L)` (the +-0.2 pair cancels exactly, so M1 is linear in lam):

| lam | weights (t = -0.40, -0.20, +0.20, +0.90) | M1 |
|---|---|---|
| 0.1 | 0.08968, 0.45, 0.45, 0.01032 | 1.5817e-06 |
| 0.2 | 0.17936, 0.40, 0.40, 0.02064 | 3.1634e-06 |
| 0.5 | 0.44840, 0.25, 0.25, 0.05160 | 7.9086e-06 |
| 1.0 | 0.89680, 0.00, 0.00, 0.10320 | 1.5817e-05 |

Zero to ~2.4e-5 in relative terms (individual terms are O(0.65)). Confirmed "high accuracy",
not exact.

## Result — minimum of eps'' (erf s = 0.01, exclusion 0.06, |x| <= 0.9999 L)

| lam | N | periodic: min | argmin | linspace: min | argmin |
|---|---|---|---|---|---|
| 0.1 | 1024 | 0.000e+00 | 0.9980 | +4.495e-06 | -0.9980 |
| 0.1 | 4096 | -1.350e-08 | -0.9995 | +1.641e-07 | -0.9995 |
| 0.1 | 6250 | -4.185e-08 | -0.9997 | +3.455e-08 | -0.9997 |
| 0.1 | 10000 | -4.088e-08 | -0.9998 | -1.101e-08 | -0.9998 |
| 0.1 | 16384 | -4.326e-08 | -0.9998 | -2.101e-08 | -0.9998 |
| **0.2** | 1024 | 0.000e+00 | 0.9980 | +5.678e-06 | -0.9980 |
| **0.2** | 4096 | -1.305e-07 | -0.9995 | +1.212e-07 | -0.9995 |
| **0.2** | 6250 | -1.282e-07 | -0.9997 | -1.984e-08 | -0.9997 |
| **0.2** | 10000 | -1.346e-07 | -0.9996 | -5.677e-08 | -0.9998 |
| **0.2** | 16384 | -1.331e-07 | -0.9996 | -8.574e-08 | -0.9996 |
| 0.5 | 1024 | -1.947e-02 | 0.7207 | -1.920e-02 | 0.7204 |
| 0.5 | 4096 | -1.946e-02 | 0.7207 | -1.940e-02 | 0.7211 |
| 0.5 | 6250 | -1.946e-02 | 0.7210 | -1.942e-02 | 0.7209 |
| 0.5 | 10000 | -1.946e-02 | 0.7210 | -1.943e-02 | 0.7210 |
| 0.5 | 16384 | -1.946e-02 | 0.7209 | -1.945e-02 | 0.7209 |
| 1.0 | 1024 | -1.006e-01 | 0.5000 | -1.001e-01 | 0.5015 |
| 1.0 | 4096 | -1.006e-01 | 0.5000 | -1.005e-01 | 0.4999 |
| 1.0 | 6250 | -1.006e-01 | 0.4998 | -1.005e-01 | 0.5001 |
| 1.0 | 10000 | -1.006e-01 | 0.5000 | -1.006e-01 | 0.4999 |
| 1.0 | 16384 | -1.006e-01 | 0.5000 | -1.006e-01 | 0.5000 |

**Grid convention: no disagreement.** periodic vs linspace differ by <2e-7 (lam=0.1,0.2)
and <6e-4 relative (lam=0.5,1.0); no sign flips, same argmin. The known linspace bug is not
in play for this construction.

**N-scaling.** lam = 0.1, 0.2: the minimum is a *vanishing* edge artifact — it sits at the
last retained node (|x| -> 0.9999 as N grows), and its magnitude is O(1e-7), shrinking with
the smoothing width (s = 0.02: -2.6e-6; 0.01: -1.3e-7; 0.005: -7.9e-9; 0.0025: exactly 0).
Cutting the edge back: min over |x| <= 0.99 is **+7.12e-05**, over |x| <= 0.98 **+2.91e-04**,
over |x| <= 0.95 **+1.76e-03** — all positive and growing inward, i.e. eps'' decreases to its
enforced zero at +-L. lam = 0.5, 1.0: the minimum is **N-independent to 3-4 digits** at a
fixed interior location (x ~ 0.721 and x = 0.500) — a genuine negative excursion, not noise.
Exclusion-radius sweep (0.02 ... 0.15) changes nothing, so it is not log-spike leakage.

## Cross-check: production ht_derivative vs hand-derived log kernel (|x| <= 0.99)

| lam | ht min @ x | kernel min @ x | abs diff |
|---|---|---|---|
| 0.1 | +5.7353e-05 @ -0.9900 | +6.0149e-05 @ -0.9900 | 2.8e-06 |
| 0.2 | +7.1188e-05 @ -0.9900 | +7.6794e-05 @ -0.9900 | 5.6e-06 |
| 0.3 | -1.4838e-03 @ +0.7919 | -1.4767e-03 @ +0.7922 | 7.1e-06 |
| 0.5 | -1.9462e-02 @ +0.7209 | -1.9402e-02 @ +0.7211 | 6.0e-05 |
| 1.0 | -1.0062e-01 @ +0.5000 | -1.0045e-01 @ +0.5000 | 1.7e-04 |

The two routes agree; the residual is the erf-smoothing/discretization error.

## lam threshold

Interior minimum is positive for lam <= 0.25 and negative from lam = 0.30 on. Bisection on a
dense (4e5-point) kernel probe over |x| < 0.999 gives the sign change at

  **lam* = 0.277754**,  with the first negative dip appearing near **x ~ +0.80**.

## Answer

**For lam = 0.2 (and lam = 0.1): YES, eps'' is non-negative** everywhere away from the steps —
min over |x| <= 0.99 is +7.1e-05 (lam = 0.2), positive on every N and both grid conventions;
the only negative values are O(1e-7) at the last node before |x| = L and they vanish as the
smoothing width shrinks.

**For lam = 0.5 and lam = 1.0: NO.** eps'' dips to -1.946e-02 at x = +0.721 and -1.006e-01 at
x = +0.500 respectively — resolution-independent, evaluator-independent, interior.

Note for the audit: the drop measure has support [-0.40, +0.90], i.e. a + b = 1.30 > L = 1,
yet lam = 0.2 (with M1 = 0) is passive. So on this family the condition "a + b <= L" is **not
necessary** for eps'' >= 0; the true boundary within the family is lam* ~ 0.2778.
