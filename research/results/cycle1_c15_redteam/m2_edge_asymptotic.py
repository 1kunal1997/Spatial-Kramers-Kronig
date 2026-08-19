"""M2: is the endpoint zero of eps'' structural / closed-form?

Predicted edge asymptotic near x = -L (s = x - (-L) = x + L small, s > 0):

    eps''(-L+s) = (1/pi) * [g(-L) - g(+L)] * s * ln(1/s) - (M1/(2L)) * s + O(s^2 ln s)

with g = -du/dx (u = eps', du/dx is its slope) and
    M1 = integral_{-L}^{L} g(t) tan(pi t / (2L)) dt.

We fit eps''(-L+s) = c1 * s*ln(1/s) + c2 * s over s in [1e-3 L, 1e-2 L] by
linear least squares (2 unknowns, many s samples) and compare (c1, c2) to the
predicted (1/pi)[g(-L)-g(+L)] and -M1/(2L).
"""
import numpy as np
from common import pg, tmm_h, grid, dihiti, OUTDIR
import os

L = 2.5
N = 2**18   # need very fine resolution near the seam to resolve s ~ 1e-3 L to 1e-2 L cleanly
k = 1.0 / 0.05  # logistic w = 0.05  ==  k = 1/w = 20

x = grid(L, N)
e_re = tmm_h.logistic(x, k, pg.NB)
e_im = dihiti(e_re, x)

# g = -du/dx  (u = eps')
du = np.gradient(e_re, x)
g = -du

g_mL = np.interp(-L, x, g)
g_pL = np.interp(L, x, g)
c1_pred = (1.0 / np.pi) * (g_mL - g_pL)

# M1 = integral g(t) tan(pi t / 2L) dt  -- tan blows up at t=+-L, but g -> 0
# there exponentially fast for the logistic (du/dx -> 0 at the edges), so the
# integrand is well-behaved except right at the very endpoints where we clip.
tan_arg = np.pi * x / (2 * L)
integrand = g * np.tan(tan_arg)
# guard the endpoints (tan diverges there; g should be ~0 there for logistic)
finite = np.isfinite(integrand)
M1 = np.trapz(integrand[finite], x[finite])
c2_pred = -M1 / (2 * L)

print(f"g(-L) = {g_mL:.6e}   g(+L) = {g_pL:.6e}")
print(f"predicted c1 = (1/pi)[g(-L)-g(+L)] = {c1_pred:.6e}")
print(f"M1 = {M1:.6e}   predicted c2 = -M1/(2L) = {c2_pred:.6e}")

# --- fit near x = -L ---
s_lo, s_hi = 1e-3 * L, 1e-2 * L
mask = (x >= -L + s_lo) & (x <= -L + s_hi)
s = x[mask] + L
y = e_im[mask]

A = np.column_stack([s * np.log(1.0 / s), s])
coef, res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
c1_fit, c2_fit = coef
yhat = A @ coef
ss_res = np.sum((y - yhat)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res / ss_tot

print(f"\nfit over s in [{s_lo:.3e}, {s_hi:.3e}]  (N points = {mask.sum()})")
print(f"fitted c1 = {c1_fit:.6e}   (predicted {c1_pred:.6e},  ratio = {c1_fit/c1_pred:.4f})")
print(f"fitted c2 = {c2_fit:.6e}   (predicted {c2_pred:.6e},  ratio = {c2_fit/c2_pred:.4f})")
print(f"R^2 of 2-term fit = {r2:.8f}")

# residual after subtracting the closed-form prediction (not fit) entirely,
# to see if the leading behaviour genuinely matches the CLOSED-FORM coefficients
y_pred_closed = c1_pred * s * np.log(1.0 / s) + c2_pred * s
resid = y - y_pred_closed
rel_resid = np.abs(resid) / np.abs(y).max()
print(f"\nusing the CLOSED-FORM (not fitted) coefficients directly:")
print(f"  max |residual| = {np.abs(resid).max():.4e}   max|eps''| in window = {np.abs(y).max():.4e}")
print(f"  max relative residual = {rel_resid.max():.4e}")

# also check the +L edge (should mirror by symmetry of the derivation, sign flips)
mask2 = (x <= L - s_lo) & (x >= L - s_hi)
s2 = L - x[mask2]
y2 = e_im[mask2]
A2 = np.column_stack([s2 * np.log(1.0 / s2), s2])
coef2, *_ = np.linalg.lstsq(A2, y2, rcond=None)
print(f"\n[+L edge check] fitted (c1,c2) = {coef2}")

# --- diagnose: for this deeply-saturated logistic (kL=50), g(t) is a bump of
# width ~1/k=0.05 centred at t=0, so g(+-L) ~ exp(-2kL) ~ exp(-100), i.e.
# analytically zero in double precision. If c1_pred and c2_pred are both ~0,
# the s*ln(1/s) and s terms vanish identically, and the true near-edge decay
# is governed by a HIGHER order term. Fit the power-law exponent directly:
# log|eps''| ~ p*log(s) + const, to see what order actually governs it.
logs = np.log(s)
logy = np.log(np.abs(y))
p_fit = np.polyfit(logs, logy, 1)
print(f"\npower-law check: log|eps''| = {p_fit[0]:.4f}*log(s) + {p_fit[1]:.4f}"
      f"  (exponent {p_fit[0]:.3f}; s*ln(1/s)~s^1(log-corr), s^2 would say 2)")

with open(os.path.join(OUTDIR, 'm2_power_law.txt'), 'w') as f:
    f.write(f"power-law fit over same window: log|eps''| = {p_fit[0]:.4f}*log(s) + {p_fit[1]:.4f}\n")

with open(os.path.join(OUTDIR, 'm2_summary.txt'), 'w') as f:
    f.write(f"logistic w=0.05 (k={k}), L={L}, N={N}\n")
    f.write(f"g(-L) = {g_mL:.6e}   g(+L) = {g_pL:.6e}\n")
    f.write(f"predicted c1 = (1/pi)[g(-L)-g(+L)] = {c1_pred:.6e}\n")
    f.write(f"M1 = {M1:.6e}   predicted c2 = -M1/(2L) = {c2_pred:.6e}\n")
    f.write(f"fit window s in [{s_lo:.3e}, {s_hi:.3e}] L, N_pts={mask.sum()}\n")
    f.write(f"fitted c1 = {c1_fit:.6e}  (ratio to predicted = {c1_fit/c1_pred:.4f})\n")
    f.write(f"fitted c2 = {c2_fit:.6e}  (ratio to predicted = {c2_fit/c2_pred:.4f})\n")
    f.write(f"R^2 of 2-term fit = {r2:.8f}\n")
    f.write(f"max relative residual using CLOSED-FORM coeffs directly = {rel_resid.max():.4e}\n")
    f.write(f"[+L edge] fitted (c1,c2) = {coef2}\n")


# ---------------------------------------------------------------------------
# NON-DEGENERATE validation: pure logistic (any k) is symmetric about x=0, so
# g(-L)=g(+L) and M1=0 IDENTICALLY by symmetry -- the fit above cannot tell
# "formula correct" from "formula vacuous" (both a shallow k=1 logistic and
# the deep k=20 one give c1_pred=c2_pred~0, confirmed: both underflow the
# same way). Use tanh_slopes (mL != mR, genuinely asymmetric edge ramps,
# du/dx -> mL, mR nonzero and DIFFERENT at the two edges) so the predicted
# (c1, c2) are honestly non-zero, to test the closed form for real.
print("\n" + "=" * 70)
print("non-degenerate check: tanh_slopes mL=2.0, mR=1.0 (asymmetric edge ramps)")
from profile_generator import PROFILES as _PR
_fn, _def = _PR['tanh_slopes']
_kw = dict(_def)
e_re2 = np.asarray(_fn(x, **_kw), float)
e_im2 = dihiti(e_re2, x)
du2 = np.gradient(e_re2, x)
g2 = -du2
g2_mL = np.interp(-L, x, g2)
g2_pL = np.interp(L, x, g2)
c1_pred2 = (1.0 / np.pi) * (g2_mL - g2_pL)
integrand2 = g2 * np.tan(np.pi * x / (2 * L))
finite2 = np.isfinite(integrand2)
M1_2 = np.trapz(integrand2[finite2], x[finite2])
c2_pred2 = -M1_2 / (2 * L)
print(f"g(-L)={g2_mL:.6e}  g(+L)={g2_pL:.6e}  c1_pred={c1_pred2:.6e}  M1={M1_2:.6e}  c2_pred={c2_pred2:.6e}")

mask_ = (x >= -L + s_lo) & (x <= -L + s_hi)
s_ = x[mask_] + L
y_ = e_im2[mask_]
A_ = np.column_stack([s_ * np.log(1.0 / s_), s_])
coef_, *_ = np.linalg.lstsq(A_, y_, rcond=None)
c1_fit2, c2_fit2 = coef_
yhat_ = A_ @ coef_
r2_2 = 1 - np.sum((y_ - yhat_)**2) / np.sum((y_ - y_.mean())**2)
print(f"fitted c1={c1_fit2:.6e} (ratio {c1_fit2/c1_pred2:.4f})  "
      f"fitted c2={c2_fit2:.6e} (ratio {c2_fit2/c2_pred2:.4f})  R^2={r2_2:.6f}")

with open(os.path.join(OUTDIR, 'm2_summary.txt'), 'a') as f:
    f.write("\n--- non-degenerate check: tanh_slopes mL=2.0 mR=1.0 ---\n")
    f.write(f"g(-L)={g2_mL:.6e}  g(+L)={g2_pL:.6e}  c1_pred={c1_pred2:.6e}\n")
    f.write(f"M1={M1_2:.6e}  c2_pred={c2_pred2:.6e}\n")
    f.write(f"fitted c1={c1_fit2:.6e} (ratio to predicted {c1_fit2/c1_pred2:.4f})\n")
    f.write(f"fitted c2={c2_fit2:.6e} (ratio to predicted {c2_fit2/c2_pred2:.4f})\n")
    f.write(f"R^2={r2_2:.6f}\n")

print("\nwrote m2_summary.txt, m2_power_law.txt")
