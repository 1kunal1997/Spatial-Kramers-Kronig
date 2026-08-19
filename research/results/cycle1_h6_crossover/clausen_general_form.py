import sys, os
_PROJECT_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
import tmm_helper as tmm_h
import json, subprocess, datetime

nb = 1.7; k_steep = 4.0; L = 2.5

# --- Clausen Cl2, copied (read-only reuse) from theory/passivity_2026Aug/verify_passivity.py ---
_M = 2**20
_k = np.arange(1, _M//2)
_spec = np.zeros(_M, complex)
_spec[1:_M//2] = 1.0/_k**2/(2j)
_spec[-1:-_M//2:-1] = -1.0/_k**2/(2j)
_grid = np.fft.ifft(_spec).real*_M
_tg = 2*np.pi*np.arange(_M)/_M
def clausen2(t):
    return np.interp(np.mod(np.asarray(t, float), 2*np.pi), _tg, _grid, period=2*np.pi)

def M_kernel(alpha, theta):
    """M(alpha,theta) = Cl2(theta+alpha) - Cl2(theta-alpha) - Cl2(pi+alpha) + Cl2(pi-alpha)"""
    return (clausen2(theta+alpha) - clausen2(theta-alpha)
            - clausen2(np.pi+alpha) + clausen2(np.pi-alpha))

def ht_d1(x_, e_re_):
    u = np.gradient(e_re_, x_)
    v = np.imag(hilbert(u))
    e_im_ = cumulative_trapezoid(v, x_, initial=0)
    e_im_ -= np.linspace(e_im_[0], e_im_[-1], len(e_re_))
    return e_im_

def build(alpha, N):
    xx = np.linspace(-L, L, N)
    window = ((1 + np.cos(np.pi * xx / L)) / 2)**3
    sigma_b = 0.5
    bump_L_loc = np.exp(-((xx + 1.0) / sigma_b)**2) * window
    bump_R_loc = np.exp(-((xx - 1.0) / sigma_b)**2) * window
    bump_L_loc /= bump_L_loc.max()
    bump_R_loc /= bump_R_loc.max()
    e_re = tmm_h.logistic(xx, k_steep, nb) + alpha * (bump_L_loc - bump_R_loc)
    return xx, e_re

def eps_pp_general_layercake(xx, e_re):
    """
    General signed layer-cake / Clausen reconstruction of eps''(x), valid for ANY
    even drop density g(x) = -deps'/dx (not just the idealized rectangular
    plateau-minus-dip family of Sec 5.1). Uses
        h(s) = g(x=s),  s in [0,L]      (g restricted to x>=0; g is even)
        h(s) = h(L) + int_s^L (-h'(a)) da      (fundamental theorem of calculus)
    i.e. h is the superposition of centered-interval indicators 1_{|x|<=a}
    weighted by the signed measure dnu(a) = -h'(a) da (+ atom h(L) at a=L).
    Since eps''(x) is linear in g, and Sec 5.1 already gives the exact
    eps''(x) produced by a single indicator 1_{|x|<=a} in closed (Clausen) form,
    superposing gives:
        eps''(x) = (L/pi^2) * [ int_0^L M(pi*a/L, theta_x) * (-h'(a)) da
                                  + h(L) * M(pi, theta_x) ]
    This is an EXACT reformulation (Thm 3.1 dressed in Clausen form), not an
    approximation -- it costs the same O(N) as DiHiTI itself (FALLBACK-1 tier,
    not a cheap certificate), but it generalizes the closed form beyond the
    idealized rectangular plateau+dip shape assumed by the narrow d*(b) formula.
    """
    N = len(xx)
    half = xx >= 0
    s = xx[half]
    h = -np.gradient(e_re, xx)[half]     # g restricted to s=|x|>=0
    dh = np.gradient(h, s)
    theta_x = np.pi * xx / L
    alpha_a = np.pi * s / L

    out = np.zeros(N)
    for i, th in enumerate(theta_x):
        Mvals = M_kernel(alpha_a, th)
        integral = np.trapz(-dh * Mvals, s)
        atom = h[-1] * M_kernel(np.pi, th)
        out[i] = (L/np.pi**2) * (integral + atom)
    return out

# --- 1. Validate the general form against DiHiTI directly (exactness check) ---
print("=== Validating general Clausen layer-cake form vs DiHiTI ===")
N = 4001  # O(N) loop over theta with O(N) inner integral = O(N^2); keep modest
for alpha in [0.0, 0.6, 1.0, 1.5]:
    xx, e_re = build(alpha, N)
    ei_true = ht_d1(xx, e_re)
    ei_pred = eps_pp_general_layercake(xx, e_re)
    err = np.sqrt(np.mean((ei_true - ei_pred)**2))
    scale = np.max(np.abs(ei_true)) + 1e-12
    print(f"  alpha={alpha}: rel rms err = {err/scale:.3e}  "
          f"(min_true={ei_true.min():.5f}, min_pred={ei_pred.min():.5f})")
