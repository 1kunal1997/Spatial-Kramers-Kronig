import sys, numpy as np
from scipy.interpolate import CubicSpline
import mpmath as mp
mp.mp.dps = 40
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/theory")
import tmm_helper as tmm_h

NB, k = mp.mpf('1.7'), mp.mpf('4.0')
L = mp.mpf('2.5')
A = NB**2 - 1
def g(t):
    e = mp.e**(k*t)
    return A*k*e/(1+e)**2
def kernel_true(t, xpt, L):
    return mp.log(abs(mp.cos(mp.pi*t/(2*L)) / mp.sin(mp.pi*(xpt-t)/(2*L))))
def eps_im_true(xpt, L):
    xpt = mp.mpf(xpt)
    f = lambda t: g(t)*kernel_true(t, xpt, L)
    val = mp.quad(f, [-L, xpt, L])
    return val/mp.pi

test_x = [-1.5, -0.7, -0.2, 0.3, 0.9, 1.7]
truth = {xp: float(eps_im_true(xp, L)) for xp in test_x}

prev = None
for N in [513, 1025, 2049, 4097, 8193, 16385, 32769]:
    x = np.linspace(-2.5, 2.5, N)
    e_re = tmm_h.logistic(x, 4.0, 1.7)
    eim = tmm_h.ht_derivative(x, e_re)
    cs = CubicSpline(x, eim)
    errs = [abs(float(cs(xp)) - truth[xp]) for xp in test_x]
    m = max(errs)
    ratio = (prev/m) if prev else float('nan')
    print(f"N={N:6d}  max err DiHiTI(spline) = {m:.4e}  ratio={ratio:.2f}")
    prev = m
