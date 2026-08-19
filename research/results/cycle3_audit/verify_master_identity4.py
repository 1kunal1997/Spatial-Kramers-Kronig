import sys, numpy as np
import mpmath as mp
mp.mp.dps = 40
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/theory")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/research/theory")
import tmm_helper as tmm_h
from t3_clausen import gap_clausen

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

test_x = [-1.5, -0.7, -0.2, 0.0, 0.3, 0.9, 1.7]
truth = {xp: float(eps_im_true(xp, L)) for xp in test_x}

for N in [513, 1025, 2049, 4097, 8193, 16385]:
    x = np.linspace(-2.5, 2.5, N)
    e_re = tmm_h.logistic(x, 4.0, 1.7)
    dom, D = gap_clausen(e_re, x)
    row = []
    for xp in test_x:
        idx = np.argmin(np.abs(x-xp))
        row.append(dom[idx]-truth[xp])
    print(f"N={N:6d} " + " ".join(f"{v:+.2e}" for v in row))
